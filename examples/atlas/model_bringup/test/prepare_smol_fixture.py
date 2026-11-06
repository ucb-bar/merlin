"""Wire an authenticated archived selected-layer fixture into the current reader.

The adapter generates storage and descriptor metadata only. Every inherited byte
is independently authenticated by the archived contract and its trusted pin.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
TRUST_RECEIPT_SHA256 = "d5af5f9d31b5199ff092c5fc1bb042c680b11ef8c27d1dabdc1ebfcbd0001214"
CONTRACT_SHA256 = "f1dfaeaa32f43dbb9597556c3d44c41eabbc1e2a7752da5c783f9f07ff071d68"


def checked(path):
    path = Path(path).absolute()
    if ".." in path.parts or any(word in part.lower() for part in path.parts for word in ("hammer", "vlsi")):
        raise ValueError("unsafe path")
    for ancestor in reversed([path, *path.parents]):
        if ancestor.is_symlink():
            raise ValueError("symlink unsupported")
    return path


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--trust-receipt", type=Path, required=True)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    source, target, report = map(checked, (args.source, args.out, args.report))
    if target.exists():
        raise ValueError("fresh output required")
    sys.path[:0] = [str(checked(args.view / "tools")), str(checked(args.view / "tools/host_iss"))]
    from bundle import ChecksumBundle
    from block_bundle import BlockBundle
    from elf_image import Image
    from smol_contract import validate_bundle

    receipt = checked(args.trust_receipt)
    receipt_raw = receipt.read_bytes()
    if digest(receipt_raw) != TRUST_RECEIPT_SHA256:
        raise ValueError("trust receipt identity differs")
    source_contract = source / "host/smol-epoch3-contract.json"
    # Historical receipt paths identify records, not the relocated filesystem.
    contracts = [name for name, expected in json.loads(receipt_raw)["source_sha256"].items()
                 if Path(name).parts[-2:] == ("host", "smol-epoch3-contract.json")
                 and expected == CONTRACT_SHA256]
    if len(contracts) != 1:
        raise ValueError("contract trust receipt differs")
    trusted = CONTRACT_SHA256
    archive = ChecksumBundle(source)
    contract_raw = archive.read("host/smol-epoch3-contract.json")
    if digest(contract_raw) != trusted:
        raise ValueError("contract identity differs")
    value = json.loads(contract_raw)
    if value["schema"] != "atlas.smol_selected_layer1.epoch3_launch_data.v1":
        raise ValueError("source schema differs")
    products, origins = {}, {}

    def inherit(name, record):
        raw = archive.read(name)
        if digest(raw) != record["sha256"] or ("bytes" in record and len(raw) != record["bytes"]):
            raise ValueError("source record differs: " + name)
        products[name] = digest(raw)
        origins[name] = dict(sha256=digest(raw), bytes=len(raw))
        return name

    def locate(record):
        candidates = sorted(name for name, expected in archive.sums.items() if expected == record["sha256"])
        for name in candidates:
            raw = archive.read(name)
            if len(raw) == record["bytes"]:
                return inherit(name, record)
        raise ValueError("authenticated source payload missing")

    inherit("host/host.elf", value["ELF"])
    inherit("host/build.json", value["ELF_build"])
    inherit("host/smol-epoch3-contract.json", dict(sha256=trusted, bytes=len(contract_raw)))
    for name, record in value["assets"].items():
        inherit("data/" + name + ".bin", record["file"])
    kernels = []
    for index, item in enumerate(value["ordered_projections"]):
        if item["index"] != index:
            raise ValueError("projection order differs")
        row = dict(index=index, dram_base=value["startup"]["device_DRAM_base"],
                   output_offset=item["output_offset"], requested_dma=item["requested_dma"])
        for field, source_field in (("source", "program_S"), ("original", "original_program_bin"),
                                    ("initial", "initial_dram"), ("expected", "expected_physical_output")):
            row[field] = locate(item[source_field])
        kernels.append(row)
    if len(kernels) != 7:
        raise ValueError("projection count differs")
    provenance = dict(schema="atlas.block_bundle.v1", workload="smol", kernel_count=7,
                      kernels=kernels, contract="host/smol-epoch3-contract.json", contract_sha256=trusted)
    storage = dict(schema="atlas.asset_overlay.v1", base_root=str(source),
                   base_sha256sums_sha256=archive.checksums_sha256, inherit=sorted(products))
    target.mkdir(parents=True)
    for name, metadata in (("provenance.json", provenance), ("storage.json", storage)):
        raw = (json.dumps(metadata, sort_keys=True, indent=2) + "\n").encode()
        (target / name).write_bytes(raw)
        products[name] = digest(raw)
    (target / "SHA256SUMS").write_text("".join(h + "  " + n + "\n" for n, h in sorted(products.items())))
    recovered = BlockBundle(target, "smol")
    validate_bundle(recovered, Image(recovered.read("host/host.elf")))
    recovered.verify_bookends()
    archive.verify_bookends()
    if receipt.read_bytes() != receipt_raw:
        raise ValueError("trust receipt changed")
    files = {}
    for name in ("provenance.json", "SHA256SUMS", "host/build.json", "host/host.elf", "host/smol-epoch3-contract.json", "storage.json"):
        path = recovered.path(name)
        raw = path.read_bytes()
        files[name] = dict(path=str(path), sha256=digest(raw), bytes=len(raw))
    size = sum(recovered.path(name).stat().st_size for name in recovered.sums)
    result = dict(status="PASS_AUTHENTICATED_ADMISSION", environment_variable="SMOL_BLOCK_BUNDLE", path=str(target),
                  source_root=str(source), schema=provenance["schema"], workload="smol", kernel_count=7,
                  files=files, inherited_payloads=origins,
                  checksum_closure=dict(listed_files=len(products), listed_bytes=size,
                                        total_bytes_including_manifest=size + files["SHA256SUMS"]["bytes"],
                                        inherited_files=len(origins), symlinks=0),
                  authenticated_contract_binding=dict(receipt=str(receipt), receipt_sha256=digest(receipt_raw),
                                                      historical_contract=contracts[0],
                                                      source_contract=str(source_contract), sha256=trusted),
                  source_sha256sums_sha256=archive.checksums_sha256,
                  adapter=dict(path=str(checked(__file__)), sha256=digest(checked(__file__).read_bytes()),
                               claim="Metadata adapter only; no payload or build metadata was generated"),
                  scope="CPU admission and mutation controls only")
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(status=result["status"], path=str(target), closure=result["checksum_closure"])))


if __name__ == "__main__":
    main()
