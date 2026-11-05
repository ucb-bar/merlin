"""Bind prepared Tiny semantic state to one statically audited RV64 image."""
import hashlib
import json

from bundle import ChecksumBundle
from placement import placement
from tiny_segment_contract import layout


def prepare(root, expected_sha256):
    reference = ChecksumBundle(root)
    if reference.checksums_sha256 != expected_sha256:
        raise ValueError("native reference checksum manifest changed")
    data = json.loads(reference.read("boundaries.json"))
    if (data.get("schema") != "atlas.tiny_semantic_boundaries.v1"
            or data.get("qualification") != "REFERENCE_BOUNDARIES_ONLY" or data.get("elf_bound") is not False
            or data.get("byte_order") != "little" or data.get("kernel_count") != 840):
        raise ValueError("prepared unbound Tiny semantic boundaries required")
    return reference, data


def bind(out, reference, metadata, image, data, provenance, sums):
    regions, _ = layout(image, data)
    if {r["name"]: r["bytes"] for r in regions} != metadata["region_bytes"]:
        raise ValueError("native/RV64 semantic live inventory differs")
    for name in reference.sums:
        member = "segments/" + name
        destination = out / member
        destination.parent.mkdir(parents=True, exist_ok=True)
        raw = reference.read(name)
        destination.write_bytes(raw)
        sums[member] = hashlib.sha256(raw).hexdigest()
    model = provenance["model"]
    document = dict(schema="atlas.segment_chain.v1", workload="tiny-full", qualification="REFERENCE_BOUNDARIES_ONLY",
        elf="host/host.elf", elf_sha256=sums["host/host.elf"], contract="contract.json",
        contract_sha256=sums["contract.json"], capture_receipt="segments/receipt.json",
        capture_receipt_sha256=sums["segments/receipt.json"], kernel_count=840, placement=placement(provenance),
        model_profile=dict(revision=model["revision"], backend="curve", curve_sha256=model["curve_sha256"]),
        live_regions=regions, boundaries=[dict(index=b["index"], launch_cursor=b["launch_cursor"],
            regions={k: "segments/" + v for k, v in b["regions"].items()}) for b in metadata["boundaries"]],
        final_outputs=metadata["final_outputs"])
    raw = (json.dumps(document, indent=2) + "\n").encode()
    (out / "segments.json").write_bytes(raw)
    sums["segments.json"] = hashlib.sha256(raw).hexdigest()
    reference.verify_bookends()
    return sums["segments.json"]
