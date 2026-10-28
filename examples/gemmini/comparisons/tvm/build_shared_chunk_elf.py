"""Compile a frozen published chunk and link checked ELFs; never run a simulator.

The existing TVM package compiler and connected-chunk harness own all lowering,
linking and instruction audits. This comparison entry point binds the published
catalog and lets the remote worker supply its compiler and adapter paths.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

CATALOG_SHA256 = "839fad75688a9c2a87b975842f072f217a1f0fbc0ff3ba26e9b1707600e37c2a"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def selected_chunk(root, name):
    root = root.resolve(strict=True)
    catalog = root / "catalog.json"
    if digest(catalog) != CATALOG_SHA256:
        raise ValueError("Require the frozen 82b321a89 shared chunk catalog")
    matches = [row for row in json.loads(catalog.read_text())["chunks"] if row["id"] == name]
    if name == "full-resnet":
        # The same published dependency includes the connected parent and its
        # independent reference; this is not a sum or replay of chunk outputs.
        matches = [dict(id=name, status="parent_build_only",
                        program="resnet-release/full-resnet/program.json",
                        fixture="resnet-release/full-resnet/fixture.npz",
                        fixture_provenance="relocated-resnet-proofs/full-resnet.json")]
    if len(matches) != 1:
        raise ValueError("Unknown published chunk")
    row = matches[0]
    if name != "full-resnet" and (row["status"] != "ready_for_tuning" or row["host_check"]["host_check"] != "pass"):
        raise ValueError("Blocked chunk is not admitted for ELF preparation")
    paths = {}
    for key in ("program", "fixture", "fixture_provenance"):
        path = (root / row[key]).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("Chunk member escapes the restored root")
        paths[key] = path
    proof = json.loads(paths["fixture_provenance"].read_text())
    if digest(paths["program"]) != proof["program"]["sha256"] or digest(paths["fixture"]) != proof["fixture_sha256"]:
        raise ValueError("Frozen program or fixture changed")
    for source in proof["reference_sources"]:
        path = Path(source["path"]).resolve(strict=True)
        if not path.is_relative_to(root) or digest(path) != source["sha256"]:
            raise ValueError("Reference source changed or was not relocated")
    return row, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tvm-source", "restored-root", "output-dir", "riscv-gcc", "adapter-receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--chunk", required=True, help="Published chunk ID, or full-resnet for the connected parent")
    parser.add_argument("--host-math-policy", required=True,
                        choices=("native", "portable_exp_trig", "portable_neural_f32"))
    parser.add_argument("--graph-fusion-level", type=int, choices=range(4), default=2)
    parser.add_argument("--graph-cse", action="store_true")
    parser.add_argument("--no-prepack-weights", action="store_true")
    parser.add_argument("--no-bias-preload", action="store_true")
    parser.add_argument("--tuning-database", type=Path)
    parser.add_argument("--tensor-memory-bytes", type=int, default=64 << 20)
    args = parser.parse_args()
    _, paths = selected_chunk(args.restored_root, args.chunk)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    app = args.tvm_source.resolve(strict=True) / "apps/gemmini"
    sys.path.insert(0, str(app))
    import measure
    from chunk_measure import build_chunk

    class BuildOnlyToolchain(measure.Toolchain):
        def memory_configuration(self):
            return dict(kind="firesim_build_only", deployment_qualified=False,
                        scope="No simulator or deployment bound; remote worker verifies the selected image")

    toolchain = BuildOnlyToolchain(gcc=args.riscv_gcc.resolve(strict=True),
                                  adapter_receipt=args.adapter_receipt.resolve(strict=True),
                                  spike_receipt=None, hardware_receipt=None, timing_policy="fixed-warm")
    package = root / "package"
    command = [sys.executable, str(app / "compile_package.py"), "--program", str(paths["program"]),
               "--output-dir", str(package), "--riscv-gcc", str(toolchain.gcc),
               "--adapter-receipt", str(toolchain.adapter_receipt), "--host-math-policy", args.host_math_policy,
               "--graph-fusion-level", str(args.graph_fusion_level),
               "--tensor-memory-bytes", str(args.tensor_memory_bytes)]
    for name in ("graph_cse", "no_prepack_weights", "no_bias_preload"):
        if getattr(args, name):
            command.append("--" + name.replace("_", "-"))
    if args.tuning_database:
        command += ["--tuning-database", str(args.tuning_database.resolve(strict=True))]
    (root / "compile-command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (root / "compile.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    artifact = build_chunk(package, paths["fixture"], paths["fixture_provenance"], root / "guest", toolchain)
    artifact.update(build_entrypoint_sha256=digest(__file__), catalog_sha256=CATALOG_SHA256,
                    chunk=args.chunk, execution="NOT_RUN", full_model_protocol_admission="NOT_ESTABLISHED")
    (root / "guest/artifact.json").write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"valid": artifact["elfs"]["valid"], "bad_oracle": artifact["elfs"]["bad-oracle"]}, indent=2))


if __name__ == "__main__":
    main()
