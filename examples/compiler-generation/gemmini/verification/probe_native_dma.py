"""Freeze one source-selected Gemmini core ARC/DMA numerical observation.

This example uses ModeLIR's existing GemminiDmaDriver; it does not implement a
second DMA model or infer full Rocket/SoC support. Inputs are independently
seeded, the golden is NumPy int64 matmul clipped by the selected INT32→INT8
mvout-accumulator command, and every one of 256 DRAM output bytes is checked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import sys
import tempfile
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(bundle: Path) -> int:
    """Re-execute from frozen RTL library, driver source, and case bytes only."""
    bundle = bundle.resolve(strict=True)
    document = runpy.run_path(str(bundle / "producer.py"))["verify"](bundle)
    support = bundle / "support"
    sys.path.insert(0, str(support))
    os.chdir(support)
    import numpy as np
    from mlc.backends.gemmini_dma import GemminiDmaDriver

    dim = document["case"]["shape"][0]
    a = np.frombuffer((bundle / "case/a.bin").read_bytes(), dtype=np.int8).reshape(dim, dim)
    b = np.frombuffer((bundle / "case/b.bin").read_bytes(), dtype=np.int8).reshape(dim, dim)
    driver = GemminiDmaDriver(bundle / "model.so", bundle / "state.json")
    driver._dim = dim
    observed = driver.matmul_tile(a, b).astype(np.int8)
    golden = (bundle / "case/golden.bin").read_bytes()
    if observed.tobytes() != golden or observed.tobytes() != (bundle / "case/observed.bin").read_bytes():
        raise ValueError("frozen native RTL numerical replay differs")
    if driver.dram.reads != document["case"]["observations"]["dram_reads"]:
        raise ValueError("frozen DMA read count differs")
    if driver.dram.writes != document["case"]["observations"]["dram_writes"]:
        raise ValueError("frozen DMA write count differs")
    print(
        json.dumps(
            {
                "bundle": str(bundle),
                "replay": "passed",
                "elements": dim * dim,
                "dram_reads": driver.dram.reads,
                "dram_writes": driver.dram.writes,
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--replay-bundle":
        return replay(Path(sys.argv[2]))
    from merlin.development.backend.optimization.native_engine_observation import seal, verify

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--allocated", type=Path, required=True)
    parser.add_argument("--dce", type=Path, required=True)
    parser.add_argument("--llvm-mlir", type=Path, required=True)
    parser.add_argument("--llvm-ir", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--arcilator", type=Path, required=True)
    parser.add_argument("--circt-opt", type=Path, required=True)
    parser.add_argument("--mlir-translate", type=Path, required=True)
    parser.add_argument("--clang", type=Path, required=True)
    parser.add_argument("--modelir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "selection",
        "facts",
        "allocated",
        "dce",
        "llvm_mlir",
        "llvm_ir",
        "state",
        "library",
        "arcilator",
        "circt_opt",
        "mlir_translate",
        "clang",
        "modelir",
    ):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.output = args.output.resolve()

    selection = json.loads(args.selection.read_text())
    raw = Path(selection["sources"]["firrtl"]["path"])
    hw = Path(selection["sources"]["core_hw"]["path"])
    if _sha(raw) != selection["sources"]["firrtl"]["sha256"]:
        raise ValueError("raw FIRRTL changed since source selection")
    if _sha(hw) != selection["sources"]["core_hw"]["sha256"]:
        raise ValueError("core HW changed since source selection")
    facts = json.loads(args.facts.read_text())
    dimensions = [row for row in facts["facts"]["arrays"] if row["name"] == "mesh"]
    if len(dimensions) != 1 or dimensions[0]["rows"] != dimensions[0]["cols"]:
        raise ValueError("no unique square mesh dimension in selected source facts")
    dim = int(dimensions[0]["rows"])
    if dim != 16:
        raise ValueError("this finite DIM=16 observation must be reauthored for a changed mesh")

    # ModeLIR imports a cache at package import time. The actual driver DIM is
    # overridden from the freshly selected facts above; that old cache is not
    # used to infer any numerical or source claim for this run.
    support = args.modelir.resolve(strict=True)
    sys.path.insert(0, str(support))
    previous = Path.cwd()
    os.chdir(support)
    try:
        import numpy as np
        from mlc.backends.gemmini_dma import GemminiDmaDriver

        a = np.random.default_rng(1).integers(-4, 4, size=(dim, dim), dtype=np.int8)
        b = np.random.default_rng(101).integers(-4, 4, size=(dim, dim), dtype=np.int8)
        golden = np.clip(a.astype(np.int64) @ b.astype(np.int64), -128, 127).astype(np.int8)
        driver = GemminiDmaDriver(args.library, args.state)
        driver._dim = dim
        observed = driver.matmul_tile(a, b).astype(np.int8)
        reads, writes = driver.dram.reads, driver.dram.writes
    finally:
        os.chdir(previous)
    if observed.shape != golden.shape or not np.array_equal(observed, golden):
        raise ValueError("fresh native RTL DMA observation disagrees with independent INT8 golden")
    if reads <= 0 or writes <= 0:
        raise ValueError("matmul did not exercise real DMA read and write paths")

    with tempfile.TemporaryDirectory(prefix="merlin-native-dma-") as temporary:
        temp = Path(temporary)
        for name, value in (
            ("a.bin", a.tobytes()),
            ("b.bin", b.tobytes()),
            ("golden.bin", golden.tobytes()),
            ("observed.bin", observed.tobytes()),
        ):
            (temp / name).write_bytes(value)
        members = {
            "selection.json": args.selection,
            "facts.json": args.facts,
            "raw.fir": raw,
            "core.hw.mlir": hw,
            "arc/model.allocated.mlir": args.allocated,
            "arc/model.dce.mlir": args.dce,
            "arc/model.llvm.mlir": args.llvm_mlir,
            "arc/model.ll": args.llvm_ir,
            "state.json": args.state,
            "model.so": args.library,
            "tools/arcilator": args.arcilator,
            "tools/circt-opt": args.circt_opt,
            "tools/mlir-translate": args.mlir_translate,
            "tools/clang": args.clang,
            "runner.py": Path(__file__),
            **{f"case/{name}": temp / name for name in ("a.bin", "b.bin", "golden.bin", "observed.bin")},
        }
        for source in sorted((support / "mlc").rglob("*.py")):
            members[f"support/{source.relative_to(support).as_posix()}"] = source
        cache = support / "runs/circt-arc/gemmini/outputs/discovered_interface.json"
        members["support/runs/circt-arc/gemmini/outputs/discovered_interface.json"] = cache
        for label, artifact, stem in (
            ("state-alloc", args.allocated, "allocate"),
            ("symbol-dce", args.dce, "strip_unused_source"),
            ("arc-to-llvm", args.llvm_mlir, "llvm"),
            ("llvm-translate", args.llvm_ir, "translate2"),
            ("native-link", args.library, "compile2"),
        ):
            for stream in ("stdout", "stderr"):
                members[f"logs/{label}.{stream}.log"] = artifact.parent / f"{stem}.{stream}.log"
        commands = [
            {
                "stage": "state-alloc",
                "argv": [
                    str(args.arcilator),
                    str(hw),
                    "--observe-registers",
                    "--observe-memories",
                    "--observe-named-values",
                    f"--state-file={args.state}",
                    "--until-after=state-alloc",
                    "--emit-mlir",
                    "-o",
                    str(args.allocated),
                ],
            },
            {
                "stage": "symbol-dce",
                "argv": [str(args.circt_opt), str(args.allocated), "--symbol-dce", "-o", str(args.dce)],
            },
            {
                "stage": "arc-to-llvm",
                "argv": [
                    str(args.circt_opt),
                    str(args.dce),
                    "--hw-convert-bitcasts",
                    "--arc-lower-arrays",
                    "--lower-arc-to-llvm",
                    "--cse",
                    "--arc-canonicalizer",
                    "-o",
                    str(args.llvm_mlir),
                ],
            },
            {
                "stage": "llvm-translate",
                "argv": [str(args.mlir_translate), str(args.llvm_mlir), "--mlir-to-llvmir", "-o", str(args.llvm_ir)],
            },
            {
                "stage": "native-link",
                "argv": [
                    str(args.clang),
                    "-x",
                    "ir",
                    "-O0",
                    "-fPIC",
                    "-shared",
                    "-Wl,--no-undefined",
                    str(args.llvm_ir),
                    "-o",
                    str(args.library),
                ],
            },
        ]
        case = {
            "shape": [dim, dim],
            "dtype": "int8",
            "element_bytes": 1,
            "compared_elements": dim * dim,
            "operation": "signed INT8 mvin A/B; weight-stationary matmul; INT32 accumulator; "
            "CONFIG_STORE acc_scale=1.0 converts/clips INT32 to INT8 on mvout_acc",
            "reference": "NumPy int64 matmul followed by clip[-128,127] and int8 cast",
            "observations": {
                "dram_reads": reads,
                "dram_writes": writes,
                "all_elements_equal": True,
                "max_absolute_error": 0,
            },
        }
        receipt = seal(
            output=args.output,
            members=members,
            case=case,
            selection=selection,
            build_commands=commands,
            scope="selected GemminiRocketConfig Gemmini accelerator core; no Rocket SoC or i32 readout",
        )
    print(
        json.dumps(
            {
                "receipt": str(receipt),
                "verified_members": len(verify(receipt.parent)["members"]),
                "compared_elements": dim * dim,
                "dram_reads": reads,
                "dram_writes": writes,
                "reference_sha256": _sha(receipt.parent / "case/golden.bin"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
