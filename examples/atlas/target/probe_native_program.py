"""Freeze one source-selected AtlasCore ARC/TileLink numerical observation.

The selected AtlasTile is sliced to the exact AtlasCore closure. ModeLIR owns
the TileLink driver; this example supplies only a finite program, independent
matrix golden, exact source/build inputs, and a source-deletion-safe replay.
This is not a whole-SoC, GSIM, or full-model qualification.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import runpy
import sys
import tempfile
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs_and_golden(program: dict) -> tuple[bytes, bytes, bytes]:
    """Independent golden for this explicit finite 32x32 E4M3 diagonal case.

    Only 0, +1 (0x38) and +2 (0x40) occur in the saved operands. We refuse a
    changed domain rather than pretending this integer-valued probe implements
    the full E4M3 numerical contract.
    """
    import numpy as np

    inputs = program["inputs"]
    if len(inputs) != 2 or [entry["base"] for entry in inputs] != [0, 1024]:
        raise ValueError("finite program input addresses changed")
    a, b = (base64.b64decode(entry["b64"], validate=True) for entry in inputs)
    if len(a) != 1024 or len(b) != 1024:
        raise ValueError("finite E4M3 matrix shape changed")
    a_values = np.frombuffer(a, dtype=np.uint8).reshape(32, 32)
    b_values = np.frombuffer(b, dtype=np.uint8).reshape(32, 32)
    if not np.isin(a_values, (0, 0x38)).all() or not np.isin(b_values, (0, 0x40)).all():
        raise ValueError("finite E4M3 operand domain changed")
    result = (a_values == 0x38).astype(np.int64) @ (2 * (b_values == 0x40).astype(np.int64))
    # BF16 round-to-nearest-even of exact integer-valued results, then tile
    # order C00, C01, C10, C11, each 16x16 row-major.
    f32_bits = result.astype(np.float32).view(np.uint32)
    rounded = ((f32_bits + 0x7FFF + ((f32_bits >> 16) & 1)) >> 16).astype("<u2")
    golden = b"".join(rounded[row : row + 16, col : col + 16].tobytes() for row in (0, 16) for col in (0, 16))
    output = program["output"]
    if output != {"base": 2048, "shape": [64, 16], "dtype": "torch.bfloat16"}:
        raise ValueError("finite program output ABI changed")
    if base64.b64decode(program["golden"]["b64"], validate=True) != golden:
        raise ValueError("saved program golden differs from independent matrix golden")
    return a, b, golden


def _run_native(
    support: Path, library: Path, state: Path, program: dict, a: bytes, b: bytes
) -> tuple[bytes, dict[str, int | bool]]:
    sys.path.insert(0, str(support))
    previous = Path.cwd()
    os.chdir(support)  # ModeLIR's legacy package bootstrap reads a local DIM cache.
    try:
        from mlc.backends.cosim_atlas import run_program

        result = run_program(
            library,
            state,
            program["words"],
            preload=[(0, a), (1024, b)],
            halt_signal="scalar/halt_now",
            max_cycles=20_000,
        )
        observed = result.slave.captured(2048, 2048)
        observations = {
            "halted": result.halted,
            "cycles": result.cycles,
            "dma_reads": result.reads,
            "dma_writes": result.writes,
            "halt_reason": result.halt_reason,
        }
    finally:
        os.chdir(previous)
        sys.path.remove(str(support))
    if not observations["halted"] or observations["dma_reads"] <= 0 or observations["dma_writes"] <= 0:
        raise ValueError("fresh AtlasCore did not complete compute and real DMA")
    return observed, observations


def _extract_exact(selected_hw: Path, extracted_hw: Path, extractor: Path) -> int:
    extract = runpy.run_path(str(extractor))["extract"]
    generated, included, missing = extract(selected_hw.read_text(), "AtlasCore")
    if missing or generated.encode() != extracted_hw.read_bytes():
        raise ValueError("compiled AtlasCore closure is not exact selected AtlasTile subtree")
    return len(included)


def _check_adapter_source(source: Path) -> dict[str, Path]:
    """Retain the authored semantics behind the explicit optimized-core ABI.

    The selected raw FIRRTL/manifest remains the build authority. These Scala
    files explain why the pruned public halt and D opcode have the chosen
    handling; their presence alone is not an elaboration certificate.
    """
    names = {
        "diplomatic/top/AtlasCore.scala": source / "diplomatic/top/AtlasCore.scala",
        "atlas/scalar/ScalarCore.scala": source / "atlas/scalar/ScalarCore.scala",
        "diplomatic/memory/DMA.scala": source / "diplomatic/memory/DMA.scala",
    }
    atlas = names["diplomatic/top/AtlasCore.scala"].read_text()
    scalar = names["atlas/scalar/ScalarCore.scala"].read_text()
    dma = names["diplomatic/memory/DMA.scala"].read_text()
    if "io.halted     := scalar.io.halted" not in atlas or "io.halted    := halt_now" not in scalar:
        raise ValueError("source halt chain no longer proves scalar/halt_now")
    if "val halt_now = halted || hostStop || illegal_detected || ecall_ebreak" not in scalar:
        raise ValueError("selected halt semantics changed")
    if any(f"io.tl.d.bits.{field}" in dma for field in ("opcode", "size")):
        raise ValueError("DMA now consumes D opcode/size; requalify adapter")
    if "io.tl.d.bits.data" not in dma or "io.tl.d.bits.source" not in dma:
        raise ValueError("DMA response source/data semantics changed")
    return names


def replay(bundle: Path) -> int:
    bundle = bundle.resolve(strict=True)
    receipt = runpy.run_path(str(bundle / "producer.py"))["verify"](bundle)
    selected_hw = bundle / "core.hw.mlir"
    extracted_hw = bundle / "arc/AtlasCore.hw.mlir"
    _extract_exact(selected_hw, extracted_hw, bundle / "extractor.py")
    _check_adapter_source(bundle / "source")
    program = json.loads((bundle / "case/program_bundle.json").read_text())
    a, b, golden = _inputs_and_golden(program)
    for name, content in (("a.bin", a), ("b.bin", b), ("golden.bin", golden)):
        if (bundle / "case" / name).read_bytes() != content:
            raise ValueError(f"frozen case {name} changed from independent input")
    observed, counts = _run_native(bundle / "support", bundle / "model.so", bundle / "state.json", program, a, b)
    if observed != golden or observed != (bundle / "case/observed.bin").read_bytes():
        raise ValueError("frozen AtlasCore numerical replay differs")
    if counts != receipt["case"]["observations"]:
        raise ValueError("frozen AtlasCore control/DMA observations differ")
    print(
        json.dumps(
            {
                "bundle": str(bundle),
                "replay": "passed",
                "elements": 1024,
                "output_sha256": hashlib.sha256(observed).hexdigest(),
                **counts,
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--replay-bundle":
        return replay(Path(sys.argv[2]))
    from merlin.experiments.phase2.native_engine_observation import seal, verify

    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "selection",
        "facts",
        "extracted",
        "allocated",
        "dce",
        "llvm_mlir",
        "llvm_ir",
        "state",
        "library",
        "program_bundle",
        "extractor",
        "arcilator",
        "circt_opt",
        "mlir_translate",
        "clang",
        "modelir",
        "atlas_source",
        "output",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    args = parser.parse_args()
    for name in vars(args):
        setattr(args, name, getattr(args, name).resolve(strict=name != "output"))
    selection = json.loads(args.selection.read_text())
    raw = Path(selection["sources"]["firrtl"]["path"])
    selected_hw = Path(selection["sources"]["core_hw"]["path"])
    if _sha(raw) != selection["sources"]["firrtl"]["sha256"]:
        raise ValueError("raw FIRRTL changed since source selection")
    if _sha(selected_hw) != selection["sources"]["core_hw"]["sha256"]:
        raise ValueError("selected AtlasTile HW changed since source selection")
    closure_modules = _extract_exact(selected_hw, args.extracted, args.extractor)
    source_proof = _check_adapter_source(args.atlas_source)
    program = json.loads(args.program_bundle.read_text())
    a, b, golden = _inputs_and_golden(program)
    observed, counts = _run_native(args.modelir, args.library, args.state, program, a, b)
    if observed != golden:
        raise ValueError("fresh native AtlasCore disagrees with independent BF16 golden")

    with tempfile.TemporaryDirectory(prefix="merlin-atlas-native-") as temporary:
        temp = Path(temporary)
        for name, value in (("a.bin", a), ("b.bin", b), ("golden.bin", golden), ("observed.bin", observed)):
            (temp / name).write_bytes(value)
        members = {
            "selection.json": args.selection,
            "facts.json": args.facts,
            "raw.fir": raw,
            "core.hw.mlir": selected_hw,
            "extractor.py": args.extractor,
            "arc/AtlasCore.hw.mlir": args.extracted,
            "arc/model.allocated.mlir": args.allocated,
            "arc/model.dce.mlir": args.dce,
            "arc/model.llvm.mlir": args.llvm_mlir,
            "arc/model.ll": args.llvm_ir,
            "state.json": args.state,
            "model.so": args.library,
            "case/program_bundle.json": args.program_bundle,
            "runner.py": Path(__file__),
            "tools/arcilator": args.arcilator,
            "tools/circt-opt": args.circt_opt,
            "tools/mlir-translate": args.mlir_translate,
            "tools/clang": args.clang,
            **{f"case/{name}": temp / name for name in ("a.bin", "b.bin", "golden.bin", "observed.bin")},
            **{f"source/{name}": path for name, path in source_proof.items()},
        }
        for source in sorted((args.modelir / "mlc").rglob("*.py")):
            members[f"support/{source.relative_to(args.modelir).as_posix()}"] = source
        cache = args.modelir / "runs/circt-arc/gemmini/outputs/discovered_interface.json"
        members["support/runs/circt-arc/gemmini/outputs/discovered_interface.json"] = cache
        for stage, stem in (
            ("extract", "extract"),
            ("state-alloc", "allocate"),
            ("symbol-dce", "dce"),
            ("arc-to-llvm", "llvm2"),
            ("llvm-translate", "translate2"),
            ("native-link", "compile2"),
        ):
            for stream in ("stdout", "stderr"):
                members[f"logs/{stage}.{stream}.log"] = args.extracted.parent / f"{stem}.{stream}.log"
        commands = [
            {
                "stage": "extract",
                "argv": [
                    "python",
                    "-P",
                    "-m",
                    "merlin.targetgen.rtl.extract_module",
                    str(selected_hw),
                    "--root",
                    "AtlasCore",
                    "--out",
                    str(args.extracted),
                ],
            },
            {
                "stage": "state-alloc",
                "argv": [
                    str(args.arcilator),
                    str(args.extracted),
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
            "shape": [64, 16],
            "dtype": "bfloat16",
            "element_bytes": 2,
            "compared_elements": 1024,
            "operation": "32x32 E4M3 finite diagonal matmul, BF16 2x2 tiles",
            "reference": "independent NumPy int64 matmul on exact 0/+1 and 0/+2 E4M3 subset; "
            "BF16 round-to-nearest-even and 16x16 tile packing",
            "observations": counts,
            "closure_modules": closure_modules,
        }
        receipt = seal(
            output=args.output,
            members=members,
            case=case,
            selection=selection,
            build_commands=commands,
            scope="selected AtlasRocketConfig AtlasCore inside AtlasTile; not AtlasTile/SoC or GSIM",
        )
    print(
        json.dumps(
            {
                "receipt": str(receipt),
                "verified_members": len(verify(receipt.parent)["members"]),
                "compared_elements": 1024,
                "closure_modules": closure_modules,
                "reference_sha256": _sha(receipt.parent / "case/golden.bin"),
                **counts,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
