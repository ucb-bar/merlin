"""Regenerate and replay supported Q/DQ preprocessing boundary receipts.

Run from the repository root with the explicit tool environment in README.md.
The fixture builders are the same ones used by the contract regression tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from merlin.common.paths import artifacts_dir, repo_root

ROOT = repo_root()
sys.path.insert(0, str(ROOT / "merlin/tests/ir"))
sys.path.insert(0, str(ROOT / "src"))

from test_lean_dequant_contract import PER_CHANNEL, SMALL, lowered, wrap  # noqa: E402
from test_lean_quantize_contract import source  # noqa: E402

from merlin.llvmlower.passes_xdsl import preprocess_text_textual  # noqa: E402
from merlin.verify import gate_checks  # noqa: E402
from merlin.verify.gate_errors import GateRejected  # noqa: E402
from merlin.verify.lean import check  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cases():
    channel = PER_CHANNEL.replace("ZERO", "i32").replace("RANGE", "quant_min = -128 : i64, quant_max = 127 : i64")
    return {"quantize": source(), "dequantize_tensor": wrap([SMALL]), "dequantize_channel": channel}


def tool_paths() -> tuple[Path, Path]:
    sdk = Path(os.environ["MERLIN_LEAN_SDK_BIN"]).resolve()
    # Keep the venv executable spelling: resolving its symlink selects the host Python.
    compiler = Path(os.environ["MERLIN_COMPILER_PYTHON"]).absolute()
    if not (sdk / "lake").is_file() or not compiler.is_file() or not check.lean_exe().is_file():
        raise RuntimeError("selected Lean SDK, compiler Python, or built checker is missing")
    return sdk, compiler


def generate(base: Path, sdk: Path, compiler: Path) -> dict:
    results = {}
    for name, program in cases().items():
        for route in ("xdsl", "textual"):
            if route == "xdsl":
                before, after = lowered(program)
            else:
                before = program
                after, stats = preprocess_text_textual(before)
                if stats["quant_ext_lowered"] + stats["dequantize_lowered"] != 1:
                    raise AssertionError(f"{name}/{route}: expected one Q/DQ rewrite, got {stats}")
            directory = base / f"{name}-{route}"
            directory.mkdir(parents=True, exist_ok=False)
            before_path, after_path = directory / "before.mlir", directory / "after.mlir"
            before_path.write_text(before)
            after_path.write_text(after)
            boundary = (
                "preprocess_text_textual.dequant_to_generic"
                if route == "textual" and stats["dequantize_lowered"]
                else "lower_quant_ext"
            )
            job = {
                "directory": str(directory),
                "label": boundary,
                "before": str(before_path),
                "after": str(after_path),
                "params": {"sdk_bin": str(sdk), "compiler_python": str(compiler), "index_bits": 64},
            }
            detail = gate_checks.tensor_refinement(job)
            chain = json.loads(Path(detail["chains"]["forward"]).read_text())
            if chain["refinement_verdict"] != "verified":
                raise AssertionError(f"{name}/{route}: {chain}")
            results[f"{name}-{route}"] = {
                "before_sha256": sha(before_path),
                "after_sha256": sha(after_path),
                "typed_receipt_sha256": sha(directory / "stage/receipt.json"),
                "support_receipt_sha256": sha(directory / "support-forward/receipt.json"),
                "claim": detail["claim"],
                "boundary": boundary,
                "strengthenings": detail["strengthenings"],
                "textual_stats": stats if route == "textual" else None,
                "refinement_verdict": chain["refinement_verdict"],
                "verdict": chain["verdict"],
            }
    return results


def replay(base: Path, compiler: Path) -> dict:
    results = {}
    for name in cases():
        for route in ("xdsl", "textual"):
            key = f"{name}-{route}"
            directory = base / key
            stage = directory / "stage"
            outcome = check.compose_chain(
                [stage],
                directory / "support-forward/receipt.json",
                "forward",
                directory / "fresh-replay.json",
                m2m_python=compiler,
                definedness=(directory / "defined-forward/receipt.json")
                if json.loads((directory / "defined-forward/receipt.json").read_text())["verdict"] == "verified"
                else None,
            )
            results[key] = {
                "before_sha256": sha(directory / "before.mlir"),
                "after_sha256": sha(directory / "after.mlir"),
                "refinement_verdict": outcome.get("refinement_verdict"),
                "verdict": outcome.get("verdict"),
                "reason": outcome.get("reason"),
            }
            if outcome.get("refinement_verdict") != "verified":
                raise AssertionError(f"{key}: fresh replay failed: {outcome}")
    return results


def mutate(base: Path, sdk: Path, compiler: Path) -> dict:
    """Keep a rejected actual xDSL quantizer mutation and its checker receipt."""
    original = base / "quantize-xdsl"
    directory = base / "mutation-quantize-mulf"
    directory.mkdir(parents=True, exist_ok=False)
    before = (original / "before.mlir").read_text()
    good = (original / "after.mlir").read_text()
    if good.count('"arith.mulf"') != 1:
        raise AssertionError("the tested quantizer no longer has one multiply")
    bad = good.replace('"arith.mulf"', '"arith.addf"', 1)
    (directory / "before.mlir").write_text(before)
    (directory / "after.mlir").write_text(bad)
    job = {
        "directory": str(directory),
        "label": "lower_quant_ext-mutated-multiply",
        "before": str(directory / "before.mlir"),
        "after": str(directory / "after.mlir"),
        "params": {"sdk_bin": str(sdk), "compiler_python": str(compiler), "index_bits": 64},
    }
    try:
        gate_checks.tensor_refinement(job)
    except GateRejected as exc:
        receipt = json.loads((directory / "stage/receipt.json").read_text())
        if receipt["verdict"] != "abstracted":
            raise AssertionError(f"mutation did not produce an explicit refusal: {receipt}") from exc
        return {
            "verdict": "rejected",
            "reason": str(exc),
            "before_sha256": sha(directory / "before.mlir"),
            "after_sha256": sha(directory / "after.mlir"),
            "receipt_sha256": sha(directory / "stage/receipt.json"),
        }
    raise AssertionError("mutated quantizer boundary was accepted")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("generate", "replay", "mutate"))
    parser.add_argument("--output", type=Path, default=artifacts_dir() / "verification/qdq-lowering")
    args = parser.parse_args()
    sdk, compiler = tool_paths()
    base = args.output.resolve()
    if args.mode == "generate":
        base.mkdir(parents=True, exist_ok=False)
        result = generate(base, sdk, compiler)
    elif args.mode == "replay":
        result = replay(base, compiler)
    else:
        result = mutate(base, sdk, compiler)
    (base / f"{args.mode}-summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
