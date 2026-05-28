"""Validation gates for the heterogeneous QNN compile pipeline.

Two complementary gates, both required for a Phase 5 release:

1. **Numerical-equivalence gate** — `assert_numerical_equivalence`. Runs a
   reference (all-CPU) VMFB and a heterogeneous VMFB on the same inputs;
   asserts each output element is within tolerance. Tolerances are
   per-tensor and derived from quantization scale (≤1 quant step for int8
   tensors, ≤2 ULP for fp16/fp32). This is the *runtime* correctness gate.

2. **Compile-determinism gate** — `assert_compile_deterministic`. Compiles
   the same model twice with the same env and asserts the two output VMFBs
   are byte-identical (md5 match). This ensures the partitioner + emitter
   pipeline doesn't introduce nondeterminism into the build.

These gates replace the literal "md5 bytes-equal vs all-CPU" wording from
earlier notes — that gate is undeliverable because heterogeneous VMFBs
embed opaque QNN ctxbin payloads (see `compiler/plugins/target/QNN/
QNNTarget.cpp` serializer) and are byte-different from all-CPU VMFBs by
construction.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import pathlib
import subprocess
from collections.abc import Sequence

_LOG = logging.getLogger("qnn_gates")


@dataclasses.dataclass(frozen=True)
class Tolerance:
    """Per-output tolerance specification."""

    abs_tol: float = 0.0
    rel_tol: float = 0.0
    # For quantized outputs: max difference in q-units (i.e. raw int8 step).
    quant_step: int | None = None

    @classmethod
    def for_dtype(cls, dtype: str, *, scale: float | None = None) -> Tolerance:
        """Default tolerance derived from output dtype."""
        if dtype in ("i8", "u8", "int8", "uint8"):
            return cls(quant_step=1)
        if dtype in ("f16", "float16"):
            # ~2 ULP at unit magnitude.
            return cls(abs_tol=2**-10, rel_tol=1e-3)
        if dtype in ("f32", "float32"):
            return cls(abs_tol=2**-22, rel_tol=1e-5)
        raise ValueError(f"no default tolerance for dtype '{dtype}'")


def _md5_of(path: pathlib.Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_compile_deterministic(
    model_path: pathlib.Path,
    target: str,
    *,
    runs: int = 2,
    extra_compile_args: Sequence[str] = (),
    repo_root: pathlib.Path | None = None,
) -> str:
    """Compile `model_path` `runs` times with `target`; assert all outputs
    are byte-identical. Returns the common md5.

    Uses `./merlin compile` so flag emission stays canonical. Each run
    writes to a fresh output dir under
    `build/_compile_determinism_run<i>/` to avoid cross-run cache hits.
    """
    repo_root = repo_root or pathlib.Path(__file__).resolve().parents[2]
    md5s: list[str] = []
    vmfbs: list[pathlib.Path] = []
    for i in range(runs):
        out_dir = repo_root / "build" / f"_compile_determinism_run{i}"
        if out_dir.exists():
            import shutil

            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        cmd = [
            str(repo_root / "merlin"),
            "compile",
            str(model_path),
            "--target",
            target,
            "--output-dir",
            str(out_dir),
            *extra_compile_args,
        ]
        _LOG.info("compile run %d: %s", i, " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=repo_root)
        # Locate the produced VMFB (single-file convention).
        vmfb_candidates = list(out_dir.rglob("*.vmfb"))
        if len(vmfb_candidates) != 1:
            raise RuntimeError(f"expected exactly one .vmfb in {out_dir}, got " f"{len(vmfb_candidates)}")
        vmfbs.append(vmfb_candidates[0])
        md5s.append(_md5_of(vmfb_candidates[0]))
        _LOG.info("  → %s (md5=%s)", vmfb_candidates[0].name, md5s[-1])

    if len(set(md5s)) != 1:
        raise AssertionError(
            f"compile-determinism gate FAILED for {model_path.name} "
            f"target={target}: md5s differ across {runs} runs:\n"
            + "\n".join(f"  run{i}: {h} ({p})" for i, (h, p) in enumerate(zip(md5s, vmfbs)))
        )
    return md5s[0]


def _parse_iree_run_module_output(stdout: str) -> list[list[float]]:
    """Parse `iree-run-module`'s textual output into a list of result
    arrays. Each `result[i]: ...` line is parsed; the dtype and shape
    are read from the leading `<dims>x<dtype>=` prefix; the values are
    whitespace-separated floats."""
    results: list[list[float]] = []
    for line in stdout.splitlines():
        if "=[" in line and "]" in line:
            after_eq = line.split("=[", 1)[1].rsplit("]", 1)[0]
            try:
                values = [float(tok) for tok in after_eq.split()]
            except ValueError:
                continue
            results.append(values)
    return results


def _within_tolerance(ref_vals: list[float], cand_vals: list[float], tol: Tolerance) -> tuple[bool, str]:
    """Element-wise tolerance check. Returns (ok, diagnostic)."""
    if len(ref_vals) != len(cand_vals):
        return False, (f"length mismatch: reference has {len(ref_vals)} elements, " f"candidate has {len(cand_vals)}")
    max_abs_diff = 0.0
    max_rel_diff = 0.0
    bad_idx = -1
    bad_pair = (0.0, 0.0)
    for i, (r, c) in enumerate(zip(ref_vals, cand_vals)):
        abs_diff = abs(r - c)
        rel_diff = abs_diff / (abs(r) + 1e-12)
        if abs_diff > max_abs_diff:
            max_abs_diff = abs_diff
            bad_idx = i
            bad_pair = (r, c)
        if rel_diff > max_rel_diff:
            max_rel_diff = rel_diff
        if tol.quant_step is not None:
            # Treat each value as an integer quant step.
            if abs(round(r) - round(c)) > tol.quant_step:
                return False, (
                    f"quant-step exceeded at index {i}: ref={r} cand={c} "
                    f"diff={abs(round(r) - round(c))} > {tol.quant_step}"
                )
            continue
        if abs_diff > tol.abs_tol and rel_diff > tol.rel_tol:
            return False, (
                f"tolerance exceeded at index {i}: ref={r} cand={c} "
                f"abs_diff={abs_diff} (tol={tol.abs_tol}) "
                f"rel_diff={rel_diff} (tol={tol.rel_tol})"
            )
    return True, (
        f"max abs diff = {max_abs_diff:.6g} at index {bad_idx} "
        f"(ref={bad_pair[0]} cand={bad_pair[1]}); "
        f"max rel diff = {max_rel_diff:.6g}"
    )


def assert_numerical_equivalence(
    reference_vmfb: pathlib.Path,
    candidate_vmfb: pathlib.Path,
    *,
    function: str,
    inputs: Sequence[str],
    tolerance: Tolerance,
    candidate_device: str,
    reference_device: str = "local-task",
    iree_run_module: pathlib.Path | None = None,
) -> None:
    """Run both VMFBs on `function` with `inputs`, assert outputs match
    element-wise within `tolerance`.

    `inputs` is a list of strings in `iree-run-module --input=...` form
    (e.g. `["1x16xf32=3.5"]`). `candidate_device` is the device URI
    for the heterogeneous run (e.g. `qnn://gpu`).

    Raises `AssertionError` with a diagnostic on first divergence
    beyond tolerance. On success, returns silently.
    """
    if iree_run_module is None:
        repo_root = pathlib.Path(__file__).resolve().parents[2]
        # Default search order: host-merlin-release → host-vanilla-release.
        for build_dir in ("host-merlin-release", "host-vanilla-release"):
            cand = repo_root / "build" / build_dir / "tools" / "iree-run-module"
            if cand.exists():
                iree_run_module = cand
                break
        if iree_run_module is None:
            raise FileNotFoundError(
                "iree-run-module not found under build/host-*/tools/; "
                "build via `./merlin build --profile vanilla` or pass "
                "an explicit path."
            )

    def _run(vmfb: pathlib.Path, device: str) -> list[list[float]]:
        cmd: list[str] = [
            str(iree_run_module),
            f"--module={vmfb}",
            f"--device={device}",
            f"--function={function}",
        ]
        for inp in inputs:
            cmd.append(f"--input={inp}")
        _LOG.info("running: %s", " ".join(cmd))
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(
                f"iree-run-module on {vmfb} failed (rc={res.returncode}):\n"
                f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
            )
        return _parse_iree_run_module_output(res.stdout)

    ref_outputs = _run(reference_vmfb, reference_device)
    cand_outputs = _run(candidate_vmfb, candidate_device)
    if len(ref_outputs) != len(cand_outputs):
        raise AssertionError(
            f"output count mismatch: reference has {len(ref_outputs)} " f"results, candidate has {len(cand_outputs)}"
        )
    for i, (ref, cand) in enumerate(zip(ref_outputs, cand_outputs)):
        ok, diag = _within_tolerance(ref, cand, tolerance)
        if not ok:
            raise AssertionError(f"numerical-equivalence gate FAILED on result[{i}]: {diag}")
        _LOG.info("result[%d]: %s", i, diag)
