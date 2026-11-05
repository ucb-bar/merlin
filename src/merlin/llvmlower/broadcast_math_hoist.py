"""Explicit source-semantic hoisting of broadcast reciprocal square roots.

Run after elementwise fusion and before bufferization. Exact affine projections
and scalar SSA dependencies establish a smaller, nonempty producer domain. No
external-call name grants purity, no arithmetic is reassociated, and tensor
value semantics leave physical alias/lifetime decisions to bufferization.
"""

from __future__ import annotations

from .host_tensor_rewrites import BROADCAST_SOURCE

FEATURE = "hoist_broadcast_source_rsqrt"
MARKER = "__merlin_broadcast_source_rsqrt_hoist__"


def _edit_pipeline(passes: list[str]) -> list[str]:
    hits = [i for i, stage in enumerate(passes) if "one-shot-bufferize" in stage]
    if len(hits) != 1 or MARKER in passes:
        raise ValueError("broadcast math hoist requires one bufferization stage")
    at = hits[0]
    return [*passes[:at], MARKER, *passes[at:]]


def ensure_registered() -> str:
    from .impr_features import ImprFeature, known, register

    if FEATURE not in known():
        register(
            ImprFeature(
                name=FEATURE,
                action_class="PASS",
                description=(
                    "Hoist typed pure source rsqrt dependency chains onto their exact "
                    "static nonempty broadcast domain after fusion; preserve every "
                    "operation, type and other SSA use. No strict FP/fenv permission."
                ),
                edit_pipeline=_edit_pipeline,
            )
        )
    return FEATURE


RUNNER_PRELUDE = (
    BROADCAST_SOURCE
    + r"""_BH_MARKER = "__merlin_broadcast_source_rsqrt_hoist__"
_BH_ORIG_RUN_STAGES = _run_stages

def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    if _BH_MARKER not in passes:
        return _BH_ORIG_RUN_STAGES(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    if passes.count(_BH_MARKER) != 1:
        raise ValueError("broadcast source math hoist marker must occur once")
    at = passes.index(_BH_MARKER)
    _BH_ORIG_RUN_STAGES(ctx, module, ','.join(passes[:at]), 0, (), (), (), pre_generalize)
    print('OK broadcast_source_rsqrt_hoist', _hoist_broadcast_source_math(ctx, module))
    _BH_ORIG_RUN_STAGES(ctx, module, ','.join(passes[at + 1:]), erase, mid, late, post_openmp, ())
"""
)


def apply_for_test(mlir_text: str) -> tuple[str, int]:
    """Execute the shipped typed rewrite through the selected upstream compiler."""
    import subprocess
    import tempfile
    from pathlib import Path

    from .toolchain import m2m_python

    with tempfile.TemporaryDirectory(prefix="merlin_broadcast_math_") as directory:
        root = Path(directory)
        source, script = root / "input.mlir", root / "run.py"
        source.write_text(mlir_text)
        script.write_text(
            "import sys\nfrom torch_mlir import ir\n"
            + RUNNER_PRELUDE.split("_BH_MARKER =", 1)[0]
            + "ctx=ir.Context()\nmodule=ir.Module.parse(open(sys.argv[1]).read(),ctx)\n"
            + "print('COUNT',_hoist_broadcast_source_math(ctx,module))\nprint('MODULE_BEGIN')\nprint(module)\n"
        )
        result = subprocess.run(
            [str(m2m_python()), str(script), str(source)], capture_output=True, text=True, timeout=120
        )
        if result.returncode:
            raise RuntimeError(f"broadcast source math hoist failed:\n{result.stdout}\n{result.stderr}")
        count = int(next(line.split()[1] for line in result.stdout.splitlines() if line.startswith("COUNT ")))
        return result.stdout.split("MODULE_BEGIN\n", 1)[1], count
