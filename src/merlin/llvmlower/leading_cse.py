"""Run the pipeline's leading ``canonicalize`` and ``cse`` where pass validation can see them.

Every lowering pipeline opens with ``canonicalize,cse`` on tensors. When ``cse`` is validated,
:func:`run` executes those two passes here, out of process with the same pass registry and fusion
guard as the build's runner, submits the exact before/after pair of ``cse``, and returns the
result for the pipeline. The pipeline then runs both passes again; :func:`run` proves that changes
nothing by running them once more here, and submits any change, so what the build lowers is
exactly what was checked. ``canonicalize`` itself is not validated (its folds are outside the
Lean checker's domain).

This is exact only when the build's pipeline opens with ``canonicalize,cse`` and its runner does
nothing to the module before the pass manager. :func:`confirm` establishes both after lowering
from what the build actually ran (the recorded runner command and runner script), not from a list
of features: a feature that only edits later stages validates, and a build whose runner rewrites
the module first, or whose runner cannot be read that way, fails the check and its artifacts are
rejected.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from merlin.verify import pass_validation

from . import fusion_guard
from .toolchain import m2m_python

PASS = "cse"
#: The label of the check that the pipeline's own leading canonicalize,cse changes nothing more.
HEAD_RERUN = "cse:pipeline-head-rerun"

_SCRIPT = (
    "import sys\n"
    "from torch_mlir import ir\n"
    "from torch_mlir.passmanager import PassManager\n"
    "ctx = ir.Context()\n"
    "with open(sys.argv[1]) as f:\n"
    "    module = ir.Module.parse(f.read(), ctx)\n"
    "PassManager.parse('builtin.module(' + sys.argv[3] + ')', ctx).run(module.operation)\n"
    "with open(sys.argv[2], 'w') as f:\n"
    "    f.write(str(module.operation))\n"
)


def _apply(text: str, passes: str, directory: Path, name: str) -> str:
    source, result = directory / f"{name}.in.mlir", directory / f"{name}.mlir"
    source.write_text(text, encoding="utf-8")
    script = directory / "apply.py"
    if not script.is_file():
        script.write_text(fusion_guard.inject(_SCRIPT), encoding="utf-8")
    proc = subprocess.run(
        [str(m2m_python()), str(script), str(source), str(result), passes],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if proc.returncode != 0 or not result.is_file():
        raise RuntimeError(f"{passes} failed before the pipeline:\n{proc.stdout}\n{proc.stderr}")
    return result.read_text(encoding="utf-8")


def run(mlir_text: str, work: Path) -> str:
    """``mlir_text`` after the pipeline's leading ``canonicalize,cse`` when ``cse`` is validated,
    else ``mlir_text`` unchanged. :func:`confirm` must follow the lowering."""
    if not pass_validation.selected(PASS):
        return mlir_text
    directory = Path(work) / "leading-cse"
    directory.mkdir(parents=True, exist_ok=True)
    canonical = _apply(mlir_text, "canonicalize", directory, "canonicalize")
    after = _apply(canonical, PASS, directory, PASS)
    pass_validation.submit(PASS, canonical, after)
    # The pipeline reruns its head on `after`; anything that changes is checked too, and reported as
    # the head not being a fixpoint rather than as cse itself.
    rerun = _apply(after, "canonicalize,cse", directory, "rerun")
    pass_validation.submit(PASS, after, rerun, label=HEAD_RERUN)
    return after


def top_level_passes(pipeline: str) -> list[str]:
    """``pipeline`` split at its top-level commas (nested pass lists and options stay whole)."""
    passes, depth, current = [], 0, []
    for character in pipeline:
        if character in "({":
            depth += 1
        elif character in ")}":
            depth -= 1
        if character == "," and depth == 0:
            passes.append("".join(current))
            current = []
        else:
            current.append(character)
    passes.append("".join(current))
    return [p for p in passes if p]


def _argv_gate(script_lines: list[str], gate: str) -> int | None:
    """N for a gate the runner defines as ``GATE = len(sys.argv) > N and sys.argv[N] == '1'``."""
    for line in script_lines:
        name, equals, value = line.partition(" = ")
        if name != gate or not equals:
            continue
        head, _, tail = value.partition(" and ")
        bound = head.removeprefix("len(sys.argv) > ")
        if not bound.isdigit() or tail != f"sys.argv[{bound}] == '1'":
            return None
        return int(bound)
    return None


def pre_pass_rewrites(script: str, argv: list[str]) -> list[str]:
    """What the runner may do to the module between parsing it and its pass manager, given the
    script's own ``sys.argv``: each active gate, or the statement that cannot be read as a gate."""
    lines = script.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip().startswith("module = ir.Module.parse(")), None)
    stop = next((i for i, line in enumerate(lines) if line.startswith("_run_stages(")), None)
    if start is None or stop is None or stop < start:
        return ["the runner has no recognised parse ... _run_stages( region"]
    found = []
    for line in lines[start + 1 : stop]:
        if not line.strip() or line.startswith((" ", "#", "_write_view(")):
            continue  # gate bodies, comments, the read-only inspection view
        gate = line.removeprefix("if ").removesuffix(":")
        if not (line.startswith("if ") and line.endswith(":") and gate.isidentifier()):
            found.append(f"unrecognised statement {line.strip()!r}")
            continue
        index = _argv_gate(lines, gate)
        if index is None:
            found.append(f"gate {gate} is not an argv flag")
        elif len(argv) > index and argv[index] == "1":
            found.append(f"gate {gate} (argv[{index}]) is on")
    return found


def confirm(work: Path) -> None:
    """After lowering: the checked ``cse`` must be the one the build's own pipeline would have run.
    Raises :class:`~merlin.verify.speculative_gate.GateFailure` (the artifacts are then rejected)."""
    if not pass_validation.selected(PASS):
        return
    from merlin.verify.speculative_gate import GateFailure

    work = Path(work)
    problems: list[str] = []
    recipe = json.loads((work / "lowering_recipe.json").read_text(encoding="utf-8"))
    runs = [
        c["argv"] for c in recipe.get("commands", []) if len(c.get("argv", [])) > 3 and c["argv"][1].endswith(".py")
    ]
    if len(runs) != 1:
        problems.append(f"expected one recorded runner command, found {len(runs)}")
    else:
        argv = runs[0]
        if top_level_passes(argv[4])[:2] != ["canonicalize", "cse"]:
            problems.append(f"the pipeline does not open with canonicalize,cse: {argv[4][:120]}")
        problems += pre_pass_rewrites(Path(argv[1]).read_text(encoding="utf-8"), argv[1:])
    if problems:
        raise GateFailure(
            {
                "check": "tensor_refinement",
                "label": PASS,
                "verdict": "unsupported",
                "reason": "the checked cse is not the one this build's pipeline runs: " + "; ".join(problems),
                "directory": str(work / "leading-cse"),
            }
        )
