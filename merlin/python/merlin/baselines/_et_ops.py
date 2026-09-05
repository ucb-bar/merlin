"""List the operators a ``.pte`` actually calls (runs UNDER the ExecuTorch venv).

An exported program names the kernels it needs in its own flatbuffer; the runtime binary registers
whichever kernel libraries it was LINKED against.  When those two sets disagree the runner aborts at
``Method::load`` with ``There are N instructions don't have corresponding operator registered`` --
a build-configuration fact that is only visible on the board unless something reads the ``.pte``
first.  This helper is that read: argv in, JSON out, so :mod:`.executorch` (in merlin's venv) can
decide WHICH kernel libraries the cross-compiled ``executor_runner`` has to link BEFORE it is built,
and report the operators no ExecuTorch kernel library provides at all as a named gap rather than as
an opaque board failure.

Counts are per INSTRUCTION, not per distinct operator, so the number here is directly comparable to
the ``N instructions`` the runtime reports.  Dependency-light on merlin, mirroring ``_et_export`` /
``_et_inspect``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# THIS SCRIPT SHADOWS THE PACKAGE IT NEEDS -- see the same note in ``_et_inspect``: this file lives
# beside merlin's own ``executorch.py``, and Python puts a script's own directory first on sys.path,
# so ``import executorch`` would resolve to that sibling. Drop our own directory first.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path
               if p and os.path.abspath(p) != _HERE] or [p for p in sys.path if p]


def operator_instruction_counts(pte: str) -> dict[str, int]:
    """``{"aten::mul.out": 4, ...}`` -- how many INSTRUCTIONS call each operator, across all methods.

    Delegate/jump/move/free instructions carry no operator and are not counted: a delegated node is
    executed by a backend (XNNPACK), never by a registered kernel, which is exactly why the count
    here is much smaller than the exported graph's node count.
    """
    from executorch.exir._serialize import _deserialize_pte_binary
    from executorch.exir.schema import KernelCall

    with open(pte, "rb") as fh:
        program = _deserialize_pte_binary(fh.read()).program

    counts: dict[str, int] = {}
    for plan in program.execution_plan:
        for chain in plan.chains:
            for instruction in chain.instructions:
                args = instruction.instr_args
                if not isinstance(args, KernelCall):
                    continue
                operator = plan.operators[args.op_index]
                # The runtime keys its registry on "<name>.<overload>" with the namespace already in
                # `name` (e.g. "aten::mul" + "out"). An empty overload is the default one.
                key = f"{operator.name}.{operator.overload}" if operator.overload else operator.name
                counts[key] = counts.get(key, 0) + 1
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pte", required=True, help="path to the exported .pte")
    ap.add_argument("--out", required=True, help="path to write the JSON operator census to")
    ns = ap.parse_args(argv)

    counts = operator_instruction_counts(ns.pte)
    payload = {"pte": os.path.abspath(ns.pte), "operators": counts,
               "n_kernel_instructions": sum(counts.values())}
    with open(ns.out, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
