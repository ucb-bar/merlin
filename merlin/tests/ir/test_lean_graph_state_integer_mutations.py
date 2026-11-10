"""Complete actual-IR mutations; malformed inputs are a separate denominator."""

from __future__ import annotations

import json
import os
import sys

from test_lean_graph_state_integer import REFERENCE, chain_source, nested_source, produce, standalone_source

from merlin.common.owned_process import run_owned
from merlin.common.paths import repo_root

VALID = (
    "constant",
    "exposed-site",
    "operand",
    "operand-order",
    "opcode",
    "flags",
    "carry-init",
    "yield-order",
    "capture",
    "step",
    "bound",
    "callee",
    "drop-observer",
    "observer-arguments",
    "return-order",
    "return-value",
    "function-context",
    "signature-width",
    "function-header",
    "module-header",
)


def mutated(tmp_path, kind):
    text = (
        standalone_source()
        if kind in ("constant", "exposed-site")
        else chain_source()
        if kind in ("flags", "opcode")
        else nested_source()
    )
    capture = produce(tmp_path, text)
    buffers = [row["path"] for row in capture["buffers"]]
    target = tmp_path / "mutated.mlir"
    script = repo_root() / "src/merlin/verify/lean_graph_state_integer_mutate.py"
    with (tmp_path / "mutation.stdout").open("x") as out, (tmp_path / "mutation.stderr").open("x") as err:
        run = run_owned(
            [sys.executable, str(script), buffers[1], str(target), kind],
            cwd=REFERENCE,
            env={**os.environ, "PYTHONPATH": str(REFERENCE / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
            stdout=out,
            stderr=err,
            timeout=60,
            check=False,
        )
    assert run.returncode == 0, (tmp_path / "mutation.stderr").read_text()
    return [buffers[0], target], json.loads((tmp_path / "mutation.stdout").read_text())
