"""Actual shared scheduling output, composed state proofs and valid-IR refusals."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from merlin.common.paths import repo_root
from merlin.verify import lean_graph_state as checker

REFERENCE = Path("/home/jeremy/Projects/merlin-graph-proof-intake")


def source(width=8, tail=False):
    extra = f"%tail=arith.addi %work,%result:i{width}" if tail else ""
    returned = "%tail" if tail else "%result"
    return f"""builtin.module {{
      func.func @kernel(%a:i{width},%b:i{width})->i{width} {{
        %first=arith.addi %a,%b:i{width}
        "state.config"(%a):(i{width})->()
        %work=arith.muli %a,%b:i{width}
        %seen="state.read"():()->i{width}
        %result=arith.addi %first,%seen:i{width}
        {extra}
        func.return {returned}:i{width}
      }}
      func.func @untouched(%arg:i{width})->i{width} {{func.return %arg:i{width}}}
    }}"""


def operation(name, inputs=(), outputs=(), reads=(), writes=(), observable=False):
    return {
        "name": name,
        "operand_types": list(inputs),
        "result_types": list(outputs),
        "reads": list(reads),
        "writes": list(writes),
        "semantic_sha256": hashlib.sha256(b"explicit original fixture total sequential frame model").hexdigest(),
        "deterministic_total": True,
        "complete_read_write_frames": True,
        "sequential_atomic_complete": True,
        "observable": observable,
    }


def contract(width=8):
    return {
        "operations": [
            operation("state.config", [f"i{width}"], writes=["configuration"]),
            operation("state.read", outputs=[f"i{width}"], reads=["configuration"]),
        ]
    }


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


def actual(root, width=8, tail=False, order=None):
    before = root / "source.mlir"
    before.write_text(source(width, tail))
    original_contract = write(root / "contract.json", contract(width))
    order = order or ([0, 2, 1, 3, 4, 5, 6] if tail else [0, 2, 1, 3, 4, 5])
    capture = root / "actual"
    command = [
        sys.executable,
        str(repo_root() / "src/merlin/verify/lean_graph_state_capture.py"),
        "--reference",
        str(REFERENCE),
        "--source",
        str(before),
        "--destination",
        str(capture),
        "--contract",
        str(original_contract),
        "--order",
        *map(str, order),
    ]
    p = subprocess.run(
        command,
        cwd=REFERENCE,
        env={**os.environ, "PYTHONPATH": str(REFERENCE / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=True,
    )
    (root / "producer.stdout").write_text(p.stdout)
    data = json.loads((capture / "capture.json").read_text())
    assert data["projection_checked"]
    return before, capture / "after.mlir", original_contract, data


def checked(root, before, after, original_contract, **kwargs):
    return checker.check_boundary(
        before, after, root / "checked", reference_root=REFERENCE, contract=original_contract, **kwargs
    )
