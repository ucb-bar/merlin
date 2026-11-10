"""Actual pinned scalar passes, complete typed graph/effect/control evidence."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from merlin.common.paths import repo_root
from merlin.verify import lean_graph_state_scalar as checker

REFERENCE = Path("/home/jeremy/Projects/merlin-graph-proof-intake")


def redundant(width=8):
    return f"""builtin.module {{
  func.func @kernel(%a:i{width}, %b:i{width}) -> i{width} {{
    %first = arith.addi %a, %b : i{width}
    %duplicate = arith.addi %a, %b : i{width}
    %dead = arith.muli %a, %b : i{width}
    %chain = arith.subi %dead, %a : i{width}
    %result = arith.muli %first, %duplicate : i{width}
    func.return %result : i{width}
  }}
}}"""


def looping(lower=0, upper=7, stride=2, *, carries=1, observe=False, dynamic=False, redundant_body=False):
    extra = "%upper:index, " if dynamic else ""
    upper_definition = "" if dynamic else f"%upper = arith.constant {upper} : index"
    initial = "%acc=%a" if carries == 1 else "%acc=%a, %state=%b"
    types = "i8" if carries == 1 else "i8,i8"
    values = "%next" if carries == 1 else "%next, %other"
    results = "%result" if carries == 1 else "%result:2"
    returned = "%result" if carries == 1 else "%result#0"
    function_types = "i8" if carries == 1 else "i8,i8"
    returned = returned if carries == 1 else "%result#0, %result#1"
    second = "" if carries == 1 else "%other = arith.addi %state, %first : i8"
    observation = '"opaque.observe"(%first, %i) : (i8,index) -> ()' if observe else ""
    duplicate = "%dup = arith.addi %a, %b : i8\n    %dead = arith.subi %a, %b : i8" if redundant_body else ""
    operand = "%dup" if redundant_body else "%first"
    return f"""builtin.module {{
  func.func @kernel({extra}%a:i8, %b:i8) -> ({function_types}) {{
    %lower = arith.constant {lower} : index
    {upper_definition}
    %step = arith.constant {stride} : index
    {results} = scf.for %i = %lower to %upper step %step iter_args({initial}) -> ({types}) {{
      %first = arith.addi %a, %b : i8
      {duplicate}
      %product = arith.muli {operand}, %b : i8
      {observation}
      %next = arith.subi %acc, %product : i8
      {second}
      scf.yield {values} : {types}
    }}
    func.return {returned} : {function_types}
  }}
}}"""


CONSTANTS = """builtin.module {
  func.func @kernel(%a:i8) -> i16 {
    %c1 = arith.constant 17 : i8
    %c2 = arith.constant 17 : i8
    %v1 = arith.addi %a, %c1 : i8
    %v2 = arith.addi %a, %c2 : i8
    %wide = arith.extui %v2 : i8 to i32
    %signed = arith.extsi %v1 : i8 to i16
    %narrow = arith.trunci %wide : i32 to i16
    %result = arith.subi %signed, %narrow : i16
    func.return %result : i16
  }
}"""

OPAQUE = """builtin.module {
  func.func @kernel(%a:i8, %b:i8) -> i8 {
    %first = arith.addi %a, %b : i8
    %duplicate = arith.addi %a, %b : i8
    "opaque.observe"(%duplicate) : (i8) -> ()
    %read1 = "opaque.read"() : () -> i8
    %read2 = "opaque.read"() : () -> i8
    %partial = arith.divsi %a, %b : i8
    "opaque.observe"(%first, %read1, %read2) : (i8,i8,i8) -> ()
    func.return %first : i8
  }
}"""

MULTI = redundant().replace(
    "\n}",
    """
  func.func @untouched(%x:i8, %y:i8) -> i8 {
    %z = arith.subi %x, %y : i8
    "opaque.observe"(%z) : (i8) -> ()
    func.return %z : i8
  }
}""",
)

EXTERNAL = (
    redundant()
    .replace(
        "    func.return %result : i8",
        """
    %called = func.call @external(%result) : (i8) -> i8
    func.return %called : i8""",
    )
    .replace("\n}", "\n  func.func private @external(i8) -> i8\n}")
)

NESTED = """builtin.module {
  func.func @kernel(%a:i8,%b:i8) -> i8 {
    %l=arith.constant 0 : index
    %u=arith.constant 3 : index
    %s=arith.constant 1 : index
    %outer=scf.for %i=%l to %u step %s iter_args(%acc=%a) -> (i8) {
      %inner=scf.for %j=%l to %u step %s iter_args(%state=%acc) -> (i8) {
        %invariant=arith.muli %a,%b : i8
        %duplicate=arith.muli %a,%b : i8
        "opaque.observe"(%i,%j,%invariant) : (index,index,i8) -> ()
        %next=arith.addi %state,%duplicate : i8
        scf.yield %next : i8
      }
      scf.yield %inner : i8
    }
    func.return %outer : i8
  }
}"""

CASES = {
    **{f"cse-i{width}": ("cse", redundant(width)) for width in (1, 8, 16, 32, 128, 256)},
    "constants-casts": ("cse", CONSTANTS),
    "equivalent-constants": (
        "cse",
        CONSTANTS.replace("%c1 = arith.constant 17", "%c1 = arith.constant -1").replace(
            "%c2 = arith.constant 17", "%c2 = arith.constant 255"
        ),
    ),
    "opaque-effects": ("cse", OPAQUE),
    "separate-functions": ("cse", MULTI),
    "external-contract": ("cse", EXTERNAL),
    "licm-nonunit": ("licm", looping()),
    "licm-zero": ("licm", looping(upper=0)),
    "licm-negative-lower": ("licm", looping(lower=-3, upper=4)),
    "licm-two-carries": ("licm", looping(carries=2, observe=True)),
    "licm-dynamic-bound": ("licm", looping(dynamic=True, observe=True)),
    "chain": ("chain", looping(observe=True, redundant_body=True)),
    "zero-chain": ("chain", looping(upper=0, observe=True, redundant_body=True)),
    "nested-chain": ("chain", NESTED),
    "identity-flags": (
        "cse",
        """builtin.module {
      func.func @kernel(%a:i8, %b:i8) -> i8 {
        %x = arith.addi %a, %b overflow<nsw> : i8
        %y = arith.addi %a, %b overflow<nsw> : i8
        func.return %a : i8
      }
    }""",
    ),
}


def produce(root, text=None, mode="cse"):
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = root / "source.mlir"
    source.write_text(redundant() if text is None else text)
    destination = root / "actual"
    script = repo_root() / "src/merlin/verify/lean_graph_state_scalar_capture.py"
    command = [
        sys.executable,
        str(script),
        "--reference",
        str(REFERENCE),
        "--source",
        str(source),
        "--destination",
        str(destination),
        "--mode",
        mode,
    ]
    result = subprocess.run(
        command,
        cwd=REFERENCE,
        env={**os.environ, "PYTHONPATH": str(REFERENCE / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    (root / "producer.stdout").write_text(result.stdout)
    (root / "producer.stderr").write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    return source, destination / "after.mlir", json.loads((destination / "projection.json").read_text())


def checked(root, before, after):
    return checker.check_boundary(before, after, root / "checked", reference_root=REFERENCE)
