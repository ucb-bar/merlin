"""Saturn-vectors conformance battery — a NON-matmul (vector/SIMD) family.

This is the cross-family generalization test: does Merlin's command-buffer / reference
abstraction express a vector workload cleanly, without the matmul-resident shape
(RES_PACK/MATMUL_RESIDENT/COMMIT/EVICT)? The vector opcodes are:
  VECTOR_MAP  {lhs, rhs, dst} attrs{combine: add|mul, activation: [relu]}  -- elementwise
  VREDUCE     {src, dst}      attrs{op: sum}                               -- reduction
Outputs are tensors declared with role "output" (not produced by a COMMIT).

Rungs (all integer -> bit-exact certifiable):
  VEC0  y = x + b                 (elementwise add)
  VEC1  y = relu(x + b)           (elementwise add + activation)
  VEC2  s = sum(x * w)            (elementwise mul -> reduction; a contraction expressed in
                                   the VECTOR family, not as matmul — the real family test)
"""
from __future__ import annotations

from typing import Any

N = 64


def _cb(tensors: dict, commands: list) -> dict[str, Any]:
    return {"abi_version": "0.1", "target": "saturn_vec", "backend": "spike",
            "tensors": tensors, "commands": commands}


def vec0(n: int = N) -> dict:
    return _cb(
        {"x": {"shape": [n], "dtype": "i32", "role": "input"},
         "b": {"shape": [n], "dtype": "i32", "role": "input"},
         "y": {"shape": [n], "dtype": "i32", "role": "output"}},
        [{"opcode": "VECTOR_MAP", "operands": {"lhs": "x", "rhs": "b", "dst": "y"},
          "attributes": {"combine": "add", "activation": []}}])


def vec1(n: int = N) -> dict:
    return _cb(
        {"x": {"shape": [n], "dtype": "i32", "role": "input"},
         "b": {"shape": [n], "dtype": "i32", "role": "input"},
         "y": {"shape": [n], "dtype": "i32", "role": "output"}},
        [{"opcode": "VECTOR_MAP", "operands": {"lhs": "x", "rhs": "b", "dst": "y"},
          "attributes": {"combine": "add", "activation": ["relu"]}}])


def vec2(n: int = N) -> dict:
    return _cb(
        {"x": {"shape": [n], "dtype": "i32", "role": "input"},
         "w": {"shape": [n], "dtype": "i32", "role": "input"},
         "s": {"shape": [1], "dtype": "i32", "role": "output"}},
        [{"opcode": "VECTOR_MAP", "operands": {"lhs": "x", "rhs": "w", "dst": "t"},
          "attributes": {"combine": "mul", "activation": []}},
         {"opcode": "VREDUCE", "operands": {"src": "t", "dst": "s"},
          "attributes": {"op": "sum"}}])


RUNGS = {
    "VEC0": (vec0, "elementwise add"),
    "VEC1": (vec1, "elementwise add + relu"),
    "VEC2": (vec2, "elementwise mul -> reduce (dot product in the vector family)"),
}


def build(rung: str) -> dict[str, Any]:
    if rung not in RUNGS:
        raise KeyError(f"unknown vector rung {rung!r}; have {sorted(RUNGS)}")
    return RUNGS[rung][0]()
