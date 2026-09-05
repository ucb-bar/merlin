"""Turn a solver counterexample into a witness the capsule bench can grade.

This is the point of the formal layer that a pass-rate cannot reach. When translation validation
refutes an obligation, z3 hands back a CONCRETE input at a concrete shape on which the compiled
program disagrees with its specification. Written out as a witness, that input rejoins the corpus and
is graded by the same ladder as every hand-derived one.

Why it matters beyond convenience: the corpus's default stimulus is degenerate (measured: an 8x8
activation has 64 elements and only 4 distinct values), so a defect that only shows on values outside
that set is invisible to a golden built from it. A counterexample witness carries the values that
actually break the program, so it cannot be degenerate by construction.

The witness is marked ``source_role: smt_counterexample`` — a first-class provenance, so nobody can
mistake a solver-generated shape for one an author chose.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Element dtype of the reference workload's operands and its accumulator.
_OPERAND_DTYPE = "i8"
_ACC_DTYPE = "i32"


@dataclass(frozen=True)
class CounterexampleTensor:
    """One input tensor recovered from a solver model."""
    name: str
    rows: int
    cols: int
    values: list[list[int]]


def parse_model(model_values: dict[str, Any]) -> dict[str, CounterexampleTensor]:
    """Recover input tensors from the solver's declared constants.

    Names are built by the encoder as ``<tensor>_<row>_<col>_<serial>``; they are split
    structurally, never pattern-matched — a too-narrow pattern silently dropping a constant would
    produce a partial input that looks complete, which is the failure this repo has a rule about.
    Anything that does not parse is REPORTED by raising, not skipped.
    """
    cells: dict[str, dict[tuple[int, int], int]] = {}
    for name, value in model_values.items():
        parts = name.split("_")
        if len(parts) < 4 or not isinstance(value, int):
            raise ValueError(f"unrecognised model constant {name!r}={value!r}; refusing to emit a "
                             "partial witness")
        tensor, row, col = parts[0], parts[1], parts[2]
        if not (row.isdigit() and col.isdigit()):
            raise ValueError(f"model constant {name!r} has no row/col index")
        cells.setdefault(tensor, {})[(int(row), int(col))] = value

    out: dict[str, CounterexampleTensor] = {}
    for tensor, elems in cells.items():
        rows = max(r for r, _ in elems) + 1
        cols = max(c for _, c in elems) + 1
        if len(elems) != rows * cols:
            raise ValueError(f"tensor {tensor!r} is missing elements: got {len(elems)} of "
                             f"{rows * cols}; a partial counterexample is not a witness")
        out[tensor] = CounterexampleTensor(
            tensor, rows, cols, [[elems[(r, c)] for c in range(cols)] for r in range(rows)])
    return out


def _interface_mlir(activations: list[str], weight: str, m: int, k: int, n: int) -> str:
    """The witness program in the frozen ``merlin_iface`` grammar."""
    lines = ['module attributes {merlin_iface.version = "0.1", merlin_iface.abi_version = "0.1"} {']
    lines.append(f'  %{weight} = merlin_iface.tensor {{name = "{weight}", role = "weight"}} '
                 f': tensor<{k}x{n}x{_OPERAND_DTYPE}>')
    for a in activations:
        lines.append(f'  %{a} = merlin_iface.tensor {{name = "{a}", role = "input"}} '
                     f': tensor<{m}x{k}x{_OPERAND_DTYPE}>')
    lines.append(f'  %{weight}_res = merlin_iface.resident_pack %{weight} '
                 f'{{layout = "packed_rhs"}} : (tensor<{k}x{n}x{_OPERAND_DTYPE}>) '
                 f'-> !merlin_iface.resident')
    for i, a in enumerate(activations):
        lines.append(f'  %acc{i} = merlin_iface.matmul %{a}, %{weight}_res '
                     f': (tensor<{m}x{k}x{_OPERAND_DTYPE}>, !merlin_iface.resident) '
                     f'-> !merlin_iface.acc<{_ACC_DTYPE}>')
        lines.append(f'  %Y{i} = merlin_iface.commit %acc{i} {{name = "Y{i}", epilogue = [], '
                     f'output_dtype = "{_ACC_DTYPE}"}} : (!merlin_iface.acc<{_ACC_DTYPE}>) '
                     f'-> tensor<{m}x{n}x{_ACC_DTYPE}>')
    lines.append(f'  merlin_iface.evict %{weight}_res : (!merlin_iface.resident) -> ()')
    lines.append("}")
    return "\n".join(lines) + "\n"


def emit_witness(verdict, *, name: str, dest: Path, obligation: str,
                 producing_pass: str) -> Path:
    """Write a witness directory reproducing a refuted obligation. Returns the directory."""
    if not getattr(verdict, "refuted", False):
        raise ValueError("only a refuted verdict carries a counterexample")
    tensors = parse_model(verdict.model_values)
    # Encoder convention: block arguments in order, trailing one is the reused weight.
    ordered = sorted(tensors)
    weight_key, activation_keys = ordered[-1], ordered[:-1]
    m, k = tensors[activation_keys[0]].rows, tensors[activation_keys[0]].cols
    n = tensors[weight_key].cols

    # Rename to the corpus's own leaf convention (W / A0 / A1 ...). Leaf tensors are materialized
    # deterministically BY NAME on both sides, so a witness carrying encoder-internal symbol names
    # would not line up with the golden.
    weight, activations = "W", [f"A{i}" for i in range(len(activation_keys))]
    values = {weight: tensors[weight_key].values}
    # NB: do not bind the loop variable to `name` here -- that is the witness's own name parameter,
    # and shadowing it silently wrote the witness into a directory named after the last activation.
    for leaf, key in zip(activations, activation_keys):
        values[leaf] = tensors[key].values

    dest = Path(dest) / name
    dest.mkdir(parents=True, exist_ok=True)

    capsule = {
        "name": name,
        "kind": "isa",
        "source_role": "smt_counterexample",
        "source_reference": (
            f"translation validation refuted {obligation!r} for {producing_pass!r}; shape and inputs "
            f"are the solver's counterexample, not an authored choice"),
        "label": "dev",
        "interface_mlir": "capsule.interface.mlir",
        "inputs": (
            [{"name": weight, "role": "weight", "shape": [k, n], "dtype": _OPERAND_DTYPE}]
            + [{"name": a, "role": "input", "shape": [m, k], "dtype": _OPERAND_DTYPE}
               for a in activations]),
        "operation": {"op": "matmul",
                      "attributes": {"lhs": activations[0], "weight": weight, "out": "Y0",
                                     "epilogue": [], "output_dtype": _ACC_DTYPE}},
        "numeric_policy": {"compare": "exact_int", "dtype": _ACC_DTYPE},
        "expected": {"instruction_classes": [], "modes": {}},
        "required_oracle_tiers": ["L0", "L1"],
    }

    from merlin.common.yaml import dump_yaml

    (dest / "capsule.yaml").write_text(dump_yaml(capsule), encoding="utf-8")
    (dest / "capsule.interface.mlir").write_text(
        _interface_mlir(activations, weight, m, k, n), encoding="utf-8")
    # The counterexample values themselves: the whole reason this witness is not degenerate.
    (dest / "counterexample_inputs.json").write_text(
        json.dumps(values, indent=1), encoding="utf-8")
    return dest
