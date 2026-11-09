"""Conservative, machine-readable inventory of captured MLIR versus the SMT source subset.

This is a textual inventory, not a parser, a proof, or a certificate. It only claims a possible
source-side encoding when every visible op and tensor type fits the known subset; the real SMT
encoder remains the authority and may abstain on further value-level preconditions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .linalg_semantics import ENCODABLE_OPS

_ELEMENT = frozenset(("i8", "i16", "i32", "i64"))
_STRUCTURAL = frozenset(("builtin.module", "func.func"))


def _leading_op(line: str) -> str | None:
    """Read an MLIR statement's leading operation name, without parsing its body."""
    statement = line.lstrip()
    if statement.startswith("%"):
        lhs, separator, rhs = statement.partition("=")
        names = [part.strip() for part in lhs.split(",")]
        if not separator or not all(
            name.startswith("%") and len(name) > 1 and all(c.isalnum() or c == "_" for c in name[1:]) for name in names
        ):
            return None
        statement = rhs.lstrip()
    if statement.startswith('"'):
        name, separator, _ = statement[1:].partition('"')
        return name if separator and name and all(c.isalnum() or c in "_." for c in name) else None
    if not statement or not statement[0].isalpha():
        return None
    name = statement.split(maxsplit=1)[0].split("(", 1)[0]
    name = name.rstrip("{,:;")
    return name if name and all(c.isalnum() or c in "_." for c in name) else None


def _tensor_spellings(text: str) -> Counter[str]:
    """Count simple `tensor<...>` spellings; nested syntax is left unclassified."""
    found: Counter[str] = Counter()
    start = 0
    while (marker := text.find("tensor<", start)) >= 0:
        begin = marker + len("tensor<")
        end = text.find(">", begin)
        if end < 0:
            break
        spelling = text[begin:end]
        if "<" not in spelling:
            found[spelling] += 1
        start = begin if "<" in spelling else end + 1
    return found


def _tensor_signature(spelling: str) -> tuple[int, str, bool]:
    parts = spelling.split("x")
    dtype = parts[-1]
    shape = parts[:-1]
    return len(shape), dtype, any(not dimension.isdigit() or int(dimension) <= 0 for dimension in shape)


def audit_capture(path: str | Path) -> dict:
    """Inventory one exact captured file; unknown syntax always becomes an abstention reason."""
    source = Path(path)
    raw = source.read_bytes()
    text = raw.decode("utf-8")
    ops: Counter[str] = Counter()
    unparsed_lines = 0
    for line in text.splitlines():
        stripped = line.lstrip()
        if not stripped or stripped.startswith(("//", "#", "}")):
            continue
        name = _leading_op(line)
        if name:
            if name in _STRUCTURAL:
                continue
            # Affine region labels and MLIR block argument declarations are not operations.
            if name.startswith("^") or name.startswith("affine_map"):
                continue
            if "." in name:
                ops[name] += 1
        elif stripped.startswith("%") and " = " in stripped:
            unparsed_lines += 1

    tensor_spellings = _tensor_spellings(text)
    types: Counter[tuple[int, str, bool]] = Counter()
    for spelling, count in tensor_spellings.items():
        types[_tensor_signature(spelling)] += count
    ranks = sorted({rank for rank, _, _ in types})
    dtypes = sorted({dtype for _, dtype, _ in types})
    unsupported_ops = {name: count for name, count in sorted(ops.items()) if name not in ENCODABLE_OPS}
    unsupported_types = sorted(
        {
            f"rank={rank},dtype={dtype},dynamic={dynamic}"
            for rank, dtype, dynamic in types
            if rank != 2 or dtype not in _ELEMENT or dynamic
        }
    )
    blockers = []
    if unsupported_ops:
        blockers.append("unencoded source operations")
    if unsupported_types:
        blockers.append("tensor types outside rank-2 concrete integer domain")
    if unparsed_lines:
        blockers.append("unparsed assignment lines")
    if not ops:
        blockers.append("no operations inventoried")
    return {
        "schema": "merlin.capture_smt_coverage_inventory.v1",
        "method": "textual_inventory_not_proof",
        "capture": str(source),
        "capture_sha256": hashlib.sha256(raw).hexdigest(),
        "source_boundary": "linalg_to_interface",
        "eligible_for_whole_module_smt_attempt": not blockers,
        "blockers": blockers,
        "operation_counts": dict(sorted(ops.items())),
        "unsupported_operation_counts": unsupported_ops,
        "tensor_ranks_observed": ranks,
        "tensor_dtypes_observed": dtypes,
        "tensor_type_occurrence_counts": dict(sorted(tensor_spellings.items())),
        "unsupported_tensor_signatures": unsupported_types,
        "unparsed_assignment_lines": unparsed_lines,
        "candidate_supported_operation_count": sum(count for name, count in ops.items() if name in ENCODABLE_OPS),
        "observed_operation_count": sum(ops.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="inventory captured MLIR against the current SMT source subset")
    parser.add_argument("captures", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([audit_capture(path) for path in args.captures], indent=2, sort_keys=True))


# Preserve historical object lookup; implementation and completion state live here.
for _compat_definition in (
    _leading_op,
    _tensor_spellings,
    _tensor_signature,
    audit_capture,
    main,
):
    _compat_definition.__module__ = "merlin.verify.model_coverage"
del _compat_definition

if __name__ == "__main__":
    main()
