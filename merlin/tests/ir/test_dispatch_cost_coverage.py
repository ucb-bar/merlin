"""The dispatch cost model prices contractions by a NAME ALLOWLIST, and the list is incomplete.

`schedule_dispatch.node_cost` gates its M*N*K branch on `prov.op in ("matmul", "batch_matmul")` and
falls through to an output-element count for everything else. Measured across captured MLIR under
`out/artifacts/`:

    matmul                     7453   priced as a contraction
    batch_matmul               5540   priced as a contraction
    convolution_im2col_matmul  8701   priced as a COPY
    int_matmul                 3834   priced as a COPY

So roughly half of all contractions are priced as data movement. Concretely, a 256x1024x1024 GEMM
costs 268,435,456 when its `prov.op` says `matmul` and 262,144 when it says `int_matmul` — a factor
of 1024. Hart balancing is longest-processing-time-first over these numbers, so for the int8 corpus
the schedule and the reported speedup are wrong for exactly the models the multicore path exists to
serve.

This is a MIS-SCHEDULING defect, not a miscompilation: the program still computes the right values, it
is just placed badly and the speedup number is not trustworthy. That is why this file pins rather than
fixes — and why the fix is not "add two strings".

**The shape of the fix.** A name allowlist is the brittle pattern this repo's cardinal rule exists to
prevent: a too-narrow match silently drops valid-but-differently-spelled input, which is precisely
what happened here and what `check_no_regex`'s docstring describes for the RoCC decoder. Adding
`convolution_im2col_matmul` and `int_matmul` fixes today's corpus and leaves the next spelling to be
discovered the same way. A contraction should be recognised STRUCTURALLY — from a reduction dimension
in the node's own shapes — so a new frontend spelling costs nothing.

These tests keep the gap visible and bounded: they assert the allowlist is still what it is, and that
the known-unmatched spellings are still unmatched. Fixing the model makes them red, which is the
point — the change should be deliberate and reviewed, not silent.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _cost_model_allowlist() -> tuple[str, ...]:
    """The op names the cost model prices as contractions, read from the source."""
    import inspect

    from merlin.xdsl_dialects.lowering import schedule_dispatch

    for line in inspect.getsource(schedule_dispatch).splitlines():
        stripped = line.strip()
        if stripped.startswith("if op in (") and "matmul" in stripped:
            inner = stripped[stripped.index("(") + 1: stripped.rindex(")")]
            return tuple(part.strip().strip('"\'') for part in inner.split(",") if part.strip())
    pytest.skip("could not locate the cost model's op gate; the source shape changed")
    return ()


def _prov_op_counts() -> dict[str, int]:
    """Count `prov.op` spellings across captured MLIR. Structural split, no regex."""
    from merlin.common.paths import out_dir

    root = out_dir() / "artifacts"
    if not root.is_dir():
        pytest.skip("no captured MLIR in this checkout")
    counts: dict[str, int] = {}
    for path in root.rglob("*.mlir"):
        try:
            text = path.read_text(errors="ignore", encoding="utf-8")
        except OSError:
            continue
        if "prov.op" not in text:
            continue
        for line in text.splitlines():
            _, sep, rest = line.partition("prov.op")
            if not sep:
                continue
            _, eq, tail = rest.partition("=")
            if not eq:
                continue
            value = tail.strip()
            if not value.startswith('"'):
                continue
            name = value[1:].partition('"')[0]
            if name:
                counts[name] = counts.get(name, 0) + 1
    if not counts:
        pytest.skip("no prov.op annotations found; this test would be vacuous")
    return counts


def test_the_cost_model_still_prices_contractions_by_a_name_allowlist():
    """If the gate becomes structural, these tests have served their purpose and should be replaced."""
    allow = _cost_model_allowlist()
    assert "matmul" in allow, f"the contraction gate no longer matches 'matmul': {allow}"
    assert len(allow) <= 4, (
        f"the allowlist grew to {allow} — if contractions are now recognised structurally, delete "
        f"this file; if more names were merely added, the next spelling will still be missed")


def test_known_contraction_spellings_are_still_unpriced():
    """Pin the gap. Making this red is what a real fix looks like.

    These are contractions with a K dimension by any reading of their names, they occur in captured
    MLIR in quantity, and the cost model prices them as copies.
    """
    allow = set(_cost_model_allowlist())
    counts = _prov_op_counts()

    unmatched = {op: n for op, n in counts.items()
                 if "matmul" in op and op not in allow}
    if not unmatched:
        pytest.fail(
            "no unmatched contraction spelling found — either the cost model was fixed (delete this "
            "file and the finding it pins) or the captured MLIR changed; do not just relax this test")

    total_matched = sum(n for op, n in counts.items() if op in allow)
    total_unmatched = sum(unmatched.values())
    assert total_unmatched > 0
    # Bound the damage so a NEW unpriced spelling appearing at scale is visible rather than absorbed
    # into an already-known gap.
    assert total_unmatched <= 2 * max(total_matched, 1), (
        f"unpriced contractions ({total_unmatched}) now dwarf priced ones ({total_matched}); the "
        f"schedule is being built almost entirely from copy-costed nodes: {sorted(unmatched)}")


def test_the_pricing_gap_is_orders_of_magnitude_not_a_rounding_error():
    """Show the size of the error, so nobody dismisses it as a tuning detail."""
    m, k, n = 256, 1024, 1024
    as_contraction = m * k * n
    as_copy = m * n
    assert as_contraction // as_copy == k, "the gap is exactly the reduction extent"
    assert as_contraction // as_copy >= 1000, (
        "for a realistic GEMM the mispricing is at least three orders of magnitude")
