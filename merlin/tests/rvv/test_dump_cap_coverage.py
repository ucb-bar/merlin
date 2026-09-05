"""How much of the model's answer a board run is allowed to print — and therefore graded on.

The K1 harness prints at most ``MERLIN_DUMP_CAP`` output elements, and the host gate compares that
printed prefix against the LEADING elements of the reference. A run whose console was truncated is
therefore scored on the truncation, and the resulting cos/rel is indistinguishable from a
whole-tensor verdict. Measured on the tracked int8 captures at the historical cap of 4096:
tiny_llama graded 4,096 of 256,000 elements (1.6%), deepjscc 4,096 of 12,288 (33%). tiny_llama's
``cos 0.8925`` was read as a codegen defect; it was token 0's logits alone, the one position W8A8
destroys for every implementation including torchao's own reference.

The fix is that ``dump_cap`` is reachable from the outside. These tests hold that:

* a caller CAN request full coverage (``dump_cap=None``), and the generated harness reflects it;
* the ceiling is DERIVED from the model's own ``MERLIN_OUT_ELEMS``, never a bigger host literal;
* the default is unchanged, so an existing caller's emitted C is byte-identical;
* a partial verdict cannot be mistaken for a complete one at the gate.
"""
from __future__ import annotations

import ast
import inspect
import struct
import sys

import numpy as np
import pytest

from merlin.common.paths import merlin_dir
from merlin.mining import k1
from merlin.runtime.backends import zephyr_model as zm

_CAP_DEFINE = "#define MERLIN_DUMP_CAP"

# The multi-program session harness is a separate, newer generator. Guard on its PRESENCE rather
# than assume it: the guard self-activates the moment the function lands, so this is not a check
# that can only pass. `_needs_session` reports which of the two it is at collection time.
_HAS_SESSION = hasattr(k1, "main_linux_session_c") and hasattr(k1, "build_k1_session_binary")
_needs_session = pytest.mark.skipif(
    not _HAS_SESSION, reason="k1 has no multi-program session harness in this revision")


def _cap_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.startswith(_CAP_DEFINE)]


# ---------------------------------------------------------------------------------------------
# 1. a caller can request full coverage, and it is DERIVED
# ---------------------------------------------------------------------------------------------

def test_default_cap_is_a_literal_ceiling() -> None:
    """The default still emits a fixed numeric ceiling — this is the state being escaped."""
    lines = _cap_lines(k1.main_linux_c())
    assert lines == [f"{_CAP_DEFINE} 4096"], lines


def test_full_coverage_is_requestable_and_derived() -> None:
    """``dump_cap=None`` must widen the cap to the model's OWN element count.

    Derived, not a bigger literal: a host-side "big enough" number is exactly the thing a larger
    model silently outgrows, and the overflow is invisible (it looks like a normal score).
    """
    lines = _cap_lines(k1.main_linux_c(dump_cap=None))
    assert lines == [f"{_CAP_DEFINE} MERLIN_OUT_ELEMS"], lines
    # and nothing numeric is left anywhere in the ceiling
    assert "4096" not in "\n".join(lines)


def test_full_coverage_changes_only_the_ceiling() -> None:
    """Requesting full output must not perturb the rest of the harness.

    Doubles as a moving frozen baseline: it stays true as the other harness work lands, while
    still failing the moment the uncapped path starts emitting different code.
    """
    capped = k1.main_linux_c(dump_cap=4096).splitlines()
    full = k1.main_linux_c(dump_cap=None).splitlines()
    assert len(capped) == len(full)
    differing = [(a, b) for a, b in zip(capped, full) if a != b]
    assert differing == [(f"{_CAP_DEFINE} 4096", f"{_CAP_DEFINE} MERLIN_OUT_ELEMS")], differing


def test_default_is_byte_identical_to_an_explicit_4096() -> None:
    """The frozen baseline: an existing caller's emitted C is unchanged by this parameter."""
    assert k1.main_linux_c() == k1.main_linux_c(dump_cap=4096)
    if _HAS_SESSION:
        assert k1.main_linux_session_c() == k1.main_linux_session_c(4096)


@_needs_session
def test_session_harness_drops_the_ceiling_too() -> None:
    """The session scheduler reports its element count at RUN time, so the uncapped form must
    remove the comparison rather than define a macro over a local variable."""
    capped = k1.main_linux_session_c(4096)
    full = k1.main_linux_session_c(None)
    assert _cap_lines(capped) == [f"{_CAP_DEFINE} 4096"]
    assert _cap_lines(full) == []
    assert "  int k = elems < MERLIN_DUMP_CAP ? (int)elems : MERLIN_DUMP_CAP;" in capped
    assert "  int k = (int)elems;" in full
    assert "MERLIN_DUMP_CAP" not in full


@pytest.mark.parametrize("bad", [0, -1])
def test_a_non_positive_cap_is_refused(bad: int) -> None:
    """A zero cap prints an empty OUT line that parses cleanly and grades as zero coverage —
    fail closed instead of emitting a harness that silently measures nothing."""
    with pytest.raises(ValueError):
        k1.main_linux_c(dump_cap=bad)
    if _HAS_SESSION:
        with pytest.raises(ValueError):
            k1.main_linux_session_c(bad)


# ---------------------------------------------------------------------------------------------
# 2. the parameter is actually THREADED — reachable from run_on_k1, not just from the generator
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "func",
    [k1.run_on_k1, k1.build_k1_binary, k1.main_linux_c]
    + ([k1.build_k1_session_binary, k1.main_linux_session_c] if _HAS_SESSION else []),
    ids=lambda f: f.__name__,
)
def test_dump_cap_is_a_parameter_with_the_historical_default(func) -> None:
    param = inspect.signature(func).parameters.get("dump_cap")
    assert param is not None, f"{func.__name__} does not accept dump_cap"
    assert param.default == 4096, f"{func.__name__} changed the default cap"


def _call_keywords(func, callee: str) -> set[str]:
    """Keyword names passed to ``callee`` from ``func``'s body, read structurally from the AST.

    A signature check alone passes on a parameter that is accepted and then dropped on the floor —
    which is the defect being fixed here (``run_on_k1`` reached ``main_linux_c`` without one).
    """
    tree = ast.parse(inspect.getsource(func))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
        if name != callee:
            continue
        found.update(kw.arg for kw in node.keywords if kw.arg)
        # a positional argument counts as threaded too, for the single-parameter generator
        if node.args:
            found.add("<positional>")
    return found


def test_build_k1_binary_forwards_dump_cap_to_the_harness() -> None:
    assert "dump_cap" in _call_keywords(k1.build_k1_binary, "main_linux_c")


@_needs_session
def test_build_k1_session_binary_forwards_dump_cap_to_the_harness() -> None:
    kws = _call_keywords(k1.build_k1_session_binary, "main_linux_session_c")
    assert "dump_cap" in kws or "<positional>" in kws


def test_run_on_k1_forwards_dump_cap_to_every_builder_it_calls() -> None:
    assert "dump_cap" in _call_keywords(k1.run_on_k1, "build_k1_binary")
    if _HAS_SESSION:
        assert "dump_cap" in _call_keywords(k1.run_on_k1, "build_k1_session_binary")


def test_no_caller_in_the_tree_hard_wires_the_generator_without_a_cap() -> None:
    """Every ``main_linux_c`` call site in library code must be reachable-from-outside.

    The defect was one unparameterised call; this catches the next one.
    """
    root = merlin_dir() / "python" / "merlin"
    offenders = []
    for path in root.rglob("*.py"):
        if path.name == "k1.py" and path.parent.name == "mining":
            continue  # the definition site, checked above
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
                if name in ("main_linux_c", "main_linux_session_c"):
                    kws = {kw.arg for kw in node.keywords}
                    if "dump_cap" not in kws and not node.args:
                        offenders.append(str(path))
    assert offenders == [], offenders


# ---------------------------------------------------------------------------------------------
# 3. a partial verdict cannot be mistaken for a complete one
# ---------------------------------------------------------------------------------------------

def _console(values: np.ndarray) -> str:
    """The exact OUT/METRIC/DONE console the generated harness prints for ``values``."""
    bits = np.ascontiguousarray(values, dtype="<f4").view(np.uint32)
    body = " ".join(str(int(b)) for b in bits)
    return f"OUT {bits.size} {body}\nMETRIC cycles 1\nDONE\n"


def test_truncated_and_complete_verdicts_are_distinguishable_at_the_gate() -> None:
    rng = np.random.default_rng(0)
    reference = rng.standard_normal(256_000).astype(np.float32)
    board = (reference * 1.001).astype(np.float32)

    truncated = zm._parse_console(_console(board[:4096]), 0)
    complete = zm._parse_console(_console(board), 0)

    g_short = zm._gate(truncated["outputs"], {"fp32": reference})
    g_full = zm._gate(complete["outputs"], {"fp32": reference})

    assert g_short["n_compared"] == 4096
    assert g_short["n_reference"] == 256_000
    assert g_short["comparison_complete"] is False
    assert g_short["compared_fraction"] == pytest.approx(4096 / 256_000)

    assert g_full["n_compared"] == 256_000
    assert g_full["comparison_complete"] is True
    assert g_full["compared_fraction"] == pytest.approx(1.0)


def test_a_declared_coverage_floor_vetoes_the_truncated_verdict() -> None:
    """With the cap reachable, a caller that needs a whole-output verdict can now demand one and
    get a refusal instead of a prefix score that looks complete."""
    rng = np.random.default_rng(1)
    reference = rng.standard_normal(256_000).astype(np.float32)
    board = (reference * 1.0001).astype(np.float32)

    short = zm._gate(zm._parse_console(_console(board[:4096]), 0)["outputs"],
                     {"fp32": reference}, min_coverage=1.0)
    full = zm._gate(zm._parse_console(_console(board), 0)["outputs"],
                    {"fp32": reference}, min_coverage=1.0)
    assert short["coverage_ok"] is False and short["ok"] is False
    assert full["coverage_ok"] is True and full["ok"] is True


# ---------------------------------------------------------------------------------------------
# 4. the transport survives the larger dump
# ---------------------------------------------------------------------------------------------

def test_a_full_lm_dump_round_trips_bit_exactly_through_the_console() -> None:
    """256,000 elements is tiny_llama's real output size: ~2.8 MB on ONE console line.

    ``_parse_console`` finds the OUT line by prefix and splits it whole, so the only way this can
    fail is a genuine transport/parse limit. It does not — verified bit-exactly, not by cosine.
    """
    rng = np.random.default_rng(2)
    values = rng.standard_normal(256_000).astype(np.float32)
    console = _console(values)
    assert len(console) > 2_500_000, len(console)
    parsed = zm._parse_console(console, 0)["outputs"]
    assert parsed.size == values.size
    assert np.array_equal(parsed.view(np.uint32), values.view(np.uint32))


def test_parsed_bits_match_the_struct_encoding_the_harness_uses() -> None:
    """The harness memcpy's each f32 to a uint32 and prints it decimal; guard that convention."""
    values = np.array([0.0, -0.0, 1.5, -3.25, np.float32(1e-38)], dtype=np.float32)
    parsed = zm._parse_console(_console(values), 0)["outputs"]
    for got, want in zip(parsed, values):
        assert struct.pack("<f", got) == struct.pack("<f", want)


def test_module_imports_without_a_board() -> None:
    """These tests must be host-only: nothing here may require the K1 to be reachable."""
    assert "merlin.mining.k1" in sys.modules
