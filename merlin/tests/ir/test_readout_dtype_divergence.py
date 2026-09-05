"""Two engines define the same readout differently, and the contract does not say which is right.

`capsule_golden._apply_epilogue` ends with `_narrow_to_dtype(t, attrs.get("output_dtype", "i32"))` —
default **i32**, narrowing **any** integer width below 32. `runtime/simulator.py` and
`runtime/reference.py` both end their COMMIT with `if attrs.get("output_dtype", "i8") == "i8"` —
default **i8**, narrowing on an **exact** match only.

They agree whenever `output_dtype` is present and is `i8` or a width >= 32, which is every shipped
capsule. They disagree when it is ABSENT, or is `i16`/`i4`/`u8`.

**Why this matters and which way the error goes.** The golden is compared against
`reference_outputs(agent_cb)` at L0. A submission that omits `output_dtype` — legal, since nothing in
the schema or the ABI requires it — gets the reference's i8 clamp applied to a result the capsule
declared as i32, the two disagree, and the capsule fails with "your command buffer does not compute
the declared operation". The failure blames the agent for a disagreement between two harness engines,
and L1 cannot catch it because both of its sides apply the same rule. That is the
`harness-blames-the-agent` shape, with a `checks-that-skip-and-report-success` blind spot underneath.

**The contract is silent**, which is why this file pins rather than fixes. `command_buffer.schema.json`
lists `output_dtype` as an optional string with no default; `command_buffer_abi.yaml` describes its
values but not its absence, in a block that spells out `REQUIRED` / `no default` for the pooling
attributes right below it. Neither engine is wrong by the declared ABI, so choosing one here would be
inventing a target fact in shared code — the cardinal rule this repo gates. The fix is a contract
decision plus one shared narrow function, not a unilateral edit.

These tests do two things: keep the defect LATENT (every shipped commit must declare the attribute),
and pin the divergence so that whoever unifies the rule sees a red test rather than silence.
"""
from __future__ import annotations

import pytest


def _golden_rule_narrows(dtype: str | None, bits: int) -> bool:
    """capsule_golden: default i32, narrow any integer width below the accumulator."""
    return _bits_of(dtype if dtype is not None else "i32") not in (None, 0) and \
        _bits_of(dtype if dtype is not None else "i32") < bits


def _runtime_rule_narrows(dtype: str | None) -> bool:
    """simulator/reference COMMIT: default i8, narrow on an exact i8 match only."""
    return (dtype if dtype is not None else "i8") == "i8"


def _bits_of(dtype: str) -> int | None:
    for prefix in ("i", "u"):
        if dtype.startswith(prefix) and dtype[1:].isdigit():
            return int(dtype[1:])
    return None


def test_the_two_readout_rules_are_still_the_ones_this_file_describes():
    """Read the rules out of the source. If either moves, this file's premise must be re-checked."""
    import inspect

    from merlin.runtime import reference, simulator
    from merlin.targetgen import capsule_golden

    golden_src = inspect.getsource(capsule_golden._apply_epilogue)
    assert '_narrow_to_dtype(t, attrs.get("output_dtype", "i32"))' in golden_src, (
        "the golden's readout rule changed; re-derive this file's claims before trusting them")
    for mod in (simulator, reference):
        assert 'attrs.get("output_dtype", "i8") == "i8"' in inspect.getsource(mod), (
            f"{mod.__name__}'s COMMIT readout rule changed; re-derive this file's claims")


@pytest.mark.parametrize("dtype,agree", [
    (None, False),    # absent: golden keeps i32, runtime clamps to i8
    ("i16", False),   # golden narrows to i16, runtime does nothing
    ("i4", False),
    ("u8", False),
    ("i8", True),     # both clamp
    ("i32", True),    # neither narrows
])
def test_the_divergence_is_exactly_where_it_is_documented(dtype, agree):
    """Characterise it, so a change that widens the divergence is visible rather than absorbed."""
    acc_bits = 32
    g = _golden_rule_narrows(dtype, acc_bits)
    r = _runtime_rule_narrows(dtype)
    # "agree" here means the two rules take the same action, not that values coincide: for a value
    # already inside the narrower range both produce the same number regardless, which is why this
    # has never surfaced by accident.
    assert (g == r) is agree, (
        f"output_dtype={dtype!r}: golden narrows={g}, runtime narrows={r}; expected agree={agree}")


def test_every_shipped_capsule_declares_output_dtype_so_the_defect_stays_latent():
    """The guard that matters today.

    The divergence is unreachable while every shipped commit declares the attribute. A capsule added
    without it would make a live L0 failure reachable for a CORRECT backend, so this test is the
    tripwire — it fails on the capsule that would arm the defect, not months later on a submission.
    """
    from merlin.common.paths import merlin_dir

    root = merlin_dir() / "contract" / "capsules"
    if not root.is_dir():
        pytest.skip("no corpus tree in this checkout")

    offenders: list[str] = []
    total = 0
    for path in root.rglob("capsule.interface.mlir"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "merlin_iface.commit" not in line:
                continue
            total += 1
            if "output_dtype" not in line:
                offenders.append(f"{path.relative_to(root)}:{lineno}")
    assert total, "no commit ops found; this test would be vacuous"
    assert not offenders, (
        f"{len(offenders)} shipped commit op(s) omit output_dtype, which arms a live L0 failure for "
        f"a CORRECT backend (the golden keeps i32, the reference clamps to i8, and the capsule fails "
        f"with 'your command buffer does not compute the declared operation'): {offenders[:5]}")


def test_our_own_encoder_mirrors_the_runtime_rule_and_says_so():
    """This package must not become a sixth answer to the question.

    `cb_semantics` mirrors the simulator because the simulator is what the corpus grades against. That
    is a deliberate choice, not an accident, and it means the encoder inherits this divergence rather
    than fixing it — recorded here so nobody reads the encoder as an independent opinion.
    """
    import inspect

    from merlin.verify import cb_semantics

    src = inspect.getsource(cb_semantics)
    assert '_COMMIT_DEFAULT_DTYPE = "i8"' in src, "the encoder no longer mirrors the runtime default"
    assert "refute a correct backend" in src, (
        "the encoder must record WHY it mirrors the runtime rule rather than the golden's")
