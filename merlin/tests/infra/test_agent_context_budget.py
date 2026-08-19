"""Context-window budgeting and diagnostic fidelity for the agentic drivers.

Both behaviours here were MEASURED defects on the gemmini arm-4 campaign (2026-08-19), and both made a
model's run unrecoverable in a way that looked like a model result:

  * opencode asked the provider for its registry ``limit.output`` (32_000) on every step, so the largest
    prompt nemotron could ever send was ``131_072 - 32_000 = 99_072`` -- 24% of its window reserved for a
    completion measured at 200-400 tokens. The round died on a provider 400 instead of compacting.
  * the verdict redactor scrubbed EVERY numeric literal, so ``emit_command_buffer rc=0`` reached the agent
    as ``rc=#``. 19 of 20 capsules received that 26-character non-message for six consecutive rounds.
"""
from __future__ import annotations

import sys

from merlin.common.paths import merlin_dir

_H = merlin_dir() / "experiments" / "capsule_bench" / "harness"
if str(_H) not in sys.path:
    sys.path.insert(0, str(_H))


def _oc():
    import opencode_agent
    return opencode_agent


def _qa():
    import qa_check
    return qa_check


# ---------------------------------------------------------------- window budgeting

def test_declared_output_limit_leaves_the_window_to_the_prompt():
    """A model with a known window must not reserve a quarter of it for output it never writes."""
    oc = _oc()
    mid = "amazon-bedrock/nvidia.nemotron-super-3-120b"
    override, _ = oc._window_config(mid)
    limit = override["amazon-bedrock"]["models"]["nvidia.nemotron-super-3-120b"]["limit"]
    assert limit["context"] == 131072, "window must be the value the provider itself enforces"
    # The measured completion is 200-400 tokens/step; the old reservation was 32_000.
    assert limit["output"] <= 8000
    usable = limit["context"] - limit["output"]
    assert usable > 99072, f"fix must RECOVER prompt budget, got {usable} vs the broken 99072"


def test_compaction_reserves_more_than_the_output_ask():
    """Compaction has to fire while a full completion still fits, or the provider refuses first."""
    oc = _oc()
    for mid in ("amazon-bedrock/nvidia.nemotron-super-3-120b", "amazon-bedrock/zai.glm-5"):
        override, compaction = oc._window_config(mid)
        out = override[mid.partition("/")[0]]["models"][mid.partition("/")[2]]["limit"]["output"]
        assert compaction["auto"] is True
        assert compaction["reserved"] >= out, "reserved buffer smaller than the output ask cannot help"
        assert compaction["prune"] is True, "superseded tool output is where the context actually goes"


def test_unknown_model_keeps_registry_defaults():
    """Never invent a window for a model we have not measured -- fail open to the registry."""
    oc = _oc()
    override, compaction = oc._window_config("amazon-bedrock/some.unmeasured-model")
    assert override == {}, "an unmeasured model must not get a fabricated context window"
    assert compaction["auto"] is True, "compaction config is model-independent and still applies"


def test_declared_windows_are_provider_measured():
    """Each window must correspond to a model we actually observed a 400 from."""
    oc = _oc()
    assert oc._CONTEXT_WINDOWS["nvidia.nemotron-super-3-120b"] == 131072
    assert oc._CONTEXT_WINDOWS["zai.glm-5"] == 202752


# ---------------------------------------------------------------- diagnostic fidelity

def test_return_code_survives_redaction():
    """rc=0 is the difference between 'crashed' and 'exited clean and wrote nothing'."""
    qa = _qa()
    assert qa._redact_detail("emit_command_buffer rc=0: ").startswith("emit_command_buffer rc=0")
    assert "rc=1" in qa._redact_detail("emit_command_buffer rc=1: ValueError: bad")
    assert "rc=-11" in qa._redact_detail("rc=-11 (SIGSEGV)")


def test_golden_values_are_still_scrubbed():
    """The carve-out must not widen into an answer leak."""
    qa = _qa()
    out = qa._redact_detail("expected 42 but got 17 at index 3")
    assert "42" not in out and "17" not in out
    assert out == "expected # but got # at index #"
    assert qa._redact_detail("mismatch_count=12 expected=3.14159") == "mismatch_count=# expected=#"


def test_scrub_is_structural_not_positional():
    """A number that merely follows the WORD rc (no '=') is not a return code and must be scrubbed."""
    qa = _qa()
    assert "#" in qa._redact_detail("rc 7 golden 7")
    # a key that only ENDS in an allowlisted name is not that key
    assert qa._redact_detail("src=5") == "src=#"


def test_actionable_hint_is_not_truncated():
    """The encoding-divergence hint's self-inspection METHOD lives past 240 chars."""
    qa = _qa()
    hint = ("on spike, your emitted artifact does not compute the declared operation "
            "The command-buffer tiers (numeric + trace) PASSED, so the divergence is in your "
            "emit_target_artifact hardware encoding - some field the command buffer cannot carry "
            "(a config scale, an accumulate/dataflow bit, a readout dtype, a DRAM address). "
            "Decode your OWN emitted artifact (the disassembler / instruction_trace.json) and check "
            "each op's operands against your intent.")
    out = qa._redact_detail(hint)
    assert "Decode your OWN emitted artifact" in out
