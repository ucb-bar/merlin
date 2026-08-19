"""The failure text an agent receives must survive redaction and truncation intact.

Two measured defects, both of which made a model look stubborn when it was simply blind:

  * the numeric scrub collapsed EVERY digit, so a parse error arrived as ``tensor<#x#xi#>`` -- the shape
    and element type, which ARE the diagnostic for a compiler task, destroyed to hide values that were
    never in the string;
  * the stderr excerpt kept only the tail, so a one-line parser error behind a long absolute path arrived
    as ``seError: /scratch/...`` -- exception type gone, path kept.

The leak guard these protect is real and stays: a bare number can echo a golden value and must scrub.
"""
import json
import sys
from pathlib import Path

from merlin.common.paths import repo_root
from merlin.targetgen.capsule_common import _stderr_excerpt

sys.path.insert(0, str(repo_root() / "merlin" / "experiments" / "capsule_bench" / "harness"))
import qa_check  # noqa: E402

scrub = qa_check._scrub_numbers


# --- what must SURVIVE: structure the agent already holds ------------------------------------------
def test_mlir_shape_and_dtype_survive():
    s = '%W = merlin_iface.tensor {name = "W"} : tensor<16x16xi8>'
    assert scrub(s) == s, "shape and element type are the diagnostic, not a golden value"


def test_other_mlir_types_survive():
    for s in ("memref<4x8xbf16>", "vector<32xf32>", "tensor<1x1x1xi32>"):
        assert scrub(s) == s, s


def test_capsule_name_and_tier_survive():
    assert scrub("A0_config_smoke failed at L2") == "A0_config_smoke failed at L2"


def test_source_location_survives():
    for s in ("input.interface.mlir:12:5", "mlir_oot/gemmini_opt.py:88:12: bad token"):
        assert scrub(s) == s, s


def test_return_code_survives():
    assert scrub("emit_command_buffer rc=0: ") == "emit_command_buffer rc=0: "


# --- what must still SCRUB: anything that could echo an answer -------------------------------------
def test_expected_and_actual_are_scrubbed():
    assert scrub("expected 42, actual 17") == "expected #, actual #"


def test_float_metric_is_scrubbed():
    assert scrub("cos 0.9997 below 0.999") == "cos # below #"


def test_value_arrays_are_scrubbed():
    assert scrub("golden[0]=255 got 254") == "golden[#]=# got #"
    assert scrub("[1, 2, 3]") == "[#, #, #]"


def test_a_colon_pair_that_is_not_a_path_is_scrubbed():
    """`3:4` after a bare word is a ratio, not a source location -- it must not ride the path carve-out."""
    assert scrub("ratio 3:4 exceeded") == "ratio #:# exceeded"
    assert scrub("at 10:30 the value 42") == "at #:# the value #"


def test_named_count_is_still_scrubbed():
    assert scrub("mismatch_count=7") == "mismatch_count=#"


# --- the stderr excerpt keeps both ends ------------------------------------------------------------
def test_short_stderr_is_untouched():
    assert _stderr_excerpt("boom") == "boom"


def test_long_stderr_keeps_the_head():
    err = "ParseError: something specific went wrong\n" + ("/very/long/path" * 200)
    out = _stderr_excerpt(err)
    assert out.startswith("ParseError: something specific"), "the exception type must survive"
    assert "elided" in out and len(out) < len(err)


def test_long_stderr_keeps_the_tail():
    err = ("filler line\n" * 200) + "FINAL: the real cause"
    assert _stderr_excerpt(err).endswith("FINAL: the real cause")


# --- multi-round token accounting -------------------------------------------------------------------
def test_result_events_accumulate_across_rounds(tmp_path):
    """A multi-round loop concatenates one transcript per round, so the file carries one ``result`` event
    per round. Taking the last one reported a 6-round claude run at 15% of its real tokens while tool
    calls counted every round -- the derived per-action metrics were nonsense."""
    from merlin.targetgen import experiment_tokens as ET

    def result(inp, out, cost):
        return json.dumps({"type": "result", "total_cost_usd": cost,
                           "modelUsage": {"m": {"inputTokens": inp, "outputTokens": out}}})

    p = tmp_path / "t.jsonl"
    p.write_text("\n".join([result(100, 10, 1.0), result(200, 20, 2.0), result(300, 30, 3.0)]))
    s = ET.parse_transcript(p, trust_cli_cost=True)
    assert s["usage_source"] == "result_event"
    assert s["tokens_input"] == 600, "input must be the sum of all three rounds"
    assert s["tokens_output"] == 60
    assert s["estimated_cost_usd"] == 6.0, "cost must sum too"


def test_single_result_event_is_unchanged(tmp_path):
    from merlin.targetgen import experiment_tokens as ET
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({"type": "result", "total_cost_usd": 1.5,
                             "modelUsage": {"m": {"inputTokens": 100, "outputTokens": 10}}}))
    s = ET.parse_transcript(p, trust_cli_cost=True)
    assert (s["tokens_input"], s["tokens_output"], s["estimated_cost_usd"]) == (100, 10, 1.5)


# ---------------------------------------------------------------- traceback positions

def test_a_traceback_line_number_survives():
    """MEASURED on a live re-run: after the rest of the scrub was fixed, every Python traceback the
    agent received still arrived as ``line #``, so it was told its own compiler raised but never where.
    A line number in a traceback is a position in the AGENT'S OWN source and cannot carry a golden."""
    s = 'parse rc=1:   File "/x/submission/mlir_oot/gemmini_opt.py", line 412\n    cmd = c.get("n")'
    assert "line 412" in qa_check._scrub_numbers(s)


def test_a_column_word_survives_too():
    assert "column 17" in qa_check._scrub_numbers("error at column 17 of the module")


def test_a_position_in_the_GOLDEN_still_scrubs():
    """The narrow word list is the point: a position in the golden output is a leak, not a diagnostic.
    ``row``/``index``/``offset`` are deliberately absent from _POSITION_WORDS."""
    for s in ("first mismatch at row 3", "mismatch at index 7", "diverges at offset 128"):
        assert "#" in qa_check._scrub_numbers(s), s


def test_values_still_scrub_next_to_a_bare_word():
    assert qa_check._scrub_numbers("expected 42, actual 17") == "expected #, actual #"
    assert qa_check._scrub_numbers("cos 0.9997") == "cos #"
