"""A multi-line ``python3 -c`` probe must be visible to the arm-4 evidence extractor.

``-c`` takes its source as ONE argument, and a shell quotes a multi-line source as a single token that
SPANS lines. ``_python_fragments`` read the command line by line, so the first line was ``python3 -c "``
-- an unbalanced quote that tokenizes to nothing -- and the fragment, with every call name in it,
disappeared silently.

MEASURED 2026-09-05 on gemmini arm-4 ``merlincirct_g4p1_20260905``: the agent selected ``rtl_checks``
out of ``qa/verdict.json`` and called ``derived_levers`` and ``load_facts`` from multi-line probes, and
four RTL conformance checks reported False for all six rounds. The SINGLE-line spelling parsed
correctly throughout, which is why the earlier fix for this same check family tested green and left
this open -- so these tests pin the multi-line spelling specifically.
"""
from __future__ import annotations

import sys

import pytest

from merlin.common.paths import repo_root

sys.path.insert(0, str(repo_root() / "merlin" / "experiments" / "capsule_bench" / "harness"))
import conformance as C  # noqa: E402


def _call(command: str, result: str = "") -> C.ToolCall:
    return C.ToolCall(name="Bash", input={"command": command}, tool_use_id="t",
                      result_present=True, succeeded=True, result_text=result)


SINGLE = 'python3 -c "import json; v=json.load(open(1)); print(v)"'
MULTI = 'python3 -c "\nimport json\nv=json.load(open(1))\nprint(v)\n"'


class TestTheFragmentIsSeenEitherWay:
    def test_the_single_line_spelling_still_parses(self):
        """Regression guard: this spelling always worked and must keep working."""
        assert C._python_fragments(_call(SINGLE))
        assert "open" in C._python_call_names(_call(SINGLE))

    def test_the_multi_line_spelling_parses_too(self):
        """The defect: this yielded NO fragment and therefore no call names at all."""
        assert C._python_fragments(_call(MULTI))
        assert "open" in C._python_call_names(_call(MULTI))

    def test_both_spellings_find_the_same_calls(self):
        assert set(C._python_call_names(_call(SINGLE))) == set(C._python_call_names(_call(MULTI)))


class TestARejoinDoesNotSwallowTheNextCommand:
    def test_a_second_probe_after_a_newline_is_still_found(self):
        """Rejoining to the END of the command merges the next command into this one's argv.

        A newline is a command separator and shlex drops it, so a tail-swallowing rejoin lost a second
        ``python3 -c`` whose segment then resolved to whatever followed a pipe.
        """
        cmd = ('python3 -c "\nfrom x import target_profile\nprint(target_profile())\n" 2>&1 | tail -20\n'
               'echo "=== facts ==="\n'
               'python3 -c "\nfrom y import load_facts\nprint(load_facts())\n"')
        names = C._python_call_names(_call(cmd))
        assert "target_profile" in names, "the first probe was lost"
        assert "load_facts" in names, "the second probe after the newline was lost"


class TestForgeryIsStillRejected:
    """The anti-forgery property must survive the fix: only EXECUTED python yields call names."""

    @pytest.mark.parametrize("cmd", [
        "echo 'load_facts(\"gemmini\")'",
        "true # derived_levers(profile)",
        "cat <<'EOF'\nload_facts('gemmini')\nEOF",
    ])
    def test_a_mention_is_not_an_invocation(self, cmd):
        assert "load_facts" not in C._python_call_names(_call(cmd))
        assert "derived_levers" not in C._python_call_names(_call(cmd))


class TestABracketLaterInTheOutputCannotHideAnEarlierList:
    def test_a_valid_list_is_found_despite_a_trailing_bracket(self):
        """``_literal_string_list`` spanned the FIRST "[" to the LAST "]", so a composed command whose
        second half also printed a bracket swallowed the first half's list and parsed as nothing."""
        text = "LEVERS ['a.b', 'c.d']\n=== facts ===\n{ \"legal\": [1, 2, 3] }"
        assert C._literal_string_list(text) == ["a.b", "c.d"]

    def test_a_result_with_no_string_list_still_returns_none(self):
        assert C._literal_string_list("{ \"legal\": [1, 2, 3] }") is None
