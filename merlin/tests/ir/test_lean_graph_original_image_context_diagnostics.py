"""Data-only tests of an unintegrated, unapproved diagnostic grammar proposal."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

if not os.environ.get("MERLIN_IMAGE_CONTEXT_DATA_TEST"):
    raise unittest.SkipTest("requires an isolated SDK-free data runner")

from lean_graph_original_image_context_diagnostic_proposal import word_decide_refusal

ROOT = Path(os.environ["MERLIN_IMAGE_CONTEXT_PRIMARY"])
FIXTURE = ROOT / (
    "out/artifacts/verification/graph-state-integer/qualification-v2/suite/"
    "test_wrong_constant_bit_certif0/lean/kernel.stdout"
)
RAW = FIXTURE.read_text()
SOURCES = {RAW.partition(":")[0]}


class ProposalTests(unittest.TestCase):
    def test_original_retained_word_refusal(self):
        self.assertTrue(word_decide_refusal(RAW, SOURCES))

    def test_multiple_complete_known_blocks(self):
        self.assertTrue(word_decide_refusal(RAW + "\n" + RAW, SOURCES))

    def test_literal_syntax_is_not_one_golden_value(self):
        self.assertTrue(word_decide_refusal(RAW.replace("word 8 0 = word 8 1", "word 16 3 = word 16 4"), SOURCES))

    def test_other_lean_propositions_remain_unsupported(self):
        for proposition in ("checkProgram source target = true", "False", "word (-1) 0 = word (-1) 1"):
            self.assertFalse(word_decide_refusal(RAW.replace("word 8 0 = word 8 1", proposition), SOURCES))


def refusal_test(change):
    def test(self):
        self.assertFalse(word_decide_refusal(change(RAW), SOURCES))

    return test


MUTATIONS = {
    "unknown_source": lambda raw: raw.replace(next(iter(SOURCES)), "/unbound/Query.lean"),
    "missing_false": lambda raw: raw.replace("is false", ""),
    "false_with_extra": lambda raw: raw.replace("is false", "is false; out of memory"),
    "wrong_header": lambda raw: raw.replace("Tactic `decide` proved that the proposition", "out of memory"),
    "unindented_body": lambda raw: raw.replace("  word", "word"),
    "tab_body": lambda raw: raw.replace("  word", "\tword"),
    "runtime_body": lambda raw: raw.replace("  word 8 0 = word 8 1", "  failed to load shared library"),
    "unicode_number": lambda raw: raw.replace("word 8", "word ８"),
    "extra_continuation": lambda raw: raw.replace("is false", "  unknown diagnostic\nis false"),
    "trailing_text": lambda raw: raw + "unknown diagnostic\n",
    "leading_text": lambda raw: "unknown diagnostic\n" + raw,
    "mixed_runtime_before": lambda raw: next(iter(SOURCES)) + ":1:0: error: out of memory\n" + raw,
    "mixed_runtime_after": lambda raw: raw + next(iter(SOURCES)) + ":1:0: error: out of memory\n",
    "mixed_interpreter": lambda raw: raw + next(iter(SOURCES)) + ":1:0: error: (interpreter) unknown declaration\n",
    "empty": lambda raw: "",
}
for _name, _change in MUTATIONS.items():
    setattr(ProposalTests, "test_refuse_" + _name, refusal_test(_change))


if __name__ == "__main__":
    unittest.main()
