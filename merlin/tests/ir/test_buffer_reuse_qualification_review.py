"""SDK-free full-definition/statement preservation, not a Lean proof parser."""

import pytest
from buffer_reuse_qualification_review import compare, inventory

SOURCE = """import Init
namespace N
def data : String := "theorem fake := 3 -- /-"
theorem claim : True := by trivial
end N
"""


def test_proof_only_change_preserves_all_original_data_and_claims():
    result = compare(SOURCE, SOURCE.replace("by trivial", "by exact True.intro"))
    assert result == {
        "definitions": 1,
        "original_theorem_headers": 1,
        "new_helper_theorems": [],
        "proofs_executed": False,
        "authority": False,
    }


def test_nested_comments_and_strings_never_manufacture_or_hide_commands():
    changed = "/- outer /- theorem hidden := bad -/ end -/\n" + SOURCE
    assert compare(SOURCE, changed)["definitions"] == 1
    assert set(inventory(SOURCE)["theorem_headers"]) == {"N.claim"}


def test_multiline_record_closing_at_column_zero_remains_part_of_definition():
    source = "import Init\ndef pair : Nat × Nat := {\n fst := 1\n snd := 2\n}\n"
    assert compare(source, source)["definitions"] == 1
    with pytest.raises(ValueError):
        compare(source, source.replace("snd := 2", "snd := 3"))


def test_declaration_order_is_preserved():
    first = SOURCE.index("def data")
    end = SOURCE.index("end N")
    data, theorem = SOURCE[first:end].split("theorem claim", 1)
    changed = SOURCE[:first] + "theorem claim" + theorem + data + SOURCE[end:]
    with pytest.raises(ValueError):
        compare(SOURCE, changed)


@pytest.mark.parametrize(
    "bad",
    [
        SOURCE.replace("True", "False"),
        SOURCE.replace("fake := 3", "fake := 4"),
        SOURCE.replace("import Init", "import Lean"),
        SOURCE.replace("end N", "end Wrong"),
        "/-x-/axiom hidden : False\n" + SOURCE,
        SOURCE.replace("end N", "/-x-/def hidden : Nat := 7\nend N"),
        SOURCE.replace("end N", "  axiom hidden : False\nend N"),
        SOURCE + "/- unfinished /- nested -/",
    ],
)
def test_unknown_semantics_statements_context_or_hidden_declarations_refuse(bad):
    with pytest.raises(ValueError):
        compare(SOURCE, bad)
