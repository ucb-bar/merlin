"""Capability disclosures preserve complete code blocks and unrelated obligations."""

from merlin_experiments.phase1 import fresh10_disclosure as disclosure


def test_markdown_capability_filter_preserves_complete_nested_fence():
    text = "Keep.\n\n````python\nimport merlin.transforms.tensor_views\n\n```\nprint(1)\n````\n\nKeep again.\n"
    assert disclosure.filter_markdown(text, library=False, references=False) == "Keep.\n\nKeep again.\n"
    assert not disclosure._cca_text("MY_CCA_HELPER and CCAish")
    assert disclosure._cca_text("Use CCA.")
    assert disclosure._library_text("merlin.transforms.tensor_views")
    assert not disclosure._library_text("merlin.transforms_extra")


def test_author_prompt_removes_only_capability_sentences():
    original = (
        "Keep bit-exact results. Use merlin.transforms.tensor_views for rewrites. "
        "Check all 89 inputs. Inspect CCA. Preserve the supplied numerical limits."
    )
    result = disclosure.author_prompt(original)
    assert "Keep bit-exact results." in result
    assert "Check all 89 inputs." in result
    assert "Preserve the supplied numerical limits." in result
    assert "Use merlin.transforms" not in result
    assert "Inspect CCA." not in result
