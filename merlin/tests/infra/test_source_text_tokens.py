"""Restricted readers distinguish source tokens from comments and quoted data."""

import pytest

from merlin.common.text_tokens import is_identifier, source_tokens


@pytest.mark.parametrize("value", ["entry", "_entry42", "a_B0"])
def test_portable_identifiers_accept_literal_ascii_names(value):
    assert is_identifier(value)


@pytest.mark.parametrize("value", [None, 1, "", "9entry", "entry.name", "entry\n", "λ", "ｅntry"])
def test_portable_identifiers_refuse_nonportable_or_partial_names(value):
    assert not is_identifier(value)


def test_lexer_preserves_literal_bytes_and_positions_without_comment_tokens():
    source = 'void/* not a name */ entry("// literal \\"quoted\\"", \'/\'); // hidden\n'
    tokens = source_tokens(source)
    assert [token.text for token in tokens] == ["void", "entry", "(", '"// literal \\"quoted\\""', ",", "'/'", ")", ";"]
    assert all(source[token.start : token.end] == token.text for token in tokens)


@pytest.mark.parametrize("source", ['"open', "'open", "/* open", '"trailing\\'])
def test_unterminated_source_tokens_refuse_instead_of_consuming_unknown_text(source):
    with pytest.raises(ValueError, match="unterminated"):
        source_tokens(source)


def test_counter_declaration_requires_actual_asm_tokens_and_the_exact_csr():
    from merlin.model_check.chain_image_counter import _declares_counter

    declaration = '__asm__ volatile ( "csrr %0, mcycle" : "=r" ( ticks ) )'
    assert _declares_counter(declaration + ";")
    assert not _declares_counter("/* " + declaration + " */")
    assert not _declares_counter("// " + declaration)
    assert not _declares_counter('"__asm__ volatile"; "csrr %0, mcycle";')
    assert not _declares_counter(declaration.replace("mcycle", "time"))
    assert not _declares_counter(declaration.replace("( ticks )", "( ticks + 1 )"))
