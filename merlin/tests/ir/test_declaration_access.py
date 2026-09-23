"""Textual declaration repair uses caller-owned argument access, not a target ABI."""

import pytest

from merlin.llvmlower.declaration_access import patch_declaration_arg_attrs, unpatched_declarations


@pytest.mark.parametrize("policy", [("write", "read"), ("read", "read", "write")])
def test_explicit_access_policy_preserves_types_and_unrelated_declarations(policy):
    types = ["tensor<4xf32>", "tensor<4xi16>", "tensor<4xi32>"][: len(policy)]
    declaration = f"func.func private @selected({', '.join(types)}) -> tensor<4xf32>"
    unrelated = "func.func private @unrelated(tensor<4xf32>, tensor<4xf32>)"
    text = "module {\n  " + declaration + "\n  " + unrelated + "\n}\n"
    patched = patch_declaration_arg_attrs(text, iter(["selected"]), argument_access=policy)
    arguments = ", ".join(
        f'{ty} {{bufferization.access = "{access}"}}' for ty, access in zip(types, policy, strict=True)
    )
    expected = f"func.func private @selected({arguments}) -> tensor<4xf32>"
    assert patched == text.replace(declaration, expected)
    assert unpatched_declarations(patched, ["selected"]) == ()
    assert unpatched_declarations(patched, ["unrelated"]) == ("unrelated",)
    assert patch_declaration_arg_attrs(patched, ["selected"], argument_access=policy) == patched


@pytest.mark.parametrize(
    "declaration",
    [
        "func.func private @other(tensor<4xf32>, tensor<4xf32>)",
        "func.func private @selected(tensor<4xf32>)",
        "func.func private @selected(tensor<4xf32>, tensor<4xf32>",
    ],
)
def test_missing_symbol_arity_mismatch_and_unclosed_declarations_remain_reported(declaration):
    text = declaration + "\n"
    patched = patch_declaration_arg_attrs(text, ["selected"], argument_access=("read", "write"))
    assert patched == text
    assert unpatched_declarations(patched, ["selected"]) == ("selected",)


def test_unpatched_report_preserves_requested_symbol_order():
    text = 'func.func private @done(tensor<4xf32> {bufferization.access = "write"})\n'
    assert unpatched_declarations(text, iter(["second", "done", "first"])) == ("second", "first")
    assert patch_declaration_arg_attrs(text, [], argument_access=("read",)) == text


def test_exact_symbol_matching_does_not_rewrite_prefixed_names():
    text = "func.func private @unit_extended(tensor<4xf32>, tensor<4xf32>)\n"
    assert patch_declaration_arg_attrs(text, ["unit"], argument_access=("read", "write")) == text
    assert unpatched_declarations(text, ["unit"]) == ("unit",)
