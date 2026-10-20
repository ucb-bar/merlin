"""Public LLVM function aliases retain their real body and all audit obligations."""

import pytest
from xdsl.parser import Parser

from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
from merlin.verify.llvm_symbols import defined_function


def _alias(name, target, typ="void", extra=""):
    return f'''"llvm.mlir.alias"() <{{sym_name = "{name}",
        alias_type = !llvm.func<{typ} (!llvm.ptr)>}}> ({{
        %address = llvm.mlir.addressof @{target} : !llvm.ptr
        {extra}
        llvm.return %address : !llvm.ptr
    }}) : () -> ()'''


def _parse(aliases, body="llvm.func @body(%p: !llvm.ptr) { llvm.return }"):
    return Parser(make_llvm_context(), "module { " + aliases + body + " }").parse_module()


def test_alias_and_alias_chain_resolve_the_original_audited_function():
    module = _parse(_alias("entry", "middle") + _alias("middle", "body"))
    original = defined_function(module, "body")
    assert defined_function(module, "entry") is original
    from merlin.targetgen.owned_host_model_execution import _llvm_function

    assert _llvm_function(module, "entry") is original


@pytest.mark.parametrize("aliases", [
    _alias("entry", "missing"),
    _alias("entry", "other") + _alias("other", "entry"),
    _alias("entry", "body", typ="i32"),
    _alias("entry", "body", extra="%extra = llvm.mlir.constant(1 : i64) : i64"),
    _alias("entry", "body") + _alias("entry", "body"),
])
def test_ambiguous_cyclic_missing_type_changing_or_effectful_aliases_refuse(aliases):
    with pytest.raises(ValueError, match="type-preserving"):
        defined_function(_parse(aliases), "entry")


def test_alias_to_a_declaration_does_not_supply_a_body():
    with pytest.raises(ValueError, match="type-preserving"):
        defined_function(_parse(_alias("entry", "body"), "llvm.func @body(!llvm.ptr)"), "entry")
