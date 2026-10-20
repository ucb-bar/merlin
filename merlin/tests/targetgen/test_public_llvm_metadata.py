"""Public LLVM metadata parsing must preserve typed control-flow verification."""

import pytest
from xdsl.parser import Parser

from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context


def _module(annotation="#llvm.loop_annotation<mustProgress = true>", extra="", condition="i1"):
    return f"""module {{
      llvm.func @kernel(%c: {condition}) {{
        "llvm.cond_br"(%c) [^yes, ^no]
          <{{operandSegmentSizes = array<i32: 1, 0, 0>, loop_annotation = {annotation}{extra}}}>
          : ({condition}) -> ()
      ^yes:
        "llvm.br"() [^no] <{{loop_annotation = {annotation}}}> : () -> ()
      ^no:
        llvm.return
      }}
    }}"""


def test_loop_metadata_preserves_branch_properties_and_successors():
    module = Parser(make_llvm_context(), _module()).parse_module()
    before = str(module)
    module.verify()
    assert str(module) == before
    branches = [op for op in module.walk() if op.name in {"llvm.br", "llvm.cond_br"}]
    assert len(branches) == 2
    assert [len(op.successors) for op in branches] == [2, 1]
    for branch in branches:
        assert branch.properties["loop_annotation"].attr_name.data == "llvm.loop_annotation"
        assert branch.properties["loop_annotation"].value.data == "mustProgress = true"
    reparsed = Parser(make_llvm_context(), str(module)).parse_module()
    reparsed.verify()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"annotation": '"not an annotation"'},
        {"annotation": "#other.hint<true>"},
        {"extra": ", unknown_property = true"},
        {"condition": "i32"},
    ],
)
def test_loop_metadata_does_not_disable_branch_verification(kwargs):
    with pytest.raises(Exception):
        Parser(make_llvm_context(), _module(**kwargs)).parse_module().verify()


def test_loop_metadata_keeps_terminator_position_check():
    text = _module().replace("^yes:", '%late = "llvm.mlir.constant"() <{value = 0 : i32}> : () -> i32\n ^yes:')
    with pytest.raises(Exception, match="terminate"):
        Parser(make_llvm_context(), text).parse_module().verify()


def test_tbaa_metadata_preserves_load_store_schema_and_original_dialect():
    from xdsl.context import Context
    from xdsl.dialects import builtin, llvm

    text = """module {
      llvm.func @kernel(%p: !llvm.ptr, %v: i32) {
        %x = "llvm.load"(%p) <{tbaa = [#llvm.tbaa_tag<offset = 0>]}>
          : (!llvm.ptr) -> i32
        "llvm.store"(%v, %p) <{tbaa = [#llvm.tbaa_tag<offset = 0>]}>
          : (i32, !llvm.ptr) -> ()
        llvm.return
      }
    }"""
    module = Parser(make_llvm_context(), text).parse_module()
    before = str(module)
    module.verify()
    assert str(module) == before
    for invalid in (
        text.replace("llvm.tbaa_tag", "other.tag"),
        text.replace("%p: !llvm.ptr", "%p: i64"),
    ):
        with pytest.raises(Exception):
            Parser(make_llvm_context(), invalid).parse_module().verify()
    original = Context(allow_unregistered=True)
    original.load_dialect(builtin.Builtin)
    original.load_dialect(llvm.LLVM)
    with pytest.raises(Exception, match="tbaa"):
        Parser(original, text).parse_module().verify()


@pytest.mark.parametrize("flags", [0, 1, 2, 3])
def test_generic_truncation_flags_keep_the_original_numeric_bits(flags):
    text = f"""module {{
      llvm.func @kernel(%x: i64) -> i32 {{
        %y = "llvm.trunc"(%x) <{{overflowFlags = {flags} : i32}}> : (i64) -> i32
        llvm.return %y : i32
      }}
    }}"""
    module = Parser(make_llvm_context(), text).parse_module()
    before = str(module)
    module.verify()
    assert str(module) == before
    truncation = next(op for op in module.walk() if op.name == "llvm.trunc")
    assert truncation.properties["overflowFlags"].value.data == flags
    reparsed = Parser(make_llvm_context(), str(module)).parse_module()
    reparsed.verify()
    assert next(op for op in reparsed.walk() if op.name == "llvm.trunc").properties["overflowFlags"].value.data == flags
    for wrong in (text.replace(f"{flags} : i32", "4 : i32"), text.replace(f"{flags} : i32", "1 : i64")):
        with pytest.raises(Exception):
            Parser(make_llvm_context(), wrong).parse_module().verify()


def test_dso_local_keeps_typed_function_custom_checks():
    text = """module {
      "llvm.func"() <{sym_name = "kernel", function_type = !llvm.func<void ()>,
        dso_local, CConv = #llvm.cconv<ccc>, linkage = #llvm.linkage<"external">}> ({
          llvm.return
      }) : () -> ()
    }"""
    module = Parser(make_llvm_context(), text).parse_module()
    module.verify()
    assert "dso_local" in next(op for op in module.walk() if op.name == "llvm.func").properties
    for wrong in (
        text.replace("dso_local,", "dso_local = 1 : i32,"),
        text.replace('#llvm.linkage<"external">', "1 : i32"),
    ):
        with pytest.raises(Exception):
            Parser(make_llvm_context(), wrong).parse_module().verify()


def test_nobuiltins_preserves_selected_llvm_function_semantics_and_typed_schema():
    text = """module {
      "llvm.func"() <{sym_name = "kernel", function_type = !llvm.func<void ()>,
        dso_local, nobuiltins = [], CConv = #llvm.cconv<ccc>,
        linkage = #llvm.linkage<"external">}> ({ llvm.return }) : () -> ()
    }"""
    module = Parser(make_llvm_context(), text).parse_module()
    before = str(module)
    module.verify()
    function = next(op for op in module.walk() if op.name == "llvm.func")
    assert len(function.properties["nobuiltins"].data) == 0
    assert "nobuiltins = []" in before
    reparsed = Parser(make_llvm_context(), before).parse_module()
    reparsed.verify()
    assert str(reparsed) == before
    for wrong in (
        text.replace("nobuiltins = []", "nobuiltins = 1 : i32"),
        text.replace("nobuiltins = []", "nobuiltins = [1 : i32]"),
        text.replace("nobuiltins = []", "unknown_property = []"),
    ):
        with pytest.raises(Exception):
            Parser(make_llvm_context(), wrong).parse_module().verify()


def _function_effects(annotation="#llvm.memory_effects<other = none, argMem = none>"):
    return f"""module {{
      "llvm.func"() <{{sym_name = "external_function", function_type = !llvm.func<f32 (f32)>,
        memory_effects = {annotation}, CConv = #llvm.cconv<ccc>,
        linkage = #llvm.linkage<external>}}> ({{}}) : () -> ()
      llvm.func @entry(%a: !llvm.ptr, %b: !llvm.ptr) {{ llvm.return }}
    }}"""


def test_function_memory_effects_roundtrip_without_mutating_upstream_schema():
    from xdsl.context import Context
    from xdsl.dialects import builtin, llvm

    from merlin.targetgen.contract.compile_only import require_pointer_entry

    text = _function_effects()
    module = Parser(make_llvm_context(), text).parse_module()
    before = str(module)
    module.verify()
    assert str(module) == before
    function = next(op for op in module.walk() if op.name == "llvm.func")
    metadata = function.properties["memory_effects"]
    assert metadata.attr_name.data == "llvm.memory_effects"
    assert metadata.value.data == "other = none, argMem = none"
    reparsed = Parser(make_llvm_context(), before).parse_module()
    reparsed.verify()
    assert str(reparsed) == before
    require_pointer_entry(text, entry_symbol="entry", pointer_arity=2)
    original = Context(allow_unregistered=True)
    original.load_dialect(builtin.Builtin)
    original.load_dialect(llvm.LLVM)
    with pytest.raises(Exception, match="memory_effects"):
        Parser(original, text).parse_module().verify()


@pytest.mark.parametrize(
    "annotation",
    ['"not effects metadata"', "#other.effects<none>", "#llvm.loop_annotation<mustProgress = true>"],
)
def test_function_memory_effects_requires_its_standard_metadata_kind(annotation):
    with pytest.raises(Exception):
        Parser(make_llvm_context(), _function_effects(annotation)).parse_module().verify()


def test_function_memory_effects_keeps_original_body_and_property_checks():
    text = _function_effects()
    for invalid in (
        text.replace("memory_effects =", "unknown_property ="),
        text.replace("%a: !llvm.ptr", "%a: i64"),
    ):
        from merlin.targetgen.contract.compile_only import require_pointer_entry

        with pytest.raises(ValueError):
            require_pointer_entry(invalid, entry_symbol="entry", pointer_arity=2)
