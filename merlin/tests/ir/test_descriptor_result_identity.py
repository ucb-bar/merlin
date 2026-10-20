"""Original typed descriptor identity, scoped allocation roots and refusal."""

from dataclasses import replace

import pytest
from xdsl.dialects import func

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.borrowed_buffer_calls import BorrowedBufferEffects
from merlin.llvmlower.descriptor_result_identity import (
    DescriptorResultIdentity,
    edit_pipeline,
    implementation_paths,
    rewrite_descriptor_result_identities,
)
from merlin.xdsl_dialects._common import text


def source(rows=2, cols=7, *, shared=False, blockarg=False):
    t = f"tensor<{rows}x{cols}xi32>"
    c_arg = f", %c:{t}" if blockarg else ""
    initialize = (
        ""
        if blockarg
        else f"%z=arith.constant 5:i32\n%e=tensor.empty():{t}\n%c=linalg.fill ins(%z:i32) outs(%e:{t})->{t}"
    )
    outputs = f"({t},{t},{t})" if shared else f"({t},{t})"
    returns = f"%r,%c,%a:{t},{t},{t}" if shared else f"%r,%a:{t},{t}"
    return f"""module {{
"func.func"() <{{sym_name="writer",sym_visibility="private",
function_type=({t},{t},{t})->{t},
arg_attrs=[{{bufferization.access="read"}},{{bufferization.access="read"}},
{{bufferization.access="write"}}]}}> ({{}}) {{llvm.emit_c_interface}} : ()->()
func.func @forward(%a:{t},%b:{t}{c_arg})->{outputs}{{
{initialize}
%r=func.call @writer(%a,%b,%c):({t},{t},{t})->{t}
return {returns}
}}
}}"""


def selection(module):
    original = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "writer")
    return DescriptorResultIdentity(
        "writer",
        "borrowed",
        "writer_tensor",
        2,
        tuple(original.function_type.inputs),
        tuple(attr.data["bufferization.access"].data for attr in original.arg_attrs),
        # Original neutral fixture declaration, independently of descriptor equality.
        BorrowedBufferEffects("neutral-fixture-blocking-borrowed-v1", True, True, True),
    )


def test_exact_source_wrapper_preserves_original_declaration_and_initialization():
    module = parse_mlir_text(source())
    original = module.body.block.first_op
    before = text(original)
    rows = rewrite_descriptor_result_identities(module, (selection(module),))
    module.verify()
    assert rows[0]["calls"] == rows[0]["protected_empty_roots"] == 1
    assert text(original) == before
    assert sum(op.name == "bufferization.alloc_tensor" for op in module.walk()) == 1
    assert sum(op.name == "linalg.fill" for op in module.walk()) == 1
    wrapper = next(op for op in module.body.block.ops if op.name == "func.func" and op.sym_name.data == "writer_tensor")
    assert [op.name for op in wrapper.body.block.ops] == ["bufferization.to_buffer"] * 3 + [
        "func.call",
        "bufferization.to_tensor",
        "func.return",
    ]
    assert wrapper.body.block.last_op.arguments[0].owner.operands[0] is list(wrapper.body.block.ops)[2].results[0]
    assert all(path.is_file() for path in implementation_paths())
    import inspect
    from pathlib import Path

    from xdsl.dialects import bufferization

    assert Path(inspect.getsourcefile(bufferization.ToBufferOp.build)).resolve() in implementation_paths()
    assert Path(inspect.getsourcefile(BorrowedBufferEffects)).resolve() in implementation_paths()


def test_missing_borrowed_effects_retains_original_call_without_new_ownership():
    module = parse_mlir_text(source())
    identity = replace(selection(module), borrowed_effects=None)
    before = text(module)
    rows = rewrite_descriptor_result_identities(module, (identity,))
    assert rows == [
        {
            "symbol": "writer",
            "calls": 0,
            "protected_empty_roots": 0,
            "borrowed_effects": None,
            "reason": "original borrowed effects are unavailable",
        }
    ]
    assert text(module) == before


@pytest.mark.parametrize("field", ["no_retained_aliases", "no_releases", "complete_before_return"])
def test_false_original_effect_premise_refuses_before_any_source_edit(field):
    module = parse_mlir_text(source())
    identity = selection(module)
    object.__setattr__(identity.borrowed_effects, field, False)
    before = text(module)
    with pytest.raises(ValueError, match="borrowed ABI"):
        rewrite_descriptor_result_identities(module, (identity,))
    assert text(module) == before


@pytest.mark.parametrize("effect", [True, {}, {"no_retained_aliases": True}])
def test_untyped_effect_data_cannot_issue_a_borrowed_contract(effect):
    module = parse_mlir_text(source())
    identity = replace(selection(module), borrowed_effects=effect)
    before = text(module)
    with pytest.raises(ValueError, match="typed borrowed"):
        rewrite_descriptor_result_identities(module, (identity,))
    assert text(module) == before


@pytest.mark.parametrize("shared,blockarg", [(True, False), (False, True)])
def test_observed_or_borrowed_initializer_declines_without_any_edit(shared, blockarg):
    module = parse_mlir_text(source(shared=shared, blockarg=blockarg))
    before = text(module)
    assert rewrite_descriptor_result_identities(module, (selection(module),)) == []
    assert text(module) == before


@pytest.mark.parametrize("kind", ["result", "access", "type", "collision", "untyped", "budget"])
def test_original_identity_cannot_be_retargeted_or_overbudget(kind):
    module = parse_mlir_text(source())
    identity = selection(module)
    before = text(module)
    limit = 100000
    if kind == "result":
        identity = replace(identity, result_argument=1)
    elif kind == "access":
        identity = replace(identity, argument_access=("read", "write", "write"))
    elif kind == "type":
        identity = replace(identity, argument_types=identity.argument_types[:2])
    elif kind == "collision":
        identity = replace(identity, wrapper_symbol="forward")
    elif kind == "untyped":
        identity = identity.__dict__
    else:
        limit = 2
    with pytest.raises((ValueError, TypeError)):
        rewrite_descriptor_result_identities(module, (identity,), max_operations=limit)
    assert text(module) == before


def test_unknown_call_and_initialized_observers_remain_opaque():
    module = parse_mlir_text(
        source()
        .replace("func.func @forward", "func.func private @unknown()\nfunc.func @forward")
        .replace("%r=func.call", "func.call @unknown():()->()\n%r=func.call")
    )
    original_unknown = next(
        op for op in module.walk() if isinstance(op, func.CallOp) and op.callee.root_reference.data == "unknown"
    )
    before = text(original_unknown)
    rewrite_descriptor_result_identities(module, (selection(module),))
    module.verify()
    assert text(original_unknown) == before


def test_companion_pipeline_uses_derived_identity_at_exact_seam():
    original = [
        "cse",
        "one-shot-bufferize{bufferize-function-boundaries}",
        "canonicalize",
        "buffer-results-to-out-params{hoist-static-allocs}",
    ]
    chosen = edit_pipeline(original)
    assert chosen == original[:3] + ["drop-equivalent-buffer-results"] + original[3:]
    assert edit_pipeline(chosen) == chosen and original[3].startswith("buffer-results")
    with pytest.raises(ValueError):
        edit_pipeline(original[::-1])


@pytest.mark.parametrize("site", ["call", "declaration", "empty"])
def test_unknown_executable_metadata_is_never_admitted(site):
    from xdsl.dialects.builtin import StringAttr

    module = parse_mlir_text(source())
    name = {"call": "func.call", "declaration": "func.func", "empty": "tensor.empty"}[site]
    operation = next(op for op in module.walk() if op.name == name)
    operation.attributes["unknown.executable"] = StringAttr("effect")
    before = text(module)
    if site == "declaration":
        with pytest.raises(ValueError, match="original declaration"):
            rewrite_descriptor_result_identities(module, (selection(module),))
    else:
        assert rewrite_descriptor_result_identities(module, (selection(module),)) == []
    assert text(module) == before


def test_shared_destination_between_two_calls_stays_on_original_route():
    original = source()
    t = "tensor<2x7xi32>"
    original = original.replace(
        f"return %r,%a:{t},{t}", f"%again=func.call @writer(%a,%b,%c):({t},{t},{t})->{t}\nreturn %again,%r:{t},{t}"
    )
    module = parse_mlir_text(original)
    before = text(module)
    assert rewrite_descriptor_result_identities(module, (selection(module),)) == []
    assert text(module) == before


def test_shadowed_symbol_keeps_its_original_body_and_call():
    original = source()
    nested = source().removeprefix("module {").rsplit("}", 1)[0]
    module = parse_mlir_text(original.rsplit("}", 1)[0] + "module @nested {" + nested + "}}")
    child = next(op for op in module.body.block.ops if op.name == "builtin.module")
    before = text(child)
    assert rewrite_descriptor_result_identities(module, (selection(module),))[0]["calls"] == 1
    assert text(child) == before
    module.verify()
