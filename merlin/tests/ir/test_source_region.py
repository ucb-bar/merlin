import pytest
from xdsl.dialects import func

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.source_region import extract_source_region
from merlin.xdsl_dialects._common import text


def source():
    return parse_mlir_text("""module {func.func @original(%a:i32,%b:i32)->(i32,i32) {
      %x=arith.addi %a,%b:i32
      %y=arith.muli %x,%b:i32
      %z=arith.subi %a,%b:i32
      return %x,%y:i32,i32
    }}""")


def test_every_live_output_original_inputs_and_order():
    module = source()
    before = text(module, generic=True)
    owner = next(o for o in module.body.block.ops if isinstance(o, func.FuncOp))
    ops = tuple(owner.body.block.ops)
    result = extract_source_region(module, ops[:2])
    assert result.inputs == tuple(owner.body.block.args)
    assert result.outputs == (ops[0].results[0], ops[1].results[0])
    assert text(module, generic=True) == before
    assert result.receipt["source_operation_ordinals"] == [0, 1]
    result.module.verify()


def test_noncontiguous_span_refuses():
    module = source()
    owner = next(o for o in module.body.block.ops if isinstance(o, func.FuncOp))
    ops = tuple(owner.body.block.ops)
    with pytest.raises(ValueError, match="contiguous"):
        extract_source_region(module, (ops[0], ops[2]))


def test_unused_span_and_original_terminator_refuse():
    module = source()
    owner = next(o for o in module.body.block.ops if isinstance(o, func.FuncOp))
    ops = tuple(owner.body.block.ops)
    with pytest.raises(ValueError, match="observed"):
        extract_source_region(module, (ops[2],))
    with pytest.raises(ValueError, match="terminator"):
        extract_source_region(module, ops)


def test_effectful_direct_callee_copied_without_purity_grant():
    module = parse_mlir_text("""module {func.func private @external(i32)->i32
      func.func @original(%a:i32)->i32 {%x=func.call @external(%a):(i32)->i32
      return %x:i32}}""")
    owner = next(o for o in module.body.block.ops if isinstance(o, func.FuncOp) and o.sym_name.data == "original")
    result = extract_source_region(module, (owner.body.block.first_op,))
    assert len(tuple(result.module.body.block.ops)) == 2
    assert result.receipt["effect_or_target_admission"] is False


def test_only_explicit_original_literal_dependencies_captured():
    module = parse_mlir_text("""module {func.func @original(%a:i32)->i32 {
      %zero=arith.constant 0:i32
      %x=arith.addi %a,%zero:i32
      return %x:i32}}""")
    owner = next(o for o in module.body.block.ops if isinstance(o, func.FuncOp))
    ops = tuple(owner.body.block.ops)
    result = extract_source_region(module, (ops[1],), capture_constants=True)
    assert result.inputs == tuple(owner.body.block.args)
    assert len(result.receipt["captured_original_literal_dependencies"]) == 1
    assert len(tuple(next(iter(result.module.body.block.ops)).body.block.ops)) == 3
    with pytest.raises(ValueError, match="byte bound"):
        extract_source_region(module, (ops[1],), capture_constants=True, max_constant_bytes=1)


def test_original_view_closure_enables_existing_cse_not_value_inference():
    from merlin.transforms.tensor_views import canonicalize_tensor_views

    module = parse_mlir_text("""module {func.func @original(%a:tensor<1x4xi8>)->(tensor<2x2xi8>,tensor<2x2xi8>) {
      %flat=tensor.collapse_shape %a [[0,1]] : tensor<1x4xi8> into tensor<4xi8>
      %one=tensor.expand_shape %flat [[0,1]] output_shape [2,2] : tensor<4xi8> into tensor<2x2xi8>
      %otherflat=tensor.collapse_shape %a [[0,1]] : tensor<1x4xi8> into tensor<4xi8>
      %two=tensor.expand_shape %otherflat [[0,1]] output_shape [2,2] : tensor<4xi8> into tensor<2x2xi8>
      return %one,%two:tensor<2x2xi8>,tensor<2x2xi8>
    }}""")
    owner = next(o for o in module.body.block.ops if isinstance(o, func.FuncOp))
    ops = tuple(owner.body.block.ops)
    region = extract_source_region(module, ops[1:4], capture_views=True)
    assert region.inputs == tuple(owner.body.block.args)
    assert len(region.receipt["captured_original_view_dependencies"]) == 1
    witnesses = canonicalize_tensor_views(region.module)
    assert len(witnesses) == 2
    returned = next(iter(region.module.body.block.ops)).body.block.last_op
    assert returned.arguments[0] is returned.arguments[1]


def sibling_source(*, unknown_effect=False, seed=0):
    multiply = """linalg.generic {indexing_maps = [affine_map<(m,n,k)->(m,k)>,
      affine_map<(m,n,k)->(k,n)>,affine_map<(m,n,k)->(m,n)>],
      iterator_types = ["parallel","parallel","reduction"]}
      ins(LHS, RHS : tensor<2x2xi8>,tensor<2x2xi8>) outs(SEED:tensor<2x2xi32>) {
      ^bb0(%x:i8,%y:i8,%s:i32):
        %xx=arith.extsi %x:i8 to i32
        %yy=arith.extsi %y:i8 to i32
        %p=arith.muli %xx,%yy:i32
        %next=arith.addi %s,%p:i32
        linalg.yield %next:i32
      } -> tensor<2x2xi32>"""
    first = multiply.replace("LHS", "%a").replace("RHS", "%b").replace("SEED", "%i")
    second = multiply.replace("LHS", "%a").replace("RHS", "%wt").replace("SEED", "%j")
    source_text = """module {DECL func.func @forward(%a:tensor<2x2xi8>,%b:tensor<2x2xi8>,%w:tensor<2x2xi8>)
      ->(tensor<2x2xf32>,tensor<2x2xi32>) {
      %zero=arith.constant SEEDVALUE:i32
      %i=tensor.splat %zero:tensor<2x2xi32>
      %q=FIRST
      %e=tensor.empty():tensor<2x2xf32>
      %f=linalg.generic {indexing_maps=[affine_map<(m,n)->(m,n)>,affine_map<(m,n)->(m,n)>],
        iterator_types=["parallel","parallel"]}
        ins(%q:tensor<2x2xi32>) outs(%e:tensor<2x2xf32>) {
        ^bb0(%x:i32,%y:f32):
          %r=arith.sitofp %x:i32 to f32
          linalg.yield %r:f32
        } -> tensor<2x2xf32>
      EFFECT
      %te=tensor.empty():tensor<2x2xi8>
      %wt=linalg.transpose ins(%w:tensor<2x2xi8>) outs(%te:tensor<2x2xi8>) permutation=[1,0]
      %z=arith.constant 0:i32
      %j=tensor.splat %z:tensor<2x2xi32>
      %k=SECOND
      return %f,%k:tensor<2x2xf32>,tensor<2x2xi32>
    }}"""
    source_text = (
        source_text.replace("FIRST", first)
        .replace("SECOND", second)
        .replace("SEEDVALUE", str(seed))
        .replace("DECL", "func.func private @opaque()" if unknown_effect else "")
        .replace("EFFECT", "func.call @opaque():()->()" if unknown_effect else "")
    )
    return parse_mlir_text(source_text)


def sibling_roots(module):
    return tuple(o for o in module.walk() if o.name == "linalg.generic" and str(o.results[0].type).endswith("xi32>"))


def test_actual_sibling_outline_preserves_host_epilogue_and_results():
    import numpy as np

    from merlin.compile.scheduling.shared_input_groups import find_shared_input_groups
    from merlin.llvmlower.source_region import outline_sibling_contractions
    from merlin.runtime.linalg_numpy import evaluate

    module = sibling_source()
    before = text(module, generic=True)
    args = [np.asarray(v, dtype=np.int8) for v in [[[1, -2], [3, 4]], [[1, 2], [0, -1]], [[3, 0], [1, 1]]]]
    expected = evaluate(module, args)
    outlined = outline_sibling_contractions(module, sibling_roots(module))
    assert text(module, generic=True) == before
    actual = evaluate(outlined.module, args)
    assert all(np.array_equal(a, b) for a, b in zip(expected, actual, strict=True))
    assert len(find_shared_input_groups(outlined.core)) == 1
    assert outlined.receipt["source_hoisted_preparation_ordinals"]
    assert outlined.receipt["original_epilogues_and_observers_preserved"]
    assert not outlined.receipt["target_implementation_or_effect_admission"]


def test_sibling_outline_refuses_unknown_crossed_effect_and_changed_seed():
    from merlin.llvmlower.source_region import outline_sibling_contractions

    module = sibling_source(unknown_effect=True)
    with pytest.raises(ValueError, match="effect"):
        outline_sibling_contractions(module, sibling_roots(module))
    module = sibling_source(seed=1)
    with pytest.raises(ValueError):
        outline_sibling_contractions(module, sibling_roots(module))


def test_sibling_outline_rederives_order_and_ssa_identity():
    from merlin.llvmlower.source_region import outline_sibling_contractions

    module = sibling_source()
    roots = sibling_roots(module)
    with pytest.raises(ValueError, match="order"):
        outline_sibling_contractions(module, tuple(reversed(roots)))
    roots[1].operands[0] = roots[1].operands[1]
    with pytest.raises(ValueError, match="identical"):
        outline_sibling_contractions(module, roots)


def test_source_owned_sibling_host_retains_mixed_public_types_and_fresh_results():
    from xdsl.dialects import func

    from merlin.llvmlower.borrowed_effects import BorrowedBufferEffects
    from merlin.llvmlower.source_region_host import emit_ranked_source_entry, prepare_sibling_source_host, tensor_abi

    source = sibling_source()
    before = text(source, generic=True)
    host = prepare_sibling_source_host(
        source, effects=BorrowedBufferEffects("independent-test-implementation", True, True, True)
    )
    assert text(source, generic=True) == before
    ins, outs = tensor_abi(source)
    assert len(ins) == 3 and [row["dtype"] for row in outs] == ["f32", "i32"]
    assert host.writer.result_arguments == (3, 4)
    assert host.receipt["original_epilogues_and_all_live_outputs_preserved"]
    core = next(
        op
        for op in host.module.body.block.ops
        if isinstance(op, func.FuncOp) and op.sym_name.data == "source_shared_core"
    )
    assert not any(op.name == "linalg.generic" for op in core.walk())
    assert not any(op.name == "arith.muli" for op in host.module.walk())
    assert any(op.name == "arith.sitofp" for op in host.module.walk())
    code = emit_ranked_source_entry(source)
    assert "_mlir_ciface_forward(&d0,&d1,&d2,&d3,&d4);" in code
    assert "spatial_disjoint" in code
    assert "(float*)" not in code
    assert "malloc" not in code


def test_source_owned_sibling_host_refuses_missing_effects_and_nonzero_seed():
    from merlin.llvmlower.borrowed_effects import BorrowedBufferEffects
    from merlin.llvmlower.source_region_host import prepare_sibling_source_host

    with pytest.raises(ValueError, match="effects"):
        prepare_sibling_source_host(sibling_source(), effects=None)
    with pytest.raises(ValueError):
        prepare_sibling_source_host(sibling_source(seed=1), effects=BorrowedBufferEffects("test", True, True, True))


def test_source_ranked_entry_refuses_symbol_injection_and_nonstatic_signature():
    from merlin.llvmlower.source_region_host import emit_ranked_source_entry

    with pytest.raises(ValueError):
        emit_ranked_source_entry(sibling_source(), entry_symbol="bad;call()")
    with pytest.raises(ValueError):
        emit_ranked_source_entry(
            parse_mlir_text("module {func.func @forward(%a:tensor<?xi8>)->tensor<?xi8>{return %a:tensor<?xi8>}}")
        )


def _source_owner_fixture(tmp_path, monkeypatch):
    import hashlib
    import json
    from dataclasses import asdict

    import yaml

    from merlin.compile import owned_shared_source as owner
    from merlin.llvmlower.borrowed_effects import BorrowedBufferEffects
    from merlin.llvmlower.source_region_host import prepare_sibling_source_host, tensor_abi
    from merlin.targetgen import frozen_source_eligibility, native_model_execution

    module = sibling_source()
    raw = (text(module, generic=True) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    effects = BorrowedBufferEffects("separately-qualified-test-callee", True, True, True)
    prepared = prepare_sibling_source_host(module, effects=effects)
    ins, outs = tensor_abi(module)
    (tmp_path / "capsule.interface.mlir").write_bytes(raw)
    (tmp_path / "original-source.mlir").write_bytes(raw)
    plan = {
        "schema": "merlin.original_sibling_source_plan.v1",
        "source_sha256": digest,
        "preparation": prepared.receipt,
        "borrowed_effects": asdict(effects),
    }
    (tmp_path / "source-plan.json").write_text(json.dumps(plan))
    def pin(n):
        return {"file": n, "sha256": hashlib.sha256((tmp_path / n).read_bytes()).hexdigest()}
    cap = {
        "name": "source_test",
        "kind": "model_slice",
        "label": "public",
        "inputs": list(ins),
        "interface_mlir": "capsule.interface.mlir",
        "linalg_mlir": "capsule.interface.mlir",
        "operation": {"op": "component_program"},
        "component_program": {
            "schema": "merlin.original_ssa_program.v1",
            "source_sha256": digest,
            "outputs": list(outs),
        },
        "software_screen": {
            "status": "admitted",
            "program_sha256": digest,
            "placements": {"accelerator": 1},
            "decisions": [
                {"status": "admitted", "placement": "accelerator", "mlir_operation": "linalg.generic", "count": 2}
            ],
        },
        "owned_host_source": {
            "schema": owner.SCHEMA,
            "source_graph": pin("original-source.mlir"),
            "plan": pin("source-plan.json"),
        },
    }
    (tmp_path / "capsule.yaml").write_text(yaml.safe_dump(cap))
    certificate = {
        "source_mlir_sha256": digest,
        "n_eligible": 2,
        "regions": [{"target_eligible": True, "semantic_family": "contraction", "carrier_op": "linalg.generic"}] * 2,
    }
    # Core source routing reopens the census from its core module; the executor keeps a wrapper.
    monkeypatch.setattr(frozen_source_eligibility, "census", lambda *a, **k: certificate)
    monkeypatch.setattr(native_model_execution, "independent_frozen_source_eligibility", lambda *a, **k: certificate)
    return {**cap, "__dir__": str(tmp_path)}, certificate


def test_original_source_owner_rederives_private_core_and_keeps_public_abi(tmp_path, monkeypatch):
    from merlin.compile import owned_shared_source as owner

    cap, _ = _source_owner_fixture(tmp_path, monkeypatch)
    path, validate = owner.compiler_input(cap, tmp_path / "generated", target="test")
    module = parse_mlir_text(path.read_text())
    assert not any(op.name == "arith.sitofp" for op in module.walk())
    validate()
    private = {"abi_version": "0.1", "target": "test", "tensors": {}, "commands": []}
    cb = owner.full_command_buffer(cap, private)
    assert set(cb["tensors"]) == {"arg0", "arg1", "arg2", "output0", "output1"}
    assert cb["tensors"]["output0"]["dtype"] == "f32"
    assert cb["tensors"]["output1"]["dtype"] == "i32"
    assert cb["params"]["source_owned_host_projection"]["device_command_buffer"] == private
    assert "params" not in private
    (tmp_path / "original-source.mlir").write_text("module {}")
    with pytest.raises(ValueError, match="changed"):
        validate()


def test_original_source_owner_refuses_different_fresh_census_and_capsule(tmp_path, monkeypatch):
    from merlin.compile import owned_shared_source as owner

    cap, certificate = _source_owner_fixture(tmp_path, monkeypatch)
    certificate["n_eligible"] = 1
    with pytest.raises(ValueError, match="eligibility"):
        owner.compiler_input(cap, tmp_path / "generated", target="test")
    certificate["n_eligible"] = 2
    cap["component_program"]["outputs"][0]["dtype"] = "i8"
    with pytest.raises(ValueError, match="frozen"):
        owner.compiler_input(cap, tmp_path / "generated", target="test")
