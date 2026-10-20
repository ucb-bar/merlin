"""Exact rounding representation: source mutation checks and ordinary execution."""

import json

import numpy as np
import pytest
from xdsl.dialects import arith, builtin

from merlin.common.mlir_query import op_name
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.automatic_host import optimize_host_quantization
from merlin.llvmlower.automatic_rounding import prepare_automatic_rounding, rewrite_rounding
from merlin.llvmlower.optional_passes import Selection, applied
from merlin.llvmlower.quant_round_check import verify_quantization_rewrite
from merlin.xdsl_dialects._common import text


def source(ft="f32", it="i8", lo=-127, hi=127):
    return f'''module {{
      func.func @forward(%a:tensor<2x17x{ft}>, %state:tensor<2x17x{ft}>)
        -> (tensor<2x17x{it}>,tensor<2x17x{ft}>) {{
        %e=tensor.empty():tensor<2x17x{it}>
        %r=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>,
          affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]}}
          ins(%a,%state:tensor<2x17x{ft}>,tensor<2x17x{ft}>) outs(%e:tensor<2x17x{it}>) {{
        ^bb0(%x:{ft},%old:{ft},%unused:{it}):
          %sum=arith.addf %x,%old:{ft}
          %round=math.roundeven %sum:{ft}
          %hi=arith.constant {hi}.0:{ft}
          %lo=arith.constant {lo}.0:{ft}
          %upper=arith.minimumf %round,%hi:{ft}
          %lower=arith.maximumf %upper,%lo:{ft}
          %q=arith.fptosi %lower:{ft} to {it}
          linalg.yield %q:{it}
        }} -> tensor<2x17x{it}>
        %state_empty=tensor.empty():tensor<2x17x{ft}>
        %next=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],
          iterator_types=["parallel","parallel"]}}
          ins(%state:tensor<2x17x{ft}>) outs(%state_empty:tensor<2x17x{ft}>) {{
        ^bb0(%old:{ft},%unused:{ft}):
          %one=arith.constant 1.0:{ft}
          %sum=arith.addf %old,%one:{ft}
          linalg.yield %sum:{ft}
        }} -> tensor<2x17x{ft}>
        return %r,%next:tensor<2x17x{it}>,tensor<2x17x{ft}>
      }}
    }}'''


def residual(**kwargs):
    module, report = optimize_host_quantization(parse_mlir_text(source(**kwargs)), numerical_policy="unchanged_source")
    assert report["rewrites"] == 1
    return module


def test_checked_pass_order_current_ir_and_idempotence():
    original = residual()
    before = text(original, generic=True)
    output, report = prepare_automatic_rounding(before, serial=True)
    assert text(original, generic=True) == before
    assert report["rewrites"] == 1
    assert [r["name"] for r in report["checks"]] == [
        "rounding_source", "rounding_representation", "rounding_conformance"]
    assert tuple(report["checks"][1]["invalidated"]) == ("source",)
    assert prepare_automatic_rounding(output, serial=True)[0] == output


@pytest.mark.parametrize("options", [{"serial": False}, {"max_added_operations": 0}, {"max_scalar_operations": 0}])
def test_unsupported_family_and_budgets_retain_exact_source(options):
    original = text(residual(), generic=True)
    output, report = prepare_automatic_rounding(original, **({"serial": True} | options))
    assert output == original and report["rewrites"] == 0


def test_disabled_and_unknown_enclosing_contracts_retain_source():
    original = text(residual(), generic=True)
    with applied(Selection.parse("-automatic-host-rounding")):
        assert prepare_automatic_rounding(original, serial=True)[0] == original
    module = residual()
    fn = next(op for op in module.walk() if op.name == "func.func")
    fn.attributes["unknown_numeric_effects"] = builtin.UnitAttr()
    original = text(module, generic=True)
    assert prepare_automatic_rounding(original, serial=True)[0] == original


def test_live_round_intermediates_are_not_duplicated():
    module = residual()
    generic = next(op for op in module.walk() if op.name == "linalg.generic")
    converted = next(op for op in generic.walk() if isinstance(op, arith.FPToSIOp))
    # A live source observer uses the intermediate outside the rounding DAG.
    extra = arith.AddiOp(converted.result, converted.result)
    generic.body.block.insert_op_before(extra, generic.body.block.last_op)
    combined = arith.AddiOp(generic.body.block.last_op.operands[0], extra.result)
    generic.body.block.insert_op_before(combined, generic.body.block.last_op)
    generic.body.block.last_op.operands = [combined.result]
    original = text(module, generic=True)
    assert prepare_automatic_rounding(original, serial=True)[0] == original


def test_scalar_ir_budget_has_an_exact_boundary():
    module = residual()
    size = len(tuple(next(op for op in module.walk() if op.name == "linalg.generic").body.block.ops))
    original = text(module, generic=True)
    assert prepare_automatic_rounding(original, serial=True, max_scalar_operations=size - 1)[0] == original
    assert prepare_automatic_rounding(original, serial=True, max_scalar_operations=size,
                                      max_added_operations=1)[1]["rewrites"] == 1


def test_global_growth_budget_can_select_a_partial_legal_alternative():
    module = residual()
    fn = next(op for op in module.walk() if op.name == "func.func")
    first = next(op for op in fn.walk() if op.name == "linalg.generic")
    copied = first.clone()
    returned = fn.body.block.last_op
    fn.body.block.insert_op_before(copied, returned)
    returned.operands = [returned.operands[0], copied.results[0], returned.operands[1]]
    fn.properties["function_type"] = builtin.FunctionType.from_lists(
        fn.function_type.inputs.data, [operand.type for operand in returned.operands])
    module.verify()
    original = text(module, generic=True)
    for budget in (1, 2):
        output, report = prepare_automatic_rounding(original, serial=True, max_added_operations=budget)
        assert report["rewrites"] == budget
        actual = parse_mlir_text(output)
        assert sum(op_name(op) == "llvm.intr.roundeven" for op in actual.walk()) == budget
        verify_quantization_rewrite(module, actual, bounded_intrinsics=True)


@pytest.mark.parametrize("fault", ["outside_integer", "fractional", "missing_bound", "float_integer_limit"])
def test_unproved_capacity_or_integral_bounds_retain_original_source(fault):
    module = residual(it="i32" if fault == "float_integer_limit" else "i8")
    upper = next(op for op in module.walk() if isinstance(op, arith.MinimumfOp))
    if fault == "missing_bound":
        upper.operands = [upper.operands[0], upper.operands[0]]
    else:
        bound = {"outside_integer": 128.0, "fractional": 126.5, "float_integer_limit": float(2**23)}[fault]
        upper.operands[1].owner.properties["value"] = builtin.FloatAttr(bound, builtin.f32)
    original = text(module, generic=True)
    assert prepare_automatic_rounding(original, serial=True)[0] == original


@pytest.mark.parametrize("fault", ["accounting", "library", "unbounded"])
def test_mutated_selection_or_library_cannot_publish_a_result(tmp_path, monkeypatch, fault):
    from merlin.llvmlower import automatic_rounding as rounding

    original = text(residual(), generic=True)
    rewrite = rounding.rewrite_rounding
    if fault == "library":
        owner = tmp_path / "automatic_rounding.py"
        owner.write_text("selected implementation")
        (tmp_path / "lower.py").write_text("selected caller")
        monkeypatch.setattr(rounding, "__file__", str(owner))

        def changed(*args, **kwargs):
            result = rewrite(*args, **kwargs)
            owner.write_text("changed implementation")
            return result

        monkeypatch.setattr(rounding, "rewrite_rounding", changed)
    elif fault == "accounting":
        def changed(*args, **kwargs):
            rewrite(*args, **kwargs)
            return 0

        monkeypatch.setattr(rounding, "rewrite_rounding", changed)
    else:
        def unsafe(op, block):
            if isinstance(op, arith.AddiOp):
                return block.args[0]
            return None

        monkeypatch.setattr(rounding, "_candidate", unsafe)
    with pytest.raises(ValueError, match="library changed|accounting|bounded"):
        prepare_automatic_rounding(original, serial=True)


@pytest.mark.parametrize("fault", ["clamp", "operand", "intrinsic", "arithmetic", "attribute", "escape"])
def test_actual_source_reconstruction_rejects_mutations(fault):
    before = residual()
    after = before.clone()
    assert rewrite_rounding(after) == 1
    rounded = next(op for op in after.walk() if op_name(op) == "llvm.intr.roundeven")
    block = rounded.parent
    if fault == "clamp":
        lower = rounded.operands[0].owner
        lower.operands = [block.args[0], lower.operands[1]]
    elif fault == "operand":
        rounded.operands = [block.args[0]]
    elif fault == "intrinsic":
        rounded.attributes["op_name__"] = builtin.StringAttr("llvm.intr.round")
    elif fault == "arithmetic":
        rounded.operands = [block.args[1]]
    elif fault == "attribute":
        rounded.attributes["unknown_effect"] = builtin.UnitAttr()
    else:
        # The newly rounded float cannot become a separate source observation.
        cast = next(use.operation for use in rounded.results[0].uses)
        block.last_op.operands = [cast.operands[0]]
    with pytest.raises(Exception, match="round|quantization|Expected|type|verify|operand"):
        verify_quantization_rewrite(before, after, bounded_intrinsics=True)


@pytest.mark.parametrize("ft,it,lo,hi", [
    ("f32", "i8", -128, 127), ("f32", "i16", -1000, 1000),
    ("f64", "i8", -127, 127), ("f64", "i32", -1048576, 1048576), ("f16", "i8", -127, 127),
])
def test_ordinary_native_complete_outputs_aliased_inputs_and_carried_state(tmp_path, ft, it, lo, hi):
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
    from merlin.llvmlower.lower import lower_model

    if not toolchain.available():
        pytest.skip("selected native lowering toolchain unavailable")
    program = text(residual(ft=ft, it=it, lo=lo, hi=hi), generic=True)
    dtype, integer = np.dtype(ft.replace("f", "float")), np.dtype(it.replace("i", "int"))
    backing = np.resize(np.array([-1000, -128.5, -127.5, -2.5, -.5, -0., 0., .5, 1.5, 2.5, 126.5, 127.5, 1000],
                                  dtype=dtype), 35)
    original = backing.copy()
    x = backing[:-1].reshape(2, 17)
    measurements = {}
    for arm, switch in [("baseline", "-automatic-host-rounding"), ("optimized", "+automatic-host-rounding")]:
        with applied(Selection.parse(switch)):
            result = lower_model(program, tmp_path / arm, targets=("host",), ir_audit=True)
        assert result.stats["automatic_host_rounding"]["rewrites"] == (arm == "optimized")
        llvm = result.ll_path.read_text()
        assert ("@llvm.roundeven." in llvm) == (arm == "optimized")
        assert (" = fsub " in llvm) == (arm == "baseline")
        invoke = HostModel.load(str(result.host_so),
                                image_policy=PrivateHostImagePolicy(result.host_so.parent.resolve()))
        state = backing[1:].reshape(2, 17)
        for _ in range(3):
            outputs = [np.full(66, 99, dtype=t) for t in (integer, dtype)]
            quantized, next_state = [a[16:-16].reshape(2, 17) for a in outputs]
            summed = (x + state).astype(dtype)
            expected = np.clip(np.rint(summed), lo, hi).astype(integer)
            expected_state = (state + np.array(1, dtype=dtype)).astype(dtype)
            invoke([(a.ctypes.data, a.shape) for a in [x, state, quantized, next_state]])
            np.testing.assert_array_equal(quantized, expected)
            np.testing.assert_array_equal(next_state.view(np.uint8), expected_state.view(np.uint8))
            np.testing.assert_array_equal(backing, original)
            for out in outputs:
                np.testing.assert_array_equal(out[:16], np.full(16, 99, dtype=out.dtype))
                np.testing.assert_array_equal(out[-16:], np.full(16, 99, dtype=out.dtype))
            state = next_state
        audit = json.loads(result.audit_index.read_text())
        stages = [row["name"] for row in audit["stages"]]
        if arm == "optimized":
            assert stages.index("host-rounding-selected") < stages.index("xdsl-parsed")
        measurements[arm] = {"image_sha256": invoke.image_sha256,
                             "source_rounding": result.stats["automatic_host_rounding"]}
    (tmp_path / "measurements.json").write_text(json.dumps(measurements, indent=2))
