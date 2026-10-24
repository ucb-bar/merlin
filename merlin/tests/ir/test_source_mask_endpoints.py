"""A mask decided by a closed value interval folds; private generator state is discarded.

Only the IR participates: the uniform draw's bounds come from its bit mask and
literal scale, the threshold from the comparison literal. Case records are
deliberately misleading and must not be read.
"""

import pytest
from mlir_fixtures import forward, generic

from merlin.frontends.linalg_mlir import parse_mlir_file
from merlin.targetgen.source_mask_endpoints import specialize


def fixture(tmp_path, *, threshold=1.0, predicate="olt", private=True, unknown=False, keep=1.0):
    uniform = """%m=arith.constant 9007199254740991 : i64
          %b=arith.andi %a,%m : i64
          %f=arith.uitofp %b : i64 to f64
          %k=arith.constant 1.1102230246251565e-16 : f64
          %v=arith.mulf %f,%k : f64"""
    draw = generic([("%state", "tensor<4xi64>")], "tensor<4xf64>", uniform, name="%u")
    store = '"ml_program.global_store"(%state) <{global=@rng}> : (tensor<4xi64>)->()\n'
    store += '"fixture.effect"() : ()->()\n' if unknown else ""
    decide = f"%t=arith.constant {threshold} : f64\n%c=arith.cmpf {predicate},%a,%t : f64"
    mask = generic([("%u", "tensor<4xf64>")], "tensor<4xi1>", decide, yields="%c", name="%mask")
    scale = f"""%zero=arith.constant 0.0 : f32
          %scale=arith.constant {keep} : f32
          %factor=arith.select %b,%scale,%zero : f32
          %p=arith.mulf %a,%factor : f32"""
    masked = [("%x", "tensor<4xf32>"), ("%mask", "tensor<4xi1>")]
    data = generic(masked, "tensor<4xf32>", scale, yields="%p", name="%data")
    load = '%state="ml_program.global_load"() <{global=@rng}> : ()->tensor<4xi64>\n'
    visibility = "private" if private else "public"
    rng = f'''"ml_program.global"() <{{sym_name="rng",sym_visibility="{visibility}",is_mutable,
        type=tensor<4xi64>,value=dense<1>:tensor<4xi64>}}> : ()->()'''
    results = "(tensor<4xf32>,tensor<4xi1>,tensor<4xf32>)"
    body = f"{load}{draw}\n{store}{mask}\n{data}"
    module = forward("%x:tensor<4xf32>", results, body, ret="%data,%mask,%x", decls=rng)
    (tmp_path / "model.mlir").write_text(str(module))
    return dict(
        cases=[
            dict(
                status="bundled",
                case=dict(
                    overload="aten.add.Tensor",
                    arguments=dict(args=dict(items=[dict(kind="tensor", values="MUST NOT READ"), 0.5, True])),
                ),
                output_indices=[0, 1],
                semantic_boundary=dict(post_indices=[2]),
            )
        ]
    )


@pytest.mark.parametrize("threshold,mask,factor", [(1.0, 1, 1.0), (0.0, 0, 0.0)])
def test_interval_decided_mask_and_private_state_scope(tmp_path, threshold, mask, factor):
    report = fixture(tmp_path, threshold=threshold)
    proof = specialize(tmp_path, report)
    assert proof and [m["mask"] for m in proof["masks"]] == [mask]
    assert proof["masks"][0]["interval"] == ["0", "9007199254740991/9007199254740992"]
    module = parse_mlir_file(tmp_path / "model.mlir")
    module.verify()
    names = [getattr(getattr(op, "op_name", None), "data", op.name) for op in module.walk()]
    assert not any(name.startswith("ml_program.") for name in names)
    assert "builtin.unregistered" not in [op.name for op in module.walk()]
    # The IR's own typed scaling is retained with the decided factor literal.
    product = next(op for op in module.walk() if op.name == "arith.mulf")
    assert any(getattr(getattr(v.owner, "value", None), "value", None) is not None for v in product.operands)
    returned = next(op for op in module.walk() if op.name == "func.return")
    assert returned.operands[1].owner.name == "tensor.splat"
    assert returned.operands[2] is next(op for op in module.walk() if op.name == "func.func").body.block.args[0]
    constants = [op.value.value.data for op in module.walk() if op.name == "arith.constant" and hasattr(op.value, "value")]
    assert factor in constants


@pytest.mark.parametrize(
    "options", [dict(threshold=0.5), dict(threshold=1.5, predicate="ogt"), dict(private=False), dict(unknown=True)]
)
def test_undecided_stochastic_or_external_effects_are_not_erased(tmp_path, options):
    report = fixture(tmp_path, **options)
    original = (tmp_path / "model.mlir").read_bytes()
    if options.get("predicate") == "ogt":
        # u > 1.5 is uniformly false over [0, 1): decided, so it folds.
        assert specialize(tmp_path, report)["masks"][0]["mask"] == 0
        return
    assert specialize(tmp_path, report) is None
    assert (tmp_path / "model.mlir").read_bytes() == original
