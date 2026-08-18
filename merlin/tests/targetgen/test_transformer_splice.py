"""Whole-model splice for a TRANSFORMER: the non-matmul ops (rmsnorm, softmax, rope, layernorm, silu,
gelu) and the fused ops (attention, geglu) are co-scheduled and GRADED end to end alongside the matmul
layers, not dropped.

These extend ``test_whole_model_splice`` past the ``{matmul, relu, add, mul}`` vocabulary. A transformer's
non-matmul ops are carried in the linalg module as ``linalg.generic`` ops tagged with a ``library_call``
naming the op family (the live-module analog of the ``prov.op`` tags m2m stamps on captured models). The
splice reads that tag STRUCTURALLY, routes each op to its lane (matmuls -> the mesh, the rest -> the
scalar/vector lane), and executes each with a small numpy implementation, threading activations between
lanes. Attention/geglu DECOMPOSE: their matmul sub-ops run on the mesh, their softmax/gelu glue inline.

THE REFERENCE PROBLEM: the toy xDSL engine cannot evaluate the transcendentals (exp/rsqrt/div) these ops
need, so a module containing them is graded against a HOST-EAGER numpy recomputation of the whole module
(f64), selected automatically by ``mesh_program_run._engine_can_lower``. This is what makes
softmax/rmsnorm/rope/attention chains gradeable at all.

Target-agnostic: the lane of each op is READ from the routing plan, never assumed from an op name; the
routing units are a synthetic f32 mesh + vector contract passed as data, so the tests bind to no target.
"""
from __future__ import annotations

import numpy as np
import pytest

from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")

_REF_TARGET = "toy_npu"

# A synthetic, target-agnostic contract: an f32 systolic mesh (matmuls land here) + an f32 vector lane
# (elementwise ops land here). The transformer families (softmax/rmsnorm/rope/attention/…) match NO unit,
# so they route to the scalar/RVV lane — exactly the honest split a matmul mesh produces for a real model.
_F32_UNITS = {
    "compute_units": [
        {"name": "mesh", "kind": "systolic", "dtypes": ["f32", "fp32"], "ops": ["matmul"],
         "accumulate": [{"in": "f32", "weight": "f32", "acc": "f32"}]},
        {"name": "vec", "kind": "vector", "dtypes": ["f32", "fp32"],
         "ops": ["relu", "add", "mul", "elementwise"], "accumulate": []},
    ]
}


def _units():
    from merlin.targetgen import compute_units as cu
    return cu.compute_units(_F32_UNITS)


# --------------------------------------------------------------------------- module builders

def _f32(shape):
    from xdsl.dialects.builtin import TensorType, f32
    return TensorType(f32, list(shape))


def _tagged_generic(family, inputs, out_type):
    """A ``linalg.generic`` tagged ``library_call=family`` — the live-module carrier for a transformer op.
    A trivial (identity-yield) body + identity indexing maps keep it a well-formed generic; the splice
    dispatches on the tag, not the body, and never lowers these modules through the engine."""
    from xdsl.dialects import tensor as td
    from xdsl.dialects.builtin import AffineMapAttr, StringAttr
    from xdsl.dialects.linalg import ops as lo
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap

    init = td.EmptyOp((), out_type)
    body = Block(arg_types=[v.type.get_element_type() for v in inputs] + [out_type.get_element_type()])
    body.add_op(lo.YieldOp(body.args[-1]))
    maps = [AffineMapAttr(AffineMap.identity(len(v.type.get_shape()))) for v in inputs]
    maps.append(AffineMapAttr(AffineMap.identity(len(out_type.get_shape()))))
    iters = [StringAttr("parallel")] * len(out_type.get_shape())
    g = lo.GenericOp(inputs=tuple(inputs), outputs=(init.tensor,), body=Region([body]),
                     indexing_maps=maps, iterator_types=iters, result_types=[out_type],
                     library_call=StringAttr(family))
    return [init, g], g.results[0]


def _module(arg_types, build):
    """Assemble a single-func module: ``build(args) -> (ops, result)``."""
    from xdsl.dialects.builtin import FunctionType, ModuleOp
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.ir import Block, Region

    blk = Block(arg_types=list(arg_types))
    ops, result = build(list(blk.args))
    blk.add_ops([*ops, ReturnOp(result)])
    fn = FuncOp("model", FunctionType.from_lists(list(arg_types), [result.type]), Region([blk]))
    return ModuleOp([fn])


def _matmul(lhs, rhs, out_type):
    from xdsl.dialects import tensor as td
    from xdsl.dialects.linalg import ops as lo
    e = td.EmptyOp((), out_type)
    mm = lo.MatmulOp(inputs=(lhs, rhs), outputs=(e.tensor,), res=(out_type,))
    return [e, mm], mm.results[0]


def _add(lhs, rhs, out_type):
    from xdsl.dialects import tensor as td
    from xdsl.dialects.linalg import ops as lo
    e = td.EmptyOp((), out_type)
    a = lo.AddOp(inputs=(lhs, rhs), outputs=(e.tensor,), res=(out_type,))
    return [e, a], a.results[0]


def _transformer_block(s=4, d=8):
    """rmsnorm(X, G) -> @W1 -> softmax -> @W2 -> + X (residual): a mini transformer sublayer with two mesh
    matmuls, two scalar norms/activations, and a residual that threads the input leaf into a later op."""
    xt, gt, wt = _f32([s, d]), _f32([d]), _f32([d, d])

    def build(args):
        X, G, W1, W2 = args
        ops, rms = _tagged_generic("rmsnorm", [X, G], xt)
        o1, t1 = _matmul(rms, W1, xt)
        o2, sm = _tagged_generic("softmax", [t1], xt)
        o3, t2 = _matmul(sm, W2, xt)
        o4, out = _add(t2, X, xt)
        return [*ops, *o1, *o2, *o3, *o4], out

    return _module([xt, gt, wt, wt], build)


def _softmax_after_matmul(s=4, k=6, n=5):
    """X @ W -> softmax: the only matmul takes the model INPUTS (integer-exact through the f32 mesh), so
    the whole model is bit-exact vs the numpy reference even though softmax is transcendental."""
    at, wt, ot = _f32([s, k]), _f32([k, n]), _f32([s, n])

    def build(args):
        A, W = args
        o1, t1 = _matmul(A, W, ot)
        o2, sm = _tagged_generic("softmax", [t1], ot)
        return [*o1, *o2], sm

    return _module([at, wt], build)


def _leaf_op(family, aux_shapes, s=4, d=8):
    """A single scalar-lane op on the inputs (rmsnorm/rope/layernorm/silu/gelu): a final op with NO matmul
    downstream, so it is bit-exact vs the numpy reference."""
    xt = _f32([s, d])
    aux = [_f32(sh) for sh in aux_shapes]

    def build(args):
        return _tagged_generic(family, args, xt)

    return _module([xt, *aux], build)


def _attention(s=4, d=8):
    """attention_full(Q, K, V) = softmax(Q·Kᵀ/√d)·V — Q·Kᵀ and P·V on the mesh, softmax on the scalar lane."""
    qt = _f32([s, d])

    def build(args):
        return _tagged_generic("attention_full", args, qt)

    return _module([qt, qt, qt], build)


def _geglu(s=4, d=8, h=8):
    """geglu(X, WG, WU) = gelu(X·WG) ⊙ (X·WU) — two mesh matmuls, a gelu gate and an elementwise product."""
    xt, wt, ot = _f32([s, d]), _f32([d, h]), _f32([s, h])

    def build(args):
        return _tagged_generic("geglu", args, ot)

    return _module([xt, wt, wt], build)


# --------------------------------------------------------------------------- op-vocabulary recognition

def test_op_family_reads_library_call_tag():
    """``_op_family`` recognizes a tagged ``linalg.generic`` structurally by its ``library_call``."""
    from merlin.targetgen import mesh_program_run as mp

    fams = [i["family"] for i in mp._ordered_compute_ops(_transformer_block())]
    assert fams == ["rmsnorm", "matmul", "softmax", "matmul", "add"]


def test_transformer_module_is_not_engine_lowerable():
    """A module with a transcendental op is NOT engine-gradeable, so the host-eager reference is selected;
    a pure matmul/elementwise module still is."""
    from merlin.targetgen import mesh_program_run as mp

    assert mp._engine_can_lower(_transformer_block()) is False
    assert mp._engine_can_lower(_softmax_after_matmul()) is False
    assert mp._engine_can_lower(_geglu()) is False


def test_program_lane_tagging_and_matmul_extents():
    """The built program tags the two matmuls onto the mesh and the norm/activation/residual onto the
    scalar lane, carries the real matmul extents, and threads the residual input leaf into the final add."""
    from merlin.targetgen import mesh_program_run as mp
    from merlin.targetgen import routing as rt

    mod = _transformer_block(s=4, d=8)
    plan = rt.route_plan_on(mp.demands_from_module(mod, "f32"), _units())
    prog = mp.build_whole_model_program(plan, _REF_TARGET, mod)

    assert [(s.family, s.lane) for s in prog.steps] == [
        ("rmsnorm", "scalar"), ("matmul", "mesh"), ("softmax", "scalar"),
        ("matmul", "mesh"), ("add", "scalar")]
    mm = [s for s in prog.steps if s.family == "matmul"]
    assert (mm[0].m, mm[0].k, mm[0].n) == (4, 8, 8)
    # the residual add consumes the model input leaf (L0) alongside the second matmul's output.
    assert prog.steps[-1].inputs[1] == "L0"


# --------------------------------------------------------------------------- graded end-to-end (engine mesh)

def test_softmax_after_matmul_is_bit_exact():
    """X@W -> softmax with small-integer operands: the matmul is integer-exact through the f32 mesh and the
    softmax is the same numpy op on both sides, so the spliced whole-model result equals the host-eager
    numpy reference BIT-FOR-BIT — and the reference was selected host-eager, not engine."""
    from merlin.targetgen import mesh_program_run as mp

    r = mp.verify_whole_model_program(_softmax_after_matmul(), target=_REF_TARGET, in_fmt="f32",
                                      units=_units(), int_operands=True)
    assert r["ref_kind"] == "host_eager"
    assert r["exact"] is True
    assert r["n_mesh"] == 1 and r["n_scalar"] == 1


@pytest.mark.parametrize("family,aux", [
    ("rmsnorm", [[8]]), ("layernorm", [[8], [8]]), ("rope", []), ("silu", []), ("gelu", []),
])
def test_scalar_family_leaf_op_is_bit_exact(family, aux):
    """Each scalar-lane family (norm / rotary / activation) as a final op reproduces the host-eager numpy
    reference bit-for-bit — the splice executes it inline, gated against the same numpy math."""
    from merlin.targetgen import mesh_program_run as mp

    r = mp.verify_whole_model_program(_leaf_op(family, aux), target=_REF_TARGET, in_fmt="f32",
                                      units=_units(), int_operands=True)
    assert r["ref_kind"] == "host_eager"
    assert r["exact"] is True
    assert r["n_mesh"] == 0 and r["n_scalar"] == 1


def test_transformer_block_matches_host_eager_reference():
    """The full mini sublayer (rmsnorm -> matmul -> softmax -> matmul -> residual add) splices correctly:
    two matmuls on the mesh, rmsnorm/softmax/add on the scalar lane, activations handed between lanes, and
    the final tensor matches the host-eager numpy reference within float tolerance (the f32 mesh truncates
    the float activations that feed the matmuls — the expected float-lane tolerance, not a bug)."""
    from merlin.targetgen import mesh_program_run as mp

    r = mp.verify_whole_model_program(_transformer_block(), target=_REF_TARGET, in_fmt="f32",
                                      units=_units())
    assert r["ref_kind"] == "host_eager"
    assert r["match"] is True
    assert r["n_mesh"] == 2 and r["n_scalar"] == 3


@pytest.mark.parametrize("mod_fn,n_matmul", [(_attention, 2), (_geglu, 2)])
def test_fused_op_decomposes_and_matches_reference(mod_fn, n_matmul):
    """A fused op (attention / geglu) DECOMPOSES: its matmul sub-ops run on the mesh and its softmax/gelu
    glue inline, and the whole-model final matches the host-eager numpy reference within tolerance."""
    from merlin.targetgen import mesh_program_run as mp

    r = mp.verify_whole_model_program(mod_fn(), target=_REF_TARGET, in_fmt="f32", units=_units())
    assert r["ref_kind"] == "host_eager"
    assert r["match"] is True
    # the fused op is a single scalar-lane step whose matmul sub-ops hit the mesh lane internally.
    assert r["n_steps"] == 1 and r["n_scalar"] == 1 and r["n_mesh"] == 0


# --------------------------------------------------------------------------- splice plumbing (no engine)

def test_fused_op_routes_matmul_subops_through_injected_mesh_executor():
    """With a mesh executor injected (standing in for a real oracle), attention's two matmul sub-ops
    dispatch through it while its softmax runs inline — proving the fused op's matmul sub-ops route to the
    mesh, not the scalar lane. Final == an independent numpy attention (within f32 tolerance)."""
    from merlin.targetgen import mesh_program_run as mp
    from merlin.targetgen import routing as rt

    mod = _attention(s=4, d=8)
    plan = rt.route_plan_on(mp.demands_from_module(mod, "f32"), _units())
    prog = mp.build_whole_model_program(plan, _REF_TARGET, mod)
    rng = np.random.default_rng(0)
    leaves = {lid: rng.standard_normal(tuple(meta["shape"])).astype(np.float32)
              for lid, meta in prog.leaves.items()}

    calls: list = []

    def fake_oracle(lhs, rhs, step):
        calls.append((np.asarray(lhs).shape, np.asarray(rhs).shape))
        return np.asarray(lhs) @ np.asarray(rhs)

    run = mp.run_whole_model_program(prog, leaves, mesh_exec=fake_oracle)
    ref = mp._host_eager_final(prog, leaves)
    assert np.allclose(run["outputs"][prog.output], ref, rtol=1e-4, atol=1e-4)
    # two mesh matmuls: Q·Kᵀ (4x8 · 8x4) then P·V (4x4 · 4x8).
    assert calls == [((4, 8), (8, 4)), ((4, 4), (4, 8))]


def test_fused_op_fails_closed_when_mesh_layer_unavailable():
    """A mesh executor that cannot run attention's matmul sub-op (returns None) raises
    MeshLayerUnavailable — the fused op fails closed rather than fabricating its output."""
    from merlin.targetgen import mesh_program_run as mp
    from merlin.targetgen import routing as rt

    mod = _attention()
    plan = rt.route_plan_on(mp.demands_from_module(mod, "f32"), _units())
    prog = mp.build_whole_model_program(plan, _REF_TARGET, mod)
    leaves = {lid: np.zeros(tuple(meta["shape"]), dtype=np.float32)
              for lid, meta in prog.leaves.items()}
    with pytest.raises(mp.MeshLayerUnavailable):
        mp.run_whole_model_program(prog, leaves, mesh_exec=lambda lhs, rhs, step: None)


# --------------------------------------------------------------------------- on real hardware (radiance)

@pytest.mark.slow
def test_transformer_matmuls_on_radiance_mesh():
    """The mini transformer sublayer runs co-scheduled on the REAL radiance cyclotron mesh: its two matmul
    layers execute on the oracle, rmsnorm/softmax/residual run inline on the scalar lane, and the final is
    gated against the host-eager numpy reference (the engine cannot evaluate softmax/rmsnorm). Skips
    honestly when the radiance OOT package or the cyclotron oracle is absent."""
    from merlin import compile_cli
    from merlin.targetgen import capsule_runner as CR
    from merlin.targetgen import compute_units as cu
    from merlin.targetgen import target_registry as tr

    target = "radiance"
    if compile_cli._default_oot_package(target) is None:
        pytest.skip(f"no OOT package materialized for {target!r} — cannot emit the mesh kernel")
    so = CR._SIM_ORACLES.get(CR._bespoke_sim_via(target))
    if so is None or not so.exclusive:
        pytest.skip(f"{target!r} declares no exclusive bespoke-sim oracle")
    ok, reason = so.available(target)
    if not ok:
        pytest.skip(f"cyclotron oracle unavailable for {target!r}: {reason}")

    # the fp32-family token radiance's SIMT mesh actually declares (f32 vs fp32 are distinct router tokens).
    units = cu.compute_units(tr.load_contract(target))
    tok = next((d for u in units if u.kind == "simt" for d in u.dtypes if d.startswith("f") and "32" in d),
               None)
    assert tok is not None, "radiance contract declares no fp32-family SIMT mesh dtype"

    res = compile_cli.run_whole_model_on_mesh(
        target, _transformer_block(), in_fmt=tok, weight_fmt=tok,
        operand_dtype=tok, accum_dtype=tok, ref_target=_REF_TARGET, seed=0, timeout=900)

    if res.get("status") == "oracle_unavailable":
        pytest.skip(res.get("reason", "radiance mesh oracle unavailable at run time"))
    assert res["ref_kind"] == "host_eager"
    assert res["status"] == "pass", res
    assert res["n_mesh"] == 2 and res["n_scalar"] == 3
    assert all(layer["oracle"] == "ok" for layer in res["per_layer"])
