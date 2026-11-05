"""Exact inverse mapping, complete operands and conservative checking scope."""

from dataclasses import dataclass, replace

import pytest
from test_compact_dense import ADAPTER, encode

from merlin.compile.scheduling.block_schedule import Compute, Preload
from merlin.compile.scheduling.dense_panel_pipeline import _collapse_compute_radices
from merlin.sched.ir import Call, Expr, Kernel, Loop, Ptr, TensorArg, Var, check_structure, select
from merlin.sched.ir.kernel import concretize, instances


def sample():
    k, m = Var("k"), Var("m")
    n = Var("stripe") * 4 + Var("within")
    preload = encode(Preload(k * 64 + n * 4, (k, n), m * 48 + n * 4,
                             select("k", 0, 0, 1), 2, 4, 3))
    preload = replace(preload, args=(*preload.args, ("audit_ptr", Ptr("audit", n * 2 + m))))
    compute = encode(Compute(m * 18 + k * 3, (m, k), 2, 3, True))
    return Kernel("inverse", (TensorArg("audit", (4096,), "i8", "read"),),
                  (Loop("k", 2, (Loop("stripe", 3, (Loop("within", 4,
                    (Loop("m", 2, (preload, compute)),)),)),)),))


def concrete(kernel):
    return [(call.instr, concretize(call, env)) for call, env in instances(kernel)]


def map_pair(kernel, edit):
    def walk(body):
        if len(body) == 2 and all(not isinstance(s, Loop) for s in body):
            return edit(body)
        return tuple(replace(s, body=walk(s.body)) if type(s) is Loop else s for s in body)
    return replace(kernel, body=walk(kernel.body))


def test_compute_collapse_has_exact_forward_inverse_and_original_descriptor_order():
    original = sample()
    actual, changes = _collapse_compute_radices(original, ADAPTER)
    assert not check_structure(actual) and actual != original
    assert len(changes) == 1 and changes[0]["joined_extent"] == 12
    assert concrete(actual) == concrete(original)
    assert [c.instr for c, _ in instances(actual)] == ["preload", "compute"] * 48
    for (before, env), (after, mapped) in zip(instances(original), instances(actual), strict=True):
        index = mapped["stripe"]
        assert (index // 4, index % 4) == (env["stripe"], env["within"])
        assert concretize(before, env) == concretize(after, mapped)
    twice, changes = _collapse_compute_radices(actual, ADAPTER)
    assert twice == actual and not changes


@dataclass(frozen=True)
class Unknown(Expr):
    pass


@pytest.mark.parametrize("extra", [Var("stripe"), Var("within"),
                                  select("within", 3, 2, 4), Unknown()])
def test_every_scalar_tag_or_partial_tail_must_obey_the_exact_coefficient_relation(extra):
    original = map_pair(sample(), lambda pair: (replace(pair[0], args=(*pair[0].args, ("tag", extra))), pair[1]))
    actual, changes = _collapse_compute_radices(original, ADAPTER)
    assert actual == original and not changes


@pytest.mark.parametrize("edit", [lambda pair: tuple(reversed(pair)),
                                   lambda pair: (replace(pair[0], unit="execute"), pair[1])])
def test_changed_pair_order_or_additional_effects_are_not_collapsed(edit):
    original = map_pair(sample(), edit)
    actual, changes = _collapse_compute_radices(original, ADAPTER)
    assert actual == original and not changes


def test_completion_calls_never_enter_the_target_compute_decoder():
    from merlin.compile.scheduling.dense_panel_pipeline import DRAIN

    original = map_pair(sample(), lambda pair: (pair[0], Call(DRAIN, ())))

    def project(call):
        if call.instr == DRAIN:
            raise AssertionError("completion must not be treated as a compute operand")
        return ADAPTER.project(call)

    actual, changes = _collapse_compute_radices(original, replace(ADAPTER, project=project))
    assert actual == original and not changes


def test_outer_inner_shadowing_and_signed_extent_overflow_cannot_change_binder_scope():
    original = sample()
    k = original.body[0]
    stripe = k.body[0]
    within = stripe.body[0]
    invalid = replace(original, body=(replace(k, body=(replace(stripe, body=(replace(within, var="stripe"),)),)),))
    with pytest.raises(ValueError, match="unshadowed"):
        _collapse_compute_radices(invalid, ADAPTER)
    huge = replace(original, body=(replace(k, body=(replace(stripe, extent=1 << 30),)),))
    actual, changes = _collapse_compute_radices(huge, replace(ADAPTER, index_bits=32))
    assert actual == huge and not changes


def test_checked_segmented_stripes_record_the_inverse_but_emit_original_commands():
    from test_dense_segmented_pipeline import chosen, inputs
    from test_segmented_loop_transforms import consume

    from merlin.compile.scheduling.dense_panel_pipeline import dense_panel_rhs_stripes
    from merlin.compile.scheduling.loop_schedule import propose_loop_transforms

    p, r, s, b, admission = inputs((4, 14, 8), 2)
    original = chosen(p, r, s, b, admission, issue_order="prefetch_rhs_between_columns")
    requests = dense_panel_rhs_stripes(original.kernel, ADAPTER, factor=2)
    actual = consume(p, r, s, b, admission, requests)
    checking = actual.realization["source_phase_fission"]
    assert checking["compute_radix_collapses"]
    assert checking["actual_kernel_sha256"] == actual.kernel.digest()
    assert actual.kernel == propose_loop_transforms(original.kernel, requests, index_bits=ADAPTER.index_bits)
    assert actual.realization["source"]["output_tiles"] == original.realization["source"]["output_tiles"]
    actual.materialize_scf().verify()
