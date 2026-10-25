"""Complete symbolic loaded-sum coverage, storage and mutation checks."""

import hashlib
import json
from dataclasses import replace

import pytest

from merlin.compile.scheduling.compact_loaded_sum import (
    LoadedSumContract,
    LoadedSumEffect,
    LoadedSumLayout,
    verify_compact_loaded_sum,
)
from merlin.sched.ir import Call, Const, Kernel, Loop, TensorArg, Var, evaluate, instances, select


def fixture(rows, columns):
    proof = {"source": "original scalar contract", "proposal": "loaded pair", "bound_lsb": 0}
    identity = hashlib.sha256(json.dumps(proof, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    layout = LoadedSumLayout(rows, columns, 4, 3, 8, 16, 80)
    contract = LoadedSumContract(
        identity, ((("scale", 11),), (("scale", 13),), (("scale", 17),)), "independent-synchronous-adapter", True, True
    )
    mt, nt = (rows + 3) // 4, (columns + 2) // 3
    args = (
        ("row", Var("r") * 4),
        ("col", Var("c") * 3),
        ("rows", select("r", mt - 1, (rows - 1) % 4 + 1, 4)),
        ("cols", select("c", nt - 1, (columns - 1) % 3 + 1, 3)),
        ("base", Const(8)),
    )
    calls = tuple(
        Call(kind, args + (("scale", Const(scale)),))
        for kind, scale in (("initialize", 11), ("accumulate", 13), ("store", 17))
    )
    kernel = Kernel(
        "sum",
        (
            TensorArg("lhs", (rows, columns), "i8", "read"),
            TensorArg("rhs", (rows, columns), "i8", "read"),
            TensorArg("output", (rows, columns), "i8", "write"),
        ),
        (Loop("r", mt, (Loop("c", nt, calls),)),),
    )

    def project(call):
        assert tuple(k for k, _ in call.args) == ("row", "col", "rows", "cols", "base", "scale")
        a = dict(call.args)
        tensor = {"initialize": "lhs", "accumulate": "rhs", "store": "output"}[call.instr]
        return LoadedSumEffect(
            call.instr,
            tensor,
            a["row"],
            a["col"],
            a["rows"],
            a["cols"],
            a["base"],
            16,
            identity,
            (("scale", evaluate(a["scale"], {})),),
            "synchronous",
        )

    return kernel, layout, contract, project, proof


@pytest.mark.parametrize("shape", [(1, 1), (4, 3), (5, 7), (9, 2)])
def test_symbolic_cover_matches_every_original_element_and_active_accumulator_byte(shape):
    kernel, layout, contract, project, proof = fixture(*shape)
    report = verify_compact_loaded_sum(kernel, layout, contract, project, scalar_proof=proof)
    covered = {kind: [] for kind in ("initialize", "accumulate", "store")}
    for call, env in instances(kernel):
        effect = project(call)
        r, c, h, w = (evaluate(x, env) for x in (effect.row, effect.column, effect.rows, effect.columns))
        for i in range(h):
            for j in range(w):
                covered[effect.kind].append((r + i, c + j))
                assert (
                    layout.accumulator_base_bytes + i * layout.accumulator_row_bytes + j * 4 + 4
                    <= layout.accumulator_capacity_bytes
                )
    expected = [(r, c) for r in range(shape[0]) for c in range(shape[1])]
    assert all(sorted(values) == expected for values in covered.values())
    assert report["output_elements"] == len(expected)


@pytest.mark.parametrize("rows, expected_calls", [(6272, 4704), (12544, 9408)])
def test_large_domains_project_only_three_effects(rows, expected_calls, monkeypatch):
    kernel, _, contract, _, proof = fixture(1, 1)
    layout = LoadedSumLayout(rows, 64, 16, 16, 0, 64, 1024)
    mt, nt = rows // 16, 4
    params = (("row", Var("r") * 16), ("col", Var("c") * 16), ("rows", Const(16)), ("cols", Const(16)))
    body = tuple(Call(kind, params) for kind in ("initialize", "accumulate", "store"))
    kernel = replace(
        kernel,
        args=tuple(replace(a, shape=(rows, 64)) for a in kernel.args),
        body=(Loop("r", mt, (Loop("c", nt, body),)),),
    )
    projected = []

    def forbidden(*args, **kwargs):
        raise AssertionError("compact verification must not enumerate instances")

    monkeypatch.setattr("merlin.sched.ir.instances", forbidden)
    monkeypatch.setattr("merlin.sched.ir.kernel.instances", forbidden)

    def project(call):
        index = ("initialize", "accumulate", "store").index(call.instr)
        projected.append(call)
        a = dict(call.args)
        return LoadedSumEffect(
            call.instr,
            ("lhs", "rhs", "output")[index],
            a["row"],
            a["col"],
            a["rows"],
            a["cols"],
            Const(0),
            64,
            contract.scalar_proof_sha256,
            contract.parameters[index],
            "synchronous",
        )

    result = verify_compact_loaded_sum(kernel, layout, contract, project, scalar_proof=proof)
    assert result["dynamic_calls"] == expected_calls
    assert len(projected) == result["checked_static_effects"] == 3


@pytest.mark.parametrize("mutation", ["missing_store", "order", "rows", "columns", "tail", "base", "scale", "async"])
def test_incomplete_reordered_or_changed_actual_schedule_refuses(mutation):
    kernel, layout, contract, project, proof = fixture(5, 7)
    (outer,) = kernel.body
    (inner,) = outer.body
    calls = list(inner.body)
    if mutation == "missing_store":
        calls.pop()
    elif mutation == "order":
        calls[0], calls[1] = calls[1], calls[0]
    elif mutation in ("rows", "columns"):
        target = outer if mutation == "rows" else inner
        changed = replace(target, extent=target.extent - 1)
        kernel = replace(kernel, body=(changed if mutation == "rows" else replace(outer, body=(changed,)),))
    elif mutation == "async":
        calls[1] = replace(calls[1], produces="later")
    else:
        key, value = {"tail": ("cols", Const(3)), "base": ("base", Const(12)), "scale": ("scale", Const(19))}[mutation]
        calls[1] = replace(calls[1], args=tuple((k, value if k == key else v) for k, v in calls[1].args))
    if mutation not in ("rows", "columns"):
        kernel = replace(kernel, body=(replace(outer, body=(replace(inner, body=tuple(calls)),)),))
    with pytest.raises(ValueError):
        verify_compact_loaded_sum(kernel, layout, contract, project, scalar_proof=proof)


def test_capacity_scalar_identity_source_signature_and_synchrony_refuse():
    kernel, layout, contract, project, proof = fixture(5, 7)
    cases = [
        (kernel, replace(layout, accumulator_capacity_bytes=67), contract, project, proof),
        (kernel, replace(layout, accumulator_row_bytes=8), contract, project, proof),
        (kernel, replace(layout, rows=1 << 31, index_bits=32), contract, project, proof),
        (kernel, layout, replace(contract, private_ownership=False), project, proof),
        (kernel, layout, replace(contract, external_buffers_disjoint=False), project, proof),
        (kernel, layout, contract, project, {**proof, "bound_lsb": 1}),
        (replace(kernel, args=kernel.args[:-1]), layout, contract, project, proof),
        (kernel, layout, contract, lambda c: replace(project(c), completion="asynchronous"), proof),
        (kernel, layout, contract, lambda c: replace(project(c), scalar_proof_sha256="0" * 64), proof),
    ]
    for args in cases:
        with pytest.raises(ValueError):
            verify_compact_loaded_sum(*args[:4], scalar_proof=args[4])


def test_mutation_during_projection_invalidates_proof():
    kernel, layout, contract, project, proof = fixture(5, 7)

    def mutate(call):
        result = project(call)
        proof["extra"] = "changed while projecting"
        return result

    with pytest.raises(ValueError, match="changed during projection"):
        verify_compact_loaded_sum(kernel, layout, contract, mutate, scalar_proof=proof)
