"""Closed pointer-source binding controls; no execution or target qualification."""

import copy
import hashlib

import pytest
from merlin_experiments.phase0.connected_integer_bounds import derive
from test_direct_source_contraction import SOURCE
from xdsl.dialects import builtin

from merlin.common import mlir_query as mq
from merlin.llvmlower.direct_source_contraction import bind_direct_source_contraction
from merlin.llvmlower.pointer_source_contractions import bind_pointer_source_contractions


def fixture(tmp_path, *, distinct_rhs=False, reordered=False):
    # Two equal-shaped LHS entries make an ordinal-based output mapping wrong
    # even when a tensor-shape check alone cannot distinguish the outputs.
    shapes = [(1, 3), (2, 3), (3, 3), (1, 3), (3, 4), (3, 4)]
    pairs = [(0, 4), (3, 5 if distinct_rhs else 4), (1, 4), (2, 4)]
    order = [0, 2, 3, 1] if reordered else list(range(4))
    signature = ", ".join(f"%a{i}: tensor<{r}x{c}xi8>" for i, (r, c) in enumerate(shapes))
    result_types = [f"tensor<{shapes[pairs[i][0]][0]}x4xi32>" for i in order]
    body = SOURCE.split("%z =", 1)[1].split("func.return", 1)[0]
    dots = []
    for i, (lhs, rhs) in enumerate(pairs):
        m = shapes[lhs][0]
        text = "%z =" + body
        text = text.replace("2x3xi8", f"{m}x3xi8").replace("2x4xi32", f"{m}x4xi32")
        for before, after in (
            ("%a", f"%a{lhs}"),
            ("%b", f"%a{rhs}"),
            ("%z", f"%z{i}"),
            ("%s ", f"%s{i} "),
            ("%s :", f"%s{i} :"),
            ("%r =", f"%r{i} ="),
        ):
            text = text.replace(before, after)
        dots.append(text)
    text = "module { func.func @forward(" + signature + ") -> (" + ", ".join(result_types) + ") {\n"
    text += "\n".join(dots) + "func.return " + ", ".join(f"%r{i}" for i in order)
    text += " : " + ", ".join(result_types) + "\n}}\n"
    path = tmp_path / "source.mlir"
    path.write_text(text)
    # Arbitrary output labels follow return positions, never a suffix convention.
    outputs = ["result_omega", "result_alpha", "result_delta", "result_beta"]
    names = [f"input_{i}" for i in range(len(shapes))]
    cb = dict(
        compiler_pointer_abi=dict(version=1, arguments=names + outputs),
        tensors={
            **{
                name: dict(shape=list(shape), dtype="i8", role="input")
                for name, shape in zip(names, shapes, strict=True)
            },
            **{
                name: dict(shape=[shapes[pairs[i][0]][0], 4], dtype="i32", role="output")
                for name, i in zip(outputs, order, strict=True)
            },
        },
    )
    proof = derive(mq.parse(text), [(-8, 8)] * len(shapes))
    proof["interface_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, cb, proof, pairs, order


@pytest.mark.parametrize("distinct_rhs", [False, True])
@pytest.mark.parametrize("reordered", [False, True])
def test_complete_shared_or_distinct_rhs_and_positional_return_names(tmp_path, distinct_rhs, reordered):
    path, cb, proof, pairs, order = fixture(tmp_path, distinct_rhs=distinct_rhs, reordered=reordered)
    before = path.read_bytes(), copy.deepcopy(cb), copy.deepcopy(proof)
    bound = bind_pointer_source_contractions(path, cb, proof)
    assert [r["ordinal"] for r in bound] == [2, 5, 8, 11]
    assert [(r["lhs"], r["rhs"]) for r in bound] == [(f"input_{a}", f"input_{b}") for a, b in pairs]
    outputs = cb["compiler_pointer_abi"]["arguments"][6:]
    assert [r["output"] for r in bound] == [outputs[order.index(i)] for i in range(4)]
    assert sum(r["m"] * r["k"] * r["n"] for r in bound) == 84
    assert before == (path.read_bytes(), cb, proof)
    assert "kernel_abi" not in cb and "params" not in cb
    with pytest.raises(ValueError):
        bind_direct_source_contraction(path, cb, proof)


@pytest.mark.parametrize(
    "fault",
    [
        "stale_source",
        "missing_proof_op",
        "extra_proof_op",
        "duplicate_proof_ordinal",
        "wrong_owner",
        "boolean_owner",
        "bad_input_abi",
        "bad_output_abi",
        "bad_output_owner",
        "bad_output_interval",
        "bad_product",
        "bad_scalar",
        "bad_extent",
        "bad_initial",
        "bad_input_interval",
        "alias",
        "missing_tensor",
        "extra_tensor",
        "version",
        "layout",
        "unknown_storage",
        "noninteger",
        "return_order_shape",
        "boolean_shape",
        "global_plan",
    ],
)
def test_stale_or_incomplete_proof_storage_and_aliases_refuse(tmp_path, fault):
    path, cb, proof, *_ = fixture(tmp_path, distinct_rhs=True, reordered=True)
    rows = proof["operations"]
    if fault == "stale_source":
        path.write_text(path.read_text() + "\n")
    elif fault == "missing_proof_op":
        rows.pop()
    elif fault == "extra_proof_op":
        rows.append(copy.deepcopy(rows[-1]))
    elif fault == "duplicate_proof_ordinal":
        rows[5]["ordinal"] = 2
    elif fault == "wrong_owner":
        rows[2]["inputs"][0] = dict(argument=1)
    elif fault == "boolean_owner":
        rows[8]["inputs"][0] = dict(argument=True)
    elif fault == "bad_input_abi":
        proof["input_abi"][0]["shape"] = [2, 3]
    elif fault == "bad_output_abi":
        proof["output_abi"].reverse()
    elif fault == "bad_output_owner":
        proof["output_owners"].reverse()
    elif fault == "bad_output_interval":
        proof["output_intervals"][0] = [0, 0]
    elif fault == "bad_product":
        rows[2]["product_intervals"] = [[0, 0]]
    elif fault == "bad_scalar":
        rows[2]["scalar_operations"][0] = "arith.extui"
    elif fault == "bad_extent":
        rows[2]["reduction_extent"] = 2
    elif fault == "bad_initial":
        rows[2]["initial_interval"] = [1, 1]
    elif fault == "bad_input_interval":
        proof["input_intervals"][0] = [-129, 127]
    elif fault == "alias":
        cb["compiler_pointer_abi"]["arguments"][-1] = "input_0"
    elif fault == "missing_tensor":
        del cb["tensors"]["input_0"]
    elif fault == "extra_tensor":
        cb["tensors"]["foreign"] = dict(cb["tensors"]["input_0"])
    elif fault == "version":
        cb["compiler_pointer_abi"]["version"] = True
    elif fault == "layout":
        cb["tensors"]["input_0"]["layout"] = "transposed"
    elif fault == "unknown_storage":
        cb["tensors"]["input_0"]["alias"] = "input_1"
    elif fault == "noninteger":
        cb["tensors"]["input_0"]["dtype"] = "f32"
    elif fault == "return_order_shape":
        outputs = cb["compiler_pointer_abi"]["arguments"][6:]
        cb["compiler_pointer_abi"]["arguments"][6:] = [outputs[i] for i in (0, 3, 1, 2)]
    elif fault == "boolean_shape":
        cb["tensors"]["input_0"]["shape"][0] = True
    else:
        cb["params"] = {"global_program_plan": {"untrusted": "not an ABI override"}}
    with pytest.raises(ValueError):
        bind_pointer_source_contractions(path, cb, proof)


@pytest.mark.parametrize(
    "fault",
    [
        "nonzero",
        "unsigned",
        "extra_arithmetic",
        "derived_operand",
        "duplicate_return",
        "unreturned_dot",
        "duplicate_dot",
        "unencoded",
        "unknown_effect",
    ],
)
def test_source_semantics_not_only_proof_records_are_checked(tmp_path, fault):
    path, cb, proof, *_ = fixture(tmp_path, distinct_rhs=True)
    text = path.read_text()
    if fault == "nonzero":
        text = text.replace("arith.constant 0", "arith.constant 1", 1)
    elif fault == "unsigned":
        text = text.replace("arith.extsi", "arith.extui", 1)
    elif fault == "extra_arithmetic":
        extra = """%extra = linalg.generic {
indexing_maps = [affine_map<(m,n)->(m,n)>, affine_map<(m,n)->(m,n)>],
iterator_types = ["parallel", "parallel"]
} ins(%r0 : tensor<1x4xi32>) outs(%s0 : tensor<1x4xi32>) {
^bb0(%value:i32,%initial:i32):
%one = arith.constant 1 : i32
%plus = arith.addi %value,%one : i32
linalg.yield %plus : i32
} -> tensor<1x4xi32>
"""
        text = text.replace("func.return %r0", extra + "func.return %extra")
    elif fault == "derived_operand":
        text = text.replace("%z0 =", "%derived = tensor.cast %a0 : tensor<1x3xi8> to tensor<1x3xi8>\n%z0 =")
        text = text.replace("ins(%a0,", "ins(%derived,", 1)
    elif fault == "duplicate_return":
        text = text.replace("%r0, %r1,", "%r1, %r1,")
    elif fault == "unreturned_dot":
        lines = text.splitlines()
        lines[0] = lines[0].replace(", tensor<3x4xi32>", "")
        i = next(i for i, line in enumerate(lines) if line.startswith("func.return"))
        lines[i] = lines[i].replace(", %r3", "").replace(", tensor<3x4xi32>", "")
        text = "\n".join(lines) + "\n"
        del cb["tensors"][cb["compiler_pointer_abi"]["arguments"].pop()]
    elif fault == "duplicate_dot":
        text = text.replace("ins(%a3,%a5", "ins(%a0,%a4")
    elif fault == "unencoded":
        text = text.replace("tensor<1x3xi8>", 'tensor<1x3xi8, "encoding">')
    if fault == "unknown_effect":
        module = mq.parse(text)
        next(mq.walk(module, "arith.constant")).attributes["unknown"] = builtin.StringAttr("effect")
        text = str(module)
    # These are valid complete source proofs, including the forbidden source
    # operation/return. The binder must refuse semantics, not a missing row.
    if fault != "unsigned":
        proof = derive(mq.parse(text), [(-8, 8)] * 6)
    path.write_text(text)
    proof["interface_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        bind_pointer_source_contractions(path, cb, proof)


def test_shared_scalar_zero_is_checked_without_assuming_three_ops_per_dot(tmp_path):
    path, cb, _proof, *_ = fixture(tmp_path, distinct_rhs=True)
    text = path.read_text()
    for i in (1, 2, 3):
        text = text.replace(f"%z{i} = arith.constant 0 : i32\n", "").replace(f"%z{i} ", "%z0 ")
    path.write_text(text)
    proof = derive(mq.parse(text), [(-8, 8)] * 6)
    proof["interface_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert [r["ordinal"] for r in bind_pointer_source_contractions(path, cb, proof)] == [2, 4, 6, 8]


@pytest.mark.parametrize("which", ["source", "command_buffer", "proof"])
def test_current_input_mutation_during_recognition_refuses(tmp_path, monkeypatch, which):
    from merlin.llvmlower import pointer_source_contractions as binding

    path, cb, proof, *_ = fixture(tmp_path, distinct_rhs=True)
    original = binding.recognize_signed_integer_matmul

    def recognize(operation, dtypes):
        result = original(operation, dtypes)
        if which == "source":
            path.write_text(path.read_text() + "\n")
        elif which == "command_buffer":
            cb["target"] = "changed while binding"
        else:
            proof["scope"] = "changed while binding"
        return result

    monkeypatch.setattr(binding, "recognize_signed_integer_matmul", recognize)
    with pytest.raises(ValueError, match="changed"):
        binding.bind_pointer_source_contractions(path, cb, proof)
