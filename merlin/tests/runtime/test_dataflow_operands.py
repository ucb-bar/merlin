"""Operand binding from a command buffer must be dataflow, not pattern-matching, and must name no target.

The single-op binders return ``None`` on a CHAIN -- their own docstring says "unsupported shape (chained
matmuls / no matmul)" -- which left every fused capsule unbindable and therefore ungradeable: flash
attention (attention_qk -> softmax -> matmul -> commit), rmsnorm+qkv, and chained matmuls. Measured on
real buffers, dataflow binds all of them.

These tests deliberately use INVENTED opcodes and no target anywhere, because the property under test is
that neither matters.
"""
from __future__ import annotations

from merlin.runtime.commandbuffer import PRODUCING_KEYS, dataflow_operands


def _cb(tensors, commands):
    return {"abi_version": "0.1", "tensors": tensors, "commands": commands}


def test_binds_a_four_stage_fused_chain():
    """The real flash-attention shape: three leaves in, two intermediates, one committed result."""
    cb = _cb(
        {"Q": {"shape": [16, 32], "dtype": "mxfp8", "role": "input"},
         "K": {"shape": [32, 32], "dtype": "mxfp8", "role": "input"},
         "V": {"shape": [32, 16], "dtype": "mxfp8", "role": "input"},
         "S": {"shape": [16, 32], "dtype": "bf16", "role": "output"},
         "P": {"shape": [16, 32], "dtype": "bf16", "role": "output"},
         "Y0": {"shape": [16, 16], "dtype": "bf16", "role": "output"}},
        [{"opcode": "ATTENTION_QK", "operands": {"q": "Q", "k": "K", "dst": "S"}},
         {"opcode": "SOFTMAX", "operands": {"src": "S", "dst": "P"}},
         {"opcode": "MATMUL", "operands": {"lhs": "P", "rhs": "V", "dst": "acc_Y0"}},
         {"opcode": "COMMIT", "operands": {"src": "acc_Y0", "dst": "Y0"}}])
    assert dataflow_operands(cb) == (["Q", "K", "V"], "Y0")


def test_declared_role_alone_cannot_decide_this():
    """THREE tensors declare role=output in that buffer; two are intermediates. Any binder that trusts
    role has three candidate outputs and no way to choose -- which is why this is dataflow."""
    cb = _cb(
        {"A": {"shape": [4, 4], "role": "input"},
         "M": {"shape": [4, 4], "role": "output"},
         "Z": {"shape": [4, 4], "role": "output"}},
        [{"opcode": "STAGE1", "operands": {"src": "A", "dst": "M"}},
         {"opcode": "STAGE2", "operands": {"src": "M", "dst": "Z"}}])
    roles_say = sorted(n for n, t in cb["tensors"].items() if t.get("role") == "output")
    assert roles_say == ["M", "Z"], "fixture no longer has the ambiguity it is testing"
    assert dataflow_operands(cb) == (["A"], "Z")


def test_opcode_names_are_irrelevant():
    """Same dataflow, opcodes invented for a hypothetical NPU. Binding must not change."""
    cb = _cb(
        {"in0": {"shape": [8, 8], "role": "input"},
         "w": {"shape": [8, 8], "role": "weight"},
         "tmp": {"shape": [8, 8], "role": "output"},
         "res": {"shape": [8, 8], "role": "output"}},
        [{"opcode": "NPU_TILE_ENQUEUE", "operands": {"a": "in0", "b": "w", "dst": "tmp"}},
         {"opcode": "NPU_DRAIN", "operands": {"src": "tmp", "out": "res"}}])
    assert dataflow_operands(cb) == (["in0", "w"], "res")


def test_an_undeclared_accumulator_is_not_mistaken_for_the_output():
    """`acc_*` is produced but absent from `tensors`; the committed result is the answer."""
    cb = _cb(
        {"A": {"shape": [2, 2], "role": "input"}, "Y": {"shape": [2, 2], "role": "output"}},
        [{"opcode": "OP", "operands": {"src": "A", "dst": "acc_scratch"}},
         {"opcode": "COMMIT", "operands": {"src": "acc_scratch", "dst": "Y"}}])
    assert dataflow_operands(cb) == (["A"], "Y")


def test_every_producing_key_spelling_is_honoured():
    """The ABI admits more than one spelling for a result; missing one silently makes a produced tensor
    look like a leaf, i.e. an intermediate would be embedded as an input."""
    for key in PRODUCING_KEYS:
        cb = _cb({"A": {"shape": [2, 2], "role": "input"}, "Y": {"shape": [2, 2], "role": "output"}},
                 [{"opcode": "OP", "operands": {"src": "A", key: "Y"}}])
        assert dataflow_operands(cb) == (["A"], "Y"), f"key {key!r} not treated as producing"


def test_returns_none_when_dataflow_genuinely_cannot_answer():
    """Never a guess: no commands, no tensors, or no declared produced tensor -> None."""
    assert dataflow_operands({"tensors": {}, "commands": []}) is None
    assert dataflow_operands(_cb({"A": {"shape": [1]}}, [])) is None
    # produces only an undeclared name -> no declared output
    assert dataflow_operands(_cb({"A": {"shape": [1], "role": "input"}},
                                 [{"opcode": "OP", "operands": {"src": "A", "dst": "nowhere"}}])) is None
    # consumes nothing declared -> no leaves
    assert dataflow_operands(_cb({"Y": {"shape": [1], "role": "output"}},
                                 [{"opcode": "OP", "operands": {"dst": "Y"}}])) is None


def test_no_target_name_in_the_module():
    """The cardinal rule, asserted on the module that now owns binding."""
    import ast
    import inspect

    from merlin.runtime import commandbuffer as CBM

    tree = ast.parse(inspect.getsource(CBM))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:]
    code = ast.unparse(tree).lower()
    for name in ("radiance", "gemmini", "atlas", "saturn", "muon"):
        assert name not in code, f"commandbuffer.py names the target {name!r} in executable code"


def test_leaf_order_is_deterministic():
    """First-consumption order, so two runs agree. (Not authoritative for a kernel ABI -- documented.)"""
    cb = _cb(
        {"b": {"shape": [2, 2], "role": "weight"}, "a": {"shape": [2, 2], "role": "input"},
         "y": {"shape": [2, 2], "role": "output"}},
        [{"opcode": "OP", "operands": {"lhs": "a", "rhs": "b", "dst": "y"}}])
    first = dataflow_operands(cb)
    assert first == dataflow_operands(cb)
    assert set(first[0]) == {"a", "b"}
