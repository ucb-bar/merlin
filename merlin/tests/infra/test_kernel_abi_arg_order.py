"""The kernel ABI's argument order has ONE definition, and every harness path agrees with it.

The contract used to carry a single prose ``arg_order`` string — weight-first — which was true of
exactly one of the three command shapes a runner-owned harness renderer dispatches to. The whole-op
renderer passes the interface's DECLARATION order and the pure-movement renderer passes ``(src, dst)``;
neither was written down anywhere. Twenty-eight shipped capsules declare their weight first, so
declaration order and weight-first coincide and nothing noticed. The four that declare the activation
first (``IFM, W, Y0``) were compiled weight-first against a harness passing activation-first: the kernel
gathered activations out of the weight buffer and MVIN'd the weight with the activation's pitch. Four
functional failures, one undocumented rule.

These tests cover the gate that makes that undetectable-by-hand divergence a build failure, including
its refusal behaviour — a check that could not run must never report success.
"""
from __future__ import annotations

import importlib.util
import types

import pytest
import yaml

from merlin.common.paths import repo_root

ROOT = repo_root()
CONTRACT = ROOT / "merlin" / "contract" / "mlir_oot_backend_contract.yaml"


def _gate():
    """The gate module, imported from build_tools (it is a script, not a package)."""
    path = ROOT / "build_tools" / "scripts" / "check_kernel_abi_arg_order.py"
    spec = importlib.util.spec_from_file_location("_check_kernel_abi_arg_order", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_harness_shape_matches_the_contract():
    problems = _gate().check()
    assert not problems, ("kernel ABI argument order drifted:\n"
                          + "\n".join(f"  - {p}" for p in problems))


def test_contract_documents_one_row_per_dispatched_command_shape():
    abi = (yaml.safe_load(CONTRACT.read_text(encoding="utf-8")) or {})["kernel_abi"]
    shapes = [row["shape"] for row in abi["arg_order_by_command_shape"]]
    assert shapes == ["movement", "native_whole_op", "resident_matmul"], (
        "the rows are tried top-down and the first match decides the ABI, so their ORDER is part of "
        f"the contract; got {shapes}")
    for row in abi["arg_order_by_command_shape"]:
        assert row.get("when"), f"{row['shape']}: no `when` — a reader could pick the wrong row"
        assert row.get("order"), f"{row['shape']}: no machine-checkable `order` token list"
        assert row.get("signature"), f"{row['shape']}: no signature"
    # The flat legacy key is still consumed (merlin.llvmlower.device_shim.kernel_abi_for renders it into
    # the emitted device shim), so it must keep saying WHICH shape it describes rather than reading as
    # the rule for all of them.
    assert "resident_matmul" in abi["arg_order"]
    assert "resident_matmul" in abi["signature"], (
        "the flat signature is the resident-matmul one and must say so; it read as THE "
        "signature for three shapes with different arities")


def test_resident_matmul_order_is_group_major_not_command_order():
    """The old prose said 'matmul lhs in command order'. With two resident weights it is not.

    The emitter processes one resident weight at a time (mvin, run its matmuls, mvout, reuse the
    scratchpad), so the lhs block is grouped by weight. A probe with one weight cannot tell the two
    apart, which is why the gate's probe interleaves two.
    """
    gate = _gate()
    cb = gate._probe_resident_matmul()
    lhs = gate.resolve_token("matmul_lhs_group_major", cb)
    command_order = [c["operands"]["lhs"] for c in cb["commands"]
                     if c["opcode"] in ("MATMUL", "MATMUL_RESIDENT")]
    assert lhs == ["A0", "A2", "A1"], lhs
    assert lhs != command_order, ("the probe no longer distinguishes group-major from command order, "
                                  "so it cannot prove which one the harness implements")
    assert gate.resolve_token("commit_outputs_group_major", cb) == ["Y0", "Y2", "Y1"]
    assert gate.resolve_token("resident_weights_in_resident_pack_order", cb) == ["W0", "W1"]


def test_whole_op_order_follows_declaration_order_not_role_order():
    """The whole-op shape is declaration-order keyed: the SAME conv declared the other way round gets
    the other pointer order. That is the fact the four failing capsules needed written down."""
    gate = _gate()
    cb = gate._probe_native_whole_op("CONV2D")
    assert gate.resolve_token("interface_external_tensors_in_declaration_order", cb) == \
        ["IFM", "W", "Y0"]
    swapped = {"tensors": {"W": cb["tensors"]["W"], "IFM": cb["tensors"]["IFM"],
                           "Y0": cb["tensors"]["Y0"]},
               "commands": cb["commands"]}
    assert gate.resolve_token("interface_external_tensors_in_declaration_order", swapped) == \
        ["W", "IFM", "Y0"]


def test_gate_catches_a_harness_that_reorders_to_weight_first():
    """DRIFT PROOF. A renderer that emits the weight first on the whole-op shape — the exact mistake
    the four conv failures were — must be reported, not passed."""
    gate = _gate()
    abi = (yaml.safe_load(CONTRACT.read_text(encoding="utf-8")) or {})["kernel_abi"]
    whole_op = next(r for r in abi["arg_order_by_command_shape"] if r["shape"] == "native_whole_op")
    fake = types.SimpleNamespace(_NATIVE_INTERFACE_OPS=frozenset(whole_op["opcodes"]))

    def render(cb, *, target):
        names = [n for n, s in cb["tensors"].items() if s.get("role") in gate._EXTERNAL_ROLES]
        weight_first = ([n for n in names if cb["tensors"][n].get("role") == "weight"]
                        + [n for n in names if cb["tensors"][n].get("role") != "weight"])
        args = ", ".join(f"(void*)T_{n}" for n in weight_first)
        return f"int main() {{\n  {target}_kernel({args});\n  return 0;\n}}\n"

    monkey = lambda: [("t", fake, render)]                              # noqa: E731
    gate._renderers = monkey
    problems = gate.check()
    assert any("t/native_whole_op" in p and "harness passes" in p for p in problems), problems


def test_gate_fails_closed_on_an_unreadable_contract(tmp_path):
    gate = _gate()
    empty = tmp_path / "contract.yaml"
    empty.write_text("version: '0.1'\n", encoding="utf-8")
    gate._CONTRACT = empty
    problems = gate.check()
    assert problems and any("does NOT report success" in p for p in problems), problems


def test_gate_fails_closed_when_no_backend_exposes_a_renderer():
    gate = _gate()
    gate._renderers = lambda: []
    problems = gate.check()
    assert problems and any("does NOT report success" in p for p in problems), problems


def test_gate_fails_closed_when_the_whole_op_roster_cannot_be_read():
    """A backend whose whole-op opcode set is unreadable cannot be certified — a fourth command shape
    must not be able to arrive undocumented just because the gate could not see the roster."""
    gate = _gate()
    real = gate._renderers()
    assert real, "no registered backend exposes render_harness; this test cannot certify anything"
    target, _owner, render = real[0]
    gate._renderers = lambda: [(target, types.SimpleNamespace(), render)]
    problems = gate.check()
    assert any(gate._WHOLE_OP_ATTR in p and "does NOT report success" in p for p in problems), problems


def test_gate_rejects_an_order_token_it_cannot_resolve():
    gate = _gate()
    with pytest.raises(gate.Unresolvable):
        gate.resolve_token("weight_first_because_i_said_so", gate._probe_movement())


def test_emitted_call_parser_survives_the_pointer_casts():
    """Every argument carries a ``(void*)`` cast, so taking the FIRST close paren truncates the list to
    one bogus argument. The parser matches by depth; this pins that."""
    gate = _gate()
    text = 'int main() {\n  t_kernel((void*)T_W, (void*)T_A0, (void*)T_Y0);\n}\n'
    assert gate.emitted_call_args(text, "t_kernel") == ["W", "A0", "Y0"]
    assert gate.emitted_call_args("extern void t_kernel();\n", "t_kernel") is None
    assert gate.emitted_call_args("  t_kernel((void*)T_W,\n", "t_kernel") is None
    assert gate.emitted_call_args("  t_kernel();\n", "t_kernel") == []
