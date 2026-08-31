"""The canonical interface parser must fail CLOSED on an op it does not define.

MEASURED, and the reason this file exists: ``parse_interface_mlir`` used to return only the commands
it recognised and report nothing at all about the rest, so a module using an op outside the frozen
grammar parsed "successfully" into a SHORTER command list. Across the shipped corpora that mis-read
15 of 160 interface capsules:

* ``movement`` (5 capsules) -- the capsule's ONLY op, so each parsed to zero commands;
* ``attention_pv`` (7 flash-attention capsules) -- each lost its second matmul, the ``P @ V``;
* ``conv2d`` (3 capsules) -- the compute vanished and only ``resident_pack``/``evict`` remained.

A backend package built on the starter kit (which tells agents to use this parser instead of writing
their own) then emits code computing the wrong thing, with nothing anywhere pointing at why. So the
contract here is two-sided: an op the grammar DOES define must reach the command list, and an op it
does not must raise with its mnemonic named. A loud unimplemented op beats a silent wrong answer.

All three gaps are now closed (``conv2d``'s own semantics are covered by ``test_conv2d_grammar``),
so the falsifiers here probe with a mnemonic nobody will ever implement: naming a real gap makes a
check like this expire the moment that gap closes, taking its subject with it.
"""
from __future__ import annotations

import textwrap

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.runtime.simulator import SimulationError, simulate
from merlin.runtime.tensor import Tensor
from merlin.targetgen.contract import interface_emit as IE
from merlin.targetgen.contract.schemas import contract_dir

_MOVEMENT = textwrap.dedent('''\
    module attributes {merlin_iface.version = "0.1", merlin_iface.target = "t", merlin_iface.abi_version = "0.1"} {
      %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<16x16xi8>
      %Y0 = merlin_iface.movement %X {name = "Y0", semantic = "mvin_mvout", output_dtype = "i32"} : (tensor<16x16xi8>) -> tensor<16x16xi32>
    }
    ''')

_FLASH = textwrap.dedent('''\
    module attributes {merlin_iface.version = "0.1", merlin_iface.target = "t", merlin_iface.abi_version = "0.1"} {
      %Q = merlin_iface.tensor {name = "Q", role = "input"} : tensor<16x32xf8E4M3FN>
      %K = merlin_iface.tensor {name = "K", role = "input"} : tensor<32x32xf8E4M3FN>
      %V = merlin_iface.tensor {name = "V", role = "input"} : tensor<32x16xf8E4M3FN>
      %S = merlin_iface.attention_qk %Q, %K {name = "S", output_dtype = "bf16"} : (tensor<16x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) -> tensor<16x32xbf16>
      %P = merlin_iface.softmax %S {name = "P", axis = 1 : i64} : (tensor<16x32xbf16>) -> tensor<16x32xbf16>
      %Y0 = merlin_iface.attention_pv %P, %V {name = "Y0", output_dtype = "bf16"} : (tensor<16x32xbf16>, tensor<32x16xf8E4M3FN>) -> tensor<16x16xbf16>
    }
    ''')

_RESIDENCY = textwrap.dedent('''\
    module attributes {merlin_iface.version = "0.1", merlin_iface.target = "t", merlin_iface.abi_version = "0.1"} {
      %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<16x16xi8>
      %A0 = merlin_iface.tensor {name = "A0", role = "input"} : tensor<16x16xi8>
      %W_res = merlin_iface.resident_pack %W {layout = "packed_rhs"} : (tensor<16x16xi8>) -> !merlin_iface.resident
      %acc0 = merlin_iface.matmul %A0, %W_res : (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
      %Y0 = merlin_iface.commit %acc0 {name = "Y0", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x16xi32>
      merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
    }
    ''')


def _iface_capsules():
    """Every shipped capsule written in the ``merlin_iface`` grammar (some are linalg instead)."""
    root = repo_root() / "merlin" / "contract" / "capsules"
    out = []
    for p in sorted(root.rglob("*.interface.mlir")):
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:                       # an un-readable holdout dir is not this test's subject
            continue
        if "merlin_iface." in text:
            out.append((p, text))
    return out


class TestFailClosed:
    def test_an_undefined_op_raises_and_names_the_mnemonic(self):
        # Not a warning and not a shorter list: the caller cannot tell a short command list from a
        # complete one, so the only honest answer is to stop and say which op could not be read.
        text = _MOVEMENT.replace("merlin_iface.movement", "merlin_iface.definitely_not_an_op")
        with pytest.raises(IE.InterfaceGrammarError) as e:
            IE.parse_interface_mlir(text)
        assert "definitely_not_an_op" in str(e.value)

    def test_an_undefined_op_beside_readable_ones_still_stops_the_parse(self):
        # The dangerous shape is not a module that is entirely unreadable — it is one where MOST ops
        # parse, so the result looks like a complete program with one stage quietly missing. That is
        # what the three conv capsules did: resident_pack + evict survived and only the compute
        # vanished. The probe mnemonic here is deliberately one nobody will implement, because naming
        # a real gap makes the falsifier expire the moment that gap closes.
        partial = textwrap.dedent('''\
            module attributes {merlin_iface.version = "0.1", merlin_iface.target = "t", merlin_iface.abi_version = "0.1"} {
              %IFM = merlin_iface.tensor {name = "IFM", role = "input"} : tensor<8x8xi8>
              %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<8x8xi8>
              %W_res = merlin_iface.resident_pack %W {layout = "packed_rhs"} : (tensor<8x8xi8>) -> !merlin_iface.resident
              %Y0 = merlin_iface.not_an_op_in_any_grammar %IFM, %W_res {name = "Y0", output_dtype = "i32"} : (tensor<8x8xi8>, !merlin_iface.resident) -> tensor<8x8xi32>
              merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
            }
            ''')
        with pytest.raises(IE.InterfaceGrammarError, match="not_an_op_in_any_grammar"):
            IE.parse_interface_mlir(partial)

    def test_the_error_is_a_ValueError_so_existing_callers_still_catch_it(self):
        assert issubclass(IE.InterfaceGrammarError, ValueError)

    def test_a_type_is_not_an_op(self):
        # `!merlin_iface.resident` / `!merlin_iface.acc<i32>` appear on nearly every line of a real
        # residency capsule. Reading them as ops would make the whole shipped corpus fail closed.
        mnems = IE.op_mnemonics(_RESIDENCY)
        assert "resident" not in mnems and "acc" not in mnems
        assert IE.undefined_op_mnemonics(_RESIDENCY) == []

    def test_a_module_attribute_is_not_an_op(self):
        mnems = IE.op_mnemonics(_RESIDENCY)
        assert "version" not in mnems and "target" not in mnems and "abi_version" not in mnems

    def test_a_wrapped_module_header_does_not_manufacture_three_undefined_ops(self):
        # Failing closed is only an improvement if it fails on real defects. A header printed across
        # lines is still a header; reading `merlin_iface.version = "0.1"` as an op would refuse a
        # perfectly conformant module, which is as unhelpful as the silent drop it replaces.
        wrapped = ('module attributes {\n'
                   '  merlin_iface.version = "0.1",\n'
                   '  merlin_iface.target = "t",\n'
                   '  merlin_iface.abi_version = "0.1"} {\n'
                   '  %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<4x4xi8>\n'
                   '  %Y0 = merlin_iface.movement %X {name = "Y0"} : (tensor<4x4xi8>) -> tensor<4x4xi8>\n'
                   '}\n')
        assert IE.undefined_op_mnemonics(wrapped) == []
        cb = IE.parse_interface_mlir(wrapped)
        assert cb["target"] == "t"
        assert [c["opcode"] for c in cb["commands"]] == ["MOVEMENT"]

    def test_the_defined_set_comes_from_the_parser_tables_not_a_second_list(self):
        # One authority. A hand-maintained second list is a thing that can drift from the code that
        # actually dispatches, which is the exact class of bug this module now guards against.
        defined = IE.defined_mnemonics()
        assert set(IE._OP_TO_OPCODE) <= defined
        assert set(IE._NAMED_OP_OPERAND_KEYS) <= defined
        assert "tensor" in defined


class TestEveryOpReachesTheCommandList:
    @pytest.mark.parametrize("text,expected", [
        (_RESIDENCY, ["RES_PACK", "MATMUL_RESIDENT", "COMMIT", "EVICT"]),
        (_MOVEMENT, ["MOVEMENT"]),
        (_FLASH, ["ATTENTION_QK", "SOFTMAX", "ATTENTION_PV"]),
    ])
    def test_opcodes_in_program_order(self, text, expected):
        assert [c["opcode"] for c in IE.parse_interface_mlir(text)["commands"]] == expected

    def test_movement_carries_its_source_and_destination(self):
        cmd = IE.parse_interface_mlir(_MOVEMENT)["commands"][0]
        assert cmd["operands"] == {"src": "X", "dst": "Y0"}
        assert cmd["attributes"]["output_dtype"] == "i32"

    def test_flash_attention_keeps_its_second_matmul(self):
        # The measured defect in one assertion: this module used to parse into TWO commands with the
        # P@V silently gone, and the parser said nothing.
        cmds = IE.parse_interface_mlir(_FLASH)["commands"]
        pv = [c for c in cmds if c["opcode"] == "ATTENTION_PV"]
        assert len(pv) == 1
        assert pv[0]["operands"] == {"p": "P", "v": "V", "dst": "Y0"}

    def test_the_shipped_corpus_either_parses_whole_or_raises(self):
        """Corpus-wide: no capsule may parse into fewer commands than it has ops.

        This is the assertion that would have caught the original defect. ``tensor`` declares a leaf
        rather than issuing a command, so it is excluded from the count; every other op must produce
        exactly one command, or the parse must have raised naming it.
        """
        capsules = _iface_capsules()
        assert len(capsules) >= 100, f"corpus looks truncated ({len(capsules)} capsules)"
        dropped, raised = [], []
        for path, text in capsules:
            ops = [m for m in IE.op_mnemonics(text) if m != "tensor"]
            try:
                cb = IE.parse_interface_mlir(text)
            except IE.InterfaceGrammarError as e:
                for m in IE.undefined_op_mnemonics(text):
                    assert m in str(e), f"{path}: error does not name {m!r}"
                raised.append(path)
                continue
            if len(cb["commands"]) != len(ops):
                dropped.append((path, ops, [c["opcode"] for c in cb["commands"]]))
        assert not dropped, f"ops silently dropped: {dropped}"
        # Nothing in the shipped corpus should still be unreadable. If a future capsule introduces a
        # mnemonic the grammar lacks, this names it rather than letting the capsule read short.
        assert not raised, ("capsules the frozen grammar cannot read: "
                            + str({str(p): IE.undefined_op_mnemonics(p.read_text(encoding="utf-8"))
                                   for p in raised}))


class TestSimulatorSemantics:
    def test_movement_is_an_identity_round_trip(self):
        # The golden engine defines `movement` as `src.to_list()`. The capsule's whole point is that
        # the data survives the trip bit-for-bit, so a clamp or a requantize here would be a wrong
        # answer that still looks numerically plausible.
        cb = {"abi_version": "0.1", "target": "t",
              "tensors": {"X": {"shape": [4, 5], "dtype": "i8", "role": "input"}},
              "commands": [{"opcode": "MOVEMENT", "operands": {"src": "X", "dst": "Y0"},
                            "attributes": {"output_dtype": "i32"}}],
              "outputs": ["Y0"]}
        got = simulate(cb)["outputs"]["Y0"]
        assert got == Tensor.deterministic("X", (4, 5), "i8").to_list()

    def test_movement_needs_a_source(self):
        cb = {"abi_version": "0.1", "target": "t",
              "tensors": {"X": {"shape": [4, 5], "dtype": "i8", "role": "input"}},
              "commands": [{"opcode": "MOVEMENT", "operands": {"dst": "Y0"}, "attributes": {}}]}
        with pytest.raises(SimulationError, match="MOVEMENT"):
            simulate(cb)

    def test_attention_pv_is_p_times_v_with_no_transpose(self):
        # Sibling of ATTENTION_QK, which contracts the trailing head dim of BOTH operands (so it
        # transposes K). PV does not: p is [m, s] and v is [s, d]. Matching the golden engine's
        # `p.matmul(v)` exactly is what keeps the functional tier from disagreeing with the oracle.
        cb = {"abi_version": "0.1", "target": "t",
              "tensors": {"P": {"shape": [4, 6], "dtype": "i8", "role": "input"},
                          "V": {"shape": [6, 3], "dtype": "i8", "role": "input"}},
              "commands": [{"opcode": "ATTENTION_PV",
                            "operands": {"p": "P", "v": "V", "dst": "Y0"},
                            "attributes": {"output_dtype": "i32"}}],
              "outputs": ["Y0"]}
        want = (Tensor.deterministic("P", (4, 6), "i8")
                .matmul(Tensor.deterministic("V", (6, 3), "i8")).to_list())
        assert simulate(cb)["outputs"]["Y0"] == want

    def test_attention_pv_rejects_a_key_count_mismatch(self):
        cb = {"abi_version": "0.1", "target": "t",
              "tensors": {"P": {"shape": [4, 6], "dtype": "i8", "role": "input"},
                          "V": {"shape": [5, 3], "dtype": "i8", "role": "input"}},
              "commands": [{"opcode": "ATTENTION_PV",
                            "operands": {"p": "P", "v": "V", "dst": "Y0"}, "attributes": {}}]}
        with pytest.raises(SimulationError, match="ATTENTION_PV"):
            simulate(cb)

    def test_a_parsed_movement_capsule_actually_runs(self):
        # End to end: the text the corpus ships -> commands -> a real output. Before the grammar row
        # this produced zero commands and therefore zero outputs, and nothing complained.
        cb = IE.parse_interface_mlir(_MOVEMENT)
        cb["tensors"]["X"]["role"] = "input"
        out = simulate(cb)["outputs"]
        assert out["Y0"] == Tensor.deterministic("X", (16, 16), "i8").to_list()


class TestContractDeclaresTheNewOps:
    """The frozen contract an agent reads must list what the parser now accepts."""

    def test_the_command_buffer_abi_declares_both_opcodes(self):
        abi = yaml.safe_load((contract_dir() / "command_buffer_abi.yaml").read_text(encoding="utf-8"))
        assert {"MOVEMENT", "ATTENTION_PV"} <= set(abi["opcodes"])

    def test_the_interface_contract_maps_both_mnemonics(self):
        spec = yaml.safe_load(
            (contract_dir() / "interface_dialect_contract.yaml").read_text(encoding="utf-8"))
        mapped = {op["name"]: op.get("maps_to") for op in spec["dialect"]["required_ops"]}
        assert mapped.get("merlin_iface.movement") == "MOVEMENT"
        assert mapped.get("merlin_iface.attention_pv") == "ATTENTION_PV"

    def test_every_contract_op_is_one_the_parser_defines(self):
        # The contract and the parser are two statements of the same grammar; if they can disagree,
        # the document an agent reads stops being evidence about the tool it will actually hit.
        spec = yaml.safe_load(
            (contract_dir() / "interface_dialect_contract.yaml").read_text(encoding="utf-8"))
        defined = IE.defined_mnemonics()
        for op in spec["dialect"]["required_ops"]:
            mnem = op["name"].split("merlin_iface.", 1)[-1]
            assert mnem in defined, f"contract requires {op['name']} but the parser does not define it"
