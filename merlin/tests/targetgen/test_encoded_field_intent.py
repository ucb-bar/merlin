"""``T0.encoded_field_intent``: does the pointer an emitted move dereferences carry the tensor the ABI
contract says that argument slot holds?

The hole this lint fills. A correctly-NAMED instruction carrying a wrong FIELD is not an illegal
instruction, so the ISA linter passes it. And the numeric and trace planes both read the command buffer
— the DECLARATION — so a package whose command buffer is right and whose emitted encoding is wrong
passes both and diverges only on the oracle, which can report it as nothing but a value error.

The invariant. The DRAM base pointer of every move must be the argument slot that
``kernel_abi.arg_order_by_command_shape`` (for THIS buffer's command shape) puts that move's tensor in.
There is one order per command shape and more than one harness path, so the order is RESOLVED from the
contract — a buffer resolving against no row, or against two, is UNKNOWN with that reason rather than
screened against a guess.

What is deliberately NOT reported: the addressing arithmetic. A kernel compiled against a permuted
pointer list computes self-consistent addresses that would be CORRECT against the order it assumed, so
flagging the stride or offset computation blames arithmetic that is right. The finding names the binding.

Every ``intended`` value below is derived from the ABI contract, the command buffer's declared tensors,
or the RTL-derived array geometry passed as ``_FACTS``. Nothing here reads a golden or an expected output
value, and the mesh is a parameter of the tests (every bound is computed from it, never from the number).

Each behaviour is exercised in BOTH directions — a conformant stream passes and the specific defect is
detected — plus the fail-closed direction, because a check that could not run must never report success.
"""
from __future__ import annotations

import copy

from merlin.targetgen import rtl_checks as RC

_CID = "T0.encoded_field_intent"

#: RTL-shaped facts, passed as the explicit override so these tests exercise the CHECK rather than the
#: fact extractor. The mesh plays the role of "whatever this target's array turns out to be".
_FACTS = {"mesh": [16, 16], "scratchpad_bytes": 262144, "legal_funct": None,
          "custom_opcode": None, "funct3": None, "from": "test override"}
_EDGE = _FACTS["mesh"][0]


# ------------------------------------------------------------------------------------- fixtures
def _capsule(m=_EDGE, k=_EDGE, n=_EDGE):
    """A capsule declaring the ACTIVATION FIRST — the declaration order a package must not mistake for
    the harness ABI, and the shape of the four real failures."""
    return {"name": "unit", "inputs": [
                {"name": "A0", "role": "input", "shape": [m, k], "dtype": "i8"},
                {"name": "W", "role": "weight", "shape": [k, n], "dtype": "i8"}],
            "operation": {"op": "matmul", "attributes": {
                "lhs": "A0", "weight": "W", "out": "Y0",
                "epilogue": [], "output_dtype": "i32"}}}


def _cb_resident(m=_EDGE, k=_EDGE, n=_EDGE, *, epilogue=None, out_dtype="i32", acc_scale=None):
    """A RES_PACK / MATMUL_RESIDENT / COMMIT buffer — the contract's ``resident_matmul`` shape, whose
    argument order is weight-first regardless of how the tensors were declared."""
    attrs = {"epilogue": list(epilogue or []), "output_dtype": out_dtype}
    if acc_scale is not None:
        attrs["acc_scale"] = acc_scale
    return {"abi_version": "0.1", "target": "unit-test-target",
            "tensors": {"A0": {"shape": [m, k], "dtype": "i8", "role": "input"},
                        "W": {"shape": [k, n], "dtype": "i8", "role": "weight"},
                        "Y0": {"shape": [m, n], "dtype": out_dtype, "role": "output"}},
            "commands": [
                {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "W_res"}},
                {"opcode": "MATMUL_RESIDENT",
                 "operands": {"lhs": "A0", "rhs": "W_res", "dst": "acc_Y0"}},
                {"opcode": "COMMIT", "operands": {"src": "acc_Y0", "dst": "Y0"},
                 "attributes": attrs},
                {"opcode": "EVICT", "operands": {"handle": "W_res"}}]}


def _cb_whole_op(m=_EDGE, n=_EDGE, d=_EDGE):
    """One whole-op command — the contract's ``native_whole_op`` shape, whose order is the interface's
    OWN declaration order rather than weight-first. Proves the check does not assume one ABI."""
    return {"abi_version": "0.1", "target": "unit-test-target",
            "tensors": {"Q": {"shape": [m, d], "dtype": "i8", "role": "input"},
                        "K": {"shape": [n, d], "dtype": "i8", "role": "input"},
                        "Y0": {"shape": [m, n], "dtype": "i32", "role": "output"}},
            "commands": [{"opcode": "ATTENTION_QK",
                          "operands": {"q": "Q", "k": "K", "dst": "Y0"},
                          "attributes": {"epilogue": [], "output_dtype": "i32"}}]}


def _trace(*, loads=None, stores=None, ld_pitch=_EDGE, st_pitch=_EDGE * 4,
           relu=False, acc_scale=1.0, readout="i32"):
    """A well-formed stream. ``loads`` / ``stores`` are ``(arg_index, byte_offset, rows, cols)``."""
    ins = [{"index": 0, "class": "FENCE", "funct": None, "decoded": {}},
           {"index": 1, "class": "CONFIG_EX", "funct": 0, "decoded": {"subtype": "EX"}},
           {"index": 2, "class": "CONFIG_ST", "funct": 0,
            "decoded": {"subtype": "ST", "out_stride_bytes": st_pitch,
                        "relu": relu, "acc_scale": acc_scale}},
           {"index": 3, "class": "CONFIG_LD", "funct": 0,
            "decoded": {"subtype": "LD", "stride": ld_pitch}}]
    for load in (loads if loads is not None else
                 [(0, 0, _EDGE, _EDGE), (1, 0, _EDGE, _EDGE)]):
        arg, off, rows, cols = load[:4]
        pitch = load[4] if len(load) > 4 else ld_pitch     # a real kernel configures each operand's own
        ins.append({"index": len(ins), "class": "CONFIG_LD", "funct": 0,
                    "decoded": {"subtype": "LD", "stride": pitch}})
        ins.append({"index": len(ins), "class": "MVIN", "funct": 2,
                    "decoded": {"rows": rows, "cols": cols,
                                "dram": {"kind": "argbase", "arg_index": arg, "offset": off}}})
    ins.append({"index": len(ins), "class": "PRELOAD", "funct": 6, "decoded": {}})
    ins.append({"index": len(ins), "class": "COMPUTE_PRELOADED", "funct": 4, "decoded": {}})
    for (arg, off, rows, cols) in (stores if stores is not None else [(2, 0, _EDGE, _EDGE)]):
        ins.append({"index": len(ins), "class": "MVOUT", "funct": 3,
                    "decoded": {"rows": rows, "cols": cols, "readout": readout,
                                "dram": {"kind": "argbase", "arg_index": arg, "offset": off}}})
    ins.append({"index": len(ins), "class": "FENCE", "funct": None, "decoded": {}})
    return {"instructions": ins, "abi": {}}


def _screen(trace, capsule, cb):
    return RC.screen(trace, capsule, _FACTS, target="unit-test-target", command_buffer=cb)


def _check(trace, capsule, cb):
    return next(c for c in _screen(trace, capsule, cb).checks if c.id == _CID)


# ------------------------------------------- the argument order comes from the contract, per shape
def test_the_argument_order_is_resolved_from_the_abi_contract_not_the_declaration_order():
    """The capsule and the buffer both DECLARE the activation first; the resident_matmul shape's
    contract order is weight-first. That divergence is the whole point of resolving from the contract."""
    names, shape, why = RC.resolve_kernel_arg_order(_cb_resident())
    assert why == ""
    assert shape == "resident_matmul"
    assert names == ["W", "A0", "Y0"]                     # NOT the declaration order [A0, W, Y0]


def test_a_whole_op_buffer_resolves_to_the_declaration_order_instead():
    """A second harness path exists, and its order is the interface's own declaration order. The check
    resolves which one applies from the buffer's command shape rather than assuming either."""
    names, shape, why = RC.resolve_kernel_arg_order(_cb_whole_op())
    assert why == "" and shape == "native_whole_op"
    assert names == ["Q", "K", "Y0"]


def test_a_movement_buffer_resolves_to_its_own_two_pointer_order():
    cb = {"abi_version": "0.1", "target": "unit-test-target",
          "tensors": {"X": {"shape": [_EDGE, _EDGE], "dtype": "i8", "role": "input"},
                      "Y0": {"shape": [_EDGE, _EDGE], "dtype": "i8", "role": "output"}},
          "commands": [{"opcode": "MOVEMENT", "operands": {"src": "X", "dst": "Y0"},
                        "attributes": {"output_dtype": "i8"}}]}
    names, shape, why = RC.resolve_kernel_arg_order(cb)
    assert why == "" and shape == "movement" and names == ["X", "Y0"]


def test_the_binding_carries_each_slots_declared_extent():
    binding, shape, why = RC.kernel_argument_binding(_cb_resident(), _capsule())
    assert why == "" and shape == "resident_matmul"
    assert [b["name"] for b in binding] == ["W", "A0", "Y0"]
    assert [b["index"] for b in binding] == [0, 1, 2]
    assert binding[2]["packed_bytes"] == _EDGE * _EDGE * 4       # i32 output


# --------------------------------------------------------------- the defect: a permuted pointer list
def _permuted_trace(k, n, m):
    """The emitted stream of a kernel compiled against the DECLARATION order [A0, W, Y0] while the
    harness calls with the contract's [W, A0, Y0]: each tensor's own byte volume, through the wrong
    pointer. Here W is [k, n] and A0 is [m, k] with different extents, so the volumes are a permutation.
    """
    return _trace(loads=[(0, 0, m, k, k), (1, 0, k, n, n)],
                  stores=[(2, 0, m, n)], st_pitch=n * 4)


def test_a_permuted_pointer_list_is_detected_and_named_as_the_binding():
    k, n, m = _EDGE, _EDGE * 2, _EDGE          # W [16, 32] = 512 B, A0 [16, 16] = 256 B
    c = _check(_permuted_trace(k, n, m), _capsule(m, k, n), _cb_resident(m, k, n))
    assert c.status == "fail"
    assert c.severity == "warn"                            # advisory: never gates
    perm = c.evidence["argument_volume_permutation"]
    assert {p["arg_index"] for p in perm} == {0, 1}
    assert c.evidence["contract_command_shape"] == "resident_matmul"


def test_the_finding_says_the_addressing_arithmetic_is_not_the_defect():
    """The explicit anti-misdiagnosis assertion: the stride/offset computation of a kernel compiled
    against a permuted pointer list is self-consistent and would be CORRECT against that order, so the
    finding must say so rather than blaming the address math."""
    k, n, m = _EDGE, _EDGE * 2, _EDGE
    c = _check(_permuted_trace(k, n, m), _capsule(m, k, n), _cb_resident(m, k, n))
    assert "addressing arithmetic itself is NOT the finding" in c.message
    assert "which argument slot each tensor is read from" in c.message
    assert "arg_order_by_command_shape" in c.fix_hint
    # and no separate row_pitch / address finding is raised against the permuted slots
    assert not [f for f in c.evidence["findings"] if f["field"] == "row_pitch"]


def test_the_conformant_direction_passes_on_the_contract_order():
    """The SAME shapes, moved through the contract's own order, must pass — a package that reads the
    contract is not penalised for a capsule that declares the activation first."""
    k, n, m = _EDGE, _EDGE * 2, _EDGE
    c = _check(_trace(loads=[(0, 0, k, n, n), (1, 0, m, k, k)],
                      stores=[(2, 0, m, n)], st_pitch=n * 4),
               _capsule(m, k, n), _cb_resident(m, k, n))
    assert c.status == "pass", c.message
    assert c.evidence["fields_compared"] > 0               # and it really did compare something


def test_a_ragged_extent_moved_through_its_tile_padded_buffer_is_not_a_finding():
    """A declared extent the array edge does not divide is legitimately laid out tile-padded, so moves
    that walk the padded pitch must NOT be reported. The extent bound is deliberately the permissive
    one, because this check does not get to pick a padding convention."""
    m = k = n = _EDGE - 1
    c = _check(_trace(loads=[(0, 0, k, n), (1, 0, m, k)], stores=[(2, 0, m, n)],
                      ld_pitch=_EDGE, st_pitch=_EDGE * 4),
               _capsule(m, k, n), _cb_resident(m, k, n))
    assert c.status == "pass", c.message


def test_an_overrun_with_no_permutation_is_reported_as_the_pointer_not_as_a_binding_swap():
    """When nothing suggests a permuted pointer list, an out-of-extent move stands on its own — and is
    reported without the binding-swap claim, which would be a guess."""
    bad = _trace(loads=[(0, _EDGE * _EDGE, _EDGE, _EDGE), (1, 0, _EDGE, _EDGE)])
    c = _check(bad, _capsule(), _cb_resident())
    assert c.status == "fail"
    assert "argument_volume_permutation" not in c.evidence
    f = next(f for f in c.evidence["findings"] if f["field"] == "dram_base_pointer")
    assert f["arg_index"] == 0 and f["tensor"] == "W"
    assert f["intended"] == f"<={_EDGE * _EDGE}"           # the tile-padded extent


# ------------------------------------------------- the other declared fields (silent unless divergent)
def test_a_store_whose_readout_dtype_is_not_the_declared_output_dtype_is_detected():
    c = _check(_trace(readout="i8"), _capsule(), _cb_resident())
    assert c.status == "fail"
    f = next(f for f in c.evidence["findings"] if f["field"] == "readout_dtype")
    assert (f["emitted"], f["intended"]) == ("i8", "i32")


def test_a_store_activation_the_declared_epilogue_does_not_name_is_detected():
    c = _check(_trace(relu=True), _capsule(), _cb_resident())
    assert c.status == "fail"
    f = next(f for f in c.evidence["findings"] if f["field"] == "store_activation")
    assert (f["emitted"], f["intended"]) == (True, False)
    # and the declared direction passes
    assert not [f for f in _check(_trace(relu=True), _capsule(),
                                  _cb_resident(epilogue=["relu"])).evidence["findings"]
                if f["field"] == "store_activation"]


def test_an_accumulator_scale_other_than_the_declared_one_is_detected():
    c = _check(_trace(acc_scale=0.5), _capsule(), _cb_resident())
    assert c.status == "fail"
    f = next(f for f in c.evidence["findings"] if f["field"] == "config_scale")
    assert (f["emitted"], f["intended"]) == (0.5, 1.0)     # no declared scaling stage => identity
    assert not [f for f in _check(_trace(acc_scale=0.5), _capsule(),
                                  _cb_resident(epilogue=["acc_scale"], acc_scale=0.5))
                .evidence["findings"] if f["field"] == "config_scale"]


def test_a_column_extent_wider_than_the_bound_tensor_declares_is_detected():
    c = _check(_trace(loads=[(0, 0, 1, _EDGE * 2), (1, 0, _EDGE, _EDGE)]), _capsule(), _cb_resident())
    assert c.status == "fail"
    f = next(f for f in c.evidence["findings"] if f["field"] == "column_extent")
    assert f["emitted"] == _EDGE * 2 and f["intended"] == f"<={_EDGE}"


def test_a_pitch_matching_neither_row_convention_is_reported_only_when_nothing_else_explains_it():
    m = k = n = _EDGE - 1
    short = (_EDGE - 1) // 2                              # neither the packed nor the padded row
    c = _check(_trace(loads=[(0, 0, k, n, short), (1, 0, m, k, k)],
                      stores=[(2, 0, m, n)], st_pitch=_EDGE * 4),
               _capsule(m, k, n), _cb_resident(m, k, n))
    assert "argument_volume_permutation" not in c.evidence     # nothing else explains it
    f = next(f for f in c.evidence["findings"] if f["field"] == "row_pitch")
    assert f["emitted"] == short
    assert f["intended"] == f"{_EDGE - 1} (packed) or {_EDGE} (tile-padded)"


def test_either_pitch_convention_is_accepted():
    m = k = n = _EDGE - 1
    for pitch in (n, _EDGE):
        c = _check(_trace(loads=[(0, 0, k, n, pitch), (1, 0, m, k, k)],
                          stores=[(2, 0, m, n)], st_pitch=_EDGE * 4),
                   _capsule(m, k, n), _cb_resident(m, k, n))
        assert not [f for f in c.evidence["findings"] if f["field"] == "row_pitch"], pitch


# ------------------------------------------------------------------------------- fail closed
def test_without_a_command_buffer_the_check_is_skipped_not_passed():
    """A check that could not run must never report success."""
    c = next(x for x in RC.screen(_trace(), _capsule(), _FACTS,
                                  target="unit-test-target").checks if x.id == _CID)
    assert c.status == "skipped" and "no command buffer" in c.message


def test_without_a_derived_mesh_the_check_is_skipped_not_passed():
    c = next(x for x in RC.screen(_trace(), _capsule(), dict(_FACTS, mesh=None),
                                  target="unit-test-target",
                                  command_buffer=_cb_resident()).checks if x.id == _CID)
    assert c.status == "skipped" and "UNKNOWN" in c.message


def test_a_buffer_matching_no_contract_shape_is_unknown_not_screened():
    cb = _cb_resident()
    cb["commands"] = [{"opcode": "SOME_FUTURE_OPCODE", "operands": {"src": "W", "dst": "Y0"}}]
    names, _shape, why = RC.resolve_kernel_arg_order(cb)
    assert names == [] and "UNKNOWN" in why
    c = _check(_trace(), _capsule(), cb)
    assert c.status == "skipped" and "UNKNOWN" in c.message


def test_a_buffer_matching_two_contract_shapes_is_unknown_not_screened():
    """More than one harness path exists. A buffer that could legitimately be called either way must be
    reported UNKNOWN rather than screened against whichever order this check happened to try first."""
    cb = _cb_resident()
    cb["tensors"]["Xm"] = {"shape": [_EDGE, _EDGE], "dtype": "i8", "role": "input"}
    cb["commands"].append({"opcode": "MOVEMENT", "operands": {"src": "Xm", "dst": "Y0"},
                           "attributes": {"output_dtype": "i8"}})
    names, _shape, why = RC.resolve_kernel_arg_order(cb)
    assert names == []
    assert "resolves against 2 kernel-ABI contract shapes" in why and "UNKNOWN" in why
    c = _check(_trace(), _capsule(), cb)
    assert c.status == "skipped" and "UNKNOWN" in c.message


def test_a_shape_whose_order_token_cannot_be_resolved_is_a_refusal_not_a_guess():
    cb = _cb_resident()
    cb["commands"][0]["operands"] = {"dst": "W_res"}       # RES_PACK with no src
    names, _shape, why = RC.resolve_kernel_arg_order(cb)
    assert names == [] and "names no src/dst pair" in why


def test_an_undeclared_dtype_makes_the_binding_undecidable_rather_than_assumed():
    cb = _cb_resident()
    cb["tensors"]["W"]["dtype"] = "some_future_format"
    binding, _shape, why = RC.kernel_argument_binding(cb, _capsule())
    assert binding == [] and "byte width" in why


def test_an_argument_index_outside_the_resolved_order_yields_a_reason_not_a_finding():
    c = _check(_trace(loads=[(9, 0, _EDGE, _EDGE)]), _capsule(), _cb_resident())
    assert [u for u in c.evidence["fields_not_derivable"] if "do not cover" in u["reason"]]
    assert not [f for f in c.evidence["findings"] if f.get("arg_index") == 9]


def test_a_multi_row_move_with_no_configured_pitch_is_undecidable_not_a_finding():
    tr = _trace()
    for i in tr["instructions"]:
        if i["class"] == "CONFIG_LD":
            i["decoded"] = {"subtype": "LD"}               # drop the load-pitch configuration
    c = _check(tr, _capsule(), _cb_resident())
    assert [u for u in c.evidence["fields_not_derivable"]
            if u["field"] == "dram_base_pointer" and "no DRAM row pitch" in u["reason"]]
    assert not [f for f in c.evidence["findings"]
                if f["field"] in ("dram_base_pointer", "row_pitch") and f["instruction"] == "MVIN"]


def test_two_declared_commits_sharing_one_store_configuration_compare_neither_field():
    """Which commit a single store configuration was meant for is not derivable, so the store-path
    fields are compared for neither -- with the reason recorded."""
    cb = _cb_resident()
    cb["commands"].insert(3, copy.deepcopy(cb["commands"][2]))
    cb["commands"][3]["attributes"]["epilogue"] = ["relu"]
    c = _check(_trace(relu=True), _capsule(), cb)
    assert {"store_activation", "config_scale"} <= {u["field"] for u in
                                                    c.evidence["fields_not_derivable"]}
    assert not [f for f in c.evidence["findings"]
                if f["field"] in ("store_activation", "config_scale")]


def test_a_dram_operand_that_is_not_a_kernel_argument_is_undecidable():
    tr = _trace()
    for i in tr["instructions"]:
        if i["class"] == "MVIN":
            i["decoded"]["dram"] = {"kind": "const", "raw": 12345}
    c = _check(tr, _capsule(), _cb_resident())
    assert [u for u in c.evidence["fields_not_derivable"]
            if "not a kernel argument" in u["reason"]]


def test_a_slot_no_move_reached_does_not_manufacture_a_permutation():
    """A zero-volume slot carries no evidence either way: it must neither be reported nor block the
    permutation test on the slots that DID move."""
    k, n, m = _EDGE, _EDGE * 2, _EDGE
    tr = _permuted_trace(k, n, m)
    for i in tr["instructions"]:
        if i["class"] == "MVOUT":
            i["decoded"]["rows"] = 0                       # a store that moves nothing
    c = _check(tr, _capsule(m, k, n), _cb_resident(m, k, n))
    perm = c.evidence.get("argument_volume_permutation") or []
    assert {p["arg_index"] for p in perm} == {0, 1}        # the store slot is simply absent


# --------------------------------------------------------------------------------- advisory only
def test_this_lint_alone_never_pushes_the_report_to_reject():
    """Advisory means advisory: a stream in which only this check fails must stay at `warn`, so a caller
    that skips an expensive oracle on `reject` can never lose an oracle run over it."""
    k, n, m = _EDGE, _EDGE * 2, _EDGE
    rep = _screen(_permuted_trace(k, n, m), _capsule(m, k, n), _cb_resident(m, k, n))
    assert next(c for c in rep.checks if c.id == _CID).status == "fail"
    assert rep.n_error == 0
    assert rep.verdict == "warn"
