"""The fused ``bias_add`` epilogue: a bias moved INTO the accumulator, accumulated onto by the mesh.

The capability was declared by the target's own capability manifest (a contraction can compose a
``bias_add`` stage) and modelled on the reference side, but the emitter's epilogue switch had no branch
for it, so every buffer declaring one died in codegen and reached the runner as ``tool_crash``. The
mechanism is not new hardware: it is the same PRELOAD/COMPUTE_PRELOADED pair the emitter already
issues, with the accumulator seeded by an MVIN instead of being overwritten by the first k-tile.

Every constant asserted below is DERIVED here the same way the emitter derives it — the accumulator
address bits from the capability manifest's ``readout_bits``, the accumulator's element container from
the CIRCT-extracted memory fact, the funct codes from the RTL funct decode table — and compared as
integers. Nothing is spelled as a literal opcode, bit or width.
"""
from __future__ import annotations

from copy import deepcopy

import pytest
import yaml

from merlin.common.paths import merlin_dir
from merlin.runtime import reference_outputs, simulate
from merlin.runtime.backends import base as _bk
from merlin.targetgen import rtl_checks
from merlin.targetgen.address_space import derive_address_space
from merlin.targetgen.contract.interface_emit import parse_interface_mlir
from merlin.targetgen.rocc.decode import decode_text
from merlin.targetgen.rtl.facts import load_facts
from merlin.targetgen.target_experiment import load_capability_manifest
from merlin.targetgen.trace_check import check

gem = _bk.get_backend("gemmini")
gm = gem.gemmini_codegen_mlir
CodegenError = gem.gemmini_codegen.CodegenError

CAPSULE = "SY_epilogue_bias_add"


# --------------------------------------------------------------------------------------------------
# facts, derived here independently of the emitter
# --------------------------------------------------------------------------------------------------
def _readout_bits() -> dict:
    return load_capability_manifest("gemmini").encoding["readout_bits"]


def _funct_code(semantic_class: str) -> int:
    enc = load_capability_manifest("gemmini").encoding
    return next(int(code) for code, cls in enc["semantic_class"].items() if cls == semantic_class)


def _accumulator_container() -> tuple[str, int]:
    store = derive_address_space("gemmini", facts=load_facts("gemmini")).store("accumulator")
    assert store is not None, "this checkout derives no accumulator memory fact"
    return store.element_dtype, store.element_bits


def _capsule_dir(name: str = CAPSULE):
    return merlin_dir() / "contract" / "capsules" / "layers" / name


def _capsule_cb(name: str = CAPSULE) -> dict:
    return parse_interface_mlir(
        (_capsule_dir(name) / "capsule.interface.mlir").read_text(encoding="utf-8"))


def _capsule_yaml(name: str = CAPSULE) -> dict:
    return yaml.safe_load((_capsule_dir(name) / "capsule.yaml").read_text(encoding="utf-8"))


def _cb(m: int, k: int, n: int, *, epilogue: list[str], bias: str | None = "B",
        bias_dtype: str | None = None, output_dtype: str = "i32", acc_scale: float | None = None):
    """A resident-matmul buffer with the requested epilogue. ``bias_dtype`` defaults to the DERIVED
    accumulator container, which is what the command-buffer ABI says a bias operand carries."""
    if bias_dtype is None:
        bias_dtype = _accumulator_container()[0]
    tensors = {"W": {"shape": [k, n], "dtype": "i8", "role": "weight"},
               "A0": {"shape": [m, k], "dtype": "i8", "role": "input"}}
    attrs: dict = {"epilogue": list(epilogue), "output_dtype": output_dtype}
    if bias is not None:
        tensors[bias] = {"shape": [n], "dtype": bias_dtype, "role": "bias"}
        attrs["bias"] = bias
    if acc_scale is not None:
        attrs["acc_scale"] = acc_scale
    return {"abi_version": "0.1", "target": "gemmini", "tensors": tensors, "commands": [
        {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "R"},
         "attributes": {"layout": "packed_rhs"}},
        {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A0", "rhs": "R", "dst": "acc"}},
        {"opcode": "COMMIT", "operands": {"src": "acc", "dst": "Y0"}, "attributes": attrs},
        {"opcode": "EVICT", "operands": {"handle": "R"}}]}


def _trace(cb: dict) -> list[dict]:
    return decode_text(gm.emit_kernel_mlir(cb)[0], target="gemmini")["instructions"]


# --------------------------------------------------------------------------------------------------
# the branch that did not exist
# --------------------------------------------------------------------------------------------------
def test_the_declared_bias_capsule_lowers_at_all():
    """FAILS without the epilogue branch: the emitter raised CodegenError and the capsule reached the
    runner as `spike invocation failed: unsupported epilogue stage 'bias_add'` / tool_crash."""
    text, args = gm.emit_kernel_mlir(_capsule_cb())
    assert "llvm.func @gemmini_kernel" in text
    # The bias is a real pointer argument, in the contract's TRAILING block.
    assert args == ["W", "A0", "Y0", "B"]


def test_emitted_classes_are_exactly_the_ones_the_capsule_declares():
    """The capsule declares PRELOAD + COMPUTE_PRELOADED and no bespoke bias opcode, because the target
    HAS no bias opcode: the bias rides the existing move-in/preload pair."""
    capsule = _capsule_yaml()
    emitted = {i["class"] for i in _trace(_capsule_cb())}
    declared = set(capsule["expected"]["instruction_classes"])
    assert declared <= emitted, sorted(declared - emitted)
    assert emitted <= declared | {"FENCE"}, sorted(emitted - declared - {"FENCE"})


def test_bias_is_moved_into_the_accumulator_with_a_repeating_row_stride():
    """The bias mvin targets the DERIVED accumulator base with neither the accumulate bit nor the
    readout-width bit set, under a ZERO DRAM row stride so every accumulator row reads the same bias
    row — the target's own repeating-bias move-in."""
    rb = _readout_bits()
    acc_base = rb["c_acc"] & ~rb["full_c_bit"]
    assert acc_base == rb["acc_i8"]                       # the two derived encodings agree

    instrs = _trace(_capsule_cb())
    stride = None
    bias_moves = []
    for ins in instrs:
        if ins["class"] == "CONFIG_LD":
            stride = ins["decoded"]["stride"]
        elif ins["class"] == "MVIN" and ins["decoded"]["addr"] & rb["acc_i8"]:
            bias_moves.append((ins, stride))
    assert len(bias_moves) == 1, "one output tile, one bias move-in"
    move, move_stride = bias_moves[0]
    assert move["funct"] == _funct_code("MVIN")
    assert move["decoded"]["addr"] == acc_base            # no accumulate bit, no full-C bit
    assert move_stride == 0                              # repeating bias
    assert move["decoded"]["dram"]["kind"] == "argbase"
    assert move["decoded"]["dram"]["arg_index"] == 3      # the trailing bias argument
    # ...and it is the ONLY move into the accumulator: the operands go to the scratchpad.
    assert sum(1 for i in instrs
               if i["class"] == "MVIN" and i["decoded"]["addr"] & rb["acc_i8"]) == 1


def test_every_k_tile_accumulates_onto_the_bias_including_the_first():
    """With a bias seeded in the accumulator, k=0 must ACCUMULATE, not overwrite — otherwise the mesh
    erases the bias it was supposed to add. Without a bias, k=0 must still overwrite."""
    rb = _readout_bits()
    deep = _cb(16, 32, 16, epilogue=["bias_add"])          # Kt = 2: a first tile and a later one
    accum = [i["decoded"]["accumulate"] for i in _trace(deep) if i["class"] == "PRELOAD"]
    assert accum == [True, True]

    plain = deepcopy(deep)
    plain["commands"][2]["attributes"] = {"epilogue": [], "output_dtype": "i32"}
    del plain["tensors"]["B"]
    plain_accum = [i["decoded"]["accumulate"] for i in _trace(plain) if i["class"] == "PRELOAD"]
    assert plain_accum == [False, True], "a bias-free kernel must still OVERWRITE on its first k-tile"
    # ...and it moves nothing into the accumulator.
    assert not [i for i in _trace(plain)
                if i["class"] == "MVIN" and i["decoded"]["addr"] & rb["acc_i8"]]


def test_a_biased_kernel_restores_the_activation_row_stride():
    """MEASURED DEFECT, kept as a gate: the bias move-in reconfigures the load unit to a zero row
    stride, and the activation row-panel move-in for the NEXT output row used to inherit it — so every
    row of that panel read the same DRAM row and the whole tile came back wrong (rows 17..31 of a
    32-row output). Every activation move must be preceded by the activation pitch."""
    cb = _cb(32, 16, 16, epilogue=["bias_add"])            # Mt = 2: a second row panel
    kp = 16                                                # padded K, the activation row pitch
    stride = None
    seen = []
    for ins in _trace(cb):
        if ins["class"] == "CONFIG_LD":
            stride = ins["decoded"]["stride"]
        elif ins["class"] == "MVIN":
            seen.append((ins["decoded"]["dram"]["arg_index"], stride))
    activation_arg = 1
    activation_strides = {s for arg, s in seen if arg == activation_arg}
    assert activation_strides == {kp}, seen


# --------------------------------------------------------------------------------------------------
# fail closed — validated by mutation
# --------------------------------------------------------------------------------------------------
def test_an_unknown_epilogue_stage_still_refuses():
    cb = _cb(16, 16, 16, epilogue=["gelu"], bias=None)
    with pytest.raises(CodegenError) as e:
        gm.emit_kernel_mlir(cb)
    assert "unsupported epilogue stage 'gelu'" in str(e.value)


def test_requant_is_still_refused_by_name():
    cb = _cb(16, 16, 16, epilogue=["requant"], bias=None)
    with pytest.raises(CodegenError):
        gm.emit_kernel_mlir(cb)


def test_a_bias_stage_naming_no_tensor_refuses_rather_than_dropping_it():
    cb = _cb(16, 16, 16, epilogue=["bias_add"])
    del cb["commands"][2]["attributes"]["bias"]
    with pytest.raises(CodegenError) as e:
        gm.emit_kernel_mlir(cb)
    assert "names no bias tensor" in str(e.value)


def test_a_bias_declared_after_another_stage_refuses():
    """The bias is the accumulator's INITIAL value, so it cannot follow relu: emitting it anyway would
    silently reassociate the declared epilogue."""
    cb = _cb(16, 16, 16, epilogue=["relu", "bias_add"])
    with pytest.raises(CodegenError) as e:
        gm.emit_kernel_mlir(cb)
    assert "FIRST epilogue stage" in str(e.value)


def test_a_bias_not_in_the_accumulator_container_refuses():
    """The ABI puts a bias in the ACCUMULATOR's dtype. A bias declared in the operand dtype would be
    DMA'd with the wrong element pitch — wrong numbers, no error."""
    acc_dtype, _bits = _accumulator_container()
    assert acc_dtype != "i8"
    cb = _cb(16, 16, 16, epilogue=["bias_add"], bias_dtype="i8")
    with pytest.raises(CodegenError) as e:
        gm.emit_kernel_mlir(cb)
    assert acc_dtype in str(e.value)


def test_a_bias_that_is_not_one_value_per_output_column_refuses():
    cb = _cb(16, 16, 16, epilogue=["bias_add"])
    cb["tensors"]["B"]["shape"] = [8]
    with pytest.raises(CodegenError) as e:
        gm.emit_kernel_mlir(cb)
    assert "one value per" in str(e.value)


# --------------------------------------------------------------------------------------------------
# the harness and the ABI
# --------------------------------------------------------------------------------------------------
def test_harness_declares_the_bias_in_the_derived_accumulator_container():
    acc_dtype, acc_bits = _accumulator_container()
    harness = gm._harness_c(_capsule_cb(), blobs={})
    assert f"static const int{acc_bits}_t T_B[16] row_align_acc(1)" in harness
    assert "gemmini_kernel((void*)T_W, (void*)T_A0, (void*)T_Y0, (void*)T_B)" in harness
    assert acc_dtype == f"i{acc_bits}"


def test_kernel_argument_order_is_the_one_the_abi_contract_resolves():
    """The emitted signature and the contract's own token expansion must be the same list — the
    contract is the single definition, and the trailing bias block is part of it."""
    cb = _capsule_cb()
    _text, args = gm.emit_kernel_mlir(cb)
    names, shape, why = rtl_checks.resolve_kernel_arg_order(cb)
    assert (names, shape) == (args, "resident_matmul"), why


def test_a_bias_free_buffer_keeps_the_three_block_signature():
    """The bias block is trailing and empty-when-absent, so nothing about an already-certified kernel's
    ABI moved."""
    cb = _cb(16, 16, 16, epilogue=[], bias=None)
    _text, args = gm.emit_kernel_mlir(cb)
    assert args == ["W", "A0", "Y0"]
    assert rtl_checks.resolve_kernel_arg_order(cb)[0] == args


def test_the_advisory_screen_binds_the_output_to_its_contract_argument():
    """The store-coverage screen used to place an output at ``len(declared inputs)``, which a capsule
    declaring a bias operand shifts by one — reporting a perfectly good store as a dropped one."""
    cb, capsule = _capsule_cb(), _capsule_yaml()
    outs, why = rtl_checks.declared_outputs(capsule, cb)
    assert outs and outs[0]["arg_index"] == 2, why
    report = rtl_checks.screen(decode_text(gm.emit_kernel_mlir(cb)[0], target="gemmini"),
                               capsule, target="gemmini", command_buffer=cb)
    by_id = {c.id: c.to_dict() for c in report.checks}
    assert by_id["T0.output_store_coverage"]["status"] == "pass", by_id["T0.output_store_coverage"]


def test_the_declared_trace_expectations_still_hold():
    cb, capsule = _capsule_cb(), _capsule_yaml()
    trace = decode_text(gm.emit_kernel_mlir(cb)[0], target="gemmini")
    assert check(trace, capsule["expected"], cb, address_model="pointer_args") == {
        "status": "pass", "violations": []}


# --------------------------------------------------------------------------------------------------
# the oracles
# --------------------------------------------------------------------------------------------------
_SHAPES = [(16, 16, 16), (16, 32, 16), (48, 48, 48), (40, 37, 35)]


@pytest.mark.slow
@pytest.mark.skipif(not gem.available("spike"), reason="spike-gemmini unavailable")
@pytest.mark.parametrize(("m", "k", "n"), _SHAPES)
def test_fused_bias_is_bit_exact_on_the_model_oracle(m, k, n, tmp_path):
    cb = _cb(m, k, n, epilogue=["bias_add"])
    res = gm.run_on_spike(cb, workdir=tmp_path, simulator="spike", timeout=600)
    assert res["correct"] is True
    assert res["outputs"] == reference_outputs(cb) == simulate(cb)["outputs"]
    # ...and the bias is genuinely applied: the same buffer without the stage computes something else.
    plain = deepcopy(cb)
    plain["commands"][2]["attributes"] = {"epilogue": [], "output_dtype": "i32"}
    assert reference_outputs(plain)["Y0"] != res["outputs"]["Y0"]


@pytest.mark.slow
@pytest.mark.skipif(not gem.available("spike"), reason="spike-gemmini unavailable")
@pytest.mark.parametrize("epilogue", [["bias_add"], ["bias"], ["bias_add", "relu"]])
def test_fused_bias_composes_with_the_other_stages(epilogue, tmp_path):
    cb = _cb(32, 32, 32, epilogue=epilogue)
    res = gm.run_on_spike(cb, workdir=tmp_path, simulator="spike", timeout=600)
    assert res["correct"] is True
    assert res["outputs"] == reference_outputs(cb) == simulate(cb)["outputs"]


@pytest.mark.slow
@pytest.mark.skipif(not gem.available("verilator"), reason="Gemmini Verilator sim unavailable")
def test_fused_bias_certifies_on_the_rtl_oracle(tmp_path):
    cb = _capsule_cb()
    res = gm.run_on_spike(cb, workdir=tmp_path, simulator="verilator", timeout=1800)
    assert res["correct"] is True
    assert res["outputs"] == reference_outputs(cb) == simulate(cb)["outputs"]
    assert res["oracle"]["derived_from_rtl"] is True


# --------------------------------------------------------------------------------------------------
# corpus-wide non-regression: the feature must not leak into the already-certified path
# --------------------------------------------------------------------------------------------------
def _gemmini_capsule_buffers():
    root = merlin_dir() / "contract" / "capsules"
    for path in sorted(root.rglob("capsule.interface.mlir")):
        text = path.read_text(encoding="utf-8")
        if 'merlin_iface.target = "gemmini"' not in text:
            continue
        try:
            cb = parse_interface_mlir(text)
        except Exception:                        # noqa: BLE001 — a corpus this emitter never sees
            continue
        yield path.parent.name, cb


# ~30 s: it emits every shipped gemmini capsule, and the deep-K members unroll thousands of tiles.
# Deselected from the fast gate by cost alone, not by importance — it runs in the full suite.
@pytest.mark.slow
def test_no_bias_free_capsule_gained_an_accumulator_move_in():
    """The bias path is reached ONLY through a declared bias stage. Every other shipped gemmini capsule
    must still emit exactly what it emitted before: nothing moved into the accumulator, and no move
    issued under the zero row stride the bias configures. 76 cycle-accurate certifications went through
    this emitter; a stray extra move-in is a silent re-certification of a different program."""
    acc_bit = _readout_bits()["acc_i8"]
    checked = 0
    for name, cb in _gemmini_capsule_buffers():
        stages = [s for c in cb.get("commands", []) if c.get("opcode") == "COMMIT"
                  for s in (c.get("attributes", {}).get("epilogue") or [])]
        if any(s in ("bias_add", "bias") for s in stages):
            continue
        try:
            instrs = _trace(cb)
        except CodegenError:
            continue                             # a shape this emitter refuses, before and after
        checked += 1
        assert not [i for i in instrs
                    if i["class"] == "MVIN" and i["decoded"]["addr"] & acc_bit], name
        assert 0 not in [i["decoded"]["stride"] for i in instrs if i["class"] == "CONFIG_LD"], name
    assert checked > 100, f"only {checked} bias-free capsules emitted; the sweep proved little"
