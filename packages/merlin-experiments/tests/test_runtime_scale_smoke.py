"""Strict semantic smoke selection; numerical success alone never admits."""

from copy import deepcopy
from pathlib import Path

import pytest
from merlin_experiments.phase0 import runtime_scale_smoke as S


def candidate():
    regime = dict(readout_strategy="f32_product_rne", weight_scale_granularity="tensor", scale_source="runtime_input")
    stages = [
        dict(kind="convert", to="f32", rounding="half_even"),
        dict(kind="scale", dtype="f32", rounding="half_even", granularity="per_command"),
        dict(kind="requant", to="i8", rounding="half_even", clamp=[-128, 127]),
    ]
    node = dict(
        form="contraction",
        realization="matmul",
        regime=regime,
        seed=False,
        residual=None,
        readout=dict(exit="requant", stages=stages),
    )
    raw = deepcopy(node)
    raw["readout"] = dict(exit="accumulator", stages=[])
    raw["regime"]["scale_source"] = "constant"
    return dict(
        candidate_id="regenerated-id",
        state="candidate",
        semantic=dict(structure="shared_input", dataflow="WS", nodes=[node, raw]),
    )


def test_runtime_smoke_selects_only_nonempty_mixed_ws_regime(capsys):
    row = candidate()
    from merlin_experiments.phase0.fusion_cases import digest

    row["candidate_id"] = digest(row["semantic"])
    assert S.select_candidates(dict(rows=[row])) == [row]
    S.print_selection([row])
    assert row["candidate_id"] in capsys.readouterr().out
    for mutate in (
        lambda r: r.update(state="declined"),
        lambda r: r["semantic"].update(dataflow="OS"),
        lambda r: r["semantic"]["nodes"][0].update(seed=True),
        lambda r: r["semantic"]["nodes"][0]["regime"].update(scale_source="constant"),
        lambda r: r["semantic"]["nodes"][0]["readout"]["stages"].append(dict(kind="pool")),
        lambda r: r["semantic"]["nodes"][1].update(readout=deepcopy(r["semantic"]["nodes"][0]["readout"])),
    ):
        bad = deepcopy(row)
        mutate(bad)
        assert not S.qualifies(bad)
    with pytest.raises(ValueError, match="empty"):
        S.select_candidates(dict(rows=[]))
    with pytest.raises(ValueError, match="empty"):
        S.select_candidates(dict(rows=[row]), "wrong-id")
    assert not S.passed(dict(outcome="pass", numeric=True, placement=False, trace=True))
    assert not S.passed(dict(outcome="pass", numeric=True, placement=True, trace=False))
    assert S.passed(dict(outcome="pass", numeric=True, placement=True, trace=True))


def test_runtime_smoke_accepts_both_accumulator_relu_orders():
    for index in (0, 1):
        row = candidate()
        row["semantic"]["nodes"][0]["readout"]["stages"].insert(
            index, dict(kind="activation", function="relu", domain="accumulator")
        )
        assert S.qualifies(row)
    row = candidate()
    row["semantic"]["nodes"][0]["readout"]["stages"].insert(
        2, dict(kind="activation", function="relu", domain="accumulator")
    )
    assert not S.qualifies(row)


def test_runtime_smoke_cli_and_labels():
    assert set(S.LABELS) == {
        "aligned",
        "partial",
        "rne_saturation",
        "conversion_boundary",
        "runtime_scale_zero",
        "runtime_scale_quarter",
        "runtime_scale_half",
        "runtime_scale_nonpow2",
    }
    with pytest.raises(SystemExit) as exc:
        S.main(["--help"])
    assert exc.value.code == 0


@pytest.mark.parametrize("relu", [False, True])
@pytest.mark.parametrize("label", S.LABELS)
def test_generated_runtime_smoke_payload_has_source_and_typed_siblings(tmp_path, relu, label):
    from types import SimpleNamespace

    import numpy as np
    from merlin_experiments.phase0.fusion_cases import build_payload, digest

    row = candidate()
    scheme = dict(
        id="fixture",
        status="resolved",
        graded=True,
        unknown=[],
        scale_value="arbitrary_f32",
        operand_sum_rounding="single_rounding",
        readout_strategy="f32_product_rne",
        weight_scale_granularity="tensor",
        columns_per_scale=4,
        scale_source="constant",
    )
    for node in row["semantic"]["nodes"]:
        node["scheme"] = scheme["id"]
    if relu:
        row["semantic"]["nodes"][0]["readout"]["stages"].insert(
            0, dict(kind="activation", function="relu", domain="accumulator")
        )
    row["candidate_id"] = digest(row["semantic"])
    path = dict(
        schema="output_path_v1",
        accumulator=dict(dtype="i32"),
        accumulator_stages=[],
        readout=row["semantic"]["nodes"][0]["readout"]["stages"],
        exits=dict(accumulator=True, requant=True),
    )
    facts = dict(
        tile_dim=4,
        columns_per_scale=4,
        dataflows=["WS"],
        operand_signed_bits=8,
        mesh_signed_bits=20,
        output_path=path,
        k_extension_within=dict(max_k=1024, partial_sum_bits=32),
    )
    context = SimpleNamespace(source_root=tmp_path, held_out_inputs=[], workload_spec={})
    payload = build_payload(row, {scheme["id"]: scheme}, facts, label, None, frozenset(), context, tmp_path / "screen")
    scales = S.validate_payload(payload)
    assert scales[0]["name"] == "node0_ReadoutScale"
    assert [r["dtype"] for r in payload["references"]] == ["i8", "i32"]
    if label == "partial":
        assert payload["references"][0]["shape"][-1] != payload["references"][1]["shape"][-1]
    assert scheme["scale_source"] == "constant"
    from merlin_experiments.phase0.runtime_scale_reference import scale_stimulus

    assert np.float32(payload["input_values"]["node0_ReadoutScale"][0]) == scale_stimulus(label)[0]


def test_runtime_dependence_requires_changed_narrow_and_invariant_raw_bytes():
    rows = [
        dict(
            candidate_id="generated",
            shape_class=label,
            source_sha256="same-source",
            outcome="pass",
            numeric=True,
            placement=True,
            trace=True,
            results=[dict(dtype="i8"), dict(dtype="i32")],
            readback=dict(bytes=[str(i), "same-raw"]),
        )
        for i, label in enumerate(S.LABELS[-4:])
    ]
    assert S.runtime_dependence(rows)[0]["status"] == "pass"
    for mutate in (
        lambda r: r[-1]["readback"]["bytes"].__setitem__(1, "changed-raw"),
        lambda r: r[-1].update(source_sha256="changed-source"),
        lambda r: r[-1].update(trace=False),
    ):
        changed = deepcopy(rows)
        mutate(changed)
        assert S.runtime_dependence(changed)[0]["status"] == "fail"
    assert S.runtime_dependence(rows[:3])[0]["status"] == "fail"


# ---------------------------------------------------------------------------
# Retired load-to-CONFIG_ST provenance.

_BASE, _KERNEL, _SCALE_PTR = 0x80000000, 0x800000FE, 0xC0000400
_POINTERS = {10: 0xC0000000, 11: 0xC0000100, 12: 0xC0000200, 13: _SCALE_PTR, 14: 0xC0000500, 15: 0xC0000600}
_ABI = {"kern": [dict(arg_index=i, register=10 + i, byte_extent=4 if i == 3 else 256) for i in range(6)]}
_QUARTER = 0x3E800000


def _i_type(opcode, f3, rd, rs1, imm=0):
    return (imm & 0xFFF) << 20 | rs1 << 15 | f3 << 12 | rd << 7 | opcode


def _config(rs1, rs2):
    # RoCC CONFIG (funct7 0) on custom-3, xs1/xs2 set, no destination.
    return rs2 << 20 | rs1 << 15 | 3 << 12 | 0x7B


def _elf(path, words, size):
    """Minimal RISC-V ELF64 with one .text section and one FUNC symbol, kern."""
    import struct

    text = bytearray()
    for pc in range(_BASE, _BASE + 0x200, 2):
        text += struct.pack("<H", 0x0001)  # c.nop filler
    for pc, word in words.items():
        width = 2 if word & 3 != 3 else 4
        text[pc - _BASE : pc - _BASE + width] = word.to_bytes(width, "little")
    strtab = b"\0kern\0"
    shstrtab = b"\0.text\0.symtab\0.strtab\0.shstrtab\0"
    symtab = bytes(24) + struct.pack("<IBBHQQ", 1, 0x12, 0, 1, _KERNEL, size)
    body, offsets = bytearray(), []
    for blob in (bytes(text), symtab, strtab, shstrtab):
        offsets.append(64 + len(body))
        body += blob
    shoff = 64 + len(body)
    header = b"\x7fELF" + bytes([2, 1, 1]) + bytes(9)
    header += struct.pack("<HHIQQQIHHHHHH", 2, 243, 1, _BASE, 0, shoff, 0, 64, 0, 0, 64, 5, 4)

    def section(name, kind, flags, addr, index, length, link=0, info=0, entsize=0):
        return struct.pack("<IIQQQQIIQQ", name, kind, flags, addr, offsets[index], length, link, info, 1, entsize)

    sections = bytes(64)
    sections += section(1, 1, 6, _BASE, 0, len(text))
    sections += section(7, 2, 0, 0, 1, len(symtab), link=3, info=1, entsize=24)
    sections += section(15, 3, 0, 0, 2, len(strtab))
    sections += section(23, 3, 0, 0, 3, len(shstrtab))
    path.write_bytes(header + body + sections)
    return path


def _line(pc, word, effects=""):
    spelled = f"0x{word:04x}" if word & 3 != 3 else f"0x{word:08x}"
    return f"core   0: 0x{pc:016x} ({spelled}) disasm\ncore   0: 3 0x{pc:016x} ({spelled}) {effects}\n"


def _program(tmp_path, mutation=None, bits=_QUARTER, caller_regs=None):
    """One (or two) calls of: c.lw a0,0(a3); c.slli a0,32; c.addi a0,16; c.li a2,6; CONFIG a2,a0."""
    sext = bits | (0xFFFFFFFF00000000 if bits >> 31 else 0)
    rs2 = bits << 32 | 16
    words = {_KERNEL: 0x0001, 0x80000100: 0x4288, 0x80000102: 0x1502, 0x80000104: 0x0541}
    words.update({0x80000106: 0x4619, 0x80000108: _config(12, 10), 0x8000010C: 0x8082})
    load = (0x80000100, 0x4288, f"x10 0x{sext:016x} mem 0x{_SCALE_PTR:016x}")
    shift = (0x80000102, 0x1502, f"x10 0x{bits << 32:016x}")
    body = [
        (_KERNEL, 0x0001, ""),
        load,
        shift,
        (0x80000104, 0x0541, f"x10 0x{rs2:016x}"),
        (0x80000106, 0x4619, "x12 0x0000000000000006"),
        (0x80000108, _config(12, 10), ""),
        (0x8000010C, 0x8082, ""),
    ]
    if mutation == "wrong_register":
        # A constant in a4 equals the loaded value; the command reads that register.
        words[0x80000106] = 0x00000737  # lui a4, 0
        body[4] = (0x80000106, 0x00000737, f"x14 0x{rs2:016x}")
        words[0x80000108] = _config(12, 14)
        body[5] = (0x80000108, _config(12, 14), "")
    elif mutation == "wrong_input":
        body[1] = (0x80000100, 0x4288, f"x10 0x{sext:016x} mem 0x{_POINTERS[12]:016x}")
    elif mutation == "missing_load":
        del body[1]
    elif mutation == "wrong_slot":
        # The loaded bits reach rs2 unshifted, i.e. the stride field.
        del body[2:4]
    elif mutation == "short_shift":
        words[0x80000102] = 0x057E  # c.slli a0, 31 keeps sign copies; provenance ends
        body[2] = (0x80000102, 0x057E, f"x10 0x{sext << 31 & (1 << 64) - 1:016x}")
        body[3] = (0x80000104, 0x0541, f"x10 0x{(sext << 31 & (1 << 64) - 1) + 16:016x}")
    elif mutation == "elf_word":
        words[0x80000102] = 0x1506  # linked bytes differ from the retired word
    elif mutation == "lwu_or":
        # lwu a0,0(a3); c.slli a0,32; or a0,a0,t0 (t0 = stride) at separate PCs.
        lwu, orr = _i_type(0x03, 6, 10, 13), 5 << 20 | 10 << 15 | 6 << 12 | 10 << 7 | 0x33
        words.update({0x80000110: lwu, 0x80000114: 0x1502, 0x80000116: orr})
        body[1:4] = [
            (0x80000110, lwu, f"x10 0x{bits:016x} mem 0x{_SCALE_PTR:016x}"),
            (0x80000114, 0x1502, f"x10 0x{bits << 32:016x}"),
            (0x80000116, orr, f"x10 0x{rs2:016x}"),
        ]
    elif mutation == "low_slot":
        # lwu a0,0(a3); or a0,a0,t0 (t0 = stride in bits 32..63): the scale in rs2[31:0].
        lwu, orr = _i_type(0x03, 6, 10, 13), 5 << 20 | 10 << 15 | 6 << 12 | 10 << 7 | 0x33
        words.update({0x80000110: lwu, 0x80000116: orr})
        body[1:4] = [
            (0x80000110, lwu, f"x10 0x{bits:016x} mem 0x{_SCALE_PTR:016x}"),
            (0x80000116, orr, f"x10 0x{16 << 32 | bits:016x}"),
        ]
    elif mutation == "flw_fmv":
        # flw fa0,0(a3); fmv.x.w a0,fa0; c.slli a0,32; c.addi a0,16.
        flw, fmv = _i_type(0x07, 2, 10, 13), 0x70 << 25 | 10 << 15 | 10 << 7 | 0x53
        words.update({0x80000110: flw, 0x80000114: fmv, 0x80000118: 0x1502, 0x8000011A: 0x0541})
        body[1:4] = [
            (0x80000110, flw, f"f10 0xffffffff{bits:08x} mem 0x{_SCALE_PTR:016x}"),
            (0x80000114, fmv, f"x10 0x{sext:016x}"),
            (0x80000118, 0x1502, f"x10 0x{bits << 32:016x}"),
            (0x8000011A, 0x0541, f"x10 0x{rs2:016x}"),
        ]
    caller = "".join(
        _line(_BASE + 2 * i, 0x00000013, f"x{r} 0x{v:016x}")
        for i, (r, v) in enumerate({**_POINTERS, 5: 16, **(caller_regs or {})}.items())
    )
    calls = [body]
    if mutation == "stale":
        # The second call reuses the first call's register instead of reloading.
        calls.append([row for row in body if row[0] != 0x80000100])
    trace = caller
    for call in calls:
        trace += "".join(_line(*row) for row in call)
        trace += _line(_BASE + 0x40, 0x00000013, "")
    (tmp_path / "retirement.log").write_text(trace)
    elf = _elf(tmp_path / "model.elf", words, 0x20)
    return elf, tmp_path / "retirement.log"


def _audit(elf, trace, bits=_QUARTER, slot=None):
    from merlin.targetgen.runtime_scale_trace import audit_runtime_scale_loads

    return audit_runtime_scale_loads(
        elf=elf,
        trace=trace,
        runtime_abi=_ABI,
        scale_inputs=[dict(kernel="kern", arg_index=3, offset=0, dtype="f32", shape=[1])],
        input_bits={3: bits},
        slot=dict(offset=32, width=32) if slot is None else slot,
        custom_opcode=0x7B,
    )


def test_runtime_trace_follows_the_derived_slot_offset(tmp_path):
    from merlin.targetgen.runtime_scale_trace import authenticate_report

    low = tmp_path / "low"
    low.mkdir()
    elf, trace = _program(low, "low_slot", caller_regs={5: 16 << 32})
    report = authenticate_report(_audit(elf, trace, slot=dict(offset=0, width=32)))
    (use,) = report["config_uses"]
    assert report["slot"] == dict(offset=0, width=32)
    assert (use["scale_bits"], use["static_bits"], use["rs2"]) == (_QUARTER, 16 << 32, 16 << 32 | _QUARTER)
    # The same program under the rs2[63:32] slot, and the high-slot program under rs2[31:0], refuse.
    with pytest.raises(ValueError, match="outside the derived acc_scale slot"):
        _audit(elf, trace)
    with pytest.raises(ValueError, match="outside the derived acc_scale slot"):
        _audit(*_program(tmp_path), slot=dict(offset=0, width=32))
    # Static bits that overlap the scale field end its provenance.
    clash = tmp_path / "clash"
    clash.mkdir()
    with pytest.raises(ValueError, match="never configures a store"):
        _audit(*_program(clash, "low_slot", caller_regs={5: 16 << 32 | 1}), slot=dict(offset=0, width=32))
    for slot in (dict(offset=33, width=32), dict(offset=-1, width=32), dict(offset=0, width=16), dict(offset=32)):
        with pytest.raises(ValueError, match="not one binary32 field of rs2"):
            _audit(*_program(tmp_path), slot=slot)


@pytest.mark.parametrize("bits", [_QUARTER, 0, 0x80000000, 0xBF000000, 1, 0x7F7FFFFF])
@pytest.mark.parametrize("form", [None, "lwu_or", "flw_fmv"])
def test_runtime_trace_requires_loaded_input_bits_and_exact_rs2_provenance(tmp_path, bits, form):
    from merlin.targetgen.runtime_scale_trace import authenticate_report

    elf, trace = _program(tmp_path, form, bits)
    report = authenticate_report(_audit(elf, trace, bits))
    (use,) = report["config_uses"]
    assert (use["arg_index"], use["scale_bits"], use["static_bits"], use["rs2"]) == (3, bits, 16, bits << 32 | 16)
    assert use["pc"] == 0x80000108 and use["command_ordinal"] == 0 and use["load_pc"] == report["loads"][0]["pc"]
    assert report["loads"][0]["address"] == _SCALE_PTR and report["loads"][0]["loaded_bits"] == bits
    assert set(report["evidence"]) == {"kern:3"}
    # The frozen input, not the program, decides the expected bits.
    with pytest.raises(ValueError, match="frozen input bits"):
        _audit(elf, trace, bits ^ 1)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("wrong_register", "never configures a store"),
        ("stale", "call 2 of kern has no retired load"),
        ("wrong_input", "no retired load of scale input 3"),
        ("missing_load", "no retired load of scale input 3"),
        ("wrong_slot", "outside the derived acc_scale slot"),
        ("short_shift", "never configures a store"),
        ("elf_word", "differs from the linked ELF"),
    ],
)
def test_runtime_trace_mutations_fail_closed(tmp_path, mutation, error):
    elf, trace = _program(tmp_path, mutation)
    with pytest.raises(ValueError, match=error):
        _audit(elf, trace)


def test_runtime_trace_refuses_multi_hart_traces(tmp_path):
    elf, trace = _program(tmp_path)
    text = trace.read_text()
    trace.write_text(text + _line(_BASE, 0x00000013, "").replace("core   0:", "core   1:"))
    with pytest.raises(ValueError, match="one hart is modeled"):
        _audit(elf, trace)


def test_runtime_trace_report_is_authenticated(tmp_path):
    from merlin.targetgen.runtime_scale_trace import authenticate_report

    report = _audit(*_program(tmp_path))
    for mutate in (
        lambda r: r["config_uses"][0].update(rs2=r["config_uses"][0]["rs2"] ^ 1),
        lambda r: r["loads"][0].update(loaded_bits=0),
        lambda r: r.update(status="fail"),
        lambda r: r["evidence"].update({"kern:3": "0" * 64}),
    ):
        changed = deepcopy(report)
        mutate(changed)
        with pytest.raises(ValueError, match="UNKNOWN"):
            authenticate_report(changed)


# ---------------------------------------------------------------------------
# Lint runtime_oracle_obligations consumer.

_POST_SCALE = "3dd87ffd661a658b7e4ae56c5fe3b5ea21934707ec38f0ede2eb2fd99c3fd5ea"


def _source_caps(relu=True):
    """An i32 sibling and a runtime-scale i8 sibling, as the source declares them."""
    runtime = dict(scale_source="runtime_input", weight_scale_granularity="tensor")
    stages = ["activation", "scale"] if relu else ["scale"]
    return [
        dict(
            operation=dict(
                attributes=dict(
                    case=dict(exit="accumulator", producer="contraction", readout_stages=[]),
                    quantization_scheme=runtime,
                )
            )
        ),
        dict(
            operation=dict(
                attributes=dict(
                    case=dict(
                        exit="requant",
                        producer="contraction",
                        readout_stages=stages,
                        multipliers=[0.5],
                        runtime_scale_input="ReadoutScale",
                    ),
                    quantization_scheme=runtime,
                )
            )
        ),
    ]


def _registry():
    from merlin_experiments.phase0 import fusion_execution as E

    return E.relu_order_registry("gemmini")


def test_relu_order_identities_are_read_from_the_reviewed_target_file(tmp_path):
    import json

    import yaml
    from merlin_experiments.phase0 import fusion_execution as E

    assert set(_registry()) == {_POST_SCALE}
    source = yaml.safe_load(
        (Path(__file__).resolve().parents[3] / "examples/gemmini/target" / E.RELU_ORDER_FILE).read_text()
    )
    entry = source["extensions"][_POST_SCALE]

    def mutated(change):
        document = deepcopy(source)
        change(document)
        return document

    cases = [
        None,
        "not: [yaml",
        mutated(lambda d: d.update(schema="other")),
        mutated(lambda d: d.update(target="atlas")),
        mutated(lambda d: d.update(extra=True)),
        mutated(lambda d: d.update(extensions=[])),
        mutated(lambda d: d["extensions"][_POST_SCALE].update(relu_scale_order="unknown")),
        mutated(lambda d: d["extensions"][_POST_SCALE].update(evidence="")),
        mutated(lambda d: d["extensions"][_POST_SCALE].update(extra=1)),
        mutated(lambda d: d["extensions"].update({_POST_SCALE.upper(): entry})),
        # An identity must be the digest its hardware pin declares.
        mutated(lambda d: d["extensions"].update({"ab" * 32: entry})),
        mutated(lambda d: d["extensions"][_POST_SCALE].update(artifact="no_such_artifact")),
    ]
    for index, document in enumerate(cases):
        path = tmp_path / f"order{index}.yaml"
        if document is not None:
            path.write_text(document if isinstance(document, str) else yaml.safe_dump(document))
        with pytest.raises(E.OracleObligationRefused, match="relu-order registry"):
            E.relu_order_registry("gemmini", path)
    with pytest.raises(E.OracleObligationRefused, match="no reviewed target directory"):
        E.relu_order_registry("no_such_target")
    # A missing registry refuses the execution case before Spike, with a retained record.
    payload, proof, catalog = _obligation_case(0.25)
    root = tmp_path / "case" / "spike-build" / "region_offload"
    root.mkdir(parents=True)
    (root / "core_aten_catalog.json").write_text(json.dumps(catalog))
    record, reason = E._oracle_discharge(
        payload, proof, dict(symbol="kern"), tmp_path / "case", _POST_SCALE, "gemmini", tmp_path / "absent.yaml"
    )
    assert record["status"] == "refused" and "relu-order registry" in reason


def _obligation_case(value, relu=True, obligations="lint"):
    import numpy as np

    payload = dict(
        nodes=_source_caps(relu),
        inputs=[dict(name="A", dtype="i8", shape=[2, 2]), dict(name="node1_ReadoutScale", dtype="f32", shape=[1])],
        input_values=dict(A=[[1, 2], [3, 4]], node1_ReadoutScale=[float(np.float32(value))]),
    )
    member = dict(relu=relu, scale_ref=[1, 0], scale_operand=dict(kind="runtime_input", source=dict(func_arg=1)))
    proof = dict(kernels=[dict(symbol="kern", shared_input=dict(members=[dict(relu=False), member]))])
    kernel = dict(
        symbol="kern",
        args=[
            dict(name="a", dtype="i8", shape=[2, 2], role="input"),
            dict(name="readout_scale0", dtype="f32", shape=[1], role="input"),
            dict(name="out0", dtype="i8", shape=[2, 2], role="output"),
        ],
        runtime_scale_arguments=["readout_scale0"],
    )
    if obligations == "lint" and relu:
        kernel["runtime_oracle_obligations"] = [
            dict(
                instruction_index=14, scale_operand=[1, 0], requires="relu_scale_order_parity_or_nonnegative_test_input"
            )
        ]
    binding = dict(symbol="kern", operands=dict(a=dict(func_arg=0), readout_scale0=dict(func_arg=1)))
    return payload, proof, dict(kernels=[kernel], bindings=[binding])


def test_negative_relu_scale_is_refused_on_the_post_scale_relu_extension():
    from merlin_experiments.phase0 import fusion_execution as E

    assert _registry()[_POST_SCALE]["relu_scale_order"] == "post_scale"
    for value in (-0.5, -1e-45, float("nan"), float("-inf")):
        with pytest.raises(E.OracleObligationRefused, match="oracle_obligation_undischarged.*not relu-order parity"):
            E.discharge_oracle_obligations(*_obligation_case(value), "kern", _POST_SCALE, relu_order=_registry())
    # Without ReLU there is no obligation: signed scales remain qualified.
    record = E.discharge_oracle_obligations(
        *_obligation_case(-0.5, relu=False), "kern", _POST_SCALE, relu_order=_registry()
    )
    assert record == dict(
        schema="merlin.runtime_oracle_discharge.v1",
        status="discharged",
        symbol="kern",
        extension_sha256=_POST_SCALE,
        obligations=[],
    )


@pytest.mark.parametrize("value", [0.0, -0.0, 0.25, 0.3, 0.5, 1.0, 1e-45, 3.4028234663852886e38])
def test_nonnegative_relu_palette_discharges_with_recorded_basis(value):
    import numpy as np
    from merlin_experiments.phase0 import fusion_execution as E

    record = E.discharge_oracle_obligations(*_obligation_case(value), "kern", _POST_SCALE, relu_order=_registry())
    (row,) = record["obligations"]
    bits = int(np.float32(value).view(np.uint32))
    assert record["status"] == "discharged" and row["basis"] == "nonnegative_test_input"
    assert row["input_bits"] == f"{bits:#010x}" and row["extension_sha256"] == _POST_SCALE
    assert row["input_class"] == ("negative_zero" if bits == 0x80000000 else "finite_nonnegative")


def test_parity_proved_extension_discharges_signed_relu():
    from merlin_experiments.phase0 import fusion_execution as E

    repaired = "ab" * 32
    with pytest.raises(E.OracleObligationRefused):
        E.discharge_oracle_obligations(*_obligation_case(-0.5), "kern", repaired, relu_order=_registry())
    identity = dict(artifact="repaired", relu_scale_order="pre_scale", evidence="signed-scale probe")
    record = E.discharge_oracle_obligations(*_obligation_case(-0.5), "kern", repaired, relu_order={repaired: identity})
    assert record["obligations"][0]["basis"] == "parity_proved_extension"


def test_missing_or_unknown_obligations_fail_closed():
    from merlin_experiments.phase0 import fusion_execution as E

    # The lint obligation was dropped, but the proof still has an active ReLU.
    with pytest.raises(E.OracleObligationRefused, match="oracle_obligation_missing"):
        E.discharge_oracle_obligations(
            *_obligation_case(0.25, obligations=None), "kern", _POST_SCALE, relu_order=_registry()
        )
    payload, proof, catalog = _obligation_case(0.25)
    catalog["kernels"][0]["runtime_oracle_obligations"][0]["requires"] = "some_future_requirement"
    with pytest.raises(E.OracleObligationRefused, match="no consumer"):
        E.discharge_oracle_obligations(payload, proof, catalog, "kern", _POST_SCALE, relu_order=_registry())
    # The source, not the proof, decides which readouts owe the obligation.
    payload, proof, catalog = _obligation_case(0.25)
    proof["kernels"][0]["shared_input"]["members"][1]["relu"] = False
    with pytest.raises(E.OracleObligationRefused, match="activation differs between source and admission"):
        E.discharge_oracle_obligations(payload, proof, catalog, "kern", _POST_SCALE, relu_order=_registry())
    payload, proof, catalog = _obligation_case(0.25)
    del proof["kernels"][0]["shared_input"]["members"][1]["scale_operand"]
    with pytest.raises(E.OracleObligationRefused, match="no proved scale_operand"):
        E.discharge_oracle_obligations(payload, proof, catalog, "kern", _POST_SCALE, relu_order=_registry())
    payload, proof, catalog = _obligation_case(0.25)
    del proof["kernels"][0]["shared_input"]
    catalog["kernels"][0].pop("runtime_oracle_obligations")
    with pytest.raises(E.OracleObligationRefused, match="no proved scale_operand"):
        E.discharge_oracle_obligations(payload, proof, catalog, "kern", _POST_SCALE, relu_order=_registry())
    # A proof that dropped its ReLU member's activation still owes the source obligation.
    payload, proof, catalog = _obligation_case(0.25, obligations=None)
    del proof["kernels"][0]["shared_input"]["members"][1]["relu"]
    with pytest.raises(E.OracleObligationRefused, match="oracle_obligation_missing"):
        E.discharge_oracle_obligations(payload, proof, catalog, "kern", _POST_SCALE, relu_order=_registry())
    # The reverse: a runtime operand the source does not declare.
    payload, proof, catalog = _obligation_case(0.25)
    payload["nodes"][1]["operation"]["attributes"]["quantization_scheme"]["scale_source"] = "constant"
    with pytest.raises(E.OracleObligationRefused, match="source does not declare"):
        E.discharge_oracle_obligations(payload, proof, catalog, "kern", _POST_SCALE, relu_order=_registry())
    # A passing execution needs the discharged record; no record never qualifies.
    for discharge in (None, {}, dict(status="refused")):
        outcome = E._qualify(dict(outcome="not_admitted", reason="pass"), discharge)
        assert outcome["outcome"] == "not_admitted" and outcome["reason"].startswith("oracle_obligation_unchecked")
    assert E._qualify(dict(outcome="not_admitted"), dict(status="discharged"))["outcome"] == "pass"


def test_execute_case_refuses_undischarged_obligation_before_spike(tmp_path, monkeypatch):
    import json

    from merlin_experiments.phase0 import fusion_execution as E

    payload, proof, catalog = _obligation_case(-0.5)
    root = tmp_path / "spike-build" / "region_offload"
    root.mkdir(parents=True)
    (root / "core_aten_catalog.json").write_text(json.dumps(catalog))
    record, reason = E._oracle_discharge(payload, proof, dict(symbol="kern"), tmp_path, _POST_SCALE, "gemmini")
    assert record["status"] == "refused" and reason.startswith("oracle_obligation_undischarged")
    assert json.loads((tmp_path / "oracle-obligations.json").read_text()) == record


def _binding_case(value=0.25):
    import numpy as np

    payload = dict(
        nodes=_source_caps(),
        references=[dict(shape=[2, 3], dtype="i32"), dict(shape=[2, 3], dtype="i8")],
        inputs=[dict(name=n, dtype=d, shape=s) for n, d, s in (("A", "i8", [2, 2]), ("W", "i8", [2, 3]))]
        + [dict(name="W1", dtype="i8", shape=[2, 3]), dict(name="node1_ReadoutScale", dtype="f32", shape=[1])],
        input_values=dict(node1_ReadoutScale=[float(np.float32(value))]),
    )
    member = dict(relu=True, scale_ref=[3, 0], scale_operand=dict(kind="runtime_input", source=dict(func_arg=3)))
    operands = dict(a=dict(func_arg=0), w0=dict(func_arg=1), w1=dict(func_arg=2), readout_scale0=dict(func_arg=3))
    proof = dict(
        bindings=[dict(symbol="kern", operands=operands)],
        kernels=[dict(symbol="kern", shared_input=dict(members=[dict(relu=False), member]))],
    )
    args = [dict(name=n, dtype="i8", shape=[2, 2], role="input") for n in ("a", "w0", "w1")]
    args += [dict(name="readout_scale0", dtype="f32", shape=[1], role="input")]
    args += [dict(name="out0", dtype="i32", shape=[2, 3], role="output")]
    args += [dict(name="out1", dtype="i8", shape=[2, 3], role="output")]
    return payload, proof, args


def test_runtime_source_without_a_proved_scale_operand_is_refused():
    """The review probe: a runtime source must never fall back to constant column bits."""
    from merlin_experiments.phase0 import fusion_execution as E

    scheme = dict(scale_source="runtime_input", weight_scale_granularity="tensor")
    case = dict(
        exit="requant",
        producer="contraction",
        readout_stages=["scale"],
        multipliers=[0.5],
        runtime_scale_input="runtime_gain",
    )
    payload = dict(
        nodes=[dict(operation=dict(attributes=dict(case=case, quantization_scheme=scheme)))],
        references=[dict(shape=[4, 4])],
        inputs=[
            dict(name="node0_a", dtype="i8", shape=[4, 4]),
            dict(name="node0_runtime_gain", dtype="f32", shape=[1]),
        ],
        input_values={"node0_runtime_gain": [0.5]},
    )
    args = [
        dict(name="a", role="input", dtype="i8", shape=[4, 4]),
        dict(name="g", role="input", dtype="f32", shape=[1]),
        dict(name="o", role="output", dtype="i8", shape=[4, 4]),
    ]
    bindings = [dict(symbol="k", operands=dict(a=dict(func_arg=0), g=dict(func_arg=1)))]
    loads = dict(evidence={"k:1": "chain-digest"})
    for proof in (
        dict(bindings=bindings),
        dict(kernels=[dict(symbol="k")], bindings=bindings),
        dict(kernels=[dict(symbol="k", schedule=dict(relu=False))], bindings=bindings),
    ):
        for scale_loads in (None, loads):
            with pytest.raises(ValueError, match="no proved scale_operand"):
                E._readout_bindings(payload, proof, "k", args, scale_loads=scale_loads)
    # The probe's own payload names its scale input outside the merged source.
    unmerged = deepcopy(payload)
    unmerged["inputs"][1]["name"] = "runtime_gain"
    with pytest.raises(ValueError, match="no unique scale input"):
        E._readout_bindings(unmerged, dict(kernels=[dict(symbol="k")], bindings=bindings), "k", args)
    # A single-kernel proof binds its runtime scale in the schedule record.
    operand = dict(kind="runtime_input", source=dict(func_arg=1))
    proof = dict(kernels=[dict(symbol="k", schedule=dict(relu=False, scale_operand=operand))], bindings=bindings)
    (part,) = E._readout_bindings(payload, proof, "k", args, loads)["partitions"]
    assert part["scale_input"]["arg_index"] == 1 and part["expected_input_bits"] == 0x3F000000
    assert "column_scale_bits" not in part
    wrong = deepcopy(proof)
    wrong["kernels"][0]["schedule"]["scale_operand"]["source"]["func_arg"] = 0
    with pytest.raises(ValueError, match="differs from the source scale input"):
        E._readout_bindings(payload, wrong, "k", args, loads)
    # The reverse: a constant source whose proof binds a runtime operand.
    constant = deepcopy(payload)
    constant["nodes"][0]["operation"]["attributes"]["quantization_scheme"]["scale_source"] = "constant"
    with pytest.raises(ValueError, match="source does not declare"):
        E._readout_bindings(constant, proof, "k", args, loads)
    ambiguous = dict(proof, kernels=proof["kernels"] * 2)
    with pytest.raises(ValueError, match="ambiguous kernel proofs"):
        E._readout_bindings(payload, ambiguous, "k", args, loads)


def test_runtime_partition_binds_source_scale_input_and_frozen_bits(monkeypatch):
    from merlin_experiments.phase0 import fusion_execution as E

    from merlin.targetgen import functional_truth_compositions as F

    monkeypatch.setattr(F, "_audit_parameters", lambda *a: pytest.fail("runtime bits came from case metadata"))
    payload, proof, args = _binding_case(-0.0)
    loads = dict(evidence={"kern:3": "chain-digest"})
    (part,) = E._readout_bindings(payload, proof, "kern", args, loads)["partitions"]
    assert part == dict(
        output=dict(kind="argbase", arg_index=5, offset=0),
        shape=[2, 3],
        scale_input=dict(kind="argbase", arg_index=3, offset=0, dtype="f32", shape=[1]),
        expected_input_bits=0x80000000,
        scale_load_evidence_id="chain-digest",
    )
    with pytest.raises(ValueError, match="no retired scale-load evidence"):
        E._readout_bindings(payload, proof, "kern", args)
    for mutate in (
        lambda p, q, a: q["kernels"][0]["shared_input"]["members"][1].update(scale_ref=[2, 0]),
        lambda p, q, a: q["bindings"][0]["operands"].update(w1=dict(func_arg=3)),
        lambda p, q, a: a[3].update(dtype="i8"),
    ):
        changed = deepcopy(_binding_case())
        mutate(*changed)
        with pytest.raises(ValueError, match="not one bound f32"):
            E._readout_bindings(*changed[:2], "kern", changed[2], loads)
    contexts = [(dict(exit="accumulator"), {}, dict(i32_exit=True)), (dict(exit="requant"), {}, {"relu": True})]
    placement = dict(stages=dict(node0_mac0=[0], node1_scale=[1]))
    outputs = [dict(kind="argbase", arg_index=i, offset=0) for i in (4, 5)]
    _, scoped = E._scoped_sibling_modes(contexts, placement, {}, [part], outputs)
    assert scoped["node0_mac0"]["i32_exit"] == dict(outputs=[outputs[0]], identity_config=True)
    assert scoped["node1_scale"]["readout_multipliers"]["scale_input"]["arg_index"] == 3
