"""Original capture replay uses selected source, decoder and immutable observer bytes."""

from __future__ import annotations

import hashlib
import json

import pytest
from merlin_experiments.phase1.feedback import reference_analysis as analysis
from merlin_experiments.phase1.feedback import reference_capture_views as views
from merlin_experiments.phase1.feedback import reference_experiments as reference
from test_reference_analysis_tool import _request, _world

from merlin.kernels.reference_capture import source_function

SOURCE = b"""// projection(
// A : i8[1, 2] @DRAM,
// W : i8[2, 2] @DRAM,
// O : i32[1, 2] @DRAM
// )
void projection(void *ctxt, const int8_t *A, const int8_t *W, int32_t *O) {}
"""

# A small, explicitly selected test observer; the integration receipt uses the
# unchanged production observer and all 37,029 original mm_o issue commands.
OBSERVER = """from merlin.kernels.tile_reference import recover_tile_region
from merlin.kernels.kernel_evidence import KernelEvidence, ArtifactPin, analyze_normalized_commands
def observe_capture(capture, *, settings, workspace):
    events = [row['event'] for row in capture['records'] if row['kind']=='command']
    recovered = recover_tile_region(events, capture['function']['declarations'], evidence_id='selected:test')
    facts = tuple(recovered.pop('facts'))
    evidence = KernelEvidence('selected:test', tuple(ArtifactPin.from_record(p) for p in capture['artifacts']),
        facts, capture_kind='native_issue', complete=True)
    commands = [dict(index=i,identity=e['kind'],roles=[e['kind']]) for i,e in enumerate(events)]
    model = analyze_normalized_commands(commands, complete=True, max_commands=settings['max_commands'])
    return dict(recovered,evidence=evidence,schedule_model=model)
"""


def _pin(path, workspace):
    return dict(path=path.relative_to(workspace).as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _fixture(tmp_path, monkeypatch):
    pytest.importorskip("tree_sitter_c")
    fixture = _world(tmp_path)
    folder = tmp_path / "cca-evidence"
    folder.mkdir()
    source = tmp_path / "reference-kernels/selected.c"
    source.write_bytes(SOURCE)
    (folder / "observer.py").write_text(OBSERVER)
    (folder / "contract.yaml").write_text("name: selected\n")
    (folder / "protocol.json").write_text(json.dumps(dict(target="selected")))
    events = [dict(kind="load", tensor="A", offset=0, stride=2, rows=1, cols=2, tile_cols=1, address=0, block_stride=1)]
    for k in range(2):
        events.append(
            dict(
                kind="load",
                tensor="W",
                offset=2 * k,
                stride=2,
                rows=1,
                cols=2,
                tile_cols=1,
                address=100,
                block_stride=1,
            )
        )
        for n in range(2):
            events += [
                dict(kind="bind", weight=100 + n, address=200 + n, accumulate=bool(k)),
                dict(kind="compute", lhs=k, rows=1, cols=1),
            ]
    events.append(
        dict(
            kind="store",
            tensor="O",
            offset=0,
            stride=8,
            address=200,
            rows=1,
            cols=2,
            tile_cols=1,
            block_stride=1,
            numeric=dict(
                accumulator_dtype="i32", bias="none", scale=1, activation="none", rounding="none", saturation="none"
            ),
        )
    )
    records = [
        dict(kind="buffer", name=name, address=address, bytes=size)
        for name, address, size in (("A", 10, 2), ("W", 20, 4), ("O", 30, 8))
    ]
    records += [dict(kind="command", event=event) for event in events] + [dict(kind="complete")]
    (folder / "capture.jsonl").write_text("\n".join(json.dumps(row) for row in records) + "\n")
    function = source_function(SOURCE, "projection")
    entry = dict(
        source=_pin(source, tmp_path),
        symbol="projection",
        function_body_sha256=function["body_sha256"],
        function_source_sha256=hashlib.sha256(function["source"].encode()).hexdigest(),
        capture=_pin(folder / "capture.jsonl", tmp_path),
        facts=_pin(fixture["facts"], tmp_path),
        contract=_pin(folder / "contract.yaml", tmp_path),
        operand_protocol=_pin(folder / "protocol.json", tmp_path),
        observer=dict(_pin(folder / "observer.py", tmp_path), factory="observe_capture"),
    )
    # Facts are public selected inputs too; copy them into the scoped grant.
    (folder / "facts.json").write_bytes(fixture["facts"].read_bytes())
    entry["facts"] = _pin(folder / "facts.json", tmp_path)
    manifest = dict(schema=views.SCHEMA, entries=[entry])
    path = folder / "manifest.json"

    def select():
        path.write_text(json.dumps(manifest))
        monkeypatch.setenv(views.SELECTION, json.dumps(dict(path=str(path), sha256=_pin(path, tmp_path)["sha256"])))

    select()
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(
        analysis,
        "_operand_decoder",
        lambda contract, facts, target, descriptor=None, **kwargs: dict(
            semantics=object(),
            target=target,
            decoder_inputs=[
                dict(path="selected_rtl_facts", sha256=_pin(fixture["facts"], tmp_path)["sha256"]),
                dict(path="selected_target_contract", sha256=_pin(folder / "contract.yaml", tmp_path)["sha256"]),
            ],
        ),
    )
    fixture.update(folder=folder, source=source, manifest=manifest, select=select, events=events)
    return fixture


def _lift(fixture):
    with reference.invocation(fixture["path"], fixture["sha"]):
        return analysis.handle(
            _request("lift", "reference-kernels/selected.c", format="c_source", symbol="projection"),
            workspace=fixture["workspace"],
            facts=fixture["facts"],
            target="selected",
        )


def test_raw_c_lift_replays_selected_capture_and_existing_tile_algebra(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    result = _lift(fixture)
    assert result["status"] == "analyzed"
    report = result["report"]
    assert report["kind"] == "captured_reference_analysis"
    assert report["semantic_region"]["complete"]
    assert report["tile_dataflow"]["compute_tiles"] == 4
    assert report["tile_dataflow"]["loaded_bytes"] == dict(A=2, W=4)
    assert report["tile_dataflow"]["lhs_retained_complete_invocation"]
    assert report["schedule_model"]["command_count"] == len(fixture["events"])
    assert report["cca"]["memory"]["onchip_resident"] == "a"
    assert report["completion"] == report["overlap"] == report["timing_status"] == "unknown"
    assert report["numerical_status"] == "not_run" and result["qualification_issued"] is False
    assert str(tmp_path) not in json.dumps(result)


def test_complete_capture_above_old_bound_replays_without_truncation(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    path = fixture["folder"] / "capture.jsonl"
    records = [json.loads(row) for row in path.read_text().splitlines()]
    # The selected test observer and output algebra still run. Redundant source
    # loads preserve the four products but must all survive event accounting.
    extra = 100_001
    records[3:3] = [dict(kind="command", event=fixture["events"][0])] * extra
    path.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    fixture["manifest"]["entries"][0]["capture"] = _pin(path, tmp_path)
    fixture["select"]()
    result = _lift(fixture)
    assert result["status"] == "analyzed", result
    report = result["report"]
    assert report["capture_records"] == len(records)
    assert report["schedule_model"]["command_count"] == len(fixture["events"]) + extra
    assert report["tile_dataflow"]["loaded_bytes"]["A"] == 2 * (extra + 1)
    assert report["tile_dataflow"]["compute_tiles"] == 4
    assert report["semantic_region"]["complete"]
    assert report["numerical_status"] == "not_run" and result["qualification_issued"] is False


def test_capture_replay_keeps_finite_native_command_bound(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(views, "MAX_CAPTURE_COMMANDS", len(fixture["events"]) - 1)
    result = _lift(fixture)
    assert result["status"] == "unknown" and result["reason"] == "analysis_capture_records_invalid"
    assert result["qualification_issued"] is False


@pytest.mark.parametrize(
    "corruption", ["source", "body", "capture", "facts", "contract", "protocol", "observer", "selection"]
)
def test_selected_capture_rejects_changed_source_and_decoder_artifacts(tmp_path, monkeypatch, corruption):
    fixture = _fixture(tmp_path, monkeypatch)
    folder, entry = fixture["folder"], fixture["manifest"]["entries"][0]
    if corruption == "source":
        fixture["source"].write_bytes(SOURCE + b"\n")
    elif corruption == "body":
        entry["function_body_sha256"] = "0" * 64
        fixture["select"]()
    elif corruption == "selection":
        (folder / "manifest.json").write_text("{}")
    else:
        filename = dict(
            capture="capture.jsonl",
            facts="facts.json",
            contract="contract.yaml",
            protocol="protocol.json",
            observer="observer.py",
        )[corruption]
        (folder / filename).write_bytes((folder / filename).read_bytes() + b" ")
    result = _lift(fixture)
    assert result["status"] == "unknown" and "report" not in result
    assert result["qualification_issued"] is False


def test_capture_replay_checks_output_coverage_not_just_pins(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    path = fixture["folder"] / "capture.jsonl"
    records = [json.loads(row) for row in path.read_text().splitlines()]
    records[-2]["event"]["cols"] = 1
    path.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    fixture["manifest"]["entries"][0]["capture"] = _pin(path, tmp_path)
    fixture["select"]()
    result = _lift(fixture)
    assert result["status"] == "unknown" and result["reason"] == "analysis_capture_replay_incomplete"
    assert "cover" in result["gap"]["detail"]["witness"]


@pytest.mark.parametrize("role", ["facts", "contract", "protocol_target"])
def test_capture_replay_rejects_another_selected_decoder_context(tmp_path, monkeypatch, role):
    fixture = _fixture(tmp_path, monkeypatch)
    if role == "protocol_target":
        path = fixture["folder"] / "protocol.json"
        path.write_text(json.dumps(dict(target="another")))
        fixture["manifest"]["entries"][0]["operand_protocol"] = _pin(path, tmp_path)
        fixture["select"]()
    else:
        original = analysis._operand_decoder

        def changed(*args, **kwargs):
            decoder = original(*args, **kwargs)
            for pin in decoder["decoder_inputs"]:
                if pin["path"] == ("selected_rtl_facts" if role == "facts" else "selected_target_contract"):
                    pin["sha256"] = "0" * 64
            return decoder

        monkeypatch.setattr(analysis, "_operand_decoder", changed)
    result = _lift(fixture)
    assert result["status"] == "unknown" and result["reason"] == "analysis_capture_decoder_mismatch"


def test_capture_selection_remains_disabled_for_control_arm(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    (fixture["folder"] / "manifest.json").write_text("malformed")
    response = analysis.handle(_request("lift", "reference-kernels/selected.c", format="c_source"), workspace=tmp_path)
    assert response["status"] == "denied" and response["reason"] == "reference_guidance_disabled"


def test_unmatched_c_source_retains_honest_syntax_only_result(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    (tmp_path / "reference-kernels/other.c").write_bytes(SOURCE)
    with reference.invocation(fixture["path"], fixture["sha"]):
        response = analysis.handle(_request("lift", "reference-kernels/other.c", format="c_source"), workspace=tmp_path)
    assert response["report"]["kind"] == "source_syntax"
    assert response["report"]["semantic_region_status"] == "unknown"


@pytest.mark.parametrize("unknown_assembly", [False, True])
def test_candidate_axes_count_actual_emission_and_refuse_unknown_commands(unknown_assembly):
    from merlin.targetgen.contract.interface_emit import emit_interface_mlir

    class Observer:
        def isa_constants(self, target):
            return dict(CUSTOM_OPCODE=123, FUNCT3=3)

        def decode_instruction(self, funct, *args):
            return ("SET_MODE" if funct == 0 else "FETCH"), {}

    cb = dict(
        abi_version="0.1",
        target="selected",
        tensors=dict(
            a=dict(shape=[1, 2], dtype="i8", role="input"),
            w=dict(shape=[2, 2], dtype="i8", role="weight"),
            o=dict(shape=[1, 2], dtype="i32", role="output"),
        ),
        commands=[
            dict(opcode="RES_PACK", operands=dict(src="w", dst="r"), attributes=dict(layout="packed_rhs")),
            dict(opcode="MATMUL_RESIDENT", operands=dict(lhs="a", rhs="r", dst="acc")),
            dict(opcode="COMMIT", operands=dict(src="acc", dst="o"), attributes=dict(epilogue=[], output_dtype="i32")),
            dict(opcode="EVICT", operands=dict(handle="r")),
        ],
    )
    emitted = """module { llvm.func @kernel_entry(%a: !llvm.ptr, %w: !llvm.ptr, %o: !llvm.ptr) {
    "llvm.inline_asm"() <{asm_string = ".insn r 0x7b, 3, 0, x0, x0, x0", constraints = "", has_side_effects}> : () -> ()
    "llvm.inline_asm"() <{asm_string = ".insn r 0x7b, 3, 1, x0, x0, x0", constraints = "", has_side_effects}> : () -> ()
      "llvm.inline_asm"() <{asm_string = "fence", constraints = "", has_side_effects}> : () -> ()
      llvm.return
    } }"""
    if unknown_assembly:
        emitted = emitted.replace("fence", "unrecognized command")
    report, cca = views.candidate_view(
        dict(source=emit_interface_mlir(cb).encode(), emitted=emitted.encode()),
        entry_symbol="kernel_entry",
        decoder=dict(target="selected", semantics=Observer(), structure=dict(
            instruction_roles={"SET_MODE": ["config"], "FETCH": ["operand_load"]})),
    )
    if unknown_assembly:
        assert report["status"] == "unknown" and cca is None
    else:
        assert report["command_counts"] == dict(SET_MODE=1, FETCH=1, FENCE=1)
        assert cca.dispatch.n_dispatches == 2 and cca.dispatch.n_fences == 1
        assert cca.dispatch.config_fraction == 0.5 and cca.provenance["source"] == "candidate"
        assert report["traffic"] == report["residency"] == "unknown"
