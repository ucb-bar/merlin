"""Whole-call scope and actual recipe/fence evidence; no simulation or tier credit."""

import copy
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import annotated_spike_evidence as E
from merlin_experiments.phase2 import owned_model_observation as O

from merlin.llvmlower.compilation_recipe import CompilationRecipe


def listing():
    return E.parse_disassembly("""00000100 <main>:
      100: csrr s0,mcycle
      104: li a0,4
      108: jal 400 <merlin_run_multi>
      10c: csrr s1,mcycle
    """)


def test_whole_call_has_its_own_scope_and_actual_invocation_not_a_kernel_alias():
    rows = listing()
    result = O.synchronous_window(rows, counter_asm="csrr %0, mcycle")
    assert result["begin_pc"] == 0x100 and result["end_pc"] == 0x10C
    assert result["invocation_call_pc"] == 0x108
    assert "completion_call_pc" not in result and "symbol_aliases" not in result
    selection = dict(selection_sha256="a" * 64, owned_model_scope=O.SCOPE_CONTRACT)
    assert O.scope_digest(selection) != E.kernel_call_scope(selection)
    with pytest.raises(ValueError, match="explicit synchronous"):
        O.scope_digest(dict(selection, owned_model_scope={}))


def test_pending_physical_tier_allows_only_audited_numerical_scope():
    record = dict(status="unverified", tiers={"L2": {"status": "pass"}}, missing_tiers=["L3"], failed_tiers=[])
    check = dict(kind="owned_host_model_execution", violations=[],
                 candidate_source_placement={"status": "clean"},
                 candidate_source_coverage={"status": "verified"},
                 candidate_completed_dispatch={"status": "verified"},
                 candidate_required_tiers={"tiers": {"L2": {"status": "pass"}}})
    result = {"model_execution_check": check}
    before = copy.deepcopy((result, record))
    assert O._completed_numerical_scope(result, record)
    assert (result, record) == before  # No pending tier or certification is upgraded.
    assert not O._completed_numerical_scope({}, record)
    for changes in [dict(missing_tiers=["L2"]), dict(failed_tiers=["L3"]),
                    dict(tiers={"L2": {"status": "unverified"}})]:
        assert not O._completed_numerical_scope(result, dict(record, **changes))
    for key in ["candidate_source_placement", "candidate_source_coverage", "candidate_completed_dispatch"]:
        damaged = copy.deepcopy(result)
        damaged["model_execution_check"][key]["status"] = "unverified"
        assert not O._completed_numerical_scope(damaged, record)


@pytest.mark.parametrize(
    "damage", ["extra_counter", "missing_counter", "indirect", "branch", "wrong_call", "extra_call"]
)
def test_ambiguous_or_bypassable_counter_scopes_refuse(damage):
    rows = listing()
    if damage == "extra_counter":
        rows += [{**rows[0], "function": "helper", "pc": 0x500}]
    elif damage == "missing_counter":
        rows.pop()
    elif damage == "indirect":
        rows[2]["mnemonic"] = "jalr"
    elif damage == "branch":
        rows[1]["mnemonic"] = "beqz"
    elif damage == "wrong_call":
        rows[2]["symbols"] = ["kernel_entry"]
    else:
        rows[1].update(mnemonic="jal", symbols=["helper"])
    with pytest.raises(ValueError):
        O.synchronous_window(rows, counter_asm="csrr %0, mcycle")


def pin(path):
    return dict(path=str(path), sha256=O.F.digest(path))


def completion_fixture(tmp_path, *, fence=True, opaque=False):
    host = tmp_path / "host.mlir"
    host.write_text("""module { llvm.func @group() llvm.func @forward() {
      llvm.call @group() : () -> ()
      llvm.return
    }}""")
    kernel = tmp_path / "kernel.mlir"
    instructions = (
        '"llvm.inline_asm"() <{asm_string = ".insn r 0xb, 0, 4, x0, x0, x0", '
        'constraints = "~{memory}", has_side_effects}> : () -> ()'
    )
    if fence:
        instructions += (
            '\n"llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects}> : () -> ()'
        )
    if opaque:
        instructions += "\nllvm.call @opaque() : () -> ()"
    kernel.write_text("module { llvm.func @opaque() llvm.func @kernel() { " + instructions + "\nllvm.return } }")
    return dict(
        device_artifacts={"group": {"llvm_mlir": pin(kernel)}},
        host_calls=dict(mandatory_call_roster={"group": 1}, generic_cfg=pin(host)),
    )


def test_all_dispatched_groups_require_closing_fence(tmp_path):
    dispatch = completion_fixture(tmp_path)
    assert len(O._completion_modules(dispatch)) == 2
    changed = copy.deepcopy(dispatch)
    changed["host_calls"]["mandatory_call_roster"]["second"] = 1
    with pytest.raises(ValueError, match="every dispatched"):
        O._completion_modules(changed)
    kernel = Path(dispatch["device_artifacts"]["group"]["llvm_mlir"]["path"])
    kernel.write_text(kernel.read_text() + "\n")
    with pytest.raises(ValueError, match="source changed"):
        O._completion_modules(dispatch)


@pytest.mark.parametrize("fence,opaque", [(False, False), (True, True)])
def test_absent_fence_or_unknown_call_cannot_grant_synchronous_completion(tmp_path, fence, opaque):
    with pytest.raises(ValueError, match="completion is unknown"):
        O._completion_modules(completion_fixture(tmp_path, fence=fence, opaque=opaque))


def test_host_indirect_or_unqualified_dispatch_remains_unknown(tmp_path):
    dispatch = completion_fixture(tmp_path)
    host = Path(dispatch["host_calls"]["generic_cfg"]["path"])
    host.write_text(host.read_text().replace("@group", "@unqualified"))
    dispatch["host_calls"]["generic_cfg"] = pin(host)
    with pytest.raises(ValueError, match="unsupported or indirect"):
        O._completion_modules(dispatch)


def recipe_fixture(tmp_path):
    source = tmp_path / "program.c"
    source.write_text("int main(void){return 0;}\n")
    elf = tmp_path / "program.elf"
    observed = CompilationRecipe(tmp_path, producer=Path(__file__).resolve())
    cc = shutil.which("cc")
    assert cc
    observed.run([cc, str(source), "-o", str(elf)], runner=subprocess.run, inputs=[source], output=elf)
    observed.completed(elf)
    owner = O.OwnedModelScope(
        elf,
        observed.path,
        O.F.digest(observed.path),
        json.dumps(dict(elf=pin(elf))).encode(),
        ((str(source), O.F.digest(source)),),
    )
    return owner, source


def test_real_link_recipe_reopened_without_fake_historical_invocation(tmp_path):
    owner, source = recipe_fixture(tmp_path)
    assert owner.producer(owner.elf, SimpleNamespace(max_elf_bytes=1 << 20)) == pin(owner.elf)
    assert not (tmp_path / "invocations").exists()
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="source evidence changed"):
        owner.producer(owner.elf, SimpleNamespace(max_elf_bytes=1 << 20))


@pytest.mark.parametrize("damage", ["elf", "recipe", "other_elf"])
def test_original_producer_custody_refuses_changes(tmp_path, damage):
    owner, _ = recipe_fixture(tmp_path)
    selected = owner.elf
    if damage == "elf":
        selected.write_bytes(selected.read_bytes() + b"changed")
    elif damage == "recipe":
        owner.recipe_path.write_text(owner.recipe_path.read_text() + "\n")
    else:
        selected = tmp_path / "other.elf"
        selected.write_bytes(owner.elf.read_bytes())
    with pytest.raises(ValueError):
        owner.producer(selected, SimpleNamespace(max_elf_bytes=1 << 20))


def test_incomplete_numerics_do_not_construct_a_scope():
    with pytest.raises(ValueError, match="completed native"):
        O.prepare(
            dict(owned_host_native_execution=dict(status="verified", tiers={"L2": {"status": "fail"}})),
            target="gemmini",
        )


def test_scope_binds_complete_contract_bytes():
    selection = dict(selection_sha256="a" * 64, owned_model_scope=O.SCOPE_CONTRACT)
    changed = copy.deepcopy(selection)
    changed["owned_model_scope"]["completion"] = "return_assumed"
    with pytest.raises(ValueError):
        O.scope_digest(changed)
    assert O.scope_digest(selection) != O.scope_digest(dict(selection, selection_sha256="b" * 64))


@pytest.mark.parametrize("tuned", [False, True])
def test_native_feedback_selects_explicit_owned_scope_without_tuned_fallback(monkeypatch, tuned):
    from merlin_experiments.phase1.feedback import joint_performance as J

    capsule = {"name": "neutral"}
    selection = dict(runtime={"path": "kernel"}, owned_model_runtime={"path": "whole"}, facts={"sha256": "a" * 64})
    if tuned:
        selection["timing_epochs"] = [
            dict(
                epoch=1,
                mode="tuned",
                runtime={"path": "tuned"},
                calibration_evidence={"path": "review"},
                coverage=dict(
                    capsule_sha256=[J.capsule_identity(capsule)],
                    scope_sha256="a" * 64,
                    memory_regime_sha256="b" * 64,
                    feature_ranges={},
                ),
            )
        ]
    monkeypatch.setattr(J, "selected", lambda: ({"sha256": "c" * 64}, selection))
    monkeypatch.setattr(J, "structural_feedback", lambda *a, **kw: {})
    monkeypatch.setattr(J, "_read", lambda *a, **kw: {})
    selected = []

    def freeze(pin, **kwargs):
        selected.append(pin)
        raise ValueError("stop before execution in selection-only test")

    monkeypatch.setattr(J, "_runtime", freeze)
    result = dict(
        owned_host_native_execution={}, numeric={"status": "pass"}, tiers={"L2": {"status": "pass", "engine": "spike"}}
    )
    row = {}
    J.attach(
        row,
        "unused",
        result=result,
        context=SimpleNamespace(target="neutral"),
        compiler="unused",
        capsule=capsule,
        remaining_s=30,
    )
    assert selected == [{"path": "tuned" if tuned else "whole"}]
    assert row["performance"]["status"] == "UNKNOWN"


from test_ordinary_archive_screen import ordinary_inputs as ordinary_inputs  # noqa: E402,F401
from test_ordinary_archive_screen import source as source  # noqa: E402,F401


def test_runtime_freeze_requires_separate_whole_model_domain(source):
    from merlin_experiments.phase2.archive_spike_runtime import freeze_runtime
    from merlin_experiments.phase2.contracts import StageGateError

    r = source["runtime"]
    selection = dict(json.loads(r.timing_json), owned_model_scope=O.SCOPE_CONTRACT)
    model = json.loads(r.model_json)
    model["domain"]["scope_sha256"] = O.scope_digest(selection)
    path = source["root"] / "whole-model.json"
    path.write_text(json.dumps(model))
    args = dict(
        review_id="explicit provisional whole-call scope",
        functional_argv=r.functional_argv,
        observer=r.observer,
        disassembler=r.disassembler,
        model_path=path,
        source_manifest=r.source_manifest,
        libraries=r.libraries,
        dependencies=r.dependencies,
        entry_symbol="merlin_run_multi",
        completion_symbol="return_after_synchronous_dispatch",
        timing_selection=selection,
    )
    fresh = freeze_runtime(**args)
    assert json.loads(fresh.model_json)["domain"]["scope_sha256"] != json.loads(r.model_json)["domain"]["scope_sha256"]
    fresh.verify()
    with pytest.raises(StageGateError):
        freeze_runtime(**dict(args, completion_symbol="kernel_complete"))
    with pytest.raises(StageGateError):
        freeze_runtime(**dict(args, model_path=r.model_path))


def test_owned_console_must_match_original_complete_numerical_execution(tmp_path):
    from hashlib import sha256

    expected = b"independently validated complete original output"
    owner = O.OwnedModelScope(
        tmp_path / "elf",
        tmp_path / "recipe",
        "a" * 64,
        json.dumps(dict(numerical_console={"sha256": sha256(expected).hexdigest()})).encode(),
        (),
    )
    owner.validate_console(expected)
    with pytest.raises(ValueError, match="original complete numerical"):
        owner.validate_console(expected + b"changed")


def binary_frame():
    payload = b"\xff\nDONE\n\x00"
    checksum = 14695981039346656037
    for byte in payload:
        checksum = ((checksum ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return (
        f"OUT_BIN_BEGIN v1 out0 1 {len(payload)} 1 u {len(payload)}\n".encode()
        + payload
        + f"OUT_BIN_END v1 {checksum:016x}\nMETRIC cycles 12\nDONE\n".encode()
    )


@pytest.mark.parametrize("damage", [None, "checksum", "truncated", "text"])
def test_guardian_binary_mode_keeps_complete_frame_and_process_gates(tmp_path, damage):
    import sys

    from merlin.common import invocation_record as I
    from merlin.perf import _census_process as P
    from merlin.perf.dynamic_census import CensusLimits, CensusUnavailable

    payload = binary_frame()
    if damage == "checksum":
        payload = payload.replace(b"\xff", b"\xfe")
    elif damage == "truncated":
        payload = payload[:45]
    argv = [sys.executable, "-c", "import os;os.write(1," + repr(payload) + ")"]

    def run():
        with I.observe(
            tmp_path / "run",
            stage="neutral_binary_protocol",
            argv=argv,
            cwd=tmp_path,
            env={},
            inputs=(),
            outputs=(),
            dependencies=(),
        ) as record:
            code = P.capture(
                argv,
                environment={},
                cwd=tmp_path,
                limits=CensusLimits.for_timeout(3),
                record=record,
                census=SimpleNamespace(line=lambda line: None),
                stdout_protocol="text" if damage == "text" else "out_bin_v1",
            )
            record.complete_streams(code)
            assert record.census_cleanup_complete
            assert (record.directory / "stdout.bin").read_bytes() == payload

    if damage is None:
        run()
    else:
        with pytest.raises(CensusUnavailable):
            run()


def test_unknown_target_completion_contract_refuses(tmp_path):
    with pytest.raises(ValueError, match="Gemmini fence contract"):
        O.prepare({}, target="other", evidence_root=tmp_path / "review")
