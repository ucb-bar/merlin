"""Saved library evidence never replaces independent host selection."""

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import compiler_library_state as S
from merlin_experiments.phase2 import paired_measurement as PM
from test_compiler_libraries import selected
from test_paired_measurement import _plan

from merlin.targetgen.compiler_library import active_compiler_library, scoped_compiler_library


def frozen(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    selection, _ = selected(source)
    binding = S.freeze_selection(selection, tmp_path / "exact/python")
    snapshot = S.verify_binding(binding, selection)
    return selection, binding, snapshot


def test_exact_snapshot_excludes_private_leaves_and_requires_host_owner(tmp_path):
    host, binding, snapshot = frozen(tmp_path)
    assert binding["origin"] == host.record()
    assert snapshot.contract == host.contract
    assert (host.root / "private-answer.txt").is_file()
    assert not (snapshot.root / "private-answer.txt").exists()
    assert not snapshot.root.stat().st_mode & 0o222
    with pytest.raises(S.StageGateError, match="independently reviewed"):
        S.verify_binding(binding, None)
    other = tmp_path / "other"
    other.mkdir()
    different, _ = selected(other)
    assert different.contract == host.contract
    with pytest.raises(S.StageGateError, match="independently reviewed"):
        S.verify_binding(binding, different)


@pytest.mark.parametrize("change", ["binding", "contract", "source", "snapshot", "mode", "extra"])
def test_forged_or_stale_sources_refuse_at_final_consumer(tmp_path, change):
    host, binding, snapshot = frozen(tmp_path)
    handoff = SimpleNamespace(
        compiler_library_binding=binding, compiler_library_selection=snapshot, compiler_library_host_selection=host
    )
    assert S.handoff_selection(handoff) is snapshot
    if change == "binding":
        binding["snapshot_sha256"] = "f" * 64
    elif change == "contract":
        binding["snapshot"]["contract"]["review_id"] += "-forged"
        binding["sha256"] = S.document_sha256({k: v for k, v in binding.items() if k != "sha256"})
    elif change == "source":
        (host.root / "merlin/portable.py").write_text("changed = True\n")
    elif change == "snapshot":
        leaf = snapshot.root / "merlin/portable.py"
        leaf.chmod(0o644)
        leaf.write_text("changed = True\n")
        leaf.chmod(0o444)
    elif change == "mode":
        snapshot.root.chmod(0o755)
    else:
        snapshot.root.chmod(0o755)
        (snapshot.root / "foreign.py").write_text("foreign = True\n")
        snapshot.root.chmod(0o555)
    with pytest.raises((S.StageGateError, ValueError), match="library|member"):
        S.handoff_selection(handoff)


def test_host_loader_requires_complete_explicit_review_and_expected_identity(tmp_path):
    host, _ = selected(tmp_path)
    contract = tmp_path / "reviewed.json"
    contract.write_text(json.dumps(host.contract.record()))
    arguments = dict(root=host.root, contract_path=contract, contract_sha256=host.contract.sha256)
    assert S.load_selection(**arguments).record() == host.record()
    assert S.load_selection(root=None, contract_path=None, contract_sha256=None) is None
    for missing in arguments:
        with pytest.raises(S.StageGateError, match="required together"):
            S.load_selection(**{**arguments, missing: None})
    with pytest.raises(S.StageGateError, match="identity"):
        S.load_selection(**{**arguments, "contract_sha256": "0" * 64})


@pytest.mark.parametrize("width", [1, 2])
def test_paired_workers_execute_both_arms_under_exact_snapshot_owner(tmp_path, width):
    host, binding, snapshot = frozen(tmp_path)
    plan = _plan(tmp_path)
    declaration = {
        **plan.declaration,
        "schema": "paired_arm4_measurement_plan_v4",
        "additional_compiler_library": copy.deepcopy(binding),
    }
    plan = replace(
        plan,
        declaration=declaration,
        declaration_sha256=PM.sha256_bytes(PM.canonical_bytes(declaration)),
        compiler_library_binding=binding,
        compiler_library_selection=snapshot,
        compiler_library_host_selection=host,
    )
    out = tmp_path / "results"
    out.mkdir()
    calls = []

    def executor(spec, *args, **kwargs):
        assert active_compiler_library() is snapshot
        calls.append(spec.arm)
        return {"execution": spec.as_dict(), "measurement": {"per_sim": {}}}

    PM.execute_schedule(
        plan,
        out,
        contract_root=tmp_path,
        timeout=1,
        target_experiment=None,
        rtl_identity={},
        hardware_counters=False,
        executor=executor,
        progress=lambda _: None,
        fanout={"effective": width},
    )
    assert sorted(calls) == ["baseline", "baseline", "candidate", "candidate"]
    assert active_compiler_library() is None
    index = json.loads((out / "raw_results.index.json").read_text())
    for row in index:
        raw = json.loads((out / row["path"]).read_text())
        assert raw["compiler_library_binding_sha256"] == binding["sha256"]


def test_legacy_plan_cannot_inherit_an_ambient_library(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    selection, _ = selected(source)
    plan = _plan(tmp_path)
    out = tmp_path / "results"
    out.mkdir()
    with scoped_compiler_library(selection), pytest.raises(PM.PC.CampaignGateError, match="active compiler library"):
        PM.execute_schedule(
            plan,
            out,
            contract_root=tmp_path,
            timeout=1,
            target_experiment=None,
            rtl_identity={},
            hardware_counters=False,
            executor=lambda *a, **k: pytest.fail("executor"),
        )
    assert not list(out.iterdir())


@pytest.mark.parametrize("additional", [False, True])
def test_paired_cli_supplies_independent_library_selection_before_final_consumption(tmp_path, monkeypatch, additional):
    from merlin_experiments.phase2 import paired_cli as CLI

    host, _ = selected(tmp_path)
    contract = tmp_path / "reviewed.json"
    contract.write_text(json.dumps(host.contract.record()))
    arguments = []
    for option in (
        "source-root",
        "functional-runs-root",
        "measurement-root",
        "contract-root",
        "candidate-record",
        "descriptor",
        "corpus-root",
        "corpus-manifest",
        "gsim-certificate",
        "rtl-facts",
    ):
        arguments += ["--" + option, str(tmp_path / option)]
    for option in (
        "functional-submission-sha256",
        "corpus-manifest-sha256",
        "corpus-capsules-sha256",
        "gsim-certificate-sha256",
    ):
        arguments += ["--" + option, "a" * 64]
    arguments += ["--functional-run-id", "base", "--run-id", "pair", "--phase", "tuning"]
    if additional:
        arguments += [
            "--compiler-library-root",
            str(host.root),
            "--compiler-library-contract",
            str(contract),
            "--compiler-library-contract-sha256",
            host.contract.sha256,
        ]
    monkeypatch.setattr(CLI, "load_target_experiment", lambda *a, **k: SimpleNamespace())

    def consume(*args, **kwargs):
        if additional:
            assert kwargs["compiler_library_selection"].record() == host.record()
        else:
            assert "compiler_library_selection" not in kwargs
        raise RuntimeError("independent consumption reached")

    monkeypatch.setattr(CLI.PI, "load_paired_inputs", consume)
    with pytest.raises(RuntimeError, match="independent consumption"):
        CLI.main(arguments)
    assert not (tmp_path / "measurement-root").exists()
