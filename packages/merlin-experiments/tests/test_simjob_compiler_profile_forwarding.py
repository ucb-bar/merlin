"""Async grading receives the same original host profile as synchronous grading."""

import json
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.brokers import simjob
from merlin_experiments.phase1.context import InvocationContext
from merlin_experiments.phase1.feedback import lifecycle as L


def context(root):
    return InvocationContext(root, root / "descriptor.yaml", root, "fixture", root, root, root, ())


@pytest.mark.parametrize("selected", [False, True])
def test_parent_forwards_original_profile_to_both_grading_brokers_only(tmp_path, monkeypatch, selected):
    names = ("merlin_experiments.phase1.brokers.selfcheck", "merlin_experiments.phase1.brokers.simjob",
             "merlin_experiments.phase1.brokers.isa_tools")
    monkeypatch.setattr(L._TR, "COMMON_CLIENTS", ())
    monkeypatch.setattr(L._TR, "COMMON_BROKERS", tuple(
        SimpleNamespace(module=name, log=f"{i}.log", channel=".qa_channel") for i, name in enumerate(names)))
    monkeypatch.setattr(L._TR, "brokers_for", lambda _: ())
    commands = []
    monkeypatch.setattr(L.subprocess, "Popen", lambda argv, **kw: commands.append(argv) or SimpleNamespace())
    manifest = tmp_path / "original-reviewed-bundle.yaml" if selected else None
    L.start_brokers(tmp_path / "workspace", L.BrokerConfig(context(tmp_path), (), tmp_path / "timing.json",
                                                          compiler_bundle_manifest=manifest))
    for name, argv in zip(names, commands, strict=True):
        assert ("--compiler-bundle-manifest" in argv) == (selected and name != names[2])
        if "--compiler-bundle-manifest" in argv:
            assert argv[argv.index("--compiler-bundle-manifest") + 1] == str(manifest)


@pytest.mark.parametrize("selected", [False, True])
def test_actual_async_dispatch_uses_parent_manifest_never_request_selection(tmp_path, monkeypatch, selected):
    ws, frozen = tmp_path / "workspace", tmp_path / "frozen"
    channel = ws / ".qa_channel"
    channel.mkdir(parents=True)
    (frozen / "submission").mkdir(parents=True)
    (frozen / "submission/compiler.py").write_text("pass\n")
    # A caller-supplied lookalike field cannot select original compiler authority.
    evil = ws / "candidate-selected.yaml"
    (channel / "simreq_probe.json").write_text(json.dumps(
        {"sim": "gsim", "capsules": "case", "compiler_bundle_manifest": str(evil)}))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(simjob, "_resolve_tiers", lambda *a, **kw: ("L2", "L3", {"case"}))
    monkeypatch.setattr(simjob, "_per_capsule_timeout", lambda *a, **kw: (30, "test"))
    monkeypatch.setattr(simjob, "_allowed_sims", lambda _: ("gsim",))
    monkeypatch.setattr(simjob, "_valid_capsules", lambda *a: ["case"])
    monkeypatch.setattr(simjob, "_freeze_direct_job_source", lambda _: (frozen / "submission", frozen, None))
    monkeypatch.setattr(simjob, "_sim_env", lambda _: {"MERLIN_OPERATOR_GRADE_PROCESSES": "1"})
    observed = {}

    class DispatchObserved(Exception):
        pass

    def intercept(argv, **kwargs):
        observed.update(argv=argv, environment=kwargs["env"])
        raise DispatchObserved

    monkeypatch.setattr(simjob, "_spawn_selfcheck", intercept)
    manifest = tmp_path / "original-reviewed-bundle.yaml"
    argv = ["--ws", str(ws)]
    if selected:
        argv += ["--compiler-bundle-manifest", str(manifest)]
    with pytest.raises(DispatchObserved):
        simjob.main(argv, context=context(tmp_path), capsules_root=tmp_path / "public")
    command = observed["argv"]
    assert ("--compiler-bundle-manifest" in command) is selected
    if selected:
        assert command[command.index("--compiler-bundle-manifest") + 1] == str(manifest)
    assert str(evil) not in command
    assert observed["environment"]["MERLIN_OPERATOR_GRADE_PROCESSES"] == "1"


def test_invalid_forwarded_profile_cannot_create_original_execution(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback.compiler_profile_runtime import selfcheck_execution

    ws = tmp_path / "workspace"
    ws.mkdir()
    forged = ws / "candidate-selected.yaml"
    forged.write_text("allowed: []\nhost_inputs: []\n")
    monkeypatch.chdir(ws)
    with pytest.raises((RuntimeError, ValueError, OSError)):
        with selfcheck_execution(forged, context=context(tmp_path), submission=ws / "submission",
                                 work_root=tmp_path / "work"):
            pytest.fail("an unverified profile cannot restore original compiler authority")
