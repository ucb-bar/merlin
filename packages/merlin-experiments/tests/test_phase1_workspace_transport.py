"""Real external copy assembly and synthetic sandbox results, never kernel-isolation claims."""

from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import workspace_transport as W
from merlin_experiments.phase1.context import InvocationContext


@pytest.fixture
def inputs(tmp_path):
    repo = tmp_path / "operator tree"
    corpus = repo / "merlin/contract/capsules/public"
    corpus.mkdir(parents=True)
    (corpus / "capsule.interface.mlir").write_text("module {}")
    (corpus / "golden.yaml").write_text("private")
    descriptor = tmp_path / "chosen descriptor.yaml"
    descriptor.write_text(f"target: fixture\ncapsule_corpus: {corpus}\n")
    context = InvocationContext(
        repo, descriptor, tmp_path, "fixture", tmp_path / "runs", tmp_path / "reports", tmp_path / "bundles", ()
    )
    return context, corpus, tmp_path / "workspace"


def test_copy_resolves_shared_grants_and_filters_all_answer_names(inputs):
    context, corpus, ws = inputs
    for name in (
        "golden.json",
        "foo.golden.mlir",
        "expected_instruction_coverage.yaml",
        "expected_command_buffer.txt",
        "model.safetensors",
        "model.safetensors.manifest.json",
    ):
        (corpus / name).write_text("private")
    hidden = corpus / "hidden"
    hidden.mkdir()
    (hidden / "capsule.yaml").write_text("held out")
    shorthand = context.repo / "merlin/experiments/tool"
    shorthand.mkdir(parents=True)
    (shorthand / "public.txt").write_text("public")
    bundle = {"allowed": [{"path": "merlin/contract"}, {"path": "experiments/tool", "as": "tool"}, {"path": "absent"}]}
    evidence = W.assemble(bundle, ws, "none", context=context)
    copied = ws / "merlin/contract/capsules/public"
    assert sorted(p.name for p in copied.iterdir()) == ["capsule.interface.mlir"]
    assert (ws / "tool/public.txt").read_text() == "public"
    assert evidence.copy_report["unresolvable_grants"] == ["absent"]
    assert W.probe(ws, bundle, "none", context=context)["pilot_golden_visible_to_agent"] == "OK"


def test_copy_denies_resolved_alias_and_nested_private_files(inputs):
    context, corpus, ws = inputs
    tool = context.repo / "merlin/experiments/tool"
    tool.mkdir(parents=True)
    (tool / "public.py").write_text("public")
    (tool / "private.py").write_text("private")
    bundle = {
        "allowed": [{"path": "experiments/tool", "as": "tool"}],
        "denied": [{"path": "merlin/experiments/tool/private.py"}],
    }
    W.assemble(bundle, ws, "none", context=context)
    assert (ws / "tool/public.py").read_text() == "public"
    assert not (ws / "tool/private.py").exists()


@pytest.mark.parametrize("alias", ["../outside", "/absolute"])
def test_destination_escape_is_refused(inputs, alias):
    context, corpus, ws = inputs
    with pytest.raises(ValueError, match="relative"):
        W.assemble({"allowed": [{"path": str(corpus), "as": alias}]}, ws, "none", context=context)


@pytest.mark.parametrize(
    "stdout,returncode,status",
    [
        ("DONE\n", 0, "OK"),
        ("", 0, "UNPROVEN"),
        ("DONE\n", 1, "UNPROVEN"),
        ("DONE\nDONE\n", 0, "UNPROVEN"),
        ("unexpected\nDONE\n", 0, "UNPROVEN"),
        ("LEAK:/a path/expected_instruction_coverage.yaml\nDONE\n", 0, "LEAK"),
    ],
)
def test_mask_requires_real_completion_and_preserves_path_spacing(inputs, monkeypatch, stdout, returncode, status):
    context, corpus, ws = inputs
    ws.mkdir()
    captured = []
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT", "/absent/ambient.yaml")
    monkeypatch.setattr(W.BW, "snapshot_input_paths", lambda *a, **kw: [corpus])

    def command(script, workspace, bundle, *, context):
        captured.append(script)
        assert workspace == ws
        return "synthetic sandbox"

    monkeypatch.setattr(W, "sandbox_command", command)
    monkeypatch.setattr(W.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=stdout, returncode=returncode))
    result = W.probe(ws, {}, "bwrap", context=context)
    assert result["pilot_golden_visible_to_agent"] == status
    assert bool(result["probe_failure"]) == (status == "UNPROVEN")
    assert "'" + str(corpus) + "'" in captured[0]
    assert "-exec sh -c" in captured[0] and "$(find" not in captured[0]
    assert not list(ws.glob(".mask-control-*"))


def test_missing_selected_descriptor_cannot_fall_back(inputs):
    context, corpus, ws = inputs
    context.descriptor.unlink()
    ws.mkdir()
    with pytest.raises(FileNotFoundError):
        W.probe(ws, {}, "none", context=context)


def test_missing_frozen_inputs_cannot_fall_back_to_live(inputs, monkeypatch):
    context, corpus, ws = inputs
    ws.mkdir()

    def refuse(*a, **kw):
        raise RuntimeError("snapshot missing")

    monkeypatch.setattr(W.BW, "snapshot_input_paths", refuse)
    with pytest.raises(RuntimeError, match="snapshot missing"):
        W.probe(ws, {}, "bwrap", context=context)


def test_copy_probe_detects_file_symlinks_into_selected_external_corpus(inputs):
    context, corpus, ws = inputs
    ws.mkdir()
    (ws / "innocent-alias").symlink_to(corpus / "golden.yaml")
    result = W.probe(ws, {}, "none", context=context)
    assert result["pilot_golden_visible_to_agent"] == "LEAK"
    assert result["symlinks_into_capsules"] == [str(ws / "innocent-alias")]


def test_frozen_assembly_filters_snapshot_not_mutable_live_membership(inputs, monkeypatch):
    context, corpus, ws = inputs
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(ws.parent / "out"))
    bundle = {"allowed": [{"path": str(corpus), "as": "public"}]}
    W.BW.materialize_bundle_inputs(ws, bundle, repo=context.repo)
    try:
        (corpus / "golden.yaml").unlink()
        (corpus / "late.txt").write_text("not admitted")
        evidence = W.assemble(bundle, ws, "bwrap", context=context)
        assert evidence.violations == []
        assert sorted(p.name for p in (ws / "public").iterdir()) == ["capsule.interface.mlir"]
        assert (ws / "public/capsule.interface.mlir").is_symlink()
        assert (ws / "public/capsule.interface.mlir").readlink() == corpus / "capsule.interface.mlir"
        # These aliases are destination names; only bwrap supplies frozen bytes there.
        assert W.BW.snapshot_input_paths(ws, bundle, [corpus], repo=context.repo)[0] != corpus
    finally:
        W.BW.remove_bundle_snapshot(ws)


def test_explicit_public_contract_below_blanket_target_deny_is_preserved(inputs):
    context, corpus, ws = inputs
    package = context.repo / "target-package"
    contracts = package / "contracts"
    contracts.mkdir(parents=True)
    (contracts / "facts.json").write_text("public facts")
    W.assemble(
        {"allowed": [{"path": str(contracts), "as": "facts"}], "denied": [{"path": str(package)}]},
        ws,
        "none",
        context=context,
    )
    assert (ws / "facts/facts.json").read_text() == "public facts"


def test_probe_timeout_never_reports_ok(inputs, monkeypatch):
    context, corpus, ws = inputs
    ws.mkdir()
    monkeypatch.setattr(W.BW, "snapshot_input_paths", lambda *a, **kw: [corpus])
    monkeypatch.setattr(W, "sandbox_command", lambda *a, **kw: "synthetic")

    def timeout(*a, **kw):
        raise W.subprocess.TimeoutExpired("synthetic", 60)

    monkeypatch.setattr(W.subprocess, "run", timeout)
    with pytest.raises(W.subprocess.TimeoutExpired):
        W.probe(ws, {}, "bwrap", context=context)
    assert not list(ws.glob(".mask-control-*"))


def test_copy_denied_external_alias_cannot_smuggle_private_file(inputs):
    context, corpus, ws = inputs
    tools = context.repo / "tools"
    tools.mkdir()
    private = context.repo / "private"
    private.mkdir()
    (private / "answer.txt").write_text("secret")
    (tools / "innocent.txt").symlink_to(private / "answer.txt")
    (tools / "public.txt").write_text("public")
    W.assemble({"allowed": [{"path": "tools"}], "denied": [{"path": "private"}]}, ws, "none", context=context)
    assert not (ws / "tools/innocent.txt").exists()
    assert (ws / "tools/public.txt").read_text() == "public"


def test_probe_refuses_ancestor_link_exposing_selected_corpus(inputs):
    context, corpus, ws = inputs
    ws.mkdir()
    (ws / "innocent").symlink_to(corpus.parent)
    result = W.probe(ws, {}, "none", context=context)
    assert result["pilot_golden_visible_to_agent"] == "LEAK"
