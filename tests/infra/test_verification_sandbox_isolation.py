from __future__ import annotations

from types import SimpleNamespace

from verification_witnesses.sandbox_isolation import (
    _NO_SANDBOX,
    _SANDBOX,
    BW,
    AnswerSurface,
    _sim_family_policy_without_its_checkout,
    answer_surfaces,
    audit_tokens,
    probe_sandbox,
    pytest,
    sim_toolchain,
    subprocess,
    weight_files,
)


@pytest.mark.usefixtures("_sim_family_policy_without_its_checkout")
def test_missing_declared_grant_refuses_snapshot(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    ws = tmp_path / "run" / "workspace"
    bundle = {"allowed": [{"path": "does/not/exist.txt"}]}
    with pytest.raises(FileNotFoundError, match="unresolvable allowed grant"):
        BW.materialize_bundle_inputs(ws, bundle, repo=repo)
    assert not BW.bundle_snapshot_root(ws).exists()


@pytest.mark.usefixtures("_sim_family_policy_without_its_checkout")
def test_snapshot_bytes_do_not_follow_later_source_edits(tmp_path):
    """Falsifier for the live-worktree race: both sandbox views stay frozen."""
    if _NO_SANDBOX:
        pytest.skip("bwrap unavailable — immutable mount tested structurally only")
    repo = tmp_path / "repo"
    source = repo / "inputs" / "contract.txt"
    source.parent.mkdir(parents=True)
    source.write_text("before\n", encoding="utf-8")
    ws = tmp_path / "run" / "workspace"
    bundle = {"allowed": [{"path": "inputs/contract.txt"}]}
    try:
        manifest = BW.materialize_bundle_inputs(ws, bundle, repo=repo)
        assert manifest["n_files"] == 1
        assert manifest["n_bytes"] == len("before\n")
        assert len(manifest["content_sha256"]) == 64
        frozen = BW.bundle_snapshot_root(ws) / manifest["grants"][0]["snapshot"]
        assert not frozen.stat().st_mode & 146
        assert not BW.bundle_snapshot_root(ws).stat().st_mode & 146
        ws.mkdir(parents=True)
        (ws / "contract.txt").symlink_to(source)
        source.write_text("after\n", encoding="utf-8")
        assert BW.materialize_bundle_inputs(ws, bundle, repo=repo) == manifest
        source.unlink()
        argv = BW.base_argv(ws, bundle, repo=repo)
        grant = [
            (argv[i + 1], argv[i + 2])
            for (i, token) in enumerate(argv[:-2])
            if token == "--ro-bind" and argv[i + 2] == str(source)
        ]
        assert grant and all((src != str(source) for (src, _) in grant))
        probe = f'''cat "{source}"; cat "{ws / "contract.txt"}"'''
        out = subprocess.run([*argv, "bash", "-c", probe], capture_output=True, text=True, timeout=30)
        assert out.returncode == 0, out.stderr
        assert out.stdout.splitlines() == ["before", "before"]
    finally:
        BW.remove_bundle_snapshot(ws)


@pytest.mark.usefixtures("_sim_family_policy_without_its_checkout")
def test_post_snapshot_answer_file_is_not_mounted_inside_frozen_parent(tmp_path, monkeypatch):
    """A live-tree addition is absent from the run snapshot and needs no answer overlay.

    The snapshot may bind a broad directory such as ``experiments/corpora``.  If another host process
    later creates a golden below that live directory, mount-table replay must inspect the frozen
    *source* of the controlling bind.  Treating the live destination as the source makes
    ``apply_answer_masks`` append a ``/dev/null`` file bind for a path absent from the snapshot; real
    bwrap then tries to create that mount point below the read-only parent and aborts before the agent
    starts.
    """
    repo = tmp_path / "repo"
    contract = repo / "experiments/corpora"
    (contract / "public/A0").mkdir(parents=True)
    (contract / "public/A0/capsule.yaml").write_text("name: A0\n", encoding="utf-8")
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(repo))
    weight = contract / "other-target/model/weights.safetensors"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"private answer bytes")
    discovered = weight_files(SimpleNamespace(capsule_corpus=None))
    assert weight in discovered

    ws = tmp_path / "run/workspace"
    bundle = {"allowed": [{"path": "experiments/corpora"}]}
    try:
        BW.materialize_bundle_inputs(ws, bundle, repo=repo)
        ws.mkdir(parents=True)
        late_golden = contract / "_perf/PK00/golden.yaml"
        late_golden.parent.mkdir(parents=True)
        late_golden.write_text("outputs: []\n", encoding="utf-8")
        surface = AnswerSurface("late golden", late_golden, "file", "golden")
        weights = [AnswerSurface("private weight", path, "file", "weight") for path in discovered]
        argv = BW.apply_answer_masks(BW.base_argv(ws, bundle, repo=repo), [surface, *weights])
        assert not BW.is_exposed(argv, weight)
        assert not BW.is_exposed(argv, late_golden)
        assert ["--ro-bind", "/dev/null", str(late_golden)] not in [
            argv[index : index + 3] for (index, token) in enumerate(argv[:-2]) if token == "--ro-bind"
        ]
        if not _NO_SANDBOX:
            completed = subprocess.run(
                [*argv, "bash", "-c", f'test ! -e "{late_golden}"'], capture_output=True, text=True, timeout=30
            )
            assert completed.returncode == 0, completed.stderr
    finally:
        BW.remove_bundle_snapshot(ws)
