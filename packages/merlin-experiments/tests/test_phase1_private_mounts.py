"""Private model inputs stay out of the agent's native bwrap argument list."""

from __future__ import annotations

from merlin.targetgen.sandbox import bwrap as BW


def test_sealed_capture_and_numerics_paths_do_not_enter_bwrap_args(tmp_path):
    repo = tmp_path / "repo"
    public = repo / "visible"
    public.mkdir(parents=True)
    (public / "input.json").write_text("{}\n")
    private = tmp_path / "private/gemmini"
    capture = private / "captures/r53b"
    numerics = private / "numerics/v1"
    capture.mkdir(parents=True)
    numerics.mkdir(parents=True)
    (capture / "sealed.json").write_text("secret\n")
    (numerics / "policy.json").write_text("secret\n")
    workspace = tmp_path / "run/workspace"
    workspace.mkdir(parents=True)
    bundle = {
        "allowed": [{"path": "visible", "mode": "ro"}],
        "private_validation_paths": [
            {"path": str(capture), "kind": "dir"},
            {"path": str(numerics), "kind": "dir"},
        ],
    }
    BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    argv = BW.base_argv(workspace, bundle, repo=repo)
    assert all(str(path) not in argv for path in (capture, numerics, capture / "sealed.json"))
    assert BW.is_exposed(argv, public / "input.json")
    assert not BW.is_exposed(argv, capture / "sealed.json")
    assert not BW.is_exposed(argv, numerics / "policy.json")
