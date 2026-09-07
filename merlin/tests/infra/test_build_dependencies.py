"""Pure build grants cannot weaken or leak across the native sandbox policy."""
from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from merlin.targetgen.sandbox.build_dependencies import HostBuildDependencies


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def setup(tmp_path):
    source, frozen, work = (tmp_path / p for p in ("source", "frozen", "work"))
    for path in (source, frozen, work):
        path.mkdir()
    helper = source / "pure.py"
    helper.write_text("ANSWER = None\n")
    worker, request = work / "worker.py", work / "request.json"
    worker.write_text("pass\n")
    request.write_text("{}")
    oracle = frozen / "runtime" / "reference.py"
    oracle.parent.mkdir()
    oracle.write_text("hidden")
    prefix = ["bwrap", "--clearenv", "--ro-bind", str(frozen), str(frozen),
              "--ro-bind", str(helper), str(frozen / "existing.py"),
              "--ro-bind", "/dev/null", str(oracle), "--chdir", str(work)]
    sandbox = {"command_prefix": prefix, "answer_surfaces": [
        {"path": str(oracle), "kind": "file", "origin": "oracle"}]}
    argv = ("/usr/bin/python3", str(worker), str(request))
    capability = HostBuildDependencies(str(source), str(frozen), argv,
        tuple((str(p), digest(p)) for p in (Path(argv[0]), worker, request)),
        ((str(helper), str(frozen / "pure.py"), digest(helper)),), str(request), str(worker))
    return capability, sandbox, source, frozen, helper


def test_before_masks_and_no_policy_mutation(setup):
    cap, policy, _, frozen, _ = setup
    original = list(policy["command_prefix"])
    actual = cap.extend(policy, cap.argv)
    assert actual.index(str(frozen / "pure.py")) < actual.index("/dev/null")
    assert policy["command_prefix"] == original
    assert len(actual) == len(original) + 3
    assert cap.extend(policy, cap.argv) == actual  # never accumulate grants


@pytest.mark.parametrize("change", ["argv", "request", "helper"])
def test_exact_worker_and_bytes_bound(setup, change):
    cap, policy, _, _, helper = setup
    argv = cap.argv
    if change == "argv":
        argv = (*argv, "--other")
    elif change == "request":
        Path(cap.request_path).write_text('{"changed": true}')
    else:
        helper.write_text("changed")
    with pytest.raises(ValueError):
        cap.extend(policy, argv)


def test_source_cannot_rename_oracle_as_pure(setup):
    cap, policy, source, frozen, _ = setup
    hidden = source / "runtime" / "reference.py"
    hidden.parent.mkdir()
    hidden.write_text("secret")
    for destination in (frozen / "pure.py", frozen / "runtime" / "reference.py"):
        bad = replace(cap, source_grants=((str(hidden), str(destination), digest(hidden)),))
        with pytest.raises(ValueError, match="mapping|masked"):
            bad.extend(policy, cap.argv)


def test_masked_directory_stays_denied(setup):
    cap, policy, source, frozen, _ = setup
    hidden = source / "backend" / "helper.py"
    hidden.parent.mkdir()
    hidden.write_text("not pure")
    policy["answer_surfaces"].append({"path": str(frozen / "backend"), "kind": "dir"})
    policy["command_prefix"] += ["--tmpfs", str(frozen / "backend")]
    bad = replace(cap, source_grants=((str(hidden), str(frozen / "backend/helper.py"), digest(hidden)),))
    with pytest.raises(ValueError, match="masked"):
        bad.extend(policy, cap.argv)


def test_destination_not_flag_or_unbound_directory(setup, tmp_path):
    cap, policy, _, frozen, helper = setup
    for dest in ("--ro-bind", str(frozen / "../escape.py")):
        with pytest.raises(ValueError):
            replace(cap, source_grants=((str(helper), dest, digest(helper)),)).extend(policy, cap.argv)
    other = tmp_path / "unbound"
    other.mkdir()
    with pytest.raises(ValueError, match="source view"):
        replace(cap, namespace_root=str(other)).extend(policy, cap.argv)


def test_exposed_original_answer_refused(setup):
    cap, policy, *_ = setup
    p = policy["command_prefix"]
    i = p.index("/dev/null") - 1
    del p[i:i+3]
    with pytest.raises(ValueError, match="answer masks"):
        cap.extend(policy, cap.argv)


def test_only_canonical_format_pair_is_grantable(setup):
    from merlin.common.paths import repo_root, schemas_dir
    cap, policy, _, frozen, _ = setup
    registry, schema = (schemas_dir() / name for name in
                        ("quant_formats.registry.yaml", "quant_format.schema.yaml"))
    rows = tuple((str(p), str(frozen / p.relative_to(repo_root())), digest(p), role)
                 for p, role in ((registry, "numeric_format_registry"),
                                 (schema, "numeric_format_schema")))
    authorized = replace(cap, source_root=str(repo_root()), source_grants=(), format_data=rows)
    prefix = authorized.extend(policy, cap.argv)
    assert len(prefix) == len(policy["command_prefix"]) + 6
    with pytest.raises(ValueError, match="exact numeric"):
        replace(authorized, format_data=rows[:1]).extend(policy, cap.argv)
    bad = (str(Path(cap.request_path)), rows[0][1], digest(Path(cap.request_path)), rows[0][3])
    with pytest.raises(ValueError, match="canonical"):
        replace(authorized, format_data=(bad, rows[1])).extend(policy, cap.argv)


def test_format_overlay_not_granted(setup, monkeypatch):
    cap, policy, *_ = setup
    monkeypatch.setenv("MERLIN_QUANT_FORMATS", cap.request_path)
    with pytest.raises(ValueError, match="overlays"):
        replace(cap, format_data=((cap.request_path, cap.request_path,
                                  digest(Path(cap.request_path)), "unknown"),)).extend(policy, cap.argv)


def test_one_pass_visibility_matches_existing_replay(setup):
    from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface
    from merlin.targetgen.sandbox.bwrap import coverage_gap
    from merlin.targetgen.sandbox.build_dependencies import _coverage_gaps
    _, policy, source, frozen, helper = setup
    surfaces = [AnswerSurface("ref", Path(policy["answer_surfaces"][0]["path"]), "file", "oracle"),
                AnswerSurface("public", frozen / "existing.py", "file", "test"),
                AnswerSurface("absent", frozen / "absent.py", "file", "test")]
    for additions in ([], ["--tmpfs", str(frozen)],
                      ["--ro-bind", str(helper), str(surfaces[0].path)]):
        prefix = policy["command_prefix"] + additions
        assert _coverage_gaps(prefix, surfaces) == coverage_gap(prefix, surfaces)


def test_reuses_real_private_overlay_for_missing_readonly_leaf(setup, tmp_path):
    import importlib
    import sys
    from merlin.common.paths import repo_root
    from merlin.targetgen.sandbox.bwrap import _mounts, is_exposed
    sys.path.insert(0, str(repo_root() / "merlin/experiments/gemmini_perf_bench/scripts"))
    controller = importlib.import_module("run_global_perf_experiment")
    cap, policy, _, frozen, helper = setup
    original = list(policy["command_prefix"])
    private = tmp_path / "private_overlay"
    private.mkdir()
    actual = cap.extend(policy, cap.argv, overlay_root=private,
                        overlay_builder=controller.compiler_dependency_mounts)
    assert original == policy["command_prefix"]
    assert not (frozen / "pure.py").exists()  # immutable original not edited
    served = [(Path(src), Path(dest)) for state, src, dest in _mounts(actual)
              if state == "expose" and Path(dest) == frozen]
    assert served and private in served[0][0].parents
    assert (served[0][0] / "pure.py").read_bytes() == helper.read_bytes()
    assert not (served[0][0].stat().st_mode & 0o222)
    assert not is_exposed(actual, frozen / "runtime/reference.py")
