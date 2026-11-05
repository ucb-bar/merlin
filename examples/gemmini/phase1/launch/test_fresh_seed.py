"""Seeded fresh campaigns: a from-scratch plan is unchanged; a seeded one needs no operator hand step."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_fresh_campaign import F, _plan, fixture_spec, library  # noqa: F401  (pytest fixtures)

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "fresh_seed_scratch_golden.json"


def _scratch_products(F, fixture):
    """Every product of a from-scratch plan, with the fixture's temporary root, commit and checkout normalised."""
    out = fixture.root / "launch-out"
    record = json.loads((out / "proposed-launch.json").read_text())
    record["launch_spec"].pop("sha256")  # the spec file embeds the temporary root
    files = {
        path.relative_to(out).as_posix(): path.read_text()
        for path in sorted(out.rglob("*"))
        if path.is_file() and path.name != "proposed-launch.json"
    }
    scripts = {mode: F.preflight_script(record, run=mode == "run") for mode in ("preflight", "run")}
    text = json.dumps({"record": record, "files": files, "scripts": scripts}, indent=1, sort_keys=True)
    text = text.replace(str(fixture.root), "<ROOT>").replace(fixture.revision, "<REV>")
    return json.loads(text.replace(str(HERE), "<LAUNCH>"))


def test_scratch_plan_is_byte_identical_to_the_recorded_golden(F, fixture_spec, library, monkeypatch):
    monkeypatch.setenv("USER", "ubuntu")
    monkeypatch.setenv("LOGNAME", "ubuntu")
    _plan(F, fixture_spec)
    products = _scratch_products(F, fixture_spec)
    if os.environ.get("MERLIN_REGENERATE_SCRATCH_GOLDEN") == "1":
        GOLDEN.write_text(json.dumps(products, indent=1, sort_keys=True) + "\n")
    assert products == json.loads(GOLDEN.read_text())


# --------------------------------------------------------------------------- seeded launches


def _seed_fixture(root: Path) -> dict:
    """A frozen seed compiler with old-harness notes and evidence, and the run that produced it."""
    seed = root / "seed"
    files = {
        "backend.py": "TUNING = 'docs/tuning_table.md'\n",
        "manifest.yaml": "version: 1\n",
        "REPORT.md": "old harness narrative\n",
        "docs/iteration_notes.md": "round 3 of the old harness\n",
        "docs/tuning_table.md": "read by the compiler\n",
        "compiler_library/old.py": "OLD = 1\n",
        "SEED_EVIDENCE/baseline-map.json": '{"slowest": ["conv"]}\n',
    }
    for name, text in files.items():
        (seed / name).parent.mkdir(parents=True, exist_ok=True)
        (seed / name).write_text(text)
    workspace = root / "seed-run/workspace"
    for name, text in {
        "OPTIMIZATION_RULES.json": '{"rules": [{"name": "resident_rhs"}]}\n',
        "ab/rejected_epochs/README.md": "rejected: 3% slower on conv\n",
        "ab/rejected_epochs/raw.log": "x" * 10,
        "variants/epochs/backend.py": "TUNING = 'changed'\n",
        "variants/epochs/manifest.yaml": "version: 1\n",
        "variants/epochs/build/obj.o": "build product\n",
        "evidence/epochs-panel.json": '{"cycles": 90}\n',
        "tools/compare.py": "print('compare')\n",
        "ITERATION_NOTES.md": "seed author notes\n",
        "TASK.md": "harness task text\n",
        "qa/perf_history.jsonl": json.dumps({"kind": "modeled", "test_case": "conv", "cycles": 120,
                                             "evidence": "modeled: Spike timing model (self-check)"}) + "\n",
        "qa/cases/conv/d1.json": json.dumps({"capsule": "conv", "performance": {
            "status": "PROVISIONAL", "complete_event_accounting": True, "estimated_whole_soc_cycles": 100,
            "elf_sha256": "e" * 64, "capsule_sha256": "c" * 64, "selection_sha256": "s" * 64,
            "model_sha256": "m" * 64, "facts_sha256": "f" * 64, "timing_epoch": 0,
            "structural_roofline": {"structural_compute_floor_cycles": 10}}, "compiler_diagnostics": {"x": 1}}),
        "qa/latest_completed_grade.json": json.dumps({"graded_source_sha256": "g" * 64, "per_capsule": [
            {"capsule": "conv", "status": "pass", "graded": {"token": "1"}, "detail": {"path": "qa/cases/conv/d1.json"},
             "performance": {"array_utilization_pct": 7.5, "bound_by": "compute"}}]}),
    }.items():
        (workspace / name).parent.mkdir(parents=True, exist_ok=True)
        (workspace / name).write_text(text)
    (root / "seed-run/run").mkdir(parents=True)
    from merlin.common.tree_hash import hash_tree

    return {
        "from": {"path": str(seed), "sha256": hash_tree(seed)["sha256"]},
        "strip": ["compiler_library"],
        "source": "${ARTIFACTS}/seed-compiler",
        "review_id": "${RUN_ID}",
        "target": "fixture-target",
        "run": {"run_id": "seedrun", "run_dir": str(root / "seed-run/run"), "workspace": str(workspace),
                "framework": str(root / "seed-framework")},
    }


def _seeded_spec(fixture, seed):
    import yaml

    spec = dict(fixture.spec, seed=seed)
    spec["variables"] = dict(spec["variables"], OPERATOR_TMP=str(fixture.root / "operator"))
    argv = [*spec["plan"]["argv"], "--sandbox", "bwrap", "--compiler-execution", "snapshot_external_v1"]
    spec["plan"] = dict(spec["plan"], argv=argv)
    path = fixture.root / "seeded.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return path


@pytest.fixture
def seeded(F, fixture_spec, library, monkeypatch):
    """The seeded plan with the export-runtime steps run in-process instead of the selected interpreter."""
    module = F.load_module("fresh_seed", HERE)
    calls = []

    def carry(ctx, seed, environment, report):
        calls.append((seed["run"]["run_id"], environment.get("MERLIN_TIER_CERT_PORTABLE")))
        return {"decision": "re-keyed: fixture", "carried": 2, "by_tier": {"L2": 1, "L3": 1}}

    monkeypatch.setattr(module, "carry_certificates", carry)
    loader = F.load_module
    monkeypatch.setattr(F, "load_module", lambda name, d: module if name == "fresh_seed" else loader(name, d))
    path = _seeded_spec(fixture_spec, _seed_fixture(fixture_spec.root))
    ctx = F.Context(path, export_id=fixture_spec.revision[:12])
    return SimpleNamespace(ctx=ctx, record=F.build_plan(ctx), module=module, calls=calls,
                           out=fixture_spec.root / "launch-out")


def test_seeded_plan_rewrites_the_fresh_launch_without_hand_steps(seeded):
    record, out = seeded.record, seeded.out
    argv = record["argv"]
    assert "--fresh-backend" not in argv
    assert argv[argv.index("--seed-submission") + 1] == str(out / "seed-compiler")
    assert record["environment"]["MERLIN_TIER_CERT_PORTABLE"] == "1"
    assert "MERLIN_TIER_CERT_PORTABLE=1" in record["reserved_argv"]
    # Seeded grades carry the screen, grade changes first and label carried passes.
    assert record["environment"]["MERLIN_SEEDED_GRADE"] == "seedrun"
    assert "MERLIN_SEEDED_GRADE=seedrun" in record["reserved_argv"]
    assert seeded.calls == [("seedrun", "1")]
    policy = json.loads((out / "policy.json").read_text())
    origin = policy["compiler_origin"]
    assert origin == record["origin"] and origin["inherit_certificates_for_identical_executables"] is True
    assert origin["previous_session"]["run_dir"].endswith("seed-run/run")
    review = json.loads((out / "seed-review.json").read_text())
    assert all(review[key] == origin[key] for key in origin if key not in ("review", "schema"))
    assert record["seed"]["carry"] == dict.fromkeys(
        ("certificates", "results", "rules", "handoff", "variants", "performance"), True)
    # The seeded order is selected explicitly: the base specs' information-first would keep it off.
    assert record["environment"]["MERLIN_CAPSULE_ORDER"] == "seeded"
    assert "MERLIN_CAPSULE_ORDER=seeded" not in record["seed"]["preflight_twin"]["reserved_argv"]
    assert record["seed"]["certificates"]["carried"] == 2
    # The unseeded twin keeps the fresh launch for the runtime's preflight-only mode.
    twin = record["seed"]["preflight_twin"]
    assert "--fresh-backend" in twin["argv"] and "--seed-submission" not in twin["argv"]
    assert "compiler_origin" not in json.loads(Path(twin["policy"]["path"]).read_text())
    assert "MERLIN_TIER_CERT_PORTABLE=1" not in twin["reserved_argv"]
    assert not any(str(item).startswith("MERLIN_SEEDED_GRADE") for item in twin["reserved_argv"])
    guidance = (out / "author-guidance.md").read_text()
    assert guidance.startswith("guidance\n") and "starting point to evolve, not a design to preserve" in guidance
    assert "Make big changes" in guidance and "SEED_INDEX.md" in guidance and "SEED_VARIANTS/" in guidance
    assert "vs_seed" in guidance and "Spike cycle-model estimates" in guidance
    assert json.loads((out / "proposed-launch.json").read_text()) == record


def test_seed_copy_strips_stale_notes_and_keeps_what_the_compiler_reads(seeded):
    copy, evidence = seeded.out / "seed-compiler", seeded.out / "seed-evidence"
    assert not (copy / "compiler_library").exists() and not (copy / "SEED_EVIDENCE").exists()
    assert not (copy / "docs/iteration_notes.md").exists()
    assert (evidence / "SEED_NOTES/docs/iteration_notes.md").read_text() == "round 3 of the old harness\n"
    assert (evidence / "SEED_NOTES/REPORT.md").read_text() == "old harness narrative\n"
    assert "SEED_NOTES/REPORT.md" in (copy / "REPORT.md").read_text()  # the run finalizes a report here
    assert (copy / "docs/tuning_table.md").is_file()  # named by backend.py
    notes = seeded.record["seed"]["evidence"]["notes"]
    assert notes == {"moved": ["REPORT.md", "docs/iteration_notes.md"], "kept_read_by_compiler": []}
    assert (evidence / "SEED_EVIDENCE/baseline-map.json").is_file()
    assert json.loads((evidence / "OPTIMIZATION_RULES.json").read_text())["rules"][0]["name"] == "resident_rhs"
    index = json.loads((evidence / "SEED_EXPERIMENTS/index.json").read_text())
    assert index["ab"]["rejected_epochs"] == {"records": ["README.md"], "files": 2}
    assert index["variants"]["epochs"]["changed_python_files"] == ["backend.py"]
    pin = seeded.record["origin"]["seed_evidence"]
    from merlin.common.tree_hash import hash_tree

    assert pin == {"path": str(evidence), "sha256": hash_tree(evidence)["sha256"]}


def test_seed_evidence_carries_variants_performance_and_an_index(seeded):
    evidence = seeded.out / "seed-evidence"
    variant = evidence / "SEED_VARIANTS/epochs"
    assert (variant / "backend.py").read_text() == "TUNING = 'changed'\n" and not (variant / "build").exists()
    outcomes = json.loads((evidence / "SEED_VARIANTS/OUTCOMES.json").read_text())["variants"]["epochs"]
    assert outcomes["changed"] == ["backend.py"] and "evidence/epochs-panel.json" in outcomes["seed_records"]
    assert (evidence / "SEED_EXPERIMENTS/evidence/epochs-panel.json").is_file()
    assert (evidence / "SEED_TOOLS/compare.py").is_file()
    assert (evidence / "SEED_NOTES/workspace/ITERATION_NOTES.md").is_file()
    assert not (evidence / "SEED_NOTES/workspace/TASK.md").exists()
    assert (evidence / "SEED_PERFORMANCE/perf_history.jsonl").is_file()
    baseline = json.loads((evidence / "SEED_PERFORMANCE/baseline.json").read_text())["cases"]["conv"]
    assert baseline["cycles"] == 100 and baseline["best_cycles"] == 120 and baseline["roofline_bound_cycles"] == 10
    assert baseline["array_utilization_pct"] == 7.5 and baseline["model_sha256"] == "m" * 64
    timing = json.loads((evidence / f"RUN_PRIVATE/seed_timing/{'e' * 64}.json").read_text())
    assert timing["records"][0]["capsule"] == "conv" and timing["records"][0]["compiler_diagnostics"] == {"x": 1}
    assert (evidence / "RUN_PRIVATE/seed_baseline.json").read_text() == (
        evidence / "SEED_PERFORMANCE/baseline.json").read_text()
    index = (evidence / "SEED_INDEX.md").read_text()
    for name in ("SEED_VARIANTS/", "SEED_PERFORMANCE/baseline.json", "PREVIOUS_SESSION.md", "SEED_TOOLS/"):
        assert f"`{name}`" in index
    counts = seeded.record["seed"]["evidence"]["performance"]
    assert counts["timed_executables"] == 1 and counts["baseline_cases_with_cycles"] == 1


def test_stale_notes_named_by_compiler_sources_stay(seeded, tmp_path):
    copy = tmp_path / "copy"
    (copy / "docs").mkdir(parents=True)
    (copy / "docs/iteration_plan.md").write_text("plan\n")
    (copy / "loader.py").write_text("open(HERE / 'docs' / 'iteration_plan.md')\n")
    result = seeded.module.strip_stale(copy, tmp_path / "notes", ["docs/*iteration*"])
    assert result == {"moved": [], "kept_read_by_compiler": ["docs/iteration_plan.md"]}


def test_seeded_preflight_checks_the_seed_and_preflights_the_unseeded_twin(F, seeded, monkeypatch, capsys):
    checks = []
    monkeypatch.setattr(seeded.module, "verify", lambda ctx, plan: checks.append(plan["run_id"]) or {"seed": "ok"})
    monkeypatch.setattr(F, "host_health_line", lambda: "HOST HEALTH ok")
    assert F.cmd_preflight(seeded.ctx, SimpleNamespace(run=False, execute=False)) == 0
    script = Path(json.loads(capsys.readouterr().out.splitlines()[-1])["script"]).read_text()
    assert checks == ["fixture-run"]
    assert "--fresh-backend" in script and "--seed-submission" not in script and "--preflight-only" in script
    assert F.cmd_preflight(seeded.ctx, SimpleNamespace(run=True, execute=False)) == 0
    run_script = Path(json.loads(capsys.readouterr().out.splitlines()[-1])["script"]).read_text()
    assert "--seed-submission" in run_script and "--preflight-only" not in run_script


def test_seeded_launch_dry_run_rechecks_the_seed_and_starts_nothing(F, seeded, monkeypatch, capsys):
    checks = []
    monkeypatch.setattr(seeded.module, "verify", lambda ctx, plan: checks.append(1) or {"seed": "ok"})
    assert F.cmd_launch(seeded.ctx, SimpleNamespace(confirm=False)) == 2
    assert checks == [1] and "refusing to launch" in capsys.readouterr().err
    assert json.loads((seeded.out / "proposed-launch.json").read_text())["execution_started"] is False


def test_seed_runtime_check_accepts_the_planned_seed(seeded, monkeypatch):
    """The check preflight runs in the campaign runtime: options, review, and a scratch placement."""
    monkeypatch.setenv("MERLIN_TIER_CERT_PORTABLE", "1")
    monkeypatch.delenv("MERLIN_JOINT_CAMPAIGN", raising=False)
    # The session's own launch checks need the complete fixture-free runtime (codex, full compiler).
    monkeypatch.setattr("merlin_experiments.phase1.session.validate_options", lambda options: None)
    compiled = []

    def compile_staged(record, options, submission, scratch):
        # The staged seed is complete when the compile check sees it (seed_compile_probe has its own tests).
        compiled.append(sorted(p.name for p in submission.iterdir()))
        return {"status": "compiled", "case": "fixture"}

    monkeypatch.setattr(seeded.module, "compile_staged", compile_staged)
    report = seeded.module.verify_in_runtime(seeded.record)
    assert report["compile"] == {"status": "compiled", "case": "fixture"}
    assert compiled and "backend.py" in compiled[0] and "manifest.yaml" in compiled[0]
    placement = report["placement"]
    assert report["seed"] == "accepted" and placement["certificate_carry_over"] is True
    assert {"PREVIOUS_SESSION.md", "OPTIMIZATION_RULES.json", "SEED_NOTES", "SEED_EVIDENCE",
            "SEED_EXPERIMENTS", "SEED_INDEX.md", "SEED_VARIANTS", "SEED_PERFORMANCE",
            "SEED_TOOLS"} <= set(placement["workspace_root"])
    assert "RUN_PRIVATE" not in placement["workspace_root"]
    assert sorted(placement["seed_evidence"]["run_private"]["staged"]) == ["seed_baseline.json", "seed_timing"]


def test_seed_spec_refuses_carry_without_its_run(F, fixture_spec, library):
    seed = _seed_fixture(fixture_spec.root)
    seed.pop("run")
    seed["carry"] = {"rules": True}
    ctx = F.Context(_seeded_spec(fixture_spec, seed), export_id=fixture_spec.revision[:12])
    with pytest.raises(ValueError, match="requires seed.run"):
        F.build_plan(ctx)


def test_a_read_only_seed_copy_can_move_its_notes_out(F, tmp_path):
    seed = F.load_module("fresh_seed", HERE)
    source = tmp_path / "frozen"
    (source / "docs").mkdir(parents=True)
    (source / "REPORT.md").write_text("old\n")
    (source / "docs/iteration_1.md").write_text("old\n")
    (source / "backend.py").write_text("X = 1\n")
    for path in [*source.rglob("*"), source]:
        path.chmod(path.stat().st_mode & ~0o222)
    try:
        dest = tmp_path / "copy"
        seed.copy_source(source, dest, [])
        moved = seed.strip_stale(dest, tmp_path / "notes", seed.NOTES)["moved"]
        assert sorted(moved) == ["REPORT.md", "docs/iteration_1.md"]
    finally:
        for path in [source, *source.rglob("*")]:
            path.chmod(path.stat().st_mode | 0o200)
