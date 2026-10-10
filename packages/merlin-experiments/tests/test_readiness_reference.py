"""Operator-selected Chipyard readiness package checks; no simulator is started here."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from merlin.common.paths import repo_root

_MODULE = repo_root() / "merlin/experiments/capsule_bench/harness/readiness_reference.py"
_SPEC = importlib.util.spec_from_file_location("readiness_reference", _MODULE)
assert _SPEC is not None and _SPEC.loader is not None
reference = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(reference)


def test_reference_backend_selection_is_explicit_and_target_bound(tmp_path):
    with pytest.raises(ValueError, match="requires --reference-backend"):
        reference.select_reference_backend(None, target="gemmini")
    with pytest.raises(ValueError, match="absolute package path"):
        reference.select_reference_backend("relative/package", target="gemmini")

    package = tmp_path / "published"
    package.mkdir()
    manifest = package / "manifest.yaml"
    manifest.write_text(
        "target: gemmini\nartifact_type: mlir_oot_target_backend\n"
        "integrity_exempt: false\nlanguage: python\n"
        "entrypoints: {tool: gemmini-opt}\npackage_id: published_v1\n",
        encoding="utf-8",
    )
    selected, metadata = reference.select_reference_backend(str(package), target="gemmini")
    assert selected == package
    assert metadata["package_id"] == "published_v1"

    with pytest.raises(ValueError, match="expected 'other'"):
        reference.select_reference_backend(str(package), target="other")
    manifest.write_text(manifest.read_text().replace("integrity_exempt: false", "integrity_exempt: true"))
    with pytest.raises(ValueError, match="integrity_exempt: false"):
        reference.select_reference_backend(str(package), target="gemmini")


def _package(root, name):
    package = root / name
    package.mkdir()
    (package / "manifest.yaml").write_text(
        "target: fixture_target\nartifact_type: mlir_oot_target_backend\n"
        "integrity_exempt: false\nlanguage: python\nentrypoints: {tool: primitive-tool}\n"
    )
    return package


@pytest.fixture
def probe_inputs(tmp_path):
    root = tmp_path / "probes"
    root.mkdir()
    for name in ("screen", "timing", "policy"):
        member = root / name
        member.mkdir()
        (member / "capsule.yaml").write_text(
            f"name: {name}\nperformance:\n  arms:\n    candidate:\n"
            "      instruction_policy:\n        prohibited_instruction_roles: [fixture_role]\n"
        )
    positive = _package(tmp_path, "positive")
    kwargs = {
        "target": "fixture_target",
        "root": str(root),
        "screen": "screen",
        "timing": "timing",
        "incorrect_backend": str(_package(tmp_path, "incorrect")),
        "prohibited_backend": str(_package(tmp_path, "prohibited")),
        "prohibited_probe": "policy",
        "timeout_s": 45,
        "timing_output": str(tmp_path / "observed-timing.json"),
        "resource_root": tmp_path / "sealed-experiment",
    }
    return positive, kwargs


def test_explicit_probes_preserve_legacy_absence_and_require_complete_selection(probe_inputs):
    _, kwargs = probe_inputs
    absent = {key: None for key in kwargs if key != "target"}
    assert reference.select_probe_inputs(target="fixture_target", **absent) is None
    with pytest.raises(ValueError, match="all root"):
        reference.select_probe_inputs(target="fixture_target", required=True, **absent)
    selected = reference.select_probe_inputs(**kwargs)
    assert selected["root"] == Path(kwargs["root"])
    assert selected["timing"] == "timing"
    reference.verify_probe_inputs(selected)


@pytest.mark.parametrize(
    "missing", ["root", "screen", "timing", "incorrect_backend", "prohibited_backend", "prohibited_probe"]
)
def test_partial_probe_selection_never_falls_back(probe_inputs, missing):
    _, kwargs = probe_inputs
    kwargs[missing] = None
    with pytest.raises(ValueError, match="all root"):
        reference.select_probe_inputs(**kwargs)


@pytest.mark.parametrize("timeout", [None, False, True, 0, -1, 601, 1.5, "45"])
def test_explicit_timeout_is_positive_bounded_and_typed(probe_inputs, timeout):
    _, kwargs = probe_inputs
    kwargs["timeout_s"] = timeout
    with pytest.raises(ValueError, match="probe-timeout-s"):
        reference.select_probe_inputs(**kwargs)


@pytest.mark.parametrize("timeout", [1, 600])
def test_explicit_timeout_endpoints_preserved(probe_inputs, timeout):
    _, kwargs = probe_inputs
    kwargs["timeout_s"] = timeout
    assert reference.select_probe_inputs(**kwargs)["timeout_s"] == timeout


@pytest.mark.parametrize(
    "kind", ["missing", "relative", "existing", "symlink", "parent-symlink", "resources", "missing-parent"]
)
def test_explicit_timing_output_cannot_mutate_selected_resources(probe_inputs, tmp_path, kind):
    _, kwargs = probe_inputs
    path = Path(kwargs["timing_output"])
    if kind == "missing":
        kwargs["timing_output"] = None
    elif kind == "relative":
        kwargs["timing_output"] = "relative.json"
    elif kind == "existing":
        path.write_text("prior retained observation")
    elif kind == "symlink":
        path.symlink_to(tmp_path / "absent")
    elif kind == "parent-symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
        kwargs["timing_output"] = str(alias / "new.json")
    elif kind == "resources":
        kwargs["resource_root"].mkdir()
        kwargs["timing_output"] = str(kwargs["resource_root"] / "new.json")
    elif kind == "missing-parent":
        kwargs["timing_output"] = str(tmp_path / "absent-parent/new.json")
    with pytest.raises(ValueError, match="timing"):
        reference.select_probe_inputs(**kwargs)


@pytest.mark.parametrize("name", ["../screen", "screen,timing", "all", ".", "..", "screen other"])
def test_probe_name_cannot_expand_or_replace_the_member(probe_inputs, name):
    _, kwargs = probe_inputs
    kwargs["screen"] = name
    with pytest.raises(ValueError, match="one direct"):
        reference.select_probe_inputs(**kwargs)


@pytest.mark.parametrize("mutation", ["name", "policy", "missing", "bad-policy", "extra-role", "duplicate-role"])
def test_probe_identity_and_same_instruction_policy_are_required(probe_inputs, mutation):
    _, kwargs = probe_inputs
    member = Path(kwargs["root"]) / "timing/capsule.yaml"
    text = member.read_text()
    if mutation == "missing":
        member.unlink()
    else:
        changes = {
            "name": ("name: timing", "name: substituted"),
            "policy": ("fixture_role", "different_role"),
            "bad-policy": (
                "instruction_policy:\n        prohibited_instruction_roles: [fixture_role]",
                "instruction_policy: []",
            ),
            "extra-role": ("[fixture_role]", "[fixture_role, another_role]"),
            "duplicate-role": ("[fixture_role]", "[fixture_role, fixture_role]"),
        }
        member.write_text(text.replace(*changes[mutation]))
    with pytest.raises(ValueError):
        reference.select_probe_inputs(**kwargs)


@pytest.mark.parametrize("member", ["capsule", "negative-manifest"])
def test_selected_probe_source_mutation_refuses(probe_inputs, member):
    _, kwargs = probe_inputs
    selected = reference.select_probe_inputs(**kwargs)
    path = (
        Path(kwargs["root"]) / "timing/capsule.yaml"
        if member == "capsule"
        else Path(kwargs["incorrect_backend"]) / "manifest.yaml"
    )
    path.write_text(path.read_text() + "# changed\n")
    with pytest.raises(ValueError, match="changed"):
        reference.verify_probe_inputs(selected)


def _report(capsule, sim, *, negative=None):
    row = {"capsule": capsule, "pass": negative is None, "barrier_tier": "L3", "barrier_status": "pass"}
    report = {
        "n_capsules": 1,
        "n_passed": int(negative is None),
        "all_pass": negative is None,
        "sim": sim,
        "per_capsule": [row],
    }
    if sim == "spike":
        report["all_pass"] = False
        row.update(barrier_tier="L2", tiers={"L2": "pass"})
    if negative == "numeric":
        row.update(
            barrier_status="fail",
            numeric={"status": "fail", "mismatch_count": 1, "missing_outputs": []},
            failure={"category": "FUNCTIONAL_MISMATCH", "plane": sim},
        )
    elif negative == "policy":
        row["failure"] = {
            "category": "PROHIBITED_INSTRUCTION",
            "plane": "instruction_policy",
            "prohibited_instructions": {"fixture_opcode": {"mask": 1}},
        }
    return report


@pytest.fixture
def oracle_driver(tmp_path, monkeypatch, probe_inputs):
    """Execute only the selected real function bodies, with every process call substituted."""
    positive, kwargs = probe_inputs
    path = _MODULE.with_name("readiness_check.py")
    parsed = ast.parse(path.read_text(), filename=str(path))
    names = ("_ok", "_na", "section", "test_oracles_endtoend", "main")
    selected = ast.Module(
        body=[node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[]
    )
    namespace = {
        "Path": Path,
        "sys": sys,
        "yaml": reference.yaml,
        "REPO": tmp_path,
        "EXP": kwargs["resource_root"],
        "TARGET": "fixture_target",
        "PY": "selected-python",
        "SCRIPTS": tmp_path,
        "results": [],
        "C": types.SimpleNamespace(DESCRIPTOR=tmp_path / "target.yaml", REPO=tmp_path),
    }
    calls, writes = [], []
    replies = {
        "screen": _report("screen", "spike"),
        "timing": _report("timing", "selected_rtl"),
        "incorrect": _report("timing", "selected_rtl", negative="numeric"),
        "prohibited": _report("policy", "selected_rtl", negative="policy"),
        "empty": {"n_capsules": 0, "error": "missing package"},
    }

    def run(argv, **options):
        calls.append((argv, options))
        submission = Path(argv[argv.index("--submission") + 1])
        sim = argv[argv.index("--sim") + 1]
        key = (
            submission.name
            if submission.name in ("incorrect", "prohibited")
            else "screen"
            if sim == "spike" and submission == positive
            else "timing"
            if submission == positive
            else "empty"
        )
        reply = copy.deepcopy(replies[key])
        returncode = reply.pop("_readiness_returncode", 0 if reply.get("all_pass") is True else 1)
        return types.SimpleNamespace(stdout="grade diagnostics\n" + json.dumps(reply), stderr="", returncode=returncode)

    namespace.update(
        subprocess=types.SimpleNamespace(run=run),
        ext_path=lambda name: None,
        _oracle_sim_via=lambda: "chipyard",
        _kill_our_simulators=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "readiness_reference", reference)
    import merlin.targetgen as targetgen

    runner = types.SimpleNamespace(describe_l3_engine=lambda target: {"available": False})
    monkeypatch.setitem(sys.modules, "merlin.targetgen.capsule_runner", runner)
    monkeypatch.setattr(targetgen, "capsule_runner", runner, raising=False)
    monkeypatch.setitem(
        sys.modules,
        "merlin_experiments.frozen_python",
        types.SimpleNamespace(inherited_python_command=lambda argv: argv),
    )
    monkeypatch.setitem(
        sys.modules,
        "merlin_experiments.phase1.timing",
        types.SimpleNamespace(
            selected_engine_binding=lambda **kw: {"engine": "selected_rtl"},
            timing_path=lambda *args: tmp_path / "timing.json",
            write_observed_timing=lambda *args, **kw: writes.append((args, kw)),
        ),
    )
    exec(compile(selected, str(path), "exec"), namespace)
    return namespace, calls, writes, replies, positive, kwargs


def _observe(driver):
    namespace, _, _, _, positive, kwargs = driver
    namespace["test_oracles_endtoend"](str(positive), probes=reference.select_probe_inputs(**kwargs))


def test_selected_probes_reach_the_ordinary_selfcheck_and_timing_writer(oracle_driver):
    _observe(oracle_driver)
    namespace, calls, writes, _, _, kwargs = oracle_driver
    assert all(argv[1:3] == ["-m", "merlin_experiments.phase1.feedback.selfcheck"] for argv, _ in calls)
    assert [argv[argv.index("--capsules") + 1] for argv, _ in calls] == [
        "screen",
        "timing",
        "timing",
        "policy",
        "screen",
    ]
    assert all(argv[argv.index("--capsules-root") + 1] == kwargs["root"] for argv, _ in calls)
    assert all(argv[argv.index("--timeout") + 1] == "45" and options["timeout"] == 165 for argv, options in calls)
    assert len(writes) == 1
    assert writes[0][1]["measured_capsule"] == "timing"
    assert writes[0][0] == (Path(kwargs["timing_output"]),)
    assert writes[0][1]["report"]["_readiness_returncode"] == 0
    assert not any(ok is False for _, ok, _ in namespace["results"])


@pytest.mark.parametrize(
    "key,mutation",
    [
        ("timing", "capsule"),
        ("timing", "sim"),
        ("timing", "count-bool"),
        ("timing", "extra-row"),
        ("timing", "passed-count"),
        ("timing", "positive-missing-output"),
        ("timing", "exit"),
        ("incorrect", "zero-results"),
        ("incorrect", "category"),
        ("incorrect", "numeric-pass"),
        ("incorrect", "mismatch-bool"),
        ("incorrect", "missing-output"),
        ("prohibited", "category"),
        ("prohibited", "plane"),
        ("prohibited", "no-hits"),
        ("screen", "pass"),
    ],
)
def test_substituted_or_incomplete_reports_cannot_publish_timing(oracle_driver, key, mutation):
    namespace, _, writes, replies, _, _ = oracle_driver
    report = replies[key]
    row = report["per_capsule"][0]
    if mutation == "capsule":
        row["capsule"] = "substituted"
    elif mutation == "sim":
        report["sim"] = "unselected_engine"
    elif mutation == "count-bool":
        report["n_capsules"] = True
    elif mutation == "extra-row":
        report["per_capsule"].append(copy.deepcopy(row))
    elif mutation == "passed-count":
        report["n_passed"] = 0
    elif mutation == "positive-missing-output":
        row["numeric"] = {"status": "pass", "missing_outputs": ["Y"]}
    elif mutation == "exit":
        report["_readiness_returncode"] = None
    elif mutation == "zero-results":
        report.update(n_capsules=0, per_capsule=[])
    elif mutation == "category":
        row["failure"]["category"] = "BUILD_FAILED"
    elif mutation == "numeric-pass":
        row["numeric"]["status"] = "pass"
    elif mutation == "mismatch-bool":
        row["numeric"]["mismatch_count"] = True
    elif mutation == "missing-output":
        row["numeric"]["missing_outputs"] = ["Y"]
    elif mutation == "plane":
        row["failure"]["plane"] = "build"
    elif mutation == "no-hits":
        row["failure"]["prohibited_instructions"] = {}
    elif mutation == "pass":
        row["pass"] = False
    try:
        _observe(oracle_driver)
    except ValueError:
        pass
    else:
        assert any(ok is False for _, ok, _ in namespace["results"])
    assert not writes


def _cli(kwargs, positive):
    mapping = {
        "root": "probe-capsules-root",
        "screen": "screen-probe",
        "timing": "timing-probe",
        "incorrect_backend": "incorrect-output-backend",
        "prohibited_backend": "prohibited-instruction-backend",
        "prohibited_probe": "prohibited-probe",
        "timeout_s": "probe-timeout-s",
        "timing_output": "oracle-timing-output",
    }
    return [
        "--oracle-probe-only",
        "--reference-backend",
        str(positive),
        *[value for key, flag in mapping.items() for value in (f"--{flag}", str(kwargs[key]))],
    ]


def test_probe_only_dispatch_does_not_run_historical_sections(oracle_driver, capsys):
    namespace, _, writes, _, positive, kwargs = oracle_driver
    assert namespace["main"](_cli(kwargs, positive)) == 0
    assert len(writes) == 1
    assert "finite selected timing/output/policy gate only; not full readiness" in capsys.readouterr().out


def test_probe_only_missing_selectors_refuse_before_any_process(oracle_driver):
    namespace, calls, writes, _, positive, _ = oracle_driver
    with pytest.raises(SystemExit) as error:
        namespace["main"](["--oracle-probe-only", "--reference-backend", str(positive)])
    assert error.value.code == 2
    assert not calls and not writes


def test_probe_only_wrong_route_refuses_without_program_oracle_fallback(oracle_driver):
    namespace, calls, writes, _, positive, kwargs = oracle_driver
    namespace["_oracle_sim_via"] = lambda: "other_selected_route"
    assert namespace["main"](_cli(kwargs, positive)) == 1
    assert not calls and not writes


def test_actual_child_boundary_rechecks_selected_probe_sources(oracle_driver):
    namespace, calls, writes, _, _, kwargs = oracle_driver
    run = namespace["subprocess"].run
    member = Path(kwargs["root"]) / "screen/capsule.yaml"

    def changed(argv, **options):
        result = run(argv, **options)
        member.write_text(member.read_text() + "# changed during child\n")
        return result

    namespace["subprocess"].run = changed
    with pytest.raises(ValueError, match="changed"):
        _observe(oracle_driver)
    assert len(calls) == 1 and not writes


def test_timing_output_created_during_grade_is_not_overwritten(oracle_driver):
    namespace, _, writes, _, _, kwargs = oracle_driver
    run = namespace["subprocess"].run
    output = Path(kwargs["timing_output"])

    def changed(argv, **options):
        result = run(argv, **options)
        output.write_text("retained unrelated artifact")
        return result

    namespace["subprocess"].run = changed
    with pytest.raises(ValueError, match="fresh absolute"):
        _observe(oracle_driver)
    assert not writes and output.read_text() == "retained unrelated artifact"


def test_legacy_oracle_members_and_deadlines_are_unchanged(oracle_driver):
    namespace, calls, writes, replies, positive, _ = oracle_driver
    root = namespace["REPO"] / "merlin/contract/capsules/isa"
    for name in ("A1_mvin_mvout", "A2_single_tile_matmul"):
        member = root / name
        member.mkdir(parents=True)
        (member / "capsule.yaml").write_text(f"name: {name}\n")
    replies["screen"]["per_capsule"][0]["capsule"] = "A1_mvin_mvout"
    replies["timing"]["per_capsule"][0]["capsule"] = "A2_single_tile_matmul"
    namespace["test_oracles_endtoend"](str(positive))
    assert [argv[argv.index("--timeout") + 1] for argv, _ in calls] == ["300", "900", "60"]
    assert all(argv[argv.index("--capsules-root") + 1] == str(root) for argv, _ in calls)
    assert writes[0][1]["measured_capsule"] == "A2_single_tile_matmul"


def test_full_readiness_default_keeps_every_existing_section(oracle_driver, capsys):
    namespace, _, _, _, _, _ = oracle_driver
    path = _MODULE.with_name("readiness_check.py")
    main = next(
        node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    checks = next(
        node.value
        for node in main.body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "checks" for t in node.targets)
    )
    names = [node.id for node in checks.orelse.elts]
    seen = []
    for name in names:

        def check(*args, _name=name, **kw):
            seen.append((_name, args, kw))
            namespace["_ok"](_name, True)

        namespace[name] = check
    assert namespace["main"]([]) == 0
    assert [name for name, _, _ in seen] == names
    assert len(names) == 16
    assert next(kw for name, _, kw in seen if name == "test_oracles_endtoend") == {"probes": None}
    assert "all tooling verified" in capsys.readouterr().out
