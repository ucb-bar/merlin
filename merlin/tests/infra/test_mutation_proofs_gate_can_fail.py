from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

from merlin.common.paths import repo_root

SCRIPTS = repo_root() / "build_tools" / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MP = _load("check_mutation_proofs")
GATES = ["build_tools/scripts/check_alpha.py", "build_tools/scripts/check_beta.py"]
PROOF_PATH = "merlin/tests/infra/test_alpha_gate_can_fail.py"
PROOF_SOURCE = (
    "def test_alpha_reports_a_bad_row():\n    module = load('check_alpha')\n    assert module.audit([{'bad': True}])\n"
)
MUTATION = "Feed audit() one row whose declared status is absent from the verdict table."
REMOVES = "Both satisfiers: the row is not ratcheted and the status maps to no clean bucket."


def _registry(**overrides) -> dict[str, dict]:
    entry = {"proof": f"{PROOF_PATH}::test_alpha_reports_a_bad_row", "mutation": MUTATION, "removes": REMOVES}
    entry.update(overrides)
    return {"build_tools/scripts/check_alpha.py": entry}


def _read(source: str = PROOF_SOURCE):
    return lambda path: source if path == PROOF_PATH else None


def _subjects(failures) -> list[str]:
    return [f.subject for f in failures]


def test_a_gate_with_no_recorded_mutation_is_reported():
    """THE PRIMARY MUTATION: a gate ships, and nobody ever said what makes it fail."""
    assert MP.audit(GATES, _registry(), ["build_tools/scripts/check_beta.py"], _read()) == []
    assert _subjects(MP.audit(GATES, _registry(), [], _read())) == ["build_tools/scripts/check_beta.py"]


def test_a_skipped_boundary_case_changes_the_real_runner_verdict(tmp_path, pinned_backend, monkeypatch):
    """Process transport is synthetic; Git custody and JUnit admission are real."""
    monkeypatch.syspath_prepend(str(SCRIPTS))
    gate = _load("check_backend_boundaries")
    source, _, git = pinned_backend
    main_wheel = tmp_path / "merlin-0-py3-none-any.whl"
    with ZipFile(main_wheel, "w") as archive:
        archive.writestr("merlin-0.dist-info/METADATA", "Name: merlin\nVersion: 0\n")
    manifest = json.loads((repo_root() / "build_tools/upstreams/backend-boundaries.json").read_text())
    consumers = manifest.get("source_consumers", [])
    for row in consumers:
        for relative in row["files"]:
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# bounded static source transport fixture\n")
    if consumers:
        git("add", "--", ".")
        git("commit", "-q", "--no-verify", "-m", "Add selected static consumer fixture paths")
    rows = manifest["backends"]
    for row in rows:
        row.update(
            revision=git("rev-parse", "HEAD"),
            xdsl="0.68.0",
            provider_module="fixture",
            tests=["python/fixture/consumer.py"],
            fixtures=[],
            selection="",
            installed_service=False,
        )
    for row in consumers:
        row["revision"] = git("rev-parse", "HEAD")
    pins = tmp_path / "boundary-pins.json"
    pins.write_text(json.dumps(manifest))
    mutate = False

    def process_fixture(argv, *, cwd, log, env, records):
        log.write_text("bounded process transport fixture\n")
        records.append({"argv": argv, "returncode": 0, "log": str(log)})
        if "build" in argv:
            destination = Path(argv[argv.index("--outdir") + 1])
            with ZipFile(destination / "fixture-0-py3-none-any.whl", "w") as archive:
                archive.writestr("fixture/__init__.py", "")
        elif "--junitxml" in argv:
            junit = Path(argv[argv.index("--junitxml") + 1])
            skipped = int(mutate and junit.name == rows[0]["id"] + ".xml")
            junit.write_text(
                f'<testsuites><testsuite tests="1" failures="0" errors="0" skipped="{skipped}"/></testsuites>'
            )

    monkeypatch.setattr(gate, "run", process_fixture)
    sources = [value for row in [*rows, *consumers] for value in ("--source", f"{row['id']}={source}")]
    common = ["--wheel", str(main_wheel), "--pins", str(pins), "--python068", sys.executable, *sources]
    clean = tmp_path / "boundary-clean"
    assert gate.main([*common, "--output", str(clean)]) == 0
    report = json.loads((clean / "report.json").read_text())
    assert report["passed"] is True and len(report["lanes"]) == len(rows)
    assert all(row["counts"] == dict(tests=1, failures=0, errors=0, skipped=0) for row in report["lanes"])
    assert len(report["static_inputs"]) == len(consumers)
    for expected, observed in zip(consumers, report["static_inputs"], strict=True):
        assert {row["path"] for row in observed["files"]} == set(expected["files"])
    mutate = True
    changed = tmp_path / "boundary-skipped"
    assert gate.main([*common, "--output", str(changed)]) == 1
    report = json.loads((changed / "report.json").read_text())
    assert report["passed"] is False and "must execute without skips" in report["error"]
    assert not report["lanes"]


def test_missing_reference_region_evidence_changes_the_real_readiness_verdict(tmp_path, monkeypatch):
    """Only the upstream analyzer transport is substituted; assess/main are real."""
    from merlin_experiments.phase1.feedback import reference_analysis

    gate = _load("check_reference_readiness")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "visible.dump").write_text("bounded analyzer transport fixture\n")
    facts, contract = (tmp_path / "facts.json", tmp_path / "contract.json")
    facts.write_text("{}")
    contract.write_text("{}")
    report = dict(
        input=dict(path="visible.dump", sha256=hashlib.sha256((workspace / "visible.dump").read_bytes()).hexdigest()),
        total=2,
        coverage=dict(accelerator_sites=7, selected_region_sites=7),
        regions=[
            dict(
                stable_id=f"region-{index}",
                accelerator_sites=sites,
                mechanism_projection=dict(
                    semantic_region_status="established",
                    source_binary_correspondence="established",
                    compatible_comparison="established",
                    unclassified_instruction_classes=[],
                ),
                cca=dict(
                    provenance=dict(
                        axis_evidence={
                            "dispatch.loop_offloaded": dict(status="inferred", evidence=["selected-isa-role-roster"])
                        }
                    )
                ),
            )
            for index, sites in enumerate((3, 4))
        ],
    )

    def analyzer_transport(request, **kwargs):
        assert kwargs["workspace"] == workspace and kwargs["reference_root"] is None
        if request.get("operation") == "inventory":
            return (copy.deepcopy(report), {})
        selected = next(row for row in report["regions"] if row["stable_id"] == request["region"])
        return (dict(report, regions=[copy.deepcopy(selected)]), {})

    monkeypatch.setattr(reference_analysis, "_analyze", analyzer_transport)
    common = [
        "--workspace",
        str(workspace),
        "--facts",
        str(facts),
        "--contract",
        str(contract),
        "--target",
        "fixture",
        "--reference",
        "visible.dump",
    ]
    clean = tmp_path / "reference-clean"
    assert gate.main([*common, "--output", str(clean)]) == 0
    ready = json.loads((clean / "readiness.json").read_text())
    assert ready["launch_ready"] is True and ready["ready_regions"] == 2
    report["regions"][1]["mechanism_projection"] = {}
    changed = tmp_path / "reference-missing"
    assert gate.main([*common, "--output", str(changed)]) == 2
    blocked = json.loads((changed / "readiness.json").read_text())
    assert blocked["launch_ready"] is False and blocked["region_count"] == 2
    assert blocked["ready_regions"] == 1
    assert blocked["missing"]["complete_upstream_semantic_region"] == 1
    assert blocked["missing"]["source_binary_correspondence"] == 1
    assert (changed / "0-region-1.json").is_file()


def _tree(root: Path, files: dict[str, str]) -> Path:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


@pytest.fixture
def pinned_backend(tmp_path):
    source = tmp_path / "provider"
    _tree(
        source,
        {
            "pyproject.toml": '[tool.setuptools]\npackages = ["fixture"]\npackage-dir = {fixture = "python/fixture"}\n',
            ".gitignore": "generated.py\n",
            "python/fixture/__init__.py": "",
            "python/fixture/consumer.py": "from merlin.perf import tracked_owner\n",
            "tools/inspect.py": "from merlin.perf import tracked_tool_owner\n",
        },
    )

    def git(*arguments):
        return subprocess.check_output(
            [
                "git",
                "-C",
                str(source),
                "-c",
                "core.autocrlf=false",
                "-c",
                "user.name=Wiring fixture",
                "-c",
                "user.email=wiring-fixture@example.invalid",
                "-c",
                "commit.gpgsign=false",
                *arguments,
            ],
            text=True,
        ).strip()

    git("init", "-q")
    git("add", "--", ".")
    git("commit", "-q", "--no-verify", "-m", "Create bounded wiring fixture")
    revision = git("rev-parse", "HEAD")
    pins = tmp_path / "build_tools/upstreams/backend-boundaries.json"
    pins.parent.mkdir(parents=True)
    pins.write_text(json.dumps({"backends": [{"id": "fixture", "revision": revision}]}))
    return (source, pins, git)
