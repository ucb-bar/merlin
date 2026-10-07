"""Champion export from a phase-2 ``best`` tag, and the generated per-target INDEX.yaml."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from merlin.common import oot_repo as O
from merlin.common import storage_lifecycle
from merlin.common.paths import phase_runs_root
from merlin.common.tree_hash import hash_tree
from merlin.common.yaml import load_yaml, write_yaml
from merlin.targetgen import champions as C
from merlin.targetgen import target_index as TI

TARGET = "fixture"


@pytest.fixture()
def out_root(tmp_path, monkeypatch):
    root = tmp_path / "out"
    for sub in ("runs", "artifacts", "build"):
        (root / sub).mkdir(parents=True)
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(root))
    return root


def _package(root: Path, body: str) -> Path:
    """A candidate compiler the publish bridge accepts (the standalone layout's root files)."""
    root.mkdir(parents=True)
    manifest = {
        "artifact_type": "mlir_oot_target_backend",
        "target": TARGET,
        "package_id": "fixture_oot_v0",
        "language": "python",
        "authoring": {"mode": "hand_curated"},
        "integrity_exempt": False,
        "entrypoints": {"tool": "fixture-opt"},
        "commands": {
            name: {"argv": ["{tool}", "{input_mlir}"]}
            for name in ("parse", "lower_interface_to_target", "emit_command_buffer", "lower_target_to_llvm")
        },
    }
    write_yaml(root / "manifest.yaml", manifest)
    tool = root / "fixture-opt"
    tool.write_text("#!/usr/bin/env python3\nprint('fixture')\n")
    tool.chmod(0o755)
    (root / "transforms.py").write_text(body)
    (root / "lowering").mkdir()
    (root / "lowering" / "isa.py").write_text("ROLES = ()\n")
    (root / "xdsl_dialects").mkdir()
    (root / "xdsl_dialects" / "__init__.py").write_text("")
    return root


def _campaign(tmp_path: Path, out_root: Path):
    pkg = _package(tmp_path / "workspace" / "pkg", "x = 1\n")
    p1_run = phase_runs_root(TARGET, 1) / "20260929T100000Z_merlin_assisted_abc1234"
    p1 = O.init(p1_run / "oot", sandbox_roots=[tmp_path / "workspace"])
    O.commit_candidate(p1, pkg, label="round 1", when="20260929T100500Z", run_id=p1_run.name)
    frozen = O.commit_candidate(p1, pkg, label="round 2", when="20260929T110000Z", run_id=p1_run.name)
    O.tag(p1, O.FROZEN_TAG)
    p2_run = phase_runs_root(TARGET, 2) / "20260929T120000Z_whole_model_measured_abc1234"
    p2 = O.init_from(p2_run / "oot", p1)
    (pkg / "transforms.py").write_text("x = 2\n")
    cand = O.commit_candidate(p2, pkg, label="candidate 1", when="20260929T130000Z", run_id=p2_run.name)
    O.tag(p2, "measured/1", cand.commit)
    O.tag(p2, O.BEST_TAG, cand.commit)
    return pkg, p1_run, frozen, p2_run, p2, cand


def _evidence(p1_run: Path, frozen, digest: str) -> dict:
    return {
        "provenance": {
            "phase1": {"run": str(p1_run), "frozen_commit": frozen.commit},
            "corpus_seal_digest": "a" * 64,
            "phase0_evidence_digest": "b" * 64,
        },
        "measurements": {
            "package_digest": digest,
            "firesim": {
                "cycles": 42_788_433,
                "machine": "fixture-board",
                "header": "5bfbb726",
                "control": {"in_batch": True, "cycles": 43_000_000},
            },
            "exactness": {"contract_sha256": "e" * 64, "label": "exact"},
        },
        "certification": {"gsim": {"verdict": "pass", "certificate_sha256": "c" * 64}},
        "isa_prohibition": {
            "scope": "whole_elf",
            "verdict": "clean",
            "prohibited_roles": ["loop_fsm"],
            "prohibited_instructions": {"8": "LOOP_A"},
        },
    }


@pytest.fixture()
def campaign(tmp_path, out_root, monkeypatch, pytestconfig):
    from merlin.targetgen import publish

    monkeypatch.setattr(publish.paths, "repo_root", lambda: pytestconfig.rootpath)
    return _campaign(tmp_path, out_root)


def test_champion_export_is_the_standalone_layout_of_best(campaign, out_root):
    pkg, p1_run, frozen, p2_run, p2, cand = campaign
    evidence = _evidence(p1_run, frozen, cand.package_digest)
    dest = C.export_champion(TARGET, p2, package_id="fixture_champion_v1", **evidence)
    assert dest == out_root / "artifacts" / "targets" / TARGET / "champions" / "fixture_champion_v1"
    assert C.layout_problems(dest) == []
    # The payload is byte-for-byte the best commit: strip the publication layer and re-hash.
    export = O.export(p2, "best", out_root / "build" / "check")
    payload = {p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()}
    assert payload - {p.relative_to(export).as_posix() for p in export.rglob("*") if p.is_file()} == {
        *C.PUBLISH_LAYER,
        *(f".merlin/{n}.json" for n in C.RECORDS),
        C.PUBLICATION_NOTE,
    }
    assert (dest / "fixture-opt").stat().st_mode & 0o100
    provenance = json.loads((dest / ".merlin/provenance.json").read_text())
    assert provenance["phase2"]["best_commit"] == cand.commit
    assert provenance["phase2"]["origin"]["commit"] == frozen.commit
    assert provenance["package_digest"] == hash_tree(export)["sha256"]
    assert load_yaml(dest / ".merlin/provenance.yaml")["generator"] == "merlin-target-publish"
    assert json.loads((dest / ".merlin/measurements.json").read_text())["firesim"]["cycles"] == 42_788_433
    assert storage_lifecycle.blockers(dest)  # retention-pinned
    with pytest.raises(C.ChampionError, match="already exported"):
        C.export_champion(TARGET, p2, package_id="fixture_champion_v1", **evidence)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda e: e["measurements"].update(package_digest="d" * 64), "measurements are of"),
        (lambda e: e["measurements"]["firesim"].pop("machine"), "firesim.machine"),
        (lambda e: e["measurements"]["firesim"]["control"].update(in_batch=False), "control.in_batch"),
        (lambda e: e["isa_prohibition"].update(verdict="violations"), "isa_prohibition.verdict"),
        (lambda e: e["isa_prohibition"].update(scope="kernel_only"), "isa_prohibition.scope"),
        # A clean verdict under a rule that prohibited nothing is no verdict.
        (lambda e: e["isa_prohibition"].update(prohibited_instructions={}), "prohibited_instructions"),
        (lambda e: e["isa_prohibition"].pop("prohibited_instructions"), "prohibited_instructions"),
        (lambda e: e["isa_prohibition"].update(prohibited_roles=[]), "isa_prohibition.prohibited_roles"),
        (lambda e: e["certification"]["gsim"].update(verdict="fail"), "gsim.verdict"),
        (lambda e: e["provenance"].pop("corpus_seal_digest"), "corpus_seal_digest"),
        (lambda e: e["provenance"]["phase1"].update(frozen_commit="e" * 40), "frozen"),
        (lambda e: e["measurements"].pop("exactness"), "measurements.exactness"),
        (lambda e: e["measurements"]["exactness"].update(label="unrecorded"), "exactness.label"),
    ],
)
def test_incomplete_or_failing_evidence_is_refused(campaign, out_root, mutate, match):
    pkg, p1_run, frozen, p2_run, p2, cand = campaign
    evidence = _evidence(p1_run, frozen, cand.package_digest)
    mutate(evidence)
    with pytest.raises(C.ChampionError, match=match):
        C.export_champion(TARGET, p2, package_id="refused", **evidence)
    assert not (out_root / "artifacts" / "targets" / TARGET / "champions" / "refused").exists()


def test_index_lists_releases_frozen_compilers_and_champions(campaign, out_root):
    pkg, p1_run, frozen, p2_run, p2, cand = campaign
    release = out_root / "artifacts" / "protocols" / TARGET / "phase0-20260929T090000Z-abc1234"
    (release / "private").mkdir(parents=True)
    (release / "private/preparation.json").write_text(
        json.dumps({"target": TARGET, "payload_sha256": "f" * 64, "source_run": "phase0-run", "prepared_at": "t"})
    )
    (release / "private/seal.json").write_text(
        json.dumps(
            {
                "review_digest": "9" * 64,
                "review": {"at": "t2", "note": "private", "reviewed_by": "op"},
                "retention_pin": "tok",
            }
        )
    )
    C.export_champion(TARGET, p2, package_id="fixture_champion_v1", **_evidence(p1_run, frozen, cand.package_digest))
    path = TI.index_path(TARGET)
    assert path == out_root / "artifacts" / "targets" / TARGET / "INDEX.yaml"
    assert TI.is_current(TARGET)  # export regenerated it
    index = load_yaml(path)
    assert index["problems"] == []
    assert [r["state"] for r in index["phase0_releases"]] == ["sealed"]
    assert "private" not in path.read_text() and "op" not in json.dumps(index["phase0_releases"])
    assert index["phase1_frozen"] == [
        {
            "run": p1_run.relative_to(out_root).as_posix(),
            "oot": (p1_run / "oot").relative_to(out_root).as_posix(),
            "frozen_commit": frozen.commit,
            "package_digest": frozen.package_digest,
            "rounds": 2,
            "frozen_at": "20260929T110000Z",
        }
    ]
    assert index["phase2_best"][0]["best_commit"] == cand.commit
    assert index["phase2_best"][0]["measured"] == ["measured/1"]
    champion = index["champions"][0]
    assert champion["firesim"] == {
        "cycles": 42_788_433,
        "machine": "fixture-board",
        "header": "5bfbb726",
        "control_in_batch": True,
    }
    assert champion["lineage"]["frozen_commit"] == frozen.commit
    # Regeneration is byte-stable, and a new record makes the written index stale.
    before = path.read_bytes()
    TI.write_index(TARGET)
    assert path.read_bytes() == before
    O.tag(p2, O.BEST_TAG, frozen.commit, move=True)
    assert not TI.is_current(TARGET)
    cited = TI.rows_citing(TARGET, p1_run)
    assert [r["run"] for r in cited["phase1_frozen"]] == [p1_run.relative_to(out_root).as_posix()]
    assert len(cited["champions"]) == 1


def test_index_cli_writes_and_checks(campaign, out_root, capsys):
    from merlin_experiments.cli import main

    assert main(["index", TARGET, "--check"]) == 1
    assert main(["index", TARGET]) == 0
    capsys.readouterr()
    assert main(["index", TARGET, "--check"]) == 0
    assert json.loads(capsys.readouterr().out)["current"] is True
    assert main(["lineage", "--target", TARGET]) == 0
    assert json.loads(capsys.readouterr().out)["phase1_frozen"][0]["rounds"] == 2
    assert main(["lineage"]) == 2
