"""Champion export for a lineage older than the sealed phase 0: the explicit legacy block, an honest
composition note, and an ``oot/`` history reconstructed from stored package bytes."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from merlin.common import oot_repo as O
from merlin.common.paths import phase_runs_root
from merlin.common.yaml import load_yaml, write_yaml
from merlin.targetgen import champions as C
from merlin.targetgen import target_index as TI

TARGET = "fixture"
#: Out-relative run directories, as a champion's records cite them.
P1_RUN = Path("runs", TARGET, "capsule-bench", "p1").as_posix()
P2_RUN = Path("runs", TARGET, "perf", "p2").as_posix()


@pytest.fixture()
def out_root(tmp_path, monkeypatch, pytestconfig):
    from merlin.targetgen import publish

    root = tmp_path / "out"
    for sub in ("runs", "artifacts", "build"):
        (root / sub).mkdir(parents=True)
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(root))
    monkeypatch.setattr(publish.paths, "repo_root", lambda: pytestconfig.rootpath)
    return root


def _package(root: Path, body: str) -> Path:
    root.mkdir(parents=True)
    write_yaml(
        root / "manifest.yaml",
        {
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
        },
    )
    tool = root / "fixture-opt"
    tool.write_text("#!/usr/bin/env python3\nprint('fixture')\n")
    tool.chmod(0o755)
    (root / "transforms.py").write_text(body)
    return root


@pytest.fixture()
def lineage(tmp_path, out_root):
    """Three stored packages -- a phase-1 pick, a merge base, a composed best -- and a history rebuilt
    from them, as a lineage that never had a harness repository leaves them."""
    store = tmp_path / "store"
    packages = {name: _package(store / name, f"x = {i}\n") for i, name in enumerate(("phase1", "base", "best"))}
    digests = {name: O.package_digest(pkg) for name, pkg in packages.items()}
    run = phase_runs_root(TARGET, 2) / "20261001T000000Z_reconstructed_lineage"
    hops = [
        {"package": packages[name], "digest": digests[name], "label": name, "when": f"2026092{i}T000000Z"}
        for i, name in enumerate(("phase1", "base", "best"))
    ]
    record = O.reconstruct(run / "oot", hops, frozen=0, best=2, reason="the runs predate the harness oot/ repo")
    return run / "oot", digests, record


def _legacy() -> dict:
    return {
        "reason": "the lineage was graded on an input bundle before phase-0 corpora were sealed",
        "predates": C.LEGACY_PREDATES,
        "bundle_manifest_sha": "5" * 64,
        "bundle_name": "fixture_public_v0",
        "run_dirs": [P1_RUN, P2_RUN],
        "dates": {"phase1_started": "2026-09-24T04:46:30Z", "composed": "2026-10-01T10:58:00Z"},
        "hops": [
            {"package": "phase1", "driver": "claudecode", "model": "model-a", "effort": "high"},
            {"package": "best", "driver": "codex", "model": "model-b"},
        ],
    }


def _evidence(frozen: str, digest: str, *, legacy: bool = True) -> dict:
    provenance = {"phase1": {"run": P1_RUN, "frozen_commit": frozen}}
    if legacy:
        provenance.update(corpus_seal_digest=None, phase0_evidence_digest=None, lineage={"legacy": _legacy()})
    else:
        provenance.update(corpus_seal_digest="a" * 64, phase0_evidence_digest="b" * 64)
    return {
        "provenance": provenance,
        "measurements": {
            "package_digest": digest,
            "firesim": {
                "cycles": 37_455_758,
                "machine": "fixture-board",
                "header": "5bfbb726",
                "control": {"in_batch": True},
            },
            "exactness": {"contract_sha256": "e" * 64, "label": "exact"},
        },
        "certification": {"gsim": {"verdict": "pass"}},
        "isa_prohibition": {
            "scope": "whole_elf",
            "verdict": "clean",
            "prohibited_roles": ["loop_descriptor"],
            "prohibited_instructions": {"8": "LOOP_A"},
        },
    }


def _export(repo: Path, evidence: dict, package_id: str = "legacy_champion") -> Path:
    return C.export_champion(
        TARGET, repo, package_id=package_id, stage_root=repo.parent / "exports", **copy.deepcopy(evidence)
    )


def test_reconstructed_history_is_digest_exact_and_labelled(lineage):
    repo, digests, record = lineage
    history = O.history(repo, O.BEST_TAG)
    assert [c.package_digest for c in history] == [digests["phase1"], digests["base"], digests["best"]]
    assert all(c.metadata["reconstructed"] is True for c in history)
    assert O.tags(repo)[O.FROZEN_TAG] == history[0].commit == record["frozen"]
    assert O.tags(repo)[O.BEST_TAG] == history[-1].commit == record["best"]
    assert O.reconstruction(repo) == record and record["reconstructed"] is True
    assert O.origin(repo) is None


def test_reconstruction_refuses_bytes_that_are_not_the_recorded_package(tmp_path):
    pkg = _package(tmp_path / "pkg", "x = 1\n")
    hop = {"package": pkg, "digest": "0" * 64, "label": "p", "when": "20260924T000000Z"}
    with pytest.raises(O.OotRepoError, match="not the recorded"):
        O.reconstruct(tmp_path / "oot", [hop], frozen=0, best=0, reason="legacy")
    hop["digest"] = O.package_digest(pkg)
    with pytest.raises(O.OotRepoError, match="lineage order"):
        O.reconstruct(tmp_path / "oot", [hop], frozen=1, best=0, reason="legacy")
    with pytest.raises(O.OotRepoError, match="say why"):
        O.reconstruct(tmp_path / "oot", [hop], frozen=0, best=0, reason=" ")
    assert not (tmp_path / "oot").exists()


def test_legacy_lineage_stands_in_for_the_seal_and_is_printed(lineage, out_root):
    repo, digests, record = lineage
    evidence = _evidence(record["frozen"], digests["best"])
    evidence["provenance"]["composition"] = C.composition(
        {"g1": digests["phase1"], "mm": digests["best"]}, digests["base"], tool="git merge-file", hand_edits=False
    )
    dest = _export(repo, evidence)
    assert C.layout_problems(dest) == []
    provenance = json.loads((dest / ".merlin/provenance.json").read_text())
    legacy = provenance["lineage"]["legacy"]
    assert legacy["mark"] == C.LEGACY_MARK
    assert legacy["stands_in_for"] == ["corpus_seal_digest", "phase0_evidence_digest"]
    assert provenance["corpus_seal_digest"] is None and provenance["phase0_evidence_digest"] is None
    assert provenance["phase2"]["reconstructed"] is True
    assert provenance["phase2"]["reconstruction"]["frozen"] == record["frozen"]
    assert provenance["composition"]["note"].startswith("composed by three-way merge of cell winners g1 ")
    assert (dest / ".merlin/CHAMPION").read_text().splitlines()[1].startswith(C.LEGACY_MARK)
    note = (dest / C.PUBLICATION_NOTE).read_text()
    assert note.split("\n\n")[1].startswith(f"> **{C.LEGACY_MARK}**")  # the reader's first paragraph
    assert f"## {C.LEGACY_MARK}" in note and "fixture_public_v0" in note and "model-b" in note
    assert "## Composition" in note and "## Reconstructed history" in note and "`reconstructed: true`" in note
    champion = load_yaml(TI.index_path(TARGET))["champions"][0]
    assert champion["lineage"]["unsealed_legacy"] is True
    assert champion["lineage"]["reconstructed"] is True and champion["lineage"]["composed"] is True
    cited = TI.rows_citing(TARGET, out_root / P2_RUN)
    assert [row["package_id"] for row in cited["champions"]] == ["legacy_champion"]


def _drop_seal_key(e):
    del e["provenance"]["corpus_seal_digest"]


def _drop_legacy(e):
    del e["provenance"]["lineage"]


def _seal_both(e):
    e["provenance"].update(corpus_seal_digest="a" * 64, phase0_evidence_digest="b" * 64)


@pytest.mark.parametrize(
    "mutate, match",
    [
        # Never defaulted: a digest left out is missing even when a legacy block is present.
        (_drop_seal_key, r"provenance\.corpus_seal_digest"),
        # A null digest with no legacy block is not a decision anybody recorded.
        (_drop_legacy, r"provenance\.corpus_seal_digest"),
        (_seal_both, "stands in for no digest"),
        (lambda e: e["provenance"]["lineage"]["legacy"].update(reason=""), r"legacy\.reason"),
        (lambda e: e["provenance"]["lineage"]["legacy"].update(predates="phase 1"), r"legacy\.predates"),
        (lambda e: e["provenance"]["lineage"]["legacy"].update(bundle_manifest_sha="5d540284"), "bundle_manifest_sha"),
        (lambda e: e["provenance"]["lineage"]["legacy"].update(run_dirs=[]), r"legacy\.run_dirs"),
        (lambda e: e["provenance"]["lineage"]["legacy"].update(dates={}), r"legacy\.dates"),
        (lambda e: e["provenance"]["lineage"]["legacy"]["hops"][1].pop("model"), r"hops\[1\]\.model"),
        # The legacy block covers the two digests and nothing else.
        (lambda e: e["isa_prohibition"].update(verdict="violations"), r"isa_prohibition\.verdict"),
        (lambda e: e["certification"]["gsim"].update(verdict="not_certified"), r"gsim\.verdict"),
        (lambda e: e["measurements"]["firesim"]["control"].update(in_batch=False), "control.in_batch"),
        (lambda e: e["provenance"]["phase1"].pop("frozen_commit"), r"phase1\.frozen_commit"),
    ],
)
def test_legacy_lineage_is_explicit_and_covers_only_the_two_digests(lineage, out_root, mutate, match):
    repo, digests, record = lineage
    evidence = _evidence(record["frozen"], digests["best"])
    mutate(evidence)
    with pytest.raises(C.ChampionError, match=match):
        _export(repo, evidence, package_id="refused")
    assert not C.champion_dir(TARGET, "refused").exists()


def test_composition_must_be_honest_about_its_base(lineage, out_root):
    repo, digests, record = lineage
    evidence = _evidence(record["frozen"], digests["best"])
    evidence["provenance"]["composition"] = C.composition({"cell": digests["best"]}, "e" * 64)
    with pytest.raises(C.ChampionError, match="composition base"):
        _export(repo, evidence, package_id="refused")
    evidence["provenance"]["composition"] = {"note": "", "method": "merge", "base": digests["base"], "parts": {}}
    with pytest.raises(C.ChampionError, match=r"composition\.note.*composition\.parts"):
        _export(repo, evidence, package_id="refused")


def test_reconstructed_frozen_must_be_the_declared_frozen(lineage, out_root):
    repo, digests, record = lineage
    # The base hop is an ancestor of best, so only the reconstruction record can catch the mislabel.
    base_commit = O.history(repo, O.BEST_TAG)[1].commit
    with pytest.raises(C.ChampionError, match="reconstructed from frozen"):
        _export(repo, _evidence(base_commit, digests["best"]), package_id="refused")


def test_a_sealed_lineage_needs_no_legacy_block(lineage, out_root):
    repo, digests, record = lineage
    dest = _export(repo, _evidence(record["frozen"], digests["best"], legacy=False), package_id="sealed")
    assert C.LEGACY_MARK not in (dest / C.PUBLICATION_NOTE).read_text()
    assert len((dest / ".merlin/CHAMPION").read_text().splitlines()) == 1
    assert "lineage" not in json.loads((dest / ".merlin/provenance.json").read_text())
