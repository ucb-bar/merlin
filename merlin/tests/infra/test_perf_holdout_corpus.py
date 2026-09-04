"""Offline proof of the performance holdout's commit/reveal boundary."""
from __future__ import annotations

import importlib.util
import json
import stat
import sys
from pathlib import Path

import pytest
import yaml

from merlin.benchharness import hash_tree
from merlin.common.paths import merlin_dir
from merlin.targetgen.corpus_spec import CorpusBinding


_SOURCE = merlin_dir() / "experiments/gemmini_perf_bench/scripts/perf_holdout_corpus.py"
_SPEC = importlib.util.spec_from_file_location("perf_holdout_corpus_under_test", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
HOLDOUT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = HOLDOUT
_SPEC.loader.exec_module(HOLDOUT)


def _facts() -> dict:
    return {
        "schema_version": "2.0",
        "generator": {"name": "merlin.targetgen.rtl.circt_introspect", "version": "test"},
        "inputs": {"core_hw_sha256": "a" * 64},
        "facts": {
            "target": "gemmini",
            "arrays": [{"name": "mesh", "rows": 16, "cols": 16,
                        "source": "mlc_discovery"}],
            "memories": [
                {"name": "scratchpad", "bytes": 262144, "depth": 4096,
                 "source": "mlc_discovery"},
                {"name": "accumulator", "bytes": 65536, "depth": 512,
                 "source": "mlc_discovery"},
            ],
            "datapaths": [
                {"name": "input", "dtype": "i8", "evidence": "UInt<8>"},
                {"name": "accumulator", "dtype": "i32", "evidence": "SInt<32>"},
            ],
            "timing": [{"module": "AccPipe", "pipeline_depth": 1,
                        "source": "mlc_hw_graph_walk"}],
        },
    }


def _frozen_inputs(tmp_path: Path) -> tuple[Path, Path]:
    facts = tmp_path / "rtl_facts.json"
    facts.write_text(json.dumps(_facts(), sort_keys=True), encoding="utf-8")
    profile = tmp_path / "_perf.yaml"
    profile.write_bytes((merlin_dir() / "contract/capsules/profiles/_perf.yaml").read_bytes())
    facts.chmod(0o444)
    profile.chmod(0o444)
    return facts, profile


def _commit(tmp_path: Path, *, seed: bytes = b"s" * 32, ids=("trial-0", "trial-1")):
    facts, profile = _frozen_inputs(tmp_path)
    agent_view = tmp_path / "agent"
    host_root = tmp_path / "host"
    agent_view.mkdir()
    host_root.mkdir()
    paths = HOLDOUT.commit_holdout(
        agent_view / "holdout_commitment.json", host_root / "private",
        rtl_facts_path=facts, perf_profile_path=profile, target="gemmini",
        candidate_ids=ids, count=4, seed=seed, agent_view_root=agent_view)
    return paths, facts, profile, agent_view, host_root


def _candidate_seal(tmp_path: Path, candidate_id: str, payload: str) -> Path:
    tree = tmp_path / f"candidate-{candidate_id}"
    tree.mkdir()
    (tree / "compiler.py").write_text(payload, encoding="utf-8")
    (tree / "compiler.py").chmod(0o444)
    tree.chmod(0o555)
    digest = hash_tree(tree)["sha256"]
    record = tmp_path / f"{candidate_id}.json"
    record.write_text(json.dumps({
        "state": "sealed",
        "candidate": {"path": str(tree), "sha256": digest, "read_only": True},
        "admission": {"consumable": True},
    }, sort_keys=True), encoding="utf-8")
    record.chmod(0o444)
    return record


def _binding() -> CorpusBinding:
    return CorpusBinding(
        target="gemmini", tile_dim=16, operand_dtype="int8", accum_dtype="i32",
        integer=True, tiers=["L2", "L3"], compare="exact_int",
        classes_for=lambda **_: ["CONFIG_EX", "CONFIG_LD", "MVIN", "CONFIG_ST",
                                  "PRELOAD", "COMPUTE_PRELOADED", "MVOUT"],
    )


def test_selection_is_deterministic_unseen_and_rtl_bounded(tmp_path: Path) -> None:
    facts, profile = _frozen_inputs(tmp_path)
    domain = HOLDOUT.derive_domain(facts, profile, target="gemmini")
    selected = HOLDOUT.select_members(b"a" * 32, domain, 4)

    assert selected == HOLDOUT.select_members(b"a" * 32, domain, 4)
    assert selected != HOLDOUT.select_members(b"b" * 32, domain, 4)
    assert len(selected) == len(set(selected)) == 4
    assert not set(selected).intersection(domain["legal_k"]["excluded_public_dev"])
    assert all(domain["legal_k"]["minimum"] <= k <= domain["legal_k"]["maximum"]
               for k in selected)
    assert domain["bounds"]["mesh"] == {"rows": 16, "cols": 16}
    assert domain["bounds"]["cost_envelope"]["maximum_K"] == 128
    assert domain["legal_k"]["cardinality"] == 109
    assert set(domain["source_sha256"]) >= {
        "rtl_circt_facts", "shared_perf_contract", "generate_corpus_py", "corpus_spec_py"}

    generalization = HOLDOUT.select_generalization_members(b"a" * 32, domain, 4)
    assert generalization == HOLDOUT.select_generalization_members(b"a" * 32, domain, 4)
    assert generalization != HOLDOUT.select_generalization_members(b"b" * 32, domain, 4)
    assert all(len({row[axis] for row in generalization}) >= 2 for axis in ("M", "N", "K"))
    assert any(row["M"] > 1 and row["N"] > 1 for row in generalization)
    assert any(row["M"] <= 16 for row in generalization)
    assert any(row["M"] > 16 for row in generalization)
    assert any(row["N"] <= 16 for row in generalization)
    assert any(row["N"] > 16 for row in generalization)
    assert any(row["M"] % 16 for row in generalization)
    assert any(row["N"] % 16 for row in generalization)
    public_shapes = {(16, 16, k) for k in (16, 32, 64, 128)}
    assert not public_shapes.intersection(
        {(row["M"], row["N"], row["K"]) for row in generalization})
    envelope = domain["bounds"]["cost_envelope"]
    for row in generalization:
        assert row["M"] * row["N"] * row["K"] <= envelope["maximum_macs"]
        assert (row["M"] * row["K"] + row["K"] * row["N"]
                <= envelope["maximum_operand_bytes"])
        assert row["M"] * row["N"] * 4 <= envelope["maximum_accumulator_bytes"]
    assert domain["generalization"]["semantic_scope"]["operation"] == "matmul"
    # Regression for the real Gemmini structural dimensions/capacities encoded
    # by this fixture: broadening remains a 772-point domain, not four authored
    # answers or the old 109-point K-only domain.
    assert domain["generalization"]["selection_domain"]["cardinality"] == 772


def test_rtl_facts_provenance_hashes_inputs_and_replays_complete_document(tmp_path: Path) -> None:
    extractor, core, fir = (tmp_path / name for name in ("extractor.py", "core.mlir", "source.fir"))
    extractor.write_text("# extractor\n", encoding="utf-8")
    core.write_text("module @Core\n", encoding="utf-8")
    fir.write_text("circuit TestHarness:\n", encoding="utf-8")
    document = _facts()
    document["inputs"].update({
        "target": "gemmini", "extractor_sha": HOLDOUT._sha256_file(extractor)[:16],
        "core_hw_sha256": HOLDOUT._sha256_file(core),
        "fir_sha": HOLDOUT._sha256_file(fir)[:16],
    })
    document["facts"]["source"] = {"fir_path": str(fir)}
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps(document), encoding="utf-8")
    facts.chmod(0o444)

    evidence = HOLDOUT.verify_rtl_facts_provenance(
        facts, target="gemmini", fact_builder=lambda **_: document,
        core_hw_resolver=lambda _: core, extractor_path=extractor)
    assert evidence["extractor_sha256"] == HOLDOUT._sha256_file(extractor)
    assert evidence["core_hw_sha256"] == HOLDOUT._sha256_file(core)
    assert evidence["firrtl_sha256"] == HOLDOUT._sha256_file(fir)


def test_rtl_facts_provenance_refuses_stale_extractor_rtl_or_replay(tmp_path: Path) -> None:
    extractor, core, fir = (tmp_path / name for name in ("extractor.py", "core.mlir", "source.fir"))
    for path, text in ((extractor, "extractor"), (core, "core"), (fir, "fir")):
        path.write_text(text, encoding="utf-8")
    document = _facts()
    document["inputs"].update({
        "target": "gemmini", "extractor_sha": HOLDOUT._sha256_file(extractor)[:16],
        "core_hw_sha256": HOLDOUT._sha256_file(core),
        "fir_sha": HOLDOUT._sha256_file(fir)[:16],
    })
    document["facts"]["source"] = {"fir_path": str(fir)}
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps(document), encoding="utf-8")
    facts.chmod(0o444)
    common = dict(target="gemmini", core_hw_resolver=lambda _: core, extractor_path=extractor)

    with pytest.raises(HOLDOUT.HoldoutError, match="live CIRCT extraction differs"):
        HOLDOUT.verify_rtl_facts_provenance(
            facts, fact_builder=lambda **_: {**document, "facts": {}}, **common)
    extractor.write_text("changed", encoding="utf-8")
    with pytest.raises(HOLDOUT.HoldoutError, match="different extractor revision"):
        HOLDOUT.verify_rtl_facts_provenance(facts, fact_builder=lambda **_: document, **common)


def test_public_commitment_leaks_neither_seed_members_nor_candidate_ids(tmp_path: Path) -> None:
    seed = bytes(range(32))
    paths, _, _, agent_view, _ = _commit(tmp_path, seed=seed, ids=("secret-a", "secret-b"))
    public_text = paths.public_commitment.read_text(encoding="utf-8")
    public = json.loads(public_text)

    assert set(public) == {
        "algorithm", "version", "domain", "cohort_counts", "seed_sha256"}
    assert public["cohort_counts"] == {
        "PK_predictor": 4, "PK_MNK_generalization": 4}
    assert seed.hex() not in public_text
    assert "selected_k" not in public_text and "PKH" not in public_text
    assert "secret-a" not in public_text and "secret-b" not in public_text
    assert list(agent_view.iterdir()) == [paths.public_commitment]
    assert stat.S_IMODE(paths.host_private_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(paths.seed.stat().st_mode) == 0o600
    assert stat.S_IMODE(paths.state.stat().st_mode) == 0o600
    private = json.loads(paths.state.read_text(encoding="utf-8"))
    assert private["candidate_ids"] == ["secret-a", "secret-b"]
    assert len(private["selected_k"]) == 4
    assert len(private["selected_generalization_shapes"]) == 4


def test_commit_refuses_results_symlinks_and_non_frozen_inputs(tmp_path: Path) -> None:
    facts, profile = _frozen_inputs(tmp_path)
    public_parent, private_parent = tmp_path / "public", tmp_path / "private"
    public_parent.mkdir()
    private_parent.mkdir()
    with pytest.raises(HOLDOUT.HoldoutError, match="prior-result"):
        HOLDOUT.commit_holdout(
            public_parent / "c.json", private_parent / "p", rtl_facts_path=facts,
            perf_profile_path=profile, target="gemmini", candidate_ids=["t0"],
            prior_result_paths=[tmp_path / "cycles.json"])

    facts.chmod(0o644)
    with pytest.raises(HOLDOUT.HoldoutError, match="writable rather than frozen"):
        HOLDOUT.derive_domain(facts, profile, target="gemmini")
    facts.chmod(0o444)
    linked = tmp_path / "facts-link.json"
    linked.symlink_to(facts)
    with pytest.raises(HOLDOUT.HoldoutError, match="not a plain file"):
        HOLDOUT.derive_domain(linked, profile, target="gemmini")


def test_reveal_refuses_stale_inputs_seed_tamper_and_incomplete_seals(tmp_path: Path) -> None:
    paths, facts, _, _, host_root = _commit(tmp_path)
    seals = {"trial-0": _candidate_seal(tmp_path, "trial-0", "zero"),
             "trial-1": _candidate_seal(tmp_path, "trial-1", "one")}
    with pytest.raises(HOLDOUT.HoldoutError, match="incomplete or foreign"):
        HOLDOUT.reveal_and_materialize(
            paths.public_commitment, paths.host_private_dir, host_root / "corpus-missing",
            candidate_seals={"trial-0": seals["trial-0"]})

    facts.chmod(0o644)
    facts.write_text(json.dumps({**_facts(), "tampered": True}), encoding="utf-8")
    facts.chmod(0o444)
    with pytest.raises(HOLDOUT.HoldoutError, match="changed after authoring began"):
        HOLDOUT.reveal_and_materialize(
            paths.public_commitment, paths.host_private_dir, host_root / "corpus-stale",
            candidate_seals=seals)

    # A separate commitment isolates the seed-opening failure from stale facts.
    second = tmp_path / "second"
    second.mkdir()
    paths2, _, _, _, host2 = _commit(second)
    paths2.seed.write_bytes(b"x" * 32)
    paths2.seed.chmod(0o600)
    seals2 = {"trial-0": _candidate_seal(second, "trial-0", "two"),
              "trial-1": _candidate_seal(second, "trial-1", "three")}
    with pytest.raises(HOLDOUT.HoldoutError, match="does not open"):
        HOLDOUT.reveal_and_materialize(
            paths2.public_commitment, paths2.host_private_dir, host2 / "corpus-seed",
            candidate_seals=seals2)


def test_reveal_uses_shared_generator_and_seals_host_only_corpus(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths, _, _, agent_view, host_root = _commit(tmp_path)
    seals = {"trial-0": _candidate_seal(tmp_path, "trial-0", "alpha"),
             "trial-1": _candidate_seal(tmp_path, "trial-1", "beta")}
    monkeypatch.setattr(HOLDOUT, "_binding", lambda generator, target, domain: _binding())

    manifest_path = HOLDOUT.reveal_and_materialize(
        paths.public_commitment, paths.host_private_dir, host_root / "corpus",
        candidate_seals=seals)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["kind"] == "generated_performance_holdout_reveal"
    assert manifest["reveal"]["seed_sha256"] == json.loads(
        paths.public_commitment.read_text(encoding="utf-8"))["seed_sha256"]
    assert len(manifest["members"]) == 8
    predictor = [row for row in manifest["members"] if row["family"] == "PK"]
    generalization = [row for row in manifest["members"] if row["family"] == "PKG"]
    assert len(predictor) == len(generalization) == 4
    assert not {row["K"] for row in predictor}.intersection(manifest["public_dev_k"])
    assert all(len({row[axis] for row in generalization}) >= 2 for axis in ("M", "N", "K"))
    assert manifest["cohorts"]["PK_predictor"]["claim"] == "PREDICTS"
    assert manifest["cohorts"]["PK_MNK_generalization"]["claim"] == "DIFFERENTIAL"
    assert manifest["cohorts"]["PK_MNK_generalization"][
        "descriptor_contract_family"] == "PK"
    assert "manifest assigns PKG" in manifest["cohorts"][
        "PK_MNK_generalization"]["identity_rule"]
    assert all(row["path"] == f"_perf/{row['name']}" for row in manifest["members"])
    assert manifest["generator"]["builder"] == "merlin.targetgen.corpus_spec.build"
    assert [row["candidate_id"] for row in manifest["candidate_seals"]] == [
        "trial-0", "trial-1"]
    capsule_dirs = sorted((manifest_path.parent / "_perf").iterdir())
    assert [path.name for path in capsule_dirs] == sorted(
        row["name"] for row in manifest["members"])
    for capsule in capsule_dirs:
        cap = (capsule / "capsule.yaml").read_text(encoding="utf-8")
        # `derived_sweep`, not `generated_seeded_holdout`: the latter is not in the schema's enum,
        # and writing it killed a campaign at the reveal step. What the field records is HOW the
        # capsule was constructed, and these are expanded from a sweep like every other member --
        # being held out is a property of when they are revealed, not of how they were built.
        assert "derived_sweep" in cap
        assert (capsule / "golden.yaml").is_file()
    for member in generalization:
        descriptor = yaml.safe_load(
            (manifest_path.parent / member["path"] / "capsule.yaml").read_text(
                encoding="utf-8"))
        inputs = {row["role"]: row["shape"] for row in descriptor["inputs"]}
        assert [inputs["input"][0], inputs["weight"][1], inputs["input"][1]] == [
            member["M"], member["N"], member["K"]]
        # Intentional, documented two-identity boundary: corpus generation is
        # governed by PK's admitted runnable contract, while the host manifest
        # assigns PKG solely as a separate differential measurement cohort.
        assert descriptor["performance"]["family"] == member[
            "descriptor_contract_family"] == "PK"
    assert not any("PKH" in path.name for path in agent_view.rglob("*"))
    for member in (manifest_path.parent, *manifest_path.parent.rglob("*")):
        assert not member.is_symlink()
        assert not member.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)


def test_reveal_refuses_mutable_seal_and_existing_output(tmp_path: Path) -> None:
    paths, _, _, _, host_root = _commit(tmp_path)
    seals = {"trial-0": _candidate_seal(tmp_path, "trial-0", "alpha"),
             "trial-1": _candidate_seal(tmp_path, "trial-1", "beta")}
    seals["trial-0"].chmod(0o644)
    with pytest.raises(HOLDOUT.HoldoutError, match="writable rather than frozen"):
        HOLDOUT.reveal_and_materialize(
            paths.public_commitment, paths.host_private_dir, host_root / "mutable-seal",
            candidate_seals=seals)
    seals["trial-0"].chmod(0o444)
    output = host_root / "already-there"
    output.mkdir()
    with pytest.raises(HOLDOUT.HoldoutError, match="not fresh"):
        HOLDOUT.reveal_and_materialize(
            paths.public_commitment, paths.host_private_dir, output, candidate_seals=seals)


def test_elaborated_config_locator_is_stable_across_hash_seeds(tmp_path):
    """A hardware fact bundle must not depend on PYTHONHASHSEED.

    Measured 2026-09-03: the config-declaration walk ranked candidates by (depth, -argcount) only, so a
    same-callee tie was broken by set-iteration order. has_max_pool was reported at Configs.scala line
    20 or line 21 depending on the process, the frozen bundle differed from a live replay about half
    the time, and the campaign refused at preflight with "live CIRCT extraction differs from the frozen
    RTL facts snapshot" -- a provenance alarm with no provenance change behind it.
    """
    import json as _json
    import os
    import subprocess
    import sys
    from merlin.common.paths import repo_root

    facts_path = repo_root() / "merlin/targets/gemmini/contracts/rtl_facts/facts.json"
    if not facts_path.is_file():
        pytest.skip("frozen RTL facts are absent in this checkout")
    program = (
        "import json;"
        "from merlin.targetgen import capability_discovery as d;"
        f"facts=json.load(open({str(facts_path)!r}));"
        "c=d.elaborated_config('gemmini', facts);"
        "f=c.fields.get('has_max_pool');"
        "print(f.line if f else None)"
    )
    seen = set()
    for seed in ("0", "1", "2"):
        environment = dict(os.environ, PYTHONHASHSEED=seed)
        proc = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True,
                              env=environment, timeout=900)
        if proc.returncode != 0:
            pytest.skip(f"elaborated config sources are unavailable here: {proc.stderr[-200:]}")
        seen.add(proc.stdout.strip())
    assert len(seen) == 1, f"locator moved with the hash seed: {sorted(seen)}"
