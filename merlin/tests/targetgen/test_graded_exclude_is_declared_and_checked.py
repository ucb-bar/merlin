"""An experiment may withhold a capsule from the PAID loop, but only by DECLARING it, and only visibly.

A whole-model capsule costs one oracle invocation per matmul layer, so its price tracks the model's size
rather than the compiler's difficulty. On radiance that turned a 12-round A/B into an unreachable ~15 h
per arm per round, and the first run where both arms cleared the op gate spent its whole budget grading
one model. `grading.exclude_capsules` scopes the loop; these tests pin the two properties that keep it
from becoming a way to hide failures:

  1. the library stays target-agnostic — it reads the list off the descriptor and knows no target's
     corpus, so a target that declares nothing is byte-identical to before;
  2. an exclusion that matches NO capsule raises, because the failure mode of a stale or mistyped name is
     a silently WIDER graded set, which is exactly the wall-clock blowout the knob exists to prevent.
"""
from __future__ import annotations

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.targetgen.contract.materialize import materialize_public_capsules
from merlin.targetgen.target_experiment import load_target_experiment

EXP = repo_root() / "merlin/experiments/capsule_bench/targets"


def _corpus(tmp_path, names):
    root = tmp_path / "corpus"
    for n in names:
        d = root / n
        d.mkdir(parents=True)
        (d / "capsule.yaml").write_text(
            yaml.safe_dump({"name": n, "kind": "op", "label": "public",
                            "required_oracle_tiers": ["L2"]}), encoding="utf-8")
        (d / "capsule.interface.mlir").write_text("// iface\n", encoding="utf-8")
    return root


# ---------------------------------------------------------------------------------------------
# the knob itself
# ---------------------------------------------------------------------------------------------
def test_no_declaration_materializes_everything(tmp_path):
    """The default must be untouched: a descriptor with no `grading:` block excludes nothing."""
    root = _corpus(tmp_path, ["A0", "A1", "A2"])
    got = materialize_public_capsules(tmp_path / "out", corpus_roots=[root])
    assert got == ["A0", "A1", "A2"]


def test_a_declared_name_is_withheld(tmp_path):
    root = _corpus(tmp_path, ["A0", "A1", "A2"])
    got = materialize_public_capsules(tmp_path / "out", corpus_roots=[root], exclude=("A1",))
    assert got == ["A0", "A2"]
    assert not (tmp_path / "out" / "A1").exists()


def test_an_exclusion_matching_nothing_fails_closed(tmp_path):
    """A typo must not quietly re-admit the expensive capsule it was meant to withhold."""
    root = _corpus(tmp_path, ["A0", "A1"])
    with pytest.raises(ValueError) as e:
        materialize_public_capsules(tmp_path / "out", corpus_roots=[root], exclude=("A1", "A9_typo"))
    assert "A9_typo" in str(e.value)
    # and it names what IS there, so the fix is obvious from the message alone
    assert "A0" in str(e.value)


def test_the_withheld_capsule_is_untouched_on_disk(tmp_path):
    """Withholding scopes the GRADE; it must never mutate or delete the corpus."""
    root = _corpus(tmp_path, ["A0", "A1"])
    before = (root / "A1" / "capsule.yaml").read_bytes()
    materialize_public_capsules(tmp_path / "out", corpus_roots=[root], exclude=("A1",))
    assert (root / "A1" / "capsule.yaml").read_bytes() == before


# ---------------------------------------------------------------------------------------------
# the descriptors that exist
# ---------------------------------------------------------------------------------------------
def test_every_declared_exclusion_names_a_real_capsule():
    """Whatever any target declares must resolve — this is the gate that keeps the list from going stale
    as corpora are regenerated, and it runs for every target, not just the one that uses the knob."""
    for desc in sorted(EXP.glob("*/target_experiment.yaml")):
        te = load_target_experiment(desc)
        if not te.graded_exclude or not te.capsule_corpus:
            continue
        roots = [te.capsule_corpus] + [repo_root() / s for s in te.corpus_siblings()]
        present = {d.name for r in roots for d in r.glob("*") if (d / "capsule.yaml").is_file()}
        missing = sorted(set(te.graded_exclude) - present)
        assert not missing, f"{desc}: grading.exclude_capsules names {missing}, not in the corpus"


def test_targets_that_declare_nothing_are_unaffected():
    """The cardinal rule: this is a per-target DECLARATION, so a target that says nothing keeps the old
    behavior exactly. Guards against the knob acquiring a default that quietly narrows another target."""
    seen = 0
    for desc in sorted(EXP.glob("*/target_experiment.yaml")):
        te = load_target_experiment(desc)
        doc = yaml.safe_load(desc.read_text())
        if "grading" not in doc:
            assert te.graded_exclude == ()
            seen += 1
    assert seen, "no descriptor without a grading block — this test would be vacuous"


def test_the_radiance_declaration_keeps_exactly_one_model_capsule():
    """The experiment CHOICE this landed for: keep the model capsule that proves whole-model mesh
    execution, drop the three that re-ask the same question at 6-20x the oracle cost."""
    desc = EXP / "radiance/target_experiment.yaml"
    if not desc.is_file():
        pytest.skip("radiance descriptor absent in this checkout")
    te = load_target_experiment(desc)
    roots = [te.capsule_corpus] + [repo_root() / s for s in te.corpus_siblings()]
    models = {d.name for r in roots for d in r.glob("*")
              if (d / "capsule.yaml").is_file() and d.name.startswith("M")}
    kept = models - set(te.graded_exclude)
    assert len(kept) == 1, f"expected one model capsule in the loop, got {sorted(kept)}"
