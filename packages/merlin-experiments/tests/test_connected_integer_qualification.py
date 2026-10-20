"""Source arithmetic admission cannot consume a JSON sidecar or missing owner."""

import copy
import os
import shutil
from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase0 import connected_integer_qualification as Q
from merlin_experiments.phase0 import model_qualification

from merlin.targetgen import golden_store


def test_missing_realization_owner_does_not_waive_original_admission():
    entry = dict(source_integer_proof=Q.SELECTION)
    with pytest.raises(ValueError, match="host-selected realization owner"):
        Q.qualify(entry, {}, Path("unused"))
    with pytest.raises(ValueError, match=model_qualification.REFUSAL) as refused:
        model_qualification.qualify(entry, {}, Path("unused"), None, Path("unused"))
    assert "host-selected realization owner" in str(refused.value.__cause__)
    for value in [None, {}, dict(source_rule={}), type("Diagnostics", (), {"source_rule": lambda self: {}})()]:
        with pytest.raises(ValueError, match="actual selected realization owner"):
            with Q.scoped_realization_owner(value):
                pytest.fail("diagnostic data became a realization owner")


def test_real_sealed_source_reopens_all_inputs_outputs_and_refuses_mutations(tmp_path):
    selected = os.environ.get("MERLIN_TEST_PUBLIC_SEALED_CAPSULE")
    if not selected:
        pytest.skip("explicit public original sealed fixture not selected")
    original = Path(selected)
    copied = tmp_path / "capsule"
    shutil.copytree(original, copied)
    cap = yaml.safe_load((copied / "capsule.yaml").read_bytes())
    proof = Q.source_proof(cap, copied)
    assert len(proof["output_owners"]) == 3
    assert proof["input_intervals"] == [[-8, 8]] * 3
    assert proof["output_intervals"] == [[-320, 320], [-40, 40], [-960, 960]]
    golden = golden_store.load_golden(copied)
    mutation = copy.deepcopy(golden)
    mutation["oracle_provenance"]["inputs"]["I0"]["decoded"][0] += 1
    golden_store.write_golden(copied, mutation)
    with pytest.raises(ValueError, match="inputs differ"):
        Q.source_proof(cap, copied)
    golden_store.write_golden(copied, golden)
    mutation = copy.deepcopy(cap)
    mutation["numeric_policy"]["outputs"]["spatial_state"]["compare"] = "bounded_int"
    with pytest.raises(ValueError, match="numeric policy"):
        Q.source_proof(mutation, copied)
    (copied / "capsule.interface.mlir").write_text("builtin.module {}")
    with pytest.raises(ValueError, match="member changed"):
        Q.source_proof(cap, copied)
