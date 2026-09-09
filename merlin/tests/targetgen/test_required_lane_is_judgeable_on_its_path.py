"""A capsule must not declare a lane its own grading path cannot judge.

`GN0_layernorm_host_only_bf16_pt` declared `lanes: {require: [scalar_rvv_lane], forbid: [on_mesh]}`.
It PASSED L2 and L3 with clean numerics and a clean trace, and was still scored `incomplete` /
LANE_CONTRACT_NOT_EVALUATED in every arm of the g3arm batch -- an unwinnable row in the denominator
that no compiler could ever satisfy.

The contradiction is structural. `elf_lanes.lane_report_from_elf` sets `observed: []`
unconditionally, on the correct reasoning that a symbol present in a binary is not one that executed;
`unjudged_lanes` credits a REQUIRED lane only when it appears in `observed` with an
EXECUTED_LANE_EVIDENCE rung; and only the whole-model dispatch ledger (`kind: model`, or a
`whole_program` command buffer) ever supplies such a rung. A `model_slice` takes the op tier ladder,
produces no ledger, and therefore cannot judge a required lane by construction.

`forbid` has no such problem -- it is settled by the linked-ELF scan -- and on a two-lane target
forbidding one lane already leaves the other as the only place work can land, so the `require` also
carried no information. Removing it fixed the capsule without weakening any refusal.
"""
from __future__ import annotations

import pathlib

import pytest
import yaml

from merlin.common.paths import merlin_dir

CAPSULES = merlin_dir() / "contract/capsules"
#: The perf cohort is graded by the timing harness, not the functional op ladder, so its members are
#: outside this rule. Named rather than pattern-matched so adding a new tree cannot silently opt out.
PERF_TREE = "_perf"


def _functional_capsules():
    out = []
    for path in sorted(CAPSULES.rglob("capsule.yaml")):
        if "_data" in path.parts or PERF_TREE in path.parts:
            continue
        try:
            doc = yaml.safe_load(path.read_text()) or {}
        except Exception:  # noqa: BLE001 -- a malformed capsule is another test's subject
            continue
        out.append((path, doc))
    return out


def test_the_corpus_has_functional_capsules_to_check():
    """Guard against the check passing because it found nothing."""
    assert len(_functional_capsules()) > 50


@pytest.mark.parametrize("strict_kind", ["model_slice", "op"])
def test_no_non_model_capsule_requires_a_lane_it_cannot_evidence(strict_kind):
    offenders = []
    for path, doc in _functional_capsules():
        if doc.get("kind") != strict_kind:
            continue
        required = (doc.get("lanes") or {}).get("require") or []
        if required:
            offenders.append(f"{doc.get('name')} ({path.parent.name}) requires {required}")
    assert not offenders, (
        "a required lane needs execution evidence that only the whole-model dispatch ledger carries, "
        "so a " + strict_kind + " declaring one can never be judged and scores `incomplete` forever: "
        + "; ".join(offenders))


def test_gn0_keeps_the_forbid_that_carries_its_assertion():
    """The fix removed the unjudgeable half only -- the measurable claim must survive."""
    path = CAPSULES / "model_slices/GN0_layernorm_host_only_bf16_pt/capsule.yaml"
    if not path.is_file():
        pytest.skip("GN0 absent from this corpus")
    lanes = (yaml.safe_load(path.read_text()) or {}).get("lanes") or {}
    assert lanes.get("forbid") == ["on_mesh"], "the host-only assertion must remain"
    assert not lanes.get("require"), "the unjudgeable required lane must not come back"


def test_the_profile_and_the_generated_capsule_agree():
    """The profile is the source of truth; a regeneration must not resurrect the require."""
    prof = CAPSULES / "profiles/gemmini.yaml"
    if not prof.is_file():
        pytest.skip("gemmini profile absent")
    text = prof.read_text()
    marker = "name: GN0_layernorm_host_only_bf16_pt"
    assert marker in text
    entry = text[text.index(marker):]
    entry = entry[:entry.find("\n- {")] if "\n- {" in entry else entry
    assert "lanes: {forbid: [on_mesh]}" in entry, "the profile entry must declare forbid only"
    assert "require: [scalar_rvv_lane]" not in entry
