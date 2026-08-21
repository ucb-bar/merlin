"""The shared finalizer is the integrity backbone: every capsule row goes through it.

It owns not_run_is_not_pass, the fail-open guard for a required set that is empty or all-N/A, the
provenance stamp, and self-validation against capsule_result.schema.json. Both the op path and the
whole-model path call it, so a defect here is a defect in every verdict the suite produces.

These tests exercise it directly. That matters: when this block lived inline in `run_capsule` its only
coverage was through capsule grades that need an oracle, so in an environment without one it was never
executed — which is how a lifted-out-of-scope import survived a full test run and only surfaced when a
real grade was attempted.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import capsule_runner as R
from merlin.targetgen.capsule_common import make_run_paths


@pytest.fixture()
def paths(tmp_path):
    return make_run_paths(tmp_path / "runs", "cap", suite="t", target="radiance",
                          dtype="fp32", benchmark="cap")


def _cap(**over):
    # `kind` is a schema enum (isa | layer | model_slice | model) -- "op" is not a member, and the
    # finalizer only WARNS on a contract violation, so a wrong fixture would quietly produce an invalid row.
    c = {"name": "cap", "kind": "isa", "label": "public", "required_oracle_tiers": ["L0", "L2"]}
    c.update(over)
    return c


def _cfg():
    return R._config_for_target("radiance", "t", "fp32")


def _finalize(paths, *, status, tiers, numeric=None, no_oracle=False, required=("L0", "L2")):
    return R._finalize_capsule_result(
        name="cap", capsule=_cap(), status=status, failure=None, tiers=tiers,
        trace_check_res={"status": "skipped", "violations": []},
        numeric=numeric or {"status": "pass"}, required=set(required), no_oracle=no_oracle,
        eff_target="radiance", paths=paths, run_id="cap", cfg=_cfg(), contract=None)


def _tier(status, mandatory=True, **kw):
    return R.TierResult("x", status, mandatory, **kw)


def test_a_row_carries_the_fields_the_contract_requires(paths):
    """capsule_result.schema.json requires capsule, status, contract_version, tiers, trace_check, numeric.
    A row missing any of them is invalid, and the validator runs INSIDE this function — so a row that
    reaches a caller has already been checked."""
    r = _finalize(paths, status="pass",
                  tiers={"L0": R.TierResult("L0", "pass", True), "L2": R.TierResult("L2", "pass", True)})
    for k in ("capsule", "status", "contract_version", "tiers", "trace_check", "numeric"):
        assert k in r, k
    assert r["toolchain_shas"], "a verdict records the toolchain it was produced with"


def test_every_status_the_finalizer_can_produce_validates(paths):
    """The assertion whose absence let a lifted import go unnoticed: exercise each outcome and let the
    self-validation inside the finalizer run for all of them."""
    from merlin.targetgen.contract import schemas

    cases = {
        "pass": {"L0": R.TierResult("L0", "pass", True), "L2": R.TierResult("L2", "pass", True)},
        "incomplete": {"L0": R.TierResult("L0", "pass", True),
                       "L2": R.TierResult("L2", "unavailable", True)},
        "not_applicable_floor": {"L0": R.TierResult("L0", "skipped", True, not_applicable=True),
                                 "L2": R.TierResult("L2", "pass", True)},
    }
    for label, tiers in cases.items():
        r = _finalize(paths, status="pass", tiers=tiers)
        schemas.validate(r, "capsule_result", contract=None)      # raises on a contract violation
        assert r["status"] in ("pass", "incomplete", "not_gradeable_no_oracle"), (label, r["status"])


def test_a_mandatory_tier_that_did_not_run_is_never_a_pass(paths):
    r = _finalize(paths, status="pass",
                  tiers={"L0": R.TierResult("L0", "pass", True),
                         "L2": R.TierResult("L2", "unavailable", True)})
    assert r["status"] == "incomplete"
    assert r["failure"]["category"] == "NOT_RUN_IS_NOT_PASS"


def test_a_tier_that_is_honestly_not_applicable_does_not_block(paths):
    """The integer L0/L1 floor on a float datapath, and a whole model's absent command buffer: a
    legitimate skip, distinct from a missing oracle."""
    r = _finalize(paths, status="pass",
                  tiers={"L0": R.TierResult("L0", "skipped", True, not_applicable=True),
                         "L2": R.TierResult("L2", "pass", True)})
    assert r["status"] == "pass"


def test_no_runnable_required_tier_refuses_to_pass_on_our_own_engine(paths):
    """The fail-open guard: with every required tier N/A there is no independent oracle, and L0/L1 are our
    own command-buffer interpretation. Passing on that alone is the fail-open this exists to stop."""
    r = _finalize(paths, status="pass",
                  tiers={"L0": R.TierResult("L0", "skipped", True, not_applicable=True),
                         "L2": R.TierResult("L2", "skipped", True, not_applicable=True)})
    assert r["status"] == "incomplete"


def test_no_oracle_withholds_the_verdict_rather_than_failing_it(paths):
    """--no-oracle asked for no numeric tier, so a missing one is not a fixable failure — it is a
    withheld verdict, and must not be handed back as a phantom `oracle_unavailable` to chase."""
    r = _finalize(paths, status="pass", no_oracle=True,
                  tiers={"L0": R.TierResult("L0", "pass", True),
                         "L2": R.TierResult("L2", "skipped", True)})
    assert r["status"] == "not_gradeable_no_oracle"


def test_extra_fields_ride_along_without_deciding_the_status(paths):
    """The whole-model path attaches a routing plan and mesh counters. They are evidence for a reader,
    never inputs to the verdict."""
    r = R._finalize_capsule_result(
        name="cap", capsule=_cap(), status="pass", failure=None,
        tiers={"L0": R.TierResult("L0", "pass", True), "L2": R.TierResult("L2", "pass", True)},
        trace_check_res={"status": "skipped", "violations": []}, numeric={"status": "pass"},
        required={"L0", "L2"}, no_oracle=False, eff_target="radiance", paths=paths, run_id="cap",
        cfg=_cfg(), contract=None, extra={"mesh_execution": {"matmul_layers_on_mesh": 15}})
    assert r["status"] == "pass"
    assert r["mesh_execution"]["matmul_layers_on_mesh"] == 15
