"""A source capsule waiting only on an operator's review of a PROPOSED host row is withheld, not a hard failure."""

from types import SimpleNamespace

from merlin_experiments.phase0.generation import disclosed_source_gap
from merlin_experiments.phase0.program_admission import _check_saved_inventory, summarize

SOURCE = {"name": "SY_source_x", "kind": "model", "source_role": "materialized_iteration_capture"}
PENDING = "unresolved SW constraints: operation review"


def _entry(host_status, host_reason, accel="unsupported", classification="compute_required"):
    return {
        "classification": classification,
        "observed_signature": {"mlir_operation": "quant_ext.quantize_per_tensor"},
        "operation": "m2m_q.quantize.default",
        "accelerator_admission": {"status": accel, "reason": "no accelerator row"},
        "host_admission": {"status": host_status, "reason": host_reason, "reviewed": False},
    }


ADMITTED = {
    "classification": "compute_required",
    "observed_signature": {},
    "operation": "ok",
    "accelerator_admission": {"status": "unsupported"},
    "host_admission": {"status": "admitted", "reviewed": True},
}


def test_screen_waiting_only_on_proposed_host_rows_is_marked():
    screen = summarize([ADMITTED, _entry("unknown", PENDING)], scope="t")
    assert screen["status"] == "unknown" and screen["pending_host_review_only"] is True


def test_any_other_unknown_or_any_refusal_is_not_marked():
    other = summarize([_entry("unknown", PENDING), _entry("unknown", "source_body requires parsed source operations")], scope="t")
    assert other["status"] == "unknown" and other["pending_host_review_only"] is False
    refused = summarize([_entry("unknown", PENDING), _entry("unsupported", "no row")], scope="t")
    assert refused["status"] == "unsupported" and refused["pending_host_review_only"] is False
    unresolved_accelerator = summarize([_entry("unknown", PENDING, accel="unknown")], scope="t")
    assert unresolved_accelerator["pending_host_review_only"] is False
    assert summarize([ADMITTED], scope="t")["pending_host_review_only"] is False


def test_a_marked_screen_is_disclosed_as_pending_review_and_an_unmarked_unknown_stays_fatal():
    row = disclosed_source_gap(SOURCE, observed={"status": "unknown", "pending_host_review_only": True, "reason": "m2m_q.quantize.default: x"})
    assert row["kind"] == "host_lane_admission"
    assert row["reason"].startswith("emitted SW operation admission (pending operator review of a proposed host row): ")
    assert disclosed_source_gap(SOURCE, observed={"status": "unknown", "pending_host_review_only": False, "reason": "x"}) is None
    assert disclosed_source_gap(SOURCE, observed={"status": "unknown", "reason": "x"}) is None
    must_refuse = disclosed_source_gap(
        SOURCE, observed={"status": "unknown", "pending_host_review_only": True, "reason": "x"}, must_refuse=True
    )
    assert must_refuse is None
    roster = {**SOURCE, "source_role": "derived_sweep"}
    assert disclosed_source_gap(roster, observed={"status": "unknown", "pending_host_review_only": True, "reason": "x"}) is None


def test_the_derivation_time_inventory_must_wait_on_review_too():
    materialized = {"capture_sha256": "a" * 64}
    screen = summarize([_entry("unknown", PENDING)], scope="t")

    def held(expected):
        evidence = SimpleNamespace(whole_program_admission={"applications": {"x": {"capture_sha256": "a" * 64, **expected}}})
        return _check_saved_inventory(screen, materialized, evidence)

    pending_ops = [{"status": "unknown", "pending_host_review": True}, {"status": "admitted"}]
    assert held({"status": "unknown", "reason": "r", "operations": pending_ops})["pending_host_review_only"] is True
    other_ops = [{"status": "unknown", "pending_host_review": True}, {"status": "unknown"}]
    assert held({"status": "unknown", "reason": "r", "operations": other_ops})["pending_host_review_only"] is False
    assert held({"status": "unsupported", "reason": "r", "operations": pending_ops})["pending_host_review_only"] is False
