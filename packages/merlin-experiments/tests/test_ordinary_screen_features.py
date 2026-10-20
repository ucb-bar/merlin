"""Own host activity can distinguish edits without historical timing labels."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import ordinary_calibration as CAL
from merlin_experiments.phase2 import ordinary_queue_calibration as QUEUE
from merlin_experiments.phase2 import ordinary_screen_features as F

from merlin.perf import dynamic_census as C
from merlin.perf.fast_estimate_validation import Observation


def test_independent_host_activity_survives_equal_static_device_sites():
    rows = []
    for executable, host in ((4, 9), (8, 2), (7, 4), (2, 3), (5, 6), (9, 8)):
        values = F.project(
            {
                "features": dict(zip(F.LEGACY_FEATURES, (1, executable), strict=True)),
                "candidate_diagnostics": {
                    "candidate_features": {
                        "schema": "merlin.candidate_features.v1",
                        "features": {"host_integer_arithmetic": host},
                    }
                },
            }
        )
        rows.append(SimpleNamespace(features=values))
    selected = F.identifiable_pointers(rows)
    assert "/candidate/host_integer_arithmetic" in selected
    assert F.LEGACY_FEATURES[0] not in selected
    assert all(row.features["/candidate/source_macs"] is None for row in rows)
    assert "/candidate/source_macs" not in selected


def test_missing_or_identical_work_does_not_become_free_execution():
    with pytest.raises(ValueError, match="cannot distinguish"):
        F.identifiable_pointers([SimpleNamespace(features=F.project({})) for _ in range(8)])


def test_public_partition_is_frozen_without_cycle_labels_or_candidate():
    class PublicPolicy:
        search_corpus = SimpleNamespace(manifest_sha256="a" * 64)

        def validate_search_inputs(self):
            return {("public", str(i)) for i in range(8)}

    value = QUEUE.partition(PublicPolicy())
    assert value == QUEUE.partition(PublicPolicy())
    train, held = {tuple(key) for key in value["calibration"]}, {tuple(key) for key in value["held_validation"]}
    assert not train & held
    assert train | held == PublicPolicy().validate_search_inputs()


def test_unfitted_constant_work_cannot_change_outside_training_domain():
    rows = []
    for value in range(1, 9):
        features = F.project(
            {
                "features": {F.LEGACY_FEATURES[1]: value},
                "candidate_diagnostics": {
                    "candidate_features": {
                        "schema": "merlin.candidate_features.v1",
                        "features": {"source_macs": 16, "host_conversion": 0},
                    }
                },
            }
        )
        rows.append(
            Observation(str(value) * 64, "b" * 64, "calibration", "c" * 64, features, 5 + 2 * value, ("d" * 64,))
        )
    fitted = CAL._fit(rows)
    assert fitted.predict(rows[0].features, domain_sha256="c" * 64).resolved
    changed = {**rows[0].features, "/candidate/host_conversion": 1}
    assert not fitted.predict(changed, domain_sha256="c" * 64).resolved


def _dynamic_compiled(count):
    return {
        "status": "observed", "elf_sha256": "a" * 64,
        "features": dict.fromkeys(F.LEGACY_FEATURES, 1),
        "dynamic_census": {
            "schema": C.SCHEMA, "status": "observed", "identity": {"elf_sha256": "a" * 64},
            "candidate_features": {"status": "descriptive", "features": {C.FEATURES[0]: count}},
            "completeness": {"stream_eof": True, "truncated": False, "descendants_reaped": True},
        },
    }


def test_observed_same_elf_activity_distinguishes_equal_static_sites_and_binds_domain():
    # Pure feature/source control. No trace, engine or held qualification is issued.
    rows = [Observation(str(count) * 64, "b" * 64, "calibration", "c" * 64,
                        F.project(_dynamic_compiled(count)), 5 + 2 * count, ("d" * 64,))
            for count in range(1, 9)]
    assert F.identifiable_pointers(rows) == (C.FEATURES[0],)
    assert all(row.features[C.FEATURES[1]] is None for row in rows)
    fitted = CAL._fit(rows)
    assert fitted.predict(rows[0].features, domain_sha256="c" * 64).resolved
    changed = {**rows[0].features, C.FEATURES[0]: 9}
    assert not fitted.predict(changed, domain_sha256="c" * 64).resolved
    changed = {**rows[0].features, C.FEATURES[1]: 1}
    assert not fitted.predict(changed, domain_sha256="c" * 64).resolved


@pytest.mark.parametrize("defect", ["foreign_elf", "unknown", "schema", "authority", "eof", "truncated", "reap"])
def test_foreign_or_incomplete_census_stays_unknown(defect):
    compiled = deepcopy(_dynamic_compiled(9))
    census = compiled["dynamic_census"]
    if defect == "foreign_elf":
        census["identity"]["elf_sha256"] = "b" * 64
    elif defect == "unknown":
        census["status"] = "UNKNOWN"
    elif defect == "schema":
        census["schema"] = "another observer"
    elif defect == "authority":
        census["candidate_features"]["status"] = "qualified"
    else:
        field = {"eof": "stream_eof", "truncated": "truncated", "reap": "descendants_reaped"}[defect]
        census["completeness"][field] = field == "truncated"
    projected = F.project(compiled)
    assert all(projected[key] is None for key in C.FEATURES)
    assert all(projected[key] == 1 for key in F.LEGACY_FEATURES)


@pytest.mark.parametrize("count", [None, -1, True, "9", float("nan"), float("inf"), 1.5])
def test_missing_or_malformed_activity_never_becomes_zero(count):
    assert F.project(_dynamic_compiled(count))[C.FEATURES[0]] is None
