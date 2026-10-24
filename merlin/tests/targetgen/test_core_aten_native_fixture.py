"""A capture that is already i8 x i8 -> i32 is an admitted contraction for numeric packaging.

``core_aten_numeric.prepare_spec_int8`` has two branches. A W8A8 capture is quantized at preparation
and reports ``quantized_contractions``; a native integer capture is admitted as is and reports only
``native_contractions``. ``prepare_numeric_fixture`` read the first counter alone, so every native
int8 capture failed packaging with "no admitted contraction" although its plan had admitted sites
and an exact integer reference (and ``core_aten_batch_grade`` already grades such captures).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

from merlin.targetgen import core_aten_capsules as C
from merlin.targetgen import core_aten_device, core_aten_numeric


def _flow(tmp_path: Path) -> dict:
    return dict(
        policy="spec_int8",
        target="fixture",
        execution_provider=str(tmp_path / "provider.py"),
        rtl_facts=str(tmp_path / "facts.json"),
        software_spec=str(tmp_path / "software-spec.yaml"),
        host_capabilities=str(tmp_path / "host-capabilities.yaml"),
        device_package=str(tmp_path / "package"),
    )


@pytest.fixture
def prepared(monkeypatch):
    plans = []

    @contextmanager
    def facts(target, path):
        yield {}

    monkeypatch.setattr(core_aten_device, "load_execution_provider", lambda target, path: object())
    monkeypatch.setattr(core_aten_device, "selected_facts", facts)
    monkeypatch.setattr(core_aten_numeric, "prepare_spec_int8", lambda *args, **kwargs: plans.pop(0))
    return plans


@pytest.mark.parametrize(
    "counts",
    [
        dict(native_contractions=2),
        dict(quantized_contractions=1),
        dict(quantized_contractions=0, native_contractions=1),
    ],
)
def test_either_admitted_contraction_kind_packages(prepared, tmp_path, counts):
    prepared.append(dict(counts, integer_reference=dict(status="prepared")))
    report = {"cases": [{}]}
    C.prepare_numeric_fixture(tmp_path, report, _flow(tmp_path))
    assert report["numeric_preparation"]["integer_reference"]["status"] == "prepared"
    assert report["cases"][0]["numeric_policy"] == "spec_int8"


@pytest.mark.parametrize(
    "plan",
    [
        dict(native_contractions=0, integer_reference=dict(status="prepared")),
        dict(quantized_contractions=0, integer_reference=dict(status="prepared")),
        dict(native_contractions=1, integer_reference=dict(status="not_run")),
        dict(native_contractions=1),
    ],
)
def test_no_admitted_contraction_or_reference_still_refuses(prepared, tmp_path, plan):
    prepared.append(plan)
    with pytest.raises(ValueError, match="no admitted contraction"):
        C.prepare_numeric_fixture(tmp_path, {"cases": [{}]}, _flow(tmp_path))


@pytest.mark.parametrize("family", ["F1", "F2", "F4", "F5"])
def test_native_preparation_packages_with_its_own_boundary_gate(prepared, tmp_path, family):
    plan = dict(
        status="admitted",
        numerical_family=family,
        exact_dtype_sites=1,
        sites=[dict(status="admitted")],
        host_floor=dict(status="required", comparison="native_eager_bits_every_boundary_result"),
    )
    prepared.append(plan)
    report = {"cases": [{}]}
    C.prepare_numeric_fixture(tmp_path, report, _flow(tmp_path))
    assert report["numeric_preparation"] == plan


def test_native_fixture_cannot_skip_native_grading(prepared, tmp_path):
    prepared.append(
        dict(status="admitted", numerical_family="F1", exact_dtype_sites=1, sites=[dict(status="admitted")])
    )
    with pytest.raises(ValueError, match="no admitted contraction"):
        C.prepare_numeric_fixture(tmp_path, {"cases": [{}]}, _flow(tmp_path))


def test_admission_lane_retains_precise_host_refusal(prepared, tmp_path):
    plan = dict(status="host", refusal_code="non_native_unit_construction")
    prepared.append(plan)
    report = {"cases": [{}]}
    assert C.prepare_numeric_fixture(tmp_path, report, _flow(tmp_path), require_device=False) == plan
    assert not C._device_fixture(plan)
