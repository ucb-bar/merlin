"""Must-accelerate lanes follow the selected software spec's admission, not an operation table."""

import copy
import json
import os
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.functional_truth import SPEC_DECLINED, expectation, select_admission
from merlin.targetgen.software_spec import load_software_spec
from merlin.targetgen.spec_fact_drift import resolve_spec

ROOT = repo_root()
SPEC = ROOT / "examples/gemmini/target/software-spec.yaml"
EVIDENCE = json.loads((ROOT / "merlin/tests/data/exactness/core_aten_193_lane_evidence.json").read_text())


def _case(overload):
    return copy.deepcopy(next(case for case in EVIDENCE["cases"] if case["overload"] == overload))


@pytest.fixture(scope="module")
def authored():
    # The authored contraction declaration needs no fact resolution; fact-derived rows stay unresolved.
    return {"spec": resolve_spec(load_software_spec(SPEC), {})[0]}


def test_native_lane_requires_a_selected_admission():
    with pytest.raises(ValueError, match="software spec admission"):
        expectation(_case("aten.mm.default"))
    assert expectation(_case("aten.abs.default"))["lane"] == "host-guard"


def test_admitted_contraction_is_a_device_obligation(authored):
    commitment = expectation(_case("aten.mm.default"), admission=authored)
    assert commitment["lane"] == "device"
    assert commitment["admission"]["declaration"] == "contraction"


def _in_place(case):
    case["mutated_arguments"] = ["args[0]"]


def _rank_five(case):
    for tensor in [*case["arguments"]["args"]["items"], case["expected"]]:
        tensor["shape"], tensor["stride"] = [1, 1, *tensor["shape"]], [0, 0, *tensor["stride"]]


@pytest.mark.parametrize("mutate", [_in_place, _rank_five], ids=["in_place", "undeclared_rank"])
def test_construction_outside_the_declared_domain_is_a_host_commitment(authored, mutate):
    case = _case("aten.bmm.default")
    mutate(case)
    commitment = expectation(case, admission=authored)
    assert commitment["truth"] == SPEC_DECLINED and commitment["lane"] == "host-guard"
    assert commitment["admission"]["status"] == "unsupported"


@pytest.mark.skipif(
    not os.environ.get("MERLIN_RTL_FACTS") or not Path(os.environ["MERLIN_RTL_FACTS"]).is_file(),
    reason="the fact-resolved spec needs the selected RTL facts (MERLIN_RTL_FACTS)",
)
def test_spec_admission_reproduces_the_recorded_device_lane():
    admission = select_admission(
        "gemmini",
        rtl_facts=Path(os.environ["MERLIN_RTL_FACTS"]),
        software_spec=SPEC,
        host_capabilities=ROOT / "examples/gemmini/platform/host-capabilities-core-aten.yaml",
    )
    device = {
        case["overload"] for case in EVIDENCE["cases"] if expectation(case, admission=admission)["lane"] == "device"
    }
    assert device == set(EVIDENCE["device_lane"])
    assert len(device) == EVIDENCE["lane_run"]["device"]
