"""Facts of a chosen target configuration bind a simulation runner only through their correspondence."""

import pytest
from test_bound_chipyard_rocc_backend import _bound, _gsim

from merlin.runtime.backends import chipyard_rocc_bound as B
from merlin.targetgen.rtl import unit_correspondence as UC

pytest_plugins = ("test_bound_chipyard_rocc_backend",)

TARGET = {"firrtl": "a" * 64, "hierarchy": "b" * 64, "core_hw": "c" * 64, "selection": "d" * 64}


def _corresponding(selection, tmp_path, status="structurally_equivalent"):
    """Re-label the selection's facts as the chosen target's, corresponding to the runner config."""
    engine, doc, receipt = _gsim(selection, tmp_path)
    contract, facts = selection
    simulated = facts["inputs"]["fir_sha256"]  # the runner's receipt FIRRTL
    facts["inputs"].update(
        fir_sha256=TARGET["firrtl"], core_hw_sha256=TARGET["core_hw"], source_bundle_sha256=TARGET["selection"]
    )
    facts["facts"]["source"].update(
        config="target_config", fir_sha256=TARGET["firrtl"], hierarchy_sha256=TARGET["hierarchy"]
    )
    facts["source_consistency"]["config"] = "target_config"
    simulation = {"firrtl": simulated, "hierarchy": "e" * 64, "core_hw": "1" * 64, "selection": "2" * 64}
    facts["facts"]["simulation_correspondence"] = {
        "schema": UC.SCHEMA,
        "status": status,
        "unit_root": "Unit",
        "configs": {
            "target": {
                "config": "target_config",
                "core_root": "Unit",
                **{role: {"path": f"/target/{role}", "sha256": sha} for role, sha in TARGET.items()},
            },
            "simulation": {
                "config": contract["runner"]["chipyard_rocc"]["config"],
                "core_root": "Unit",
                **{role: {"path": f"/simulation/{role}", "sha256": sha} for role, sha in simulation.items()},
            },
        },
    }
    return engine, doc, receipt


def test_corresponding_target_facts_bind_and_the_receipt_must_carry_the_simulation_firrtl(selection, tmp_path):
    _corresponding(selection, tmp_path)
    bound = _bound(selection)
    identity = bound.execution_engine_identity("gsim")
    simulated = selection[1]["facts"]["simulation_correspondence"]["configs"]["simulation"]["firrtl"]["sha256"]
    assert identity["receipt"]["firrtl_sha256"] == simulated != selection[1]["inputs"]["fir_sha256"]
    assert bound.gsim_selected_firrtl_status()[0] is True


def test_mismatched_simulation_firrtl_is_refused_by_the_executed_engine(selection, tmp_path):
    _corresponding(selection, tmp_path)
    selection[1]["facts"]["simulation_correspondence"]["configs"]["simulation"]["firrtl"]["sha256"] = "9" * 64
    bound = _bound(selection)
    with pytest.raises(B.RoCCExecutionError, match="no strict receipt for the original FIRRTL"):
        bound.execution_engine_identity("gsim")
    assert bound.gsim_selected_firrtl_status()[0] is False


def test_simulation_side_of_another_configuration_is_refused(selection, tmp_path):
    _corresponding(selection, tmp_path)
    selection[1]["facts"]["simulation_correspondence"]["configs"]["simulation"]["config"] = "other_config"
    with pytest.raises(B.RoCCExecutionError, match="another target or configuration"):
        _bound(selection)


@pytest.mark.parametrize("role", ["firrtl", "hierarchy", "core_hw", "selection"])
def test_mismatched_target_side_is_refused(selection, tmp_path, role):
    _corresponding(selection, tmp_path)
    selection[1]["facts"]["simulation_correspondence"]["configs"]["target"][role]["sha256"] = "8" * 64
    with pytest.raises(B.RoCCExecutionError, match="another target or configuration"):
        _bound(selection)


def test_missing_correspondence_is_refused(selection, tmp_path):
    _corresponding(selection, tmp_path)
    del selection[1]["facts"]["simulation_correspondence"]
    with pytest.raises(B.RoCCExecutionError, match="another target or configuration"):
        _bound(selection)


@pytest.mark.parametrize("status", ["different", "unknown", None])
def test_non_equivalent_correspondence_is_refused(selection, tmp_path, status):
    _corresponding(selection, tmp_path, status=status)
    with pytest.raises(B.RoCCExecutionError, match="another target or configuration"):
        _bound(selection)


def test_disagreeing_facts_configurations_are_refused(selection, tmp_path):
    _corresponding(selection, tmp_path)
    selection[1]["source_consistency"]["config"] = selection[0]["runner"]["chipyard_rocc"]["config"]
    with pytest.raises(B.RoCCExecutionError, match="another target or configuration"):
        _bound(selection)
