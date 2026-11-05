"""A host-only probe is screened as host work: never with the accelerator's accumulator, and data
movement is left to the written program's admission rather than refused for lacking a compute declaration."""

from __future__ import annotations

import pytest
import yaml
from merlin_experiments.phase0.requirements import _screen_defaults
from merlin_experiments.phase0.software_screen import screen_entry

from merlin.common.paths import repo_root

TARGET = repo_root() / "examples" / "gemmini" / "target"
_PROBE = {
    "cat": "model_slices",
    "kind": "model_slice",
    "operand_dtype": "f32",
    "lanes": {"forbid": ["on_mesh"]},
    "generalization": {"must_accelerate": False, "eligible": False, "generalization_axis": "host_lane"},
    "M": 16,
    "K": 32,
    "N": 16,
}
_SEMANTICS = {"operand_dtype": "int8", "accumulator_dtype": "int32"}


def _host_capabilities():
    """The selected host profile exactly as Phase 0 evidence selects it (needs the minted host package)."""
    from merlin.targetgen.target_experiment import load_target_experiment

    te = load_target_experiment(TARGET / "descriptor.yaml")
    out = {}
    for name, lane in te.host_lanes.profiles.items():
        try:
            _path, capabilities, identity = lane.resolve_capabilities(root=repo_root(), descriptor=te.path)
        except ValueError as exc:
            pytest.skip(f"the host lane package is not minted in this checkout: {exc}")
        out[name] = {**identity, "capability_spec": capabilities}
    return out


def _spec():
    return yaml.safe_load((TARGET / "software-spec.yaml").read_text())


def test_a_host_contraction_probe_is_not_screened_with_the_accelerator_accumulator():
    probe = dict(_PROBE, name="SY_host_lane_contraction_f32", op="batch_matmul", frontend_op="aten.bmm.default")
    assert _screen_defaults(probe, _SEMANTICS) == {"operand_dtype": "f32"}
    for entry in (probe, dict(probe, accum_dtype="f32")):
        decision = screen_entry(
            _spec(), entry, defaults=_screen_defaults(entry, _SEMANTICS), host_capabilities=_host_capabilities()
        )
        assert decision["status"] != "unsupported", decision
        assert "int32" not in str(decision)


def test_accelerator_candidates_keep_the_accelerator_semantics():
    assert _screen_defaults({"op": "matmul", "operand_dtype": "int8"}, _SEMANTICS) is _SEMANTICS


def test_a_host_movement_probe_is_left_to_the_written_programs_admission():
    probe = dict(_PROBE, name="SY_host_lane_movement_f32", op="permute", frontend_op="aten.permute.default")
    decision = screen_entry(_spec(), probe, defaults=_SEMANTICS, host_capabilities=_host_capabilities())
    assert decision["status"] == "unknown" and "data movement" in decision["reason"]
