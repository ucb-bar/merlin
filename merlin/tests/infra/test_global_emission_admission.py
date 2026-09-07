"""A successful compiler process may still decline the whole graph.

Exercise production admission before any decoding, machine audit or timing.
"""
import json
import sys

import pytest

from merlin.benchharness import hash_tree
from merlin.common.paths import merlin_dir
from merlin.targetgen import oot_runner

_SCRIPTS = merlin_dir() / "experiments/gemmini_perf_bench/scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import perf_agent_stage as PAS


@pytest.mark.parametrize("arm", ["baseline", "candidate", "cached_baseline"])
@pytest.mark.parametrize("declined", [
    {"op": "host_lane", "reason": "straight-line budget exceeded", "shape": [1, 1000]},
    {},
])
def test_declined_arm_is_not_a_successful_empty_model(tmp_path, monkeypatch, arm, declined):
    baseline, candidate, source = (tmp_path / name for name in ("base", "candidate", "source"))
    for path in (baseline, candidate, source):
        path.mkdir()
    (source / "capsule.yaml").write_text(json.dumps({"id": "external_model"}))
    (source / "capsule.interface.mlir").write_text("module {}")
    sentinel = PAS.StageE2ESentinel("external_model", str(source), str(source),
        PAS._exact_tree_record(source)["sha256"], (), ())
    calls = []
    payload = json.dumps({"declined": declined, "commands": []})
    lowered = "module {}"

    def emit(package, interface, scratch, tag, timeout):
        calls.append(tag)
        return 0, lowered, payload if tag == arm else json.dumps({"commands": []})

    def no_accounting(*args, **kwargs):
        pytest.fail("declined model reached structural accounting")

    monkeypatch.setattr(oot_runner, "load_package", lambda path: path)
    monkeypatch.setattr(PAS, "analyze_command_buffers", no_accounting)
    retained = None
    if arm == "cached_baseline":
        retained = {"identity": {"baseline_sha256": hash_tree(baseline)["sha256"],
                    "capsule_sha256": sentinel.capsule_sha256, "target": "fixture_device"},
                    "lowered_text": lowered, "lowered_sha256": PAS._sha256(lowered.encode()),
                    "command_buffer_text": payload, "command_buffer_sha256": PAS._sha256(payload.encode())}
    with pytest.raises(PAS.StageGateError, match="lowering declined; no structural or cycle comparison"):
        PAS.analyze_whole_model_emission(baseline, candidate, sentinel, timeout_s=10,
            peak_macs_per_cycle=None, achievable_macs_per_cycle=None, target="fixture_device",
            emit_pair_runner=emit, baseline_artifacts=retained)
    assert calls == {"baseline": ["baseline"], "candidate": ["baseline", "candidate"],
                     "cached_baseline": []}[arm]
