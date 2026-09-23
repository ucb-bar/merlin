"""A successful compiler process may still decline the whole graph.

Exercise production admission before any decoding, machine audit or timing.
"""

import json
import sys

import pytest
from merlin_experiments.phase2 import contracts as CONTRACTS
from merlin_experiments.phase2 import contracts as P2_CONTRACTS
from merlin_experiments.phase2 import emission_analysis as EA
from merlin_experiments.phase2 import emission_diagnostics as ED
from merlin_experiments.phase2 import stage_inputs as INPUTS

from merlin.benchharness import hash_tree
from merlin.common.digest import sha256_bytes
from merlin.common.paths import merlin_dir
from merlin.targetgen import oot_runner

_SCRIPTS = merlin_dir() / "experiments/gemmini_perf_bench/scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


@pytest.mark.parametrize("arm", ["baseline", "candidate", "cached_baseline"])
@pytest.mark.parametrize(
    "declined",
    [
        {"op": "host_lane", "reason": "straight-line budget exceeded", "shape": [1, 1000]},
        {},
    ],
)
def test_declined_arm_is_not_a_successful_empty_model(tmp_path, monkeypatch, arm, declined):
    baseline, candidate, source = (tmp_path / name for name in ("base", "candidate", "source"))
    for path in (baseline, candidate, source):
        path.mkdir()
    (baseline / "implementation.txt").write_text("baseline\n")
    (candidate / "implementation.txt").write_text("candidate\n")
    (source / "capsule.yaml").write_text(json.dumps({"id": "external_model"}))
    (source / "capsule.interface.mlir").write_text("module {}")
    sentinel = INPUTS.StageE2ESentinel(
        "external_model", str(source), str(source), P2_CONTRACTS.exact_tree_record(source)["sha256"], (), ()
    )
    calls = []
    payload = json.dumps({"declined": declined, "commands": []})
    lowered = "module {}"

    def emit(package, interface, scratch, tag, timeout):
        calls.append(tag)
        return 0, lowered, payload if tag == arm else json.dumps({"commands": []})

    def no_accounting(*args, **kwargs):
        pytest.fail("declined model reached structural accounting")

    monkeypatch.setattr(oot_runner, "load_package", lambda path: path)
    monkeypatch.setattr(ED, "analyze_command_buffers", no_accounting)
    retained = None
    if arm == "cached_baseline":
        retained = {
            "identity": {
                "baseline_sha256": hash_tree(baseline)["sha256"],
                "capsule_sha256": sentinel.capsule_sha256,
                "target": "fixture_device",
            },
            "lowered_text": lowered,
            "lowered_sha256": sha256_bytes(lowered.encode()),
            "command_buffer_text": payload,
            "command_buffer_sha256": sha256_bytes(payload.encode()),
        }
    with pytest.raises(CONTRACTS.StageGateError, match="lowering declined; no structural or cycle comparison"):
        EA.analyze_whole_model_emission(
            baseline,
            candidate,
            sentinel,
            timeout_s=10,
            peak_macs_per_cycle=None,
            achievable_macs_per_cycle=None,
            target="fixture_device",
            emit_pair_runner=emit,
            baseline_artifacts=retained,
            contract_root=merlin_dir() / "contract",
        )
    assert calls == {"baseline": ["baseline"], "candidate": ["baseline", "candidate"], "cached_baseline": []}[arm]
