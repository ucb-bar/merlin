"""Normal live calibration wiring with harmless original process products.

External engine/numerical observations are synthetic. These controls establish
no simulator, physical model, runtime isolation or timing qualification.
"""

import os
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import ordinary_calibration as CAL
from merlin_experiments.phase2 import ordinary_fast_feedback as FAST
from merlin_experiments.phase2 import ordinary_queue_calibration as QUEUE
from merlin_experiments.phase2 import ordinary_search_policy as OS
from merlin_experiments.phase2 import paired_measurement as PME
from merlin_experiments.phase2.contracts import sha256_file
from merlin_experiments.phase2.evaluation_pool import AssignedResources, JobResources
from merlin_experiments.phase2.retained_revisions import RetainedRevisionStore
from merlin_experiments.phase2.search_coordinator import ComponentSearchCoordinator
from merlin_experiments.phase2.search_inputs import ORDINARY_KIND
from test_ordinary_calibration_join import _ORIGINAL_RUN, _products
from test_ordinary_fast_feedback import extend_corpus
from test_ordinary_search_policy import ordinary_inputs as ordinary_inputs


def test_normal_completed_products_install_only_unknown_screen_and_keep_original_selection(
    ordinary_inputs, tmp_path, monkeypatch
):
    cpus = tuple(sorted(os.sched_getaffinity(0)))[:2]
    if len(cpus) < 2:
        pytest.skip("two CPUs required for original harmless routing controls")
    extend_corpus(ordinary_inputs, tmp_path)
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "storage"))

    def engines(package, kernel, kernel_dir, runs, timeout, target, **kwargs):
        elf = _products(runs)
        return {
            "status": "pass",
            "numeric": "pass",
            "per_sim": {"spike": {"correct": True}, "gsim": {"correct": True, "cycles": 10}},
            "gsim_qualification": {"admitted": True, "decision": kwargs["decision"].to_dict()},
            "gsim_execution": {"gsim": {"elf": str(elf), "elf_sha256": sha256_file(elf), "cycles": 10}},
        }

    def cheap(capsule, package, **kwargs):
        _products(Path(kwargs["runs_root"]))
        return {"tiers": {tier: {"status": "pass"} for tier in ("L0", "L1", "L2")}, "status": "incomplete"}

    monkeypatch.setattr(PME, "run_execution", _ORIGINAL_RUN)
    monkeypatch.setattr(PME, "_run_el4_compiler_engines", engines)
    monkeypatch.setattr(PME.PC, "package_sandbox_policy", lambda *args: None)
    monkeypatch.setattr(PME.PC, "boxed_entrypoints", lambda *args: nullcontext())
    monkeypatch.setattr(
        FAST.CR, "load_capsule", lambda *args, **kwargs: {"required_oracle_tiers": ["L0", "L1", "L2", "L3"]}
    )
    monkeypatch.setattr(FAST.CR, "run_capsule", cheap)
    monkeypatch.setattr(FAST.CR, "simulator_adapter", lambda *args: "synthetic-only")
    monkeypatch.setattr(
        FAST,
        "get_backend",
        lambda target: SimpleNamespace(
            harness_build_recipe=lambda: SimpleNamespace(
                require_kernel_stack_frame=lambda: SimpleNamespace(entry_symbol="entry")
            )
        ),
    )
    policy = OS.prepare_ordinary_search_policy(**ordinary_inputs.arguments)
    store = RetainedRevisionStore(
        tmp_path / "sources",
        edit_authority=policy.edit_authority,
        max_bytes=16_384,
        max_files=16,
        max_total_bytes=65_536,
    )
    pool = policy.enable_evaluation_pool(
        revisions=store,
        resources=AssignedResources(cpus, 8 << 30, 32 << 20, 1, author_cores=1),
        job_resources=JobResources(1, 2 << 30, 4 << 20),
        root=tmp_path / "queue",
    )
    try:
        coordinator = policy.search_coordinator = ComponentSearchCoordinator(pool)
        first = store.freeze(policy.candidate, parent_revision=pool.baseline_revision.revision_id)
        (policy.candidate / "compiler.py").write_text("def schedule():\n    return 2\n")
        second = store.freeze(policy.candidate, parent_revision=first.revision_id)
        for revision in (first, second):
            for kind in (FAST.KIND, ORDINARY_KIND):
                job = pool.submit(revision.revision_id, kind=kind, timeout_s=20)
                deadline = time.monotonic() + 25
                while not pool.status(job["job_id"])["terminal"] and time.monotonic() < deadline:
                    time.sleep(0.02)
                assert pool.status(job["job_id"])["state"] == "completed"
        closure, measured = policy.search_closure_sha256, coordinator.status()["incumbent"]
        result = QUEUE.install(coordinator, first.revision_id + "," + second.revision_id, timeout_s=20)
        assert result["status"] == "UNKNOWN"  # Identical programs cannot identify a calibrated model.
        assert result["reasons"]
        assert result["screening_net_benefit"].startswith("UNKNOWN")
        assert policy.search_closure_sha256 == closure
        assert coordinator.status()["incumbent"] == measured
        coordinator.calibration.verify(policy)
        path, _ = next(pin for pin in coordinator.calibration.artifacts if pin[0].endswith("qualification.json"))
        Path(path).write_text("{}")
        with pytest.raises(CAL.StageGateError, match="evidence changed"):
            coordinator.calibration.verify(policy)
    finally:
        pool.close()
