"""Ordinary retained source revisions compile and execute a checked shared pass.

External functional/timing admission remains explicitly synthetic, as in the
ordinary queue fixture. Actual instruction bytes, native outputs/settings and
traffic are observed independently; command counts are never timing scores.
"""

import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from merlin_experiments.phase2 import ordinary_search_policy as OS
from merlin_experiments.phase2 import paired_measurement as PME
from merlin_experiments.phase2.evaluation_pool import AssignedResources, JobResources
from merlin_experiments.phase2.retained_revisions import RetainedRevisionStore
from merlin_experiments.phase2.search_inputs import ORDINARY_KIND
from test_ordinary_search_policy import ordinary_inputs as ordinary_inputs

from merlin.common import bundle_snapshot
from merlin.common.paths import repo_root
from merlin.targetgen.package_runtime import _py_imports_merlin

BASELINE = """def schedule(shape, resources):
    from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets
    return ScheduleBudgets(max_fusions=0, eliminate_configuration=False,
                           eliminate_copies=False, optimize_storage=True)
"""
TUNED = """def schedule(shape, resources):
    from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets
    return ScheduleBudgets(max_fusions=0,
                           eliminate_configuration=resources["persistent_fields"] >= 2 and all(x > 0 for x in shape),
                           eliminate_copies=False, optimize_storage=True)
"""
NEW_PASS_POLICY = """def schedule(shape, resources):
    from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets
    from passes.configuration import eliminate
    return ScheduleBudgets(max_fusions=0, eliminate_configuration=eliminate(shape, resources),
                           eliminate_copies=False, optimize_storage=True)
"""
NEW_PASS = """def eliminate(shape, resources):
    return resources["persistent_fields"] >= 2 and all(x > 0 for x in shape)
"""
NEW_TRANSFORM_POLICY = """def schedule(shape, resources):
    from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets
    from merlin.compile.scheduling.pointwise_source import PointwiseScheduleRewrite
    from passes.configuration import compose
    return (ScheduleBudgets(max_fusions=0, eliminate_configuration=False,
                            eliminate_copies=False, optimize_storage=True),
            (PointwiseScheduleRewrite("compose-source", compose),))
"""
NEW_TRANSFORM = """def compose(candidate, context):
    from dataclasses import replace
    from merlin.compile.scheduling.pointwise_source import PointwiseScheduleCandidate
    pairs = tuple(zip(candidate.kernel.body, context.configuration_effects, strict=True))
    arithmetic = tuple(call for call, effect in pairs if not effect.pure_write)
    if len(arithmetic) != 2:
        return candidate
    for capability in context.fusion_capabilities:
        fused = capability.emit(*arithmetic)
        if fused is not None:
            setup = tuple(call for call, effect in pairs if effect.pure_write)
            return PointwiseScheduleCandidate(replace(candidate.kernel, body=(*setup, fused)), ())
    return candidate
"""


@pytest.mark.parametrize(
    "ordinary_inputs,transformation",
    [
        (BASELINE, False),
        (
            {
                "source": BASELINE,
                "new_python_components": [
                    {"directory": "passes", "surface_ids": ["schedule"], "reason": "new candidate compiler passes"}
                ],
            },
            False,
        ),
        (
            {
                "source": BASELINE,
                "new_python_components": [
                    {"directory": "passes", "surface_ids": ["schedule"], "reason": "new candidate transformations"}
                ],
            },
            True,
        ),
    ],
    indirect=["ordinary_inputs"],
    ids=["checked-shared-policy", "new-pass-module", "new-transformation"],
)
def test_normal_queue_retains_policy_and_changes_realized_configuration(
    ordinary_inputs, tmp_path, monkeypatch, transformation
):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "storage"))
    monkeypatch.syspath_prepend(str(repo_root() / "merlin/tests/targetgen"))
    monkeypatch.syspath_prepend(str(repo_root() / "merlin/tests/infra"))
    release_tests = importlib.import_module("test_compiler_tool_profile_v6")
    configured = importlib.import_module("test_configured_pointwise")
    source = importlib.import_module("test_pointwise_source")
    owner, release = release_tests.original_owner(tmp_path / "profile")
    frozen = bundle_snapshot.snapshot_input_paths(
        owner.workspace, owner.bundle, [release / "merlin/__init__.py"], repo=repo_root()
    )[0].parent.parent
    dependencies = sorted(
        {
            str(Path(importlib.util.find_spec(name).origin).parent.parent)
            for name in ("xdsl", "numpy", "yaml", "z3", "pytest")
        }
    )
    native_root = tmp_path / "native"
    native_root.mkdir()
    native = source.execute.__wrapped__(
        native_root, SimpleNamespace(param=dict(source=configured.EXECUTOR, counters=5))
    )
    original_execute = PME.run_execution

    def execute(spec, workspace, timeout, target, rtl_identity, **kwargs):
        result = original_execute(spec, workspace, timeout, target, rtl_identity, **kwargs)
        workspace.mkdir(parents=True, exist_ok=True)
        with owner.compiler_imports() as admission:
            text = (spec.package / "compiler.py").read_text()
            assert _py_imports_merlin(text, compiler_api=admission) is None
            config = workspace / "worker.json"
            result_path = workspace / "worker-result.json"
            config.write_text(
                json.dumps(
                    dict(
                        bundle=str(frozen),
                        work=str(workspace),
                        fixtures=str(repo_root() / "merlin/tests/infra"),
                        dependencies=dependencies,
                        candidate=str(spec.package / "compiler.py"),
                        result=str(result_path),
                    )
                )
            )
            worker = repo_root() / "merlin/tests/fixtures/phase2_shared_pass_worker.py"
            run = subprocess.run(
                [sys.executable, "-I", "-S", "-B", str(worker), str(config)],
                capture_output=True,
                text=True,
                timeout=min(timeout, 60),
            )
            assert run.returncode == 0, run.stdout + run.stderr
            observed = json.loads(result_path.read_bytes())
            expected_sources = {
                row["module"]: row["sha256"]
                for row in owner.profile["shared_scheduling_library"]["members"]
                if row["module"] is not None
            }
            assert observed["source_owners"] == expected_sources
            assert observed["bytecode_disabled"]
            assert all(Path(path).is_relative_to(frozen) for path in observed["module_origins"].values())
            observations = []
            # Reopen child-published bytes through a separately loaded decoder
            # and native executor. The selected policy/plan is not evidence.
            for index, item in enumerate(observed["observations"]):
                m, n = item["shape"]
                separate = item["separate"]
                case_root = workspace / ("reopened-" + str(index))
                case_root.mkdir()
                options = configured.case(case_root, m=m, n=n, separate=separate)
                image = Path(item["image_path"]).read_bytes()
                calls = options["encoding"].decode_calls(image)
                x = np.arange(m * n, dtype=np.int32) * 37 - 350
                bias = np.full(m * n, 17, dtype=np.int32)
                expected = np.clip(x.astype(np.int64) + bias, -128, 127).astype("i1").tobytes()
                output, counts = native((image,), {"x": x.tobytes(), "b0": bias.tobytes()}, separate=separate)
                assert output == expected + bytes([193] * (1032 - m * n)) == bytes.fromhex(item["output_hex"])
                assert len(calls) == counts[0] == item["commands"]
                assert list(counts[1:3]) == item["transfer_bytes"]
                assert list(counts[3:]) == item["final_settings"] == [configured.LOW, configured.HIGH]
                observations.append(
                    dict(
                        shape=[m, n],
                        separate=separate,
                        commands=counts[0],
                        transfer_bytes=list(counts[1:3]),
                        final_settings=list(counts[3:]),
                        image_sha256=hashlib.sha256(image).hexdigest(),
                        encoded_bytes=len(image),
                        output_sha256=hashlib.sha256(output).hexdigest(),
                        rewrites=item["rewrites"],
                    )
                )
        (workspace / "shared-pass-evidence.json").write_text(
            json.dumps(
                dict(
                    arm=spec.arm,
                    library_sha256=observed["library_sha256"],
                    source_owners=observed["source_owners"],
                    module_origins=observed["module_origins"],
                    observations=observations,
                    timing="synthetic fixture observation; no performance qualification",
                )
            )
        )
        return result

    monkeypatch.setattr(PME, "run_execution", execute)
    policy = OS.prepare_ordinary_search_policy(**ordinary_inputs.arguments)
    cpus = tuple(sorted(os.sched_getaffinity(0)))[:1]
    store = RetainedRevisionStore(
        tmp_path / "sources",
        edit_authority=policy.edit_authority,
        max_bytes=16_384,
        max_files=16,
        max_total_bytes=65_536,
    )
    pool = policy.enable_evaluation_pool(
        revisions=store,
        resources=AssignedResources(cpus, 4 << 30, 32 << 20, 1, author_cores=0),
        job_resources=JobResources(1, 2 << 30, 2 << 20, 1),
        root=tmp_path / "queue",
    )
    try:
        initial = store.freeze(policy.candidate)
        new_pass = policy.edit_authority.contract["schema"] == "compiler_edit_contract_v2"
        pass_source = NEW_TRANSFORM if transformation else NEW_PASS
        if new_pass:
            (policy.candidate / "passes").mkdir()
            (policy.candidate / "passes/__init__.py").write_text("")
            (policy.candidate / "passes/configuration.py").write_text(pass_source)
        compiler_source = NEW_TRANSFORM_POLICY if transformation else NEW_PASS_POLICY if new_pass else TUNED
        (policy.candidate / "compiler.py").write_text(compiler_source)
        revision = store.freeze(policy.candidate)
        job = pool.submit(revision.revision_id, kind=ORDINARY_KIND, timeout_s=120)
        # A later live edit must not change the queued immutable compiler.
        (policy.candidate / "compiler.py").write_text("def schedule(*args):\n    raise RuntimeError('live edit')\n")
        if new_pass:
            (policy.candidate / "passes/configuration.py").write_text("raise RuntimeError('later pass edit')\n")
            assert (revision.source / "passes/configuration.py").read_text() == pass_source
        deadline = time.monotonic() + 125
        while not job["terminal"] and time.monotonic() < deadline:
            time.sleep(0.05)
            job = pool.status(job["job_id"])
        assert job["state"] == "completed", job
        assert job["result"]["candidate_sha256"] == revision.source_sha256
        evidence = [
            json.loads(path.read_text()) for path in (pool.root / job["job_id"]).rglob("shared-pass-evidence.json")
        ]
        assert {row["arm"] for row in evidence} == {"baseline", "candidate"}
        assert {row["library_sha256"] for row in evidence} == {
            owner.profile["shared_scheduling_library"]["library_sha256"]
        }
        for baseline in (row for row in evidence if row["arm"] == "baseline"):
            candidate = next(row for row in evidence if row["arm"] == "candidate")
            for old, new in zip(baseline["observations"], candidate["observations"], strict=True):
                assert old["commands"] == 18 and new["commands"] == (17 if transformation else 4)
                assert old["encoded_bytes"] > new["encoded_bytes"]
                for key in ("shape", "separate", "final_settings", "output_sha256"):
                    assert old[key] == new[key]
                if transformation:
                    elements = old["shape"][0] * old["shape"][1]
                    assert old["transfer_bytes"] == [12 * elements, 5 * elements]
                    assert new["transfer_bytes"] == [8 * elements, elements]
                    assert old["rewrites"] == [] and new["rewrites"] == ["compose-source"]
                else:
                    assert old["transfer_bytes"] == new["transfer_bytes"]
        store.restore(initial.revision_id, policy.candidate)
        assert (policy.candidate / "compiler.py").read_text() == BASELINE
        assert not (policy.candidate / "passes").exists()
    finally:
        pool.close()
