"""Real library/scheduling smoke for the automatic learning owner boundary.

The neutral declared instruction machine is a test executor, not qualified
hardware. No timing or performance score is fabricated. The proposal source only
selects scheduling decisions; the owner checks actual dialect and command IR.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.compile.scheduling.compact_verification import verify_compact_dense
from merlin.compile.scheduling.loop_schedule import LoopTransform
from merlin.compile.scheduling.schedule_ir import parse_dense_schedule_plan, read_dense_schedule_plan
from merlin.kernels.reconstruction import reconstruct
from merlin.mining.reconstruction_learning import (
    CheckResult,
    GateEvidence,
    GeneralizationPartition,
    LearningBudget,
    LearningHooks,
    LearningTask,
    run_learning,
)
from merlin.mining.reconstruction_proposal import CompilerPatch, CompilerProposalAttempt, CompilerSurface
from merlin.transforms.certificates import ir_digest
from merlin.transforms.pass_exploration import ExplorationGeometry


def _neutral_fixture():
    # This import is test/operator-only; production modules never import tests.
    path = repo_root() / "merlin/tests/kernels/test_reconstruction.py"
    spec = importlib.util.spec_from_file_location("neutral_reconstruction_operator_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def run_operator_smoke(workspace: Path):
    """Return retained real artifacts and a report; no agent or timing launch.

    An operator can call this directly with a task-local directory. Two bounded
    deterministic proposals emulate agent outputs, including one rejected gap.
    All reconstruction, symbolic checks and independent arithmetic really run.
    """
    pytest.importorskip("z3", reason="actual compact storage/dependency verification needs z3-solver")
    fixture = _neutral_fixture()
    workspace.mkdir(parents=True, exist_ok=True)
    compiler = workspace / "compiler-source"
    (compiler / "src/passes").mkdir(parents=True, exist_ok=True)
    (compiler / "src/representations").mkdir(exist_ok=True)
    pass_path = "src/passes/selection.py"
    representation_path = "src/representations/plan.py"
    baseline_pass = (
        "from representations.plan import admit\nFACTOR = 1\n"
        "def selected(enabled=True):\n    return admit(FACTOR if enabled else 1)\n"
    )
    (compiler / pass_path).write_text(baseline_pass)
    (compiler / representation_path).write_text("def admit(factor):\n    return []\n")
    proposed_pass = baseline_pass.replace("FACTOR = 1", "FACTOR = 2")
    proposed_representation = (
        "def admit(factor):\n    return [] if factor <= 1 else "
        "[{'action': 'unroll', 'path': ['outer_m'], 'factor': factor}]\n"
    )
    patch = CompilerPatch(((pass_path, proposed_pass), (representation_path, proposed_representation)))
    shapes = {"training": (8, 6, 8), "validation": (9, 7, 9), "held_out": (4, 9, 5)}
    goals = ("schedule.loop_transformed", "schedule.consumed")
    traces, frozen = [], {}
    outcomes = {}

    def requests(candidate, enabled=True):
        code = (
            "import json\nfrom passes.selection import selected\n"
            f"print(json.dumps(selected({enabled!r}), sort_keys=True))\n"
        )
        child = candidate.run([sys.executable, "-c", code])
        assert child.returncode == 0, child.stderr
        return tuple(
            LoopTransform(row["action"], tuple(row["path"]), factor=row["factor"]) for row in json.loads(child.stdout)
        )

    def compile_shape(candidate, shape, enabled=True):
        transforms = requests(candidate, enabled)
        selected = fixture.adapter(shape, loop_transform_sets=(transforms,))
        # One explicit library composition; proposal controls the editable plan,
        # not which output the oracle selects after seeing execution results.
        binding = selected.bindings[1]
        family = replace(binding.family, knobs=tuple((name, (values[0],)) for name, values in binding.family.knobs))
        selected = replace(
            selected, bindings=(selected.bindings[0], replace(binding, family=family), selected.bindings[2])
        )
        result = reconstruct(selected, goals, max_candidates=1)
        path = result.paths[0]
        assert path.candidate.emission is not None, result.summary()
        assert path.candidate.evaluation.status == "compiled"
        traces.append(
            dict(
                shape=shape,
                compiler_sha256=candidate.compiler_sha256,
                enabled=enabled,
                source_sha256=ir_digest(selected.source),
                program_sha256=_sha(path.candidate.emission.data),
                replay=path.summary()["replay"],
                observation=path.observation.summary(),
            )
        )
        return selected, result, path

    def reconstruct_owner(candidate, task, budget):
        selected, result, path = compile_shape(candidate, shapes["training"])
        outcomes[candidate.compiler_sha256] = (selected, result)
        if candidate.patch_sha256 is None:
            for split, shape in shapes.items():
                ordinary = compile_shape(candidate, shape, enabled=False)[2]
                frozen[split] = ordinary.candidate.emission.data
        return result

    def verify_owner(candidate, result, task, split):
        shape = shapes[split]
        selected, current, path = compile_shape(candidate, shape)
        product = path.candidate.evaluation.result
        artifact = product.artifact
        kernel = artifact.kernel
        artifact.revalidate()
        receipt = product.receipt
        source_hash = ir_digest(selected.source)
        assert source_hash == receipt["source"]["source_sha256"] == receipt["after"]["source_sha256"]
        assert source_hash == ir_digest(product.module)
        # Independently interpret concrete instruction operands, including
        # training/validation/held-out tails, against dense integer arithmetic.
        fixture.execute(kernel, shape)
        problem = fixture.DenseProblem(*shape, 2, 3, 4, 1, 1, 4)
        resources = fixture.DenseResources(512, 512, 2, 2, 2, partial_output_tiles=True, partial_output_blocks=True)
        storage = fixture.DenseLoopStorage(
            fixture.DenseMemoryView("left", "left_store", 3, 0, 32),
            fixture.DenseMemoryView("right", "right_store", 4, 0, 32),
            fixture.DenseMemoryView("sum", "sum_store", 16, 0, 32),
        )
        target = fixture.DenseLoopAdapter(
            fixture.encode, fixture.project, "neutral-test-config", "ordered-modular-test", True, True, 64
        )
        schedule = artifact.realization["schedule_ir"]
        decision = read_dense_schedule_plan(
            parse_dense_schedule_plan(schedule["text"]),
            problem,
            resources,
            storage,
            target,
            source_identity=source_hash,
            dtypes=("i8", "i8", "i32"),
            static_budget=256,
        )
        verification = verify_compact_dense(
            kernel, problem, resources, decision.storage, target, static_budget=256, timeout_ms=20000
        )
        assert verification["kernel_sha256"] == kernel.digest()
        assert f":schedule={decision.plan_sha256}:" in artifact.checked_emission.context.execution
        assert decision.plan_sha256 == schedule["sha256"]
        assert any('"schedule.plan"' in row["text"] for row in path.intermediates)
        emitted = path.candidate.emission
        assert emitted.schedule_sha256 == kernel.digest() and emitted.data == kernel.text().encode()
        ordinary = compile_shape(candidate, shape, enabled=False)[2].candidate.emission.data
        assert ordinary == frozen[split]
        changed = emitted.data != ordinary
        mechanism = path.satisfied
        checks = (
            CheckResult("source_semantics", "pass", (source_hash, "complete dense SSA region admission")),
            CheckResult("numerical", "pass", (kernel.digest(), f"independent command execution exact: {shape}")),
            CheckResult("storage", "pass", (kernel.digest(), json.dumps(verification, sort_keys=True))),
            CheckResult(
                "target",
                "pass",
                (
                    "NEUTRAL DECLARED MACHINE; no hardware qualification",
                    target.configuration_identity,
                    "actual operands projected and checked",
                ),
            ),
            CheckResult(
                "scheduling_dialect", "pass", (decision.plan_sha256, artifact.checked_emission.context.execution)
            ),
            CheckResult("emission", "pass", (_sha(emitted.data), emitted.schedule_sha256)),
            CheckResult(
                "mechanism",
                "pass" if mechanism else "fail",
                (json.dumps(path.observation.summary()),),
                "actual emitted loop-transform goal absent" if not mechanism else "",
            ),
            CheckResult("frozen_baseline", "pass", (_sha(ordinary), "feature-disabled emission byte-identical")),
            CheckResult(
                "noninert",
                "pass" if changed else "fail",
                (_sha(emitted.data), _sha(ordinary)),
                "unchanged checked command stream" if not changed else "",
            ),
        )
        traces.append(
            dict(
                split=split,
                numerical="pass",
                source_sha256=source_hash,
                schedule_sha256=decision.plan_sha256,
                command_sha256=kernel.digest(),
                storage=verification,
                neutral_machine_only=True,
                hardware_qualification=False,
                changed=changed,
                mechanism=mechanism,
            )
        )
        return GateEvidence(candidate.compiler_sha256, _sha(emitted.data), checks)

    proposals = []

    def propose(task, **kwargs):
        proposals.append(kwargs)
        if len(proposals) == 1:
            return CompilerProposalAttempt(CompilerPatch((patch.files[0],), "select existing unroll primitive"))
        assert kwargs["feedback"][0]["stage"] == "reconstruction"
        assert "held_out" not in json.dumps(task)
        return CompilerProposalAttempt(patch)

    def no_timing(*args):
        raise AssertionError("no real timing provider was configured")

    def geometry(shape):
        m, k, n = shape
        return ExplorationGeometry("integer_dense", ((m, k), (k, n)))

    partition = GeneralizationPartition(
        *(tuple((geometry(shapes[split]),)) for split in ("training", "validation", "held_out"))
    )
    report = run_learning(
        LearningTask("loop-consumer-integration", "adapter_gap", goals),
        compiler_root=compiler,
        source_roots=("src",),
        surfaces=(CompilerSurface("src/passes", "pass"), CompilerSurface("src/representations", "representation")),
        partition=partition,
        hooks=LearningHooks(reconstruct_owner, verify_owner, lambda *args: (), no_timing),
        workspace=workspace / "learning",
        environment_sha256=_sha(Path(__file__).read_bytes()),
        proposer=propose,
        budget=LearningBudget(max_candidates=1),
    ).summary()
    report.update(
        operator_scope="neutral declared machine; actual scheduling dialect and symbolic checks",
        timing="UNKNOWN; no provider configured",
        hardware_qualification=False,
        operator_trace=traces,
    )
    (workspace / "operator-report.json").write_text(json.dumps(report, sort_keys=True, indent=2))
    return report


def test_real_library_automatic_multifile_loop_consumes_schedule_and_generalizes(tmp_path):
    report = run_operator_smoke(tmp_path)
    assert report["status"] == "accepted", report
    assert len(report["attempts"]) == 2
    assert report["attempts"][0]["rejection"]["stage"] == "reconstruction"
    assert report["attempts"][1]["rejection"] is None
    assert report["held_out"]["gate"] is None
    assert report["attempts"][1]["evaluation"]["ranking"] == []
    splits = [row for row in report["operator_trace"] if "split" in row]
    assert [row["split"] for row in splits] == ["training", "training", "validation", "held_out"]
    assert all(row["numerical"] == "pass" and row["hardware_qualification"] is False for row in splits)
    assert all(row["mechanism"] and row["changed"] for row in splits[1:])
    assert report["performance_qualification"] is False
