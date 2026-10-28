"""Evidence-guided experiments reuse the broker and refuse stale/self-graded results."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from merlin.compile import tuning_knobs
from merlin.mining.reconstruction_experiments import (
    ExperimentBudget,
    build_experiment_packet,
    execution_context_identity,
    packet_identity,
    workspace_tuning,
)
from merlin.mining.reconstruction_learning import REQUIRED_CHECKS
from merlin.mining.reconstruction_proposal import digest

GOAL = "memory.lhs_retained_full_reduction"


def selected_context():
    return dict(
        compiler_sha256="a" * 64,
        source_sha256="b" * 64,
        environment_sha256="c" * 64,
        reference_sha256="d" * 64,
        visibility="operator_only",
        evaluator=dict(identity="independent-test-owner", sha256="e" * 64, trusted_result_sha256=[]),
        baseline=dict(submission="submission"),
        tuning=dict(
            current=dict(reuse="reload", output_width=3, unrelated=False),
            domains=dict(reuse=dict(values=["reload", "resident"], goals=[GOAL])),
        ),
        capsules=[
            dict(
                id="arbitrary-public-identifier",
                goals=[GOAL],
                split="training",
                selfcheck_argv=[
                    "python3",
                    "agent_selfcheck.py",
                    "--submission",
                    "{submission}",
                    "--capsules",
                    "arbitrary-public-identifier",
                    "--suite",
                    "scalability",
                    "--sim",
                    "spike",
                    "--wait-budget",
                    "5",
                ],
            )
        ],
    )


def missing_mechanism():
    return dict(
        gaps=[
            dict(
                kind="bounded_search_exhaustion",
                goals=[GOAL],
                reason="bounded paths lack complete operand retention",
                next_action="discover_or_bind_existing_library_alternatives",
                witnesses=[dict(program_sha256="f" * 64)],
            )
        ]
    )


def test_region_trial_requires_its_selected_capsule_binding():
    context = selected_context()
    index = dict(regions=[dict(region_id="selected-source", **missing_mechanism())])
    task = build_experiment_packet(index, context=context)["tasks"][0]
    assert not task["requests"]
    assert "binding.region_capsule_applicability" in task["next_action"]["gaps"]
    context["capsules"][0]["region_ids"] = ["selected-source"]
    task = build_experiment_packet(index, context=context)["tasks"][0]
    assert task["status"] == "ready" and len(task["requests"]) == 2
    assert task["region_id"] == "selected-source"


def receipt(request, context, *, cycles=100, gates=None, established=()):
    return dict(
        context_identity=execution_context_identity(context),
        request_id=request["request_id"],
        evaluator={key: context["evaluator"][key] for key in ("identity", "sha256")},
        parent_compiler_sha256=context["compiler_sha256"],
        source_sha256=context["source_sha256"],
        environment_sha256=context["environment_sha256"],
        reference_sha256=context["reference_sha256"],
        settings_sha256=request["settings_sha256"],
        compiler_sha256="1" * 64,
        program_sha256=digest(dict(settings=request["settings"])),
        artifact_pins=[dict(path="owner/artifact.json", sha256="3" * 64)],
        gates=gates if gates is not None else dict.fromkeys(REQUIRED_CHECKS, "pass"),
        measurement=dict(kind="measured", cycles=cycles, complete_region=True, domain_sha256="4" * 64),
        established=list(established),
    )


def test_requests_hold_other_factors_fixed_and_use_existing_private_knob_copy():
    context = selected_context()
    packet = build_experiment_packet(missing_mechanism(), context=context)
    task = packet["tasks"][0]
    control, candidate = task["requests"]
    assert task["learning_task"]["route"] == "library_parameters_or_order"
    assert not task["learning_task"]["missing_pass_certainty"]
    assert candidate["settings"] == dict(reuse="resident", output_width=3, unrelated=False)
    assert candidate["tuning_overrides"] == {"reuse": "resident"}
    assert candidate["broker_payload"]["tuning_overrides"] == candidate["tuning_overrides"]
    assert candidate["broker_payload"]["sim"] == "spike"
    assert control["settings"] == context["tuning"]["current"]
    assert control["tuning_overrides"] == {}
    assert candidate["baseline_request_id"] == control["request_id"]
    original = context["capsules"][0]["selfcheck_argv"]
    prefix = ["submission" if item == "{submission}" else item for item in original]
    assert control["argv"][: len(prefix)] == candidate["argv"][: len(prefix)] == prefix
    assert control["argv"][len(prefix) :] == ["--reference-request", control["request_id"]]
    assert candidate["argv"][len(prefix) :] == [
        "--tuning-overrides",
        '{"reuse":"resident"}',
        "--reference-request",
        candidate["request_id"],
    ]
    assert candidate["broker_payload"]["reference_experiment_request_id"] == candidate["request_id"]
    assert candidate["submission"] == "submission"
    assert candidate["changed_factors"] == ["reuse"]
    assert task["acceptance"]["policy_consumer"] == "merlin.transforms.shape_policy"
    assert packet["identity"] == packet_identity(packet)
    assert len(task["hypotheses"]) == 2
    json.dumps(packet, allow_nan=False)


def test_request_identity_is_stable_before_display_identity_arguments_are_added():
    packet = build_experiment_packet(missing_mechanism(), context=selected_context())
    for request in packet["tasks"][0]["requests"]:
        semantic = copy.deepcopy(request)
        request_id = semantic.pop("request_id")
        semantic.pop("baseline_request_id", None)
        semantic["broker_payload"].pop("reference_experiment_request_id")
        semantic["argv"] = semantic["argv"][:-2]
        if semantic["tuning_overrides"]:
            semantic["argv"] = semantic["argv"][:-2]
        assert digest(semantic)[:32] == request_id


@pytest.mark.parametrize("flag", ["--tuning-overrides", "--reference-request"])
def test_selected_command_cannot_hide_existing_overrides_or_a_different_request_identity(flag):
    context = selected_context()
    context["capsules"][0]["selfcheck_argv"].extend([flag, "other"])
    task = build_experiment_packet(missing_mechanism(), context=context)["tasks"][0]
    assert task["status"] == "blocked" and task["requests"] == []


@pytest.mark.parametrize("selection", [dict(max_tasks=17), dict(max_variants=9), dict(max_requests=33)])
def test_producer_budget_cannot_exceed_existing_feedback_consumer_bounds(selection):
    with pytest.raises(ValueError, match="feedback bounds"):
        ExperimentBudget(**selection)


def test_tuning_intake_uses_declared_settings_and_only_explicit_mechanisms(tmp_path):
    root = tmp_path / "submission"
    root.mkdir()
    settings = root / "actual-settings.json"
    settings.write_text(json.dumps(dict(first="off", second=1)))
    declaration = root / "TUNING_KNOBS.json"
    declaration.write_text(
        json.dumps(
            dict(
                schema="merlin.tuning_knobs.v1",
                settings_file=settings.name,
                knobs=dict(
                    first=dict(values=["off", "on"], mechanism="declared.library.mechanism"),
                    second=dict(values=[1, 2], note="the name of this knob proves nothing"),
                ),
            )
        )
    )
    tuning = workspace_tuning(tmp_path)
    assert tuning["current"] == dict(first="off", second=1)
    assert tuning["domains"]["first"]["goals"] == ["declared.library.mechanism"]
    assert tuning["domains"]["second"]["goals"] == []
    assert tuning["artifacts"][1]["sha256"] == hashlib.sha256(settings.read_bytes()).hexdigest()


def tuning_tree(tmp_path, *, values=None, current=True, **declaration_changes):
    root = tmp_path / "tree"
    root.mkdir()
    declaration = dict(
        schema=tuning_knobs.SCHEMA,
        settings_file="settings.json",
        knobs=dict(choice=dict(values=values or [True, 1], mechanism="declared.mechanism", goals=[GOAL])),
    )
    declaration.update(declaration_changes)
    (root / tuning_knobs.DECLARATION).write_text(json.dumps(declaration))
    (root / "settings.json").write_text(json.dumps(dict(choice=current, unrelated="preserved")))
    return root


@pytest.mark.parametrize("path", ["../escape.json", ".hidden.json", "settings.txt", ""])
def test_tuning_intake_uses_existing_settings_path_validation(tmp_path, path):
    root = tuning_tree(tmp_path, settings_file=path)
    assert not tuning_knobs.check(root)["valid"]
    with pytest.raises(tuning_knobs.KnobError):
        workspace_tuning(root)


@pytest.mark.parametrize("part", [tuning_knobs.DECLARATION, "settings.json"])
def test_tuning_intake_refuses_symlinks_as_the_existing_compiler_does(tmp_path, part):
    root = tuning_tree(tmp_path)
    actual = tmp_path / ("actual-" + part)
    (root / part).rename(actual)
    (root / part).symlink_to(actual)
    assert not tuning_knobs.check(root)["valid"]
    with pytest.raises(tuning_knobs.KnobError, match="symlink"):
        workspace_tuning(root)


@pytest.mark.parametrize(
    "invalid", [dict(schema="unrecognized"), dict(values=list(range(tuning_knobs.MAX_VALUES + 1)))]
)
def test_tuning_intake_refuses_schema_and_bounds_rejected_by_existing_tooling(tmp_path, invalid):
    root = tuning_tree(tmp_path, **invalid)
    assert not tuning_knobs.check(root)["valid"]
    with pytest.raises(tuning_knobs.KnobError):
        workspace_tuning(root)


def test_tuning_intake_preserves_exact_bool_int_values_and_selected_goal_extension(tmp_path):
    root = tuning_tree(tmp_path)
    tuning = workspace_tuning(root)
    assert type(tuning["current"]["choice"]) is bool
    assert [type(value) for value in tuning["domains"]["choice"]["values"]] == [bool, int]
    assert tuning["domains"]["choice"]["goals"] == [GOAL]
    assert tuning["current"]["unrelated"] == "preserved"
    (root / "settings.json").write_text(json.dumps(dict(choice=1)))
    raw = json.loads((root / tuning_knobs.DECLARATION).read_text())
    raw["knobs"]["choice"]["values"] = [True]
    (root / tuning_knobs.DECLARATION).write_text(json.dumps(raw))
    with pytest.raises(tuning_knobs.KnobError, match="not a declared value"):
        workspace_tuning(root)


def test_missing_binding_is_a_gap_and_capsule_names_cannot_establish_applicability():
    context = selected_context()
    context["capsules"][0]["id"] = "looks-like-full-lhs-resident"
    context["capsules"][0]["goals"] = []
    context.pop("evaluator")
    task = build_experiment_packet(missing_mechanism(), context=context)["tasks"][0]
    assert task["status"] == "blocked" and not task["requests"]
    assert "binding.independent_evaluator" in task["next_action"]["gaps"]
    assert "binding.public_capsule_applicability" in task["next_action"]["gaps"]


def test_no_shell_commands_or_hidden_final_holdout_requests():
    context = selected_context()
    context["capsules"][0]["split"] = "held_out"
    task = build_experiment_packet(missing_mechanism(), context=context)["tasks"][0]
    assert not task["requests"]
    assert "binding.development_capsule" in task["next_action"]["gaps"]
    context["capsules"][0]["split"] = "training"
    context["capsules"][0]["selfcheck_argv"][0] = "bash"
    task = build_experiment_packet(missing_mechanism(), context=context)["tasks"][0]
    assert not task["requests"]
    assert any(g.startswith("binding.exact_selfcheck_argv") for g in task["next_action"]["gaps"])


def test_raw_self_reported_status_cannot_suppress_or_resolve_a_request():
    context = selected_context()
    original = build_experiment_packet(missing_mechanism(), context=context)
    request = original["tasks"][0]["requests"][1]
    forged = receipt(request, context, established=[GOAL])
    forged["status"] = "complete"
    packet = build_experiment_packet(missing_mechanism(), context=context, history=[forged])
    assert packet["tasks"][0]["status"] == "ready"
    assert packet["tasks"][0]["requests"][1]["status"] == "ready"


def test_trusted_same_context_regression_is_not_requested_again():
    context = selected_context()
    first = build_experiment_packet(missing_mechanism(), context=context)
    requests = first["tasks"][0]["requests"]
    results = [receipt(requests[0], context, cycles=100), receipt(requests[1], context, cycles=125, established=[GOAL])]
    context["evaluator"]["trusted_result_sha256"] = [digest(row) for row in results]
    packet = build_experiment_packet(missing_mechanism(), context=context, history=results)
    task = packet["tasks"][0]
    assert first["execution_context_identity"] == packet["execution_context_identity"]
    assert packet["identity"] != first["identity"]
    assert task["requests"][1]["request_id"] == requests[1]["request_id"]
    assert task["requests"][1]["status"] == "rejected"
    assert task["status"] == "rejected"
    assert task["mechanism_status"] == "independently_admitted"
    assert task["performance_status"] == "measured_nonimprovement"
    assert not [row for row in task["requests"] if row["status"] == "ready"]


@pytest.mark.parametrize("changed", ["compiler_sha256", "source_sha256", "environment_sha256", "reference_sha256"])
def test_changed_context_invalidates_old_attempt_suppression(changed):
    context = selected_context()
    request = build_experiment_packet(missing_mechanism(), context=context)["tasks"][0]["requests"][1]
    failed = receipt(request, context, gates={"numerical": "fail"})
    context["evaluator"]["trusted_result_sha256"] = [digest(failed)]
    context[changed] = "9" * 64
    task = build_experiment_packet(missing_mechanism(), context=context, history=[failed])["tasks"][0]
    assert task["requests"][1]["status"] == "ready"


def test_receipt_artifact_and_settings_joins_are_required_even_for_trusted_hashes():
    context = selected_context()
    request = build_experiment_packet(missing_mechanism(), context=context)["tasks"][0]["requests"][1]
    for patch in (
        dict(settings_sha256="0" * 64),
        dict(artifact_pins=[]),
        dict(program_sha256=None),
        dict(evaluator=dict(identity="other", sha256="e" * 64)),
    ):
        invalid = {**receipt(request, context, established=[GOAL]), **patch}
        context["evaluator"]["trusted_result_sha256"] = [digest(invalid)]
        task = build_experiment_packet(missing_mechanism(), context=context, history=[invalid])["tasks"][0]
        assert task["requests"][1]["status"] == "ready"


def test_static_or_provisional_numbers_never_select_a_performance_winner():
    context = selected_context()
    requests = build_experiment_packet(missing_mechanism(), context=context)["tasks"][0]["requests"]
    results = [receipt(requests[0], context), receipt(requests[1], context, cycles=1, established=[GOAL])]
    results[1]["measurement"]["kind"] = "provisional"
    context["evaluator"]["trusted_result_sha256"] = [digest(row) for row in results]
    task = build_experiment_packet(missing_mechanism(), context=context, history=results)["tasks"][0]
    assert task["status"] == "unranked"
    assert task["performance_status"] == "unknown"
    assert task["requests"][1]["status"] == "unranked"
    assert "measured_comparison" not in task["requests"][1]


def test_learning_task_references_shared_context_without_copying_tuning_or_receipt_roster():
    context = selected_context()
    context["evaluator"]["trusted_result_sha256"] = ["0" * 64] * 100
    packet = build_experiment_packet(missing_mechanism(), context=context)
    learning = packet["tasks"][0]["learning_task"]
    assert learning["context"] == dict(
        execution_context_identity=packet["execution_context_identity"],
        diagnosis="bounded paths lack complete operand retention",
        next_action="discover_or_bind_existing_library_alternatives",
    )
    assert packet["context"] == context


def test_admitted_unranked_packet_is_consumed_by_ordinary_feedback(tmp_path):
    feedback = pytest.importorskip("merlin_experiments.phase1.feedback.reference_experiments")
    context = selected_context()
    context["visibility"] = "public"
    requests = build_experiment_packet(missing_mechanism(), context=context)["tasks"][0]["requests"]
    results = [receipt(requests[0], context), receipt(requests[1], context, established=[GOAL])]
    results[1]["measurement"]["kind"] = "provisional"
    context["evaluator"]["trusted_result_sha256"] = [digest(row) for row in results]
    packet = build_experiment_packet(missing_mechanism(), context=context, history=results)
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(packet))
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            dict(
                schema=feedback.SCHEMA,
                packet=dict(path=str(packet_path), sha256=hashlib.sha256(packet_path.read_bytes()).hexdigest()),
                current_context=context,
            )
        )
    )
    with feedback.invocation(str(selection_path), hashlib.sha256(selection_path.read_bytes()).hexdigest()):
        projection = feedback.project(current_compiler_sha256=context["compiler_sha256"])
    assert projection["task_status"] == "unranked"
    assert projection["status"] == "pending_experiment"
    assert not projection["qualification_issued"]


def test_normalized_prior_regression_creates_a_task_without_fabricating_reference_mismatch():
    context = selected_context()
    history = [
        dict(
            outcome="measured_regression",
            changed_factors=[dict(name="reuse", before="reload", after="resident")],
            before=dict(settings=dict(reuse="reload")),
            after=dict(settings=dict(reuse="resident")),
            artifact_pins=[dict(path="retained/rejection.json", sha256="f" * 64)],
            measurement=dict(kind="measured", before_cycles=1067, after_cycles=1514, same_domain=True),
        )
    ]
    packet = build_experiment_packet({}, context=context, history=history)
    task = packet["tasks"][0]
    assert task["kind"] == "prior_regression" and task["goals"] == [GOAL]
    assert task["requests"][1]["status"] == "ready"  # Old scope does not suppress a new-context experiment.
    assert not task["prior_attempts"][0]["context_match"]
    assert task["evidence"][0]["value"]["measurement"]["after_cycles"] == 1514
    assert not any(row["kind"] == "scoped_mechanism_difference" for row in packet["tasks"])


def test_modeled_unchanged_control_retains_independent_policy_qualification_obligation():
    history = [dict(outcome="provisional_unchanged", next_action="qualification_needed",
                    measurement=dict(kind="provisional", before_cycles=166327, after_cycles=166327),
                    artifact_pins=[dict(path="retained/comparison.json", sha256="f" * 64)])]
    task, = build_experiment_packet({}, context=selected_context(), history=history)["tasks"]
    assert task["kind"] == "qualification_gap"
    assert task["prior_attempts"][0]["outcome"] == "provisional_unchanged"
    assert task["performance_status"] == "unknown" and task["status"] == "blocked"
    assert not task["requests"]


def test_budget_is_global_and_nonexistent_pass_is_never_inferred_from_bounded_failure():
    context = selected_context()
    context["tuning"]["domains"]["reuse"]["values"] += ["other", "third"]
    index = missing_mechanism()
    index["gaps"].append(dict(kind="adapter_gap", goals=[GOAL], reason="another diagnosed boundary"))
    packet = build_experiment_packet(
        index, context=context, budget=ExperimentBudget(max_tasks=4, max_variants=1, max_requests=3)
    )
    assert sum(row["role"] == "variant" for task in packet["tasks"] for row in task["requests"]) == 1
    assert sum(len(task["requests"]) for task in packet["tasks"]) <= 3
    assert packet["tasks"][1]["status"] == "blocked"
    assert not packet["tasks"][0]["learning_task"]["missing_pass_certainty"]


def test_duplicate_diagnoses_merge_witnesses_and_packet_hash_protects_task_content():
    index = missing_mechanism()
    index["reconstruction"] = dict(gaps=index["gaps"])
    packet = build_experiment_packet(index, context=selected_context())
    assert len(packet["tasks"]) == 1
    changed = copy.deepcopy(packet)
    changed["tasks"][0]["requests"][1]["settings"]["unrelated"] = True
    assert packet_identity(changed) != packet["identity"]


def test_private_reference_evidence_cannot_be_projected_into_public_context():
    context = selected_context()
    context["visibility"] = "public_reference"
    with pytest.raises(ValueError, match="public visibility"):
        build_experiment_packet(dict(inventory=dict(references=[dict(visibility="operator_only")])), context=context)


def test_unknown_gates_do_not_turn_missing_evidence_into_a_failed_transformation():
    context = selected_context()
    request = build_experiment_packet(missing_mechanism(), context=context)["tasks"][0]["requests"][1]
    result = receipt(request, context, gates={"numerical": "unknown"})
    context["evaluator"]["trusted_result_sha256"] = [digest(result)]
    task = build_experiment_packet(missing_mechanism(), context=context, history=[result])["tasks"][0]
    assert task["requests"][1]["status"] == "unranked"
    assert task["status"] != "rejected"


def test_unchanged_program_and_conflicting_owner_receipts_are_not_admitted():
    context = selected_context()
    requests = build_experiment_packet(missing_mechanism(), context=context)["tasks"][0]["requests"]
    baseline = receipt(requests[0], context)
    variant = receipt(requests[1], context, established=[GOAL])
    variant["program_sha256"] = baseline["program_sha256"]
    context["evaluator"]["trusted_result_sha256"] = [digest(row) for row in (baseline, variant)]
    task = build_experiment_packet(missing_mechanism(), context=context, history=[baseline, variant])["tasks"][0]
    assert task["mechanism_status"] == "unknown"
    assert "evidence.unchanged_emitted_program" in task["next_action"]["gaps"]
    other = {**variant, "program_sha256": "0" * 64}
    context["evaluator"]["trusted_result_sha256"].append(digest(other))
    task = build_experiment_packet(missing_mechanism(), context=context, history=[baseline, variant, other])["tasks"][0]
    assert "evidence.conflicting_owner_receipts" in task["next_action"]["gaps"]
    assert task["mechanism_status"] == "unknown"


def test_attribution_history_requests_owner_evidence_instead_of_another_optimization():
    prior = dict(
        outcome="unresolved",
        changed_factors=None,
        next_action="attribution_evidence_repair",
        gates=dict(attribution="unknown"),
        artifact_pins=[dict(path="owner/audit.json", sha256="0" * 64)],
    )
    task = build_experiment_packet({}, context=selected_context(), history=[prior])["tasks"][0]
    assert task["kind"] == "attribution_gap"
    assert task["goals"] == ["evidence.attribution"]
    assert task["next_action"]["action"] == "attribution_evidence_repair"
    assert not task["requests"]


def test_actual_static_analysis_and_comparison_become_scoped_evidence_tasks(tmp_path):
    analysis = pytest.importorskip("merlin_experiments.phase1.feedback.reference_analysis")
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps(dict(interfaces=[dict(name="funct_decode_table", custom_opcode=123,
        legal_funct=[1], names={"1": "load"})])))
    raw = b"00000100 <kernel>:\n 100: 0200307b .insn r 0x7b,3,1,x0,x0,x0\n 104: 00008082 ret\n"
    pin = dict(path="reference-kernels/kernel.dis", sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    ccas = []
    report = analysis._lift(raw, "disassembly", pin, facts=facts, offset=0, limit=1, ccas=ccas, candidate=False)
    report["input"] = pin  # The ordinary _analyze caller attaches its freshly read artifact pin.
    packet = build_experiment_packet(report, context=selected_context())
    task, = packet["tasks"]
    assert task["kind"] == "adapter_gap"
    assert task["goals"] == ["evidence.runtime_operand_projection"]
    assert task["evidence"][0]["scope"] == "listed_instruction_sites"
    assert task["evidence"][0]["input_pins"]["input"] == pin
    fact_sha = hashlib.sha256(facts.read_bytes()).hexdigest()
    assert task["evidence"][0]["input_pins"]["selected_facts"]["sha256"] == fact_sha
    assert task["mechanism_status"] == task["performance_status"] == "unknown"
    assert not task["requests"]
    comparison = analysis._compare(report, report, ccas[0], ccas[0])
    tasks = build_experiment_packet(comparison, context=selected_context())["tasks"]
    assert comparison["cca"]["denominator"] == 0
    assert all(row["kind"] == "missing_evidence" for row in tasks)
    assert {goal for row in tasks for goal in row["goals"]} >= {
        "evidence.cca_comparison", "source.complete_fused_region", "scheduling.schedule_consumption"}
    assert not any(row["kind"] == "scoped_mechanism_difference" for row in tasks)
    assert all(row["performance_status"] == "unknown" and not row["requests"] for row in tasks)


def test_supported_analysis_difference_can_use_only_its_declared_knob_binding():
    report = dict(kind="reference_candidate_comparison", count_basis="listed_instruction_sites",
        reference=dict(input=dict(path="reference-kernels/kernel.dis", sha256="1" * 64)),
        candidate=dict(input=dict(path="variants/trial/kernel.dis", sha256="2" * 64)),
        cca=dict(status="compared", compared_axes=[GOAL],
                 divergences=[dict(axis=GOAL, expert=True, ours=False, evidence=["scoped original witness"])],
                 unknowns=[]))
    task, = build_experiment_packet(report, context=selected_context())["tasks"]
    assert task["kind"] == "scoped_mechanism_difference" and len(task["requests"]) == 2
    assert task["evidence"][0]["status"] == "observed_structural"
    assert task["evidence"][0]["scope"] == "listed_instruction_sites"
    assert task["evidence"][0]["input_pins"]["candidate"] == report["candidate"]
    assert task["performance_status"] == task["mechanism_status"] == "unknown"
    assert task["acceptance"]["performance"]["static_counts_are_scores"] is False
    report["cca"]["compared_axes"] = []
    unknown, = build_experiment_packet(report, context=selected_context())["tasks"]
    assert unknown["kind"] == "missing_evidence" and not unknown["requests"]


@pytest.mark.parametrize("kind,fields,goal,scope", [
    ("source_syntax", dict(semantic_region_status="unknown", functions=[dict(symbol="f", loop_count=2)]),
     "source.complete_fused_region", "source_syntax"),
    ("interface_source_analysis", dict(products=2, groups=[dict(rhs="shared")]),
     "evidence.source_to_emission_binding", "declared_source_contract"),
    ("declared_semantic_region", dict(region=dict(operations=[])),
     "evidence.source_to_emission_binding", "declared_source_contract"),
    ("declared_semantic_comparison", dict(declared_contracts=dict(status="equivalent")),
     "evidence.source_to_emission_binding", "declared_source_contract"),
    ("schedule_decisions", dict(realized=False, decisions=[dict(operation="schedule.plan")]),
     "scheduling.schedule_consumption", "schedule_intent"),
])
def test_source_and_schedule_reports_do_not_imply_emitted_mechanisms(kind, fields, goal, scope):
    report = dict(kind=kind, input=dict(path="submission/input", sha256="1" * 64), **fields)
    task, = build_experiment_packet(report, context=selected_context())["tasks"]
    assert task["goals"] == [goal] and task["evidence"][0]["scope"] == scope
    assert task["evidence"][0]["input_pins"]["input"] == report["input"]
    assert task["status"] == "blocked" and not task["requests"]
    assert task["performance_status"] == task["mechanism_status"] == "unknown"


def test_actual_source_and_schedule_parsers_feed_the_existing_builder(tmp_path):
    analysis = pytest.importorskip("merlin_experiments.phase1.feedback.reference_analysis")
    source = analysis._source_summary(b"def f(x):\n    for i in range(3):\n        tile(x, i)\n",
                                      "python_schedule", 0, 1)
    source_task, = build_experiment_packet(source, context=selected_context())["tasks"]
    assert source["functions"][0]["loop_count"] == 1
    assert source_task["kind"] == "missing_evidence"
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "plan.mlir").write_text(
        'builtin.module { "schedule.plan"() <{sym_name = "candidate"}> ({}) : () -> () }')
    report, _ = analysis._analyze(dict(path="submission/plan.mlir", format="schedule_mlir"),
        workspace=tmp_path, reference_root=None, facts=None, candidate=True)
    task, = build_experiment_packet(report, context=selected_context())["tasks"]
    assert report["decisions"][0]["operation"] == "schedule.plan"
    assert task["evidence"][0]["value"] is False
    assert task["next_action"]["action"] == "consume_schedule_plan_and_reanalyze_emitted_artifacts"


def test_analysis_response_unwraps_gaps_with_original_pointers_and_no_completion_credit():
    gap = dict(kind="adapter_gap", goals=["evidence.native_issue"], reason="original_native_capture_absent")
    report = dict(kind="artifact_cross_view_analysis", inputs=dict(source=dict(sha256="1" * 64)), gaps=[gap])
    response = dict(schema="merlin.reference_analysis.v1", status="analyzed", report=report)
    task, = build_experiment_packet(response, context=selected_context())["tasks"]
    assert task["evidence"][0]["pointer"] == "/report/gaps/0"
    assert task["evidence"][0]["sha256"] == digest(gap)
    assert task["evidence"][0]["scope"] == "cross_view_diagnostic"
    assert task["mechanism_status"] == task["performance_status"] == "unknown"
    denied = dict(response, status="denied", gaps=[gap])
    assert not build_experiment_packet(denied, context=selected_context())["tasks"]
    unavailable = dict(schema=response["schema"], status="unknown", gap=gap)
    task, = build_experiment_packet(unavailable, context=selected_context())["tasks"]
    assert task["evidence"][0]["pointer"] == "/gap" and task["evidence"][0]["status"] == "unknown"


def test_static_count_delta_alone_is_not_an_optimization_task_or_a_speedup():
    report = dict(kind="reference_candidate_comparison", instruction_site_delta=-100,
        command_site_comparison=dict(load=dict(reference=200, candidate=100)),
        cca=dict(status="unknown", denominator=0, compared_axes=[], divergences=[], unknowns=[]))
    packet = build_experiment_packet(report, context=selected_context())
    assert not packet["tasks"]


def test_missing_static_projection_is_unknown_and_does_not_disappear():
    report = dict(kind="static_instruction_analysis", regions=[dict(region_id=0, listed_instructions=1)])
    task, = build_experiment_packet(report, context=selected_context())["tasks"]
    assert task["kind"] == "adapter_gap" and task["evidence"][0]["status"] == "unknown"
    assert not task["requests"]


def test_cross_view_source_mismatch_and_recorded_difference_keep_separate_obligations():
    report = dict(kind="artifact_cross_view_comparison", declared_source_join="unknown_or_mismatch",
        gaps=[dict(kind="scoped_mechanism_difference", goals=[GOAL], reason="recorded issue mixes differ")])
    tasks = build_experiment_packet(report, context=selected_context())["tasks"]
    difference = next(row for row in tasks if row["kind"] == "scoped_mechanism_difference")
    source = next(row for row in tasks if row["kind"] == "missing_evidence")
    assert difference["next_action"]["route"] == "library_parameters_or_order"
    assert source["task_id"] in difference["next_action"]["depends_on"]
    assert difference["mechanism_status"] == difference["performance_status"] == "unknown"
    assert "learning_task" not in difference
    declaration = dict(kind="declared_semantic_comparison", declared_contracts=dict(status="mismatch"))
    tasks = build_experiment_packet(declaration, context=selected_context())["tasks"]
    assert any(row["kind"] == "semantic_mismatch" for row in tasks)
