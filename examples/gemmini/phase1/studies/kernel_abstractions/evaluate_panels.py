"""Operator-only evaluation plans and complete-cell collection; never starts a job.

The emitted commands call the existing study runner's ordinary selfcheck evaluator.
Heldout inputs and frozen compiler selections stay in this private output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from merlin_experiments.phase1.feedback.joint_performance_report import compare

from merlin.common.tree_hash import hash_tree

PLAN = "private.kernel_abstractions.evaluation_plan.v1"
FROZEN = "private.kernel_abstractions.frozen_sources.v1"


def _write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _pin(path):
    path = Path(path).absolute()
    if path.is_symlink() or path != path.resolve() or not path.is_file():
        raise ValueError(f"input must be a direct canonical file: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(path=str(path), sha256=digest)


def _read(path, expected=None):
    pin = _pin(path)
    if expected is not None and pin["sha256"] != expected:
        raise ValueError(f"selected bytes changed: {path}")
    value = json.loads(Path(path).read_bytes())
    if _pin(path) != pin:
        raise ValueError("input changed while reading")
    return value


def _tree(path):
    path = Path(path).absolute()
    if not path.is_dir() or path != path.resolve() or any(p.is_symlink() for p in path.rglob("*")):
        raise ValueError("frozen compiler must be a canonical directory without links")
    value = hash_tree(path)
    if not value["n_files"]:
        raise ValueError("frozen compiler is empty")
    return dict(path=str(path), **value)


def _reopen(pin, *, tree=False):
    actual = _tree(pin["path"]) if tree else _pin(pin["path"])
    if actual != pin:
        raise ValueError(f"frozen input changed: {pin['path']}")


def _frozen_source(value):
    """A plain compiler path does not silently freeze RULES or tuning settings.

    An operator may instead select {path, rules:{path,sha256},
    settings:{path,sha256}, frozen_unix}. These bytes remain separate pins;
    freezing their contents is not a claim that the rules generalize.
    """
    if isinstance(value, str):
        return dict(
            source=_tree(value),
            freeze=dict(
                status="UNKNOWN",
                requirements_complete=False,
                reason="RULES, selected settings and freeze timestamp were not independently pinned",
            ),
        )
    if (
        not isinstance(value, dict)
        or set(value) != {"path", "rules", "settings", "frozen_unix"}
        or not _positive(value["frozen_unix"])
    ):
        raise ValueError("freeze selection requires compiler, RULES, settings and recorded timestamp")
    metadata = {}
    for key in ("rules", "settings"):
        selected = {field: value[key][field] for field in ("path", "sha256")}
        _reopen(selected)
        metadata[key] = selected
    return dict(
        source=_tree(value["path"]),
        freeze=dict(
            status="pinned",
            requirements_complete=True,
            frozen_unix=value["frozen_unix"],
            **metadata,
            qualification_issued=False,
        ),
    )


def _context(config, env):
    """Common selected execution inputs, apart from run-local scratch paths.

    Actual model comparisons still require the observer's exact scope/domain keys;
    this preparation identity does not establish measured equivalence.
    """
    keys = ("MERLIN_JOINT_PERFORMANCE_SELECTION", "MERLIN_TARGET_CONTRACT")
    selected = {}
    for key in keys:
        value = env.get(key)
        if not value:
            raise ValueError(f"evaluation requires explicit {key}")
        if key.endswith("SELECTION"):
            value = json.loads(value)
            _read(value["path"], value["sha256"])
            selected[key] = value["sha256"]
        else:
            selected[key] = _pin(value)["sha256"]
    for key in ("framework", "python", "full_compiler", "library_contract", "read_only_dependencies"):
        selected[key] = config.get(key)
    selected["facts_sha256"] = _pin(config["facts"])["sha256"]
    return selected


def plan(runs_path, panel_path, inputs_path, frozen_path, output, *, timeout_s=900):
    """Emit ordinary evaluations with a common baseline or explicit initial baselines per arm."""
    if type(timeout_s) is not int or timeout_s <= 0:
        raise ValueError("positive per-request timeout required")
    output = Path(output).absolute()
    if output.exists():
        raise ValueError("select a new private evaluation directory")
    runs, panel, inputs, frozen = map(_read, (runs_path, panel_path, inputs_path, frozen_path))
    if frozen.get("schema") != FROZEN or not isinstance(frozen.get("subjects"), dict):
        raise ValueError("operator must explicitly select frozen compiler paths")
    if panel["inputs_manifest"]["sha256"] != _pin(inputs_path)["sha256"]:
        raise ValueError("panel belongs to different input manifest")
    if runs["inputs"]["sha256"] != _pin(inputs_path)["sha256"]:
        raise ValueError("runs belong to different input manifest")
    roster = {"train": panel["selection"]["training_case_ids"], "heldout": panel["selection"]["heldout_case_ids"]}
    if any(
        not isinstance(names, list) or not names or any(not isinstance(name, str) or not name for name in names)
        for names in roster.values()
    ):
        raise ValueError("panel requires nonempty training and heldout case lists")
    names = [name for split in roster.values() for name in split]
    if len(set(names)) != len(names):
        raise ValueError("panel cases must be distinct")
    cases = {row["case"]: row for row in inputs["cases"]}
    for split, selected in roster.items():
        for name in selected:
            if name not in cases or cases[name]["split"] != split:
                raise ValueError("panel split differs from frozen source manifest")
            _readable = _pin(cases[name]["source"]["path"])
            if _readable["sha256"] != cases[name]["source"]["sha256"]:
                raise ValueError("panel source changed")
    rows = runs["runs"]
    configs = [str(Path(row["config"]).absolute()) for row in rows]
    if not rows or len(set(configs)) != len(rows) or set(configs) != set(frozen["subjects"]):
        raise ValueError("frozen selections must cover every run exactly once")
    per_arm = "baselines" in frozen
    if per_arm:
        if (
            "baseline" in frozen
            or not isinstance(frozen["baselines"], dict)
            or set(frozen["baselines"]) != set(configs)
        ):
            raise ValueError("per-arm baseline mapping must cover every run exactly once, without a common baseline")
    elif "baseline" not in frozen:
        raise ValueError("select a common baseline or an explicit per-arm baseline mapping")
    subjects, contexts = [], []
    pins = [_pin(p) for p in (runs_path, panel_path, inputs_path, frozen_path)]
    for index, (row, config_path) in enumerate(zip(rows, configs, strict=True)):
        config = _read(config_path, row["config_sha256"])
        if config.get("training_cases") != roster["train"]:
            raise ValueError("evaluation training panel differs from the author run")
        env_path = Path(config["private"]) / "evaluator-environment.json"
        env = _read(env_path)
        contexts.append(_context(config, env))
        runner = config["runner"]
        _read_runner = _pin(runner["path"])
        if _read_runner["sha256"] != runner["sha256"]:
            raise ValueError("selected evaluator runner changed")
        pins.extend((_pin(config_path), _pin(env_path), _read_runner))
        subjects.append(
            dict(
                id=f"run-{index:02d}",
                replica=row.get("replica"),
                treatment=row.get("treatment", config.get("tooling")),
                assistance={
                    key: config.get(key) for key in ("reference_guidance", "library_guidance", "schedule_guidance")
                },
                role="final_compiler",
                run_config=_pin(config_path),
                baseline_subject=f"baseline-{index:02d}" if per_arm else "baseline",
                **_frozen_source(frozen["subjects"][config_path]),
                config=config,
                env=env,
            )
        )
    common_contexts = [
        {k: v for k, v in context.items() if not per_arm or k != "full_compiler"} for context in contexts
    ]
    if any(context != common_contexts[0] for context in common_contexts[1:]):
        raise ValueError(
            "evaluations require the same selected execution environment; only declared per-arm "
            "full_compiler selections may differ"
        )
    baselines = []
    for index, subject in enumerate(subjects if per_arm else subjects[:1]):
        value = frozen["baselines"][configs[index]] if per_arm else frozen["baseline"]
        baselines.append(
            dict(
                id=subject["baseline_subject"],
                role="initial_baseline",
                replica=subject["replica"] if per_arm else None,
                treatment="initial_selected_tool_environment" if per_arm else "common_current_environment_baseline",
                assistance={},
                run_config=subject["run_config"] if per_arm else None,
                baseline_subject=None,
                **_frozen_source(value),
                config=subject["config"],
                env=subject["env"],
            )
        )
    # Reuse the ordinary descriptor/suite validator, including full source-tree hashes.
    from merlin_experiments.corpus.public_scalability import selected as selected_suite

    from merlin.targetgen.target_experiment import load_target_experiment

    suites = {}
    for split in roster:
        suite = inputs["suites"][split]
        descriptor = suite["descriptor"]
        if _pin(descriptor["path"])["sha256"] != descriptor["sha256"]:
            raise ValueError("suite descriptor changed")
        root, record = selected_suite(load_target_experiment(descriptor["path"]))
        if str(root) != suite["root"]["path"] or not set(roster[split]) <= set(record["cases"]):
            raise ValueError("selected suite omits panel cases")
        suites[split] = dict(descriptor=_pin(descriptor["path"]), root=suite["root"])
        pins.append(suites[split]["descriptor"])
    output.mkdir(parents=True, mode=0o700)
    jobs = []
    for subject in (*baselines, *subjects):
        for split, selected in roster.items():
            directory = output / (subject["id"] + "-" + split)
            directory.mkdir(mode=0o700)
            (directory / "tmp").mkdir(mode=0o700)
            config = dict(
                subject["config"],
                private=str(directory),
                descriptor=suites[split]["descriptor"]["path"],
                suite=suites[split]["root"]["path"],
                reference_guidance="off",
                reference_map=None,
                training_cases=selected,
                inputs=str(Path(inputs_path).absolute()),
            )
            env = dict(subject["env"], TMPDIR=str(directory / "tmp"), MERLIN_TARGET_EXPERIMENT=config["descriptor"])
            for key in tuple(env):
                if key.startswith("MERLIN_REFERENCE"):
                    env.pop(key)
            _write(directory / "evaluator-environment.json", env)
            config["environment"] = env
            config_path = directory / "config.json"
            _write(config_path, config)
            result = directory / "result.json"
            argv = [
                config["python"],
                "-B",
                config["runner"]["path"],
                "evaluate",
                "--config",
                str(config_path),
                "--submission",
                subject["source"]["path"],
                "--capsules",
                ",".join(selected),
                "--out",
                str(result),
                "--timeout",
                str(timeout_s),
                "--tiers",
                "L0,L1,L2",
            ]
            jobs.append(
                dict(
                    id=directory.name,
                    subject=subject["id"],
                    split=split,
                    source=subject["source"],
                    suite_sha256=suites[split]["root"]["sha256"],
                    cases=selected,
                    config=_pin(config_path),
                    environment=_pin(directory / "evaluator-environment.json"),
                    result=str(result),
                    argv=argv,
                )
            )
    result = dict(
        schema=PLAN,
        inputs=pins,
        execution_context=common_contexts[0],
        full_compiler_selections={
            path: context["full_compiler"] for path, context in zip(configs, contexts, strict=True)
        },
        baseline_mode="per_arm" if per_arm else "common",
        baseline_bindings={path: subject["baseline_subject"] for path, subject in zip(configs, subjects, strict=True)},
        suites=suites,
        subjects=[
            {k: v for k, v in subject.items() if k not in ("config", "env")} for subject in (*baselines, *subjects)
        ],
        cases=[cases[name] for name in names],
        jobs=jobs,
        executed=False,
        baseline_reuse=(
            "each final compiler is compared with its own initial selected tool/dependency closure"
            if per_arm
            else "one common baseline across arms in this selected environment; old F6 scores are not imported"
        ),
        migration=dict(
            status="separate_review_required",
            included_in_agent_improvement=False,
            meaning="Initial baselines include selected tool migration. Their differences are not author improvements.",
        ),
        physical_timing="UNKNOWN; these commands request ordinary numeric and provisional model feedback",
    )
    _write(output / "plan.json", result)
    return result


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _issued_qualification(result_path, result, *, source_sha256, suite_sha256):
    """Reopen the existing evaluator's rich rows behind its compact public response."""
    from merlin.common import invocation_record as I

    result_path = Path(result_path)
    directory = result_path.with_suffix(".scalability")
    issued = _read(directory / "issued.json")
    issuer_path = Path(issued["issuer"]["path"])
    qualification_path = directory / "qualification.json"
    if not issuer_path.resolve().is_relative_to(result_path.parent / "issuers") or issued["result"]["path"] != str(
        qualification_path
    ):
        raise ValueError("issuer is outside this private check")
    _read(issuer_path, issued["issuer"]["sha256"])
    invocation = I.verify(issuer_path)
    qualification = _read(qualification_path, issued["result"]["sha256"])
    arguments = invocation.get("arguments", {})
    if (
        invocation.get("stage") != "public_spike_scalability"
        or arguments.get("candidate_sha256") != source_sha256
        or arguments.get("suite_sha256") != suite_sha256
        or any(
            value.get("candidate_sha256") != source_sha256 or value.get("suite_sha256") != suite_sha256
            for value in (qualification, result)
        )
        or not isinstance(qualification.get("observed"), dict)
    ):
        raise ValueError("private result does not bind the actual checked source and suite")
    return qualification, invocation, issued


def collect(plan_path, *, physical_collections=()):
    """Read ordinary evaluator responses, retaining every planned cell."""
    plan = _read(plan_path)
    if plan.get("schema") != PLAN:
        raise ValueError("unsupported evaluation plan")
    for pin in plan["inputs"]:
        _reopen(pin)
    for subject in plan["subjects"]:
        _reopen(subject["source"], tree=True)
        if (subject.get("freeze") or {}).get("requirements_complete") is True:
            for key in ("rules", "settings"):
                _reopen(subject["freeze"][key])
    cases = {row["case"]: row for row in plan["cases"]}
    subjects = {row["id"]: row for row in plan["subjects"]}
    if plan.get("baseline_mode") == "per_arm":
        expected = {
            s["run_config"]["path"]: s["baseline_subject"]
            for s in subjects.values()
            if s.get("role") == "final_compiler"
        }
        if (
            plan.get("baseline_bindings") != expected
            or len(set(expected.values())) != len(expected)
            or any(subjects.get(name, {}).get("role") != "initial_baseline" for name in expected.values())
        ):
            raise ValueError("per-arm baseline mapping must retain each selected initial dependency closure")
    cells = []
    for job in plan["jobs"]:
        _reopen(job["config"])
        _reopen(job["environment"])
        response, error, response_pin, qualification_pin = None, None, None, None
        if Path(job["result"]).exists():
            try:
                response = _read(job["result"])
                response_pin = _pin(job["result"])
                if not isinstance(response, dict):
                    raise ValueError("evaluator response must be a mapping")
                observed = response.get("cases", [])
                if not isinstance(observed, list) or any(
                    not isinstance(row, dict)
                    or not isinstance(row.get("capsule"), str)
                    or any(
                        row.get(key) is not None and not isinstance(row[key], dict)
                        for key in ("numeric", "tiers", "performance", "failure")
                    )
                    for row in observed
                ):
                    raise ValueError("malformed evaluator case records")
                observed_names = [row.get("capsule") for row in observed]
                if (
                    response.get("candidate_sha256") != job["source"]["sha256"]
                    or response.get("suite_sha256") != job["suite_sha256"]
                    or set(response.get("requested_cases", [])) != set(job["cases"])
                    or len(set(observed_names)) != len(observed_names)
                    or not set(observed_names) <= set(job["cases"])
                ):
                    raise ValueError("result source, suite or case identity differs")
                if any(
                    row.get("status") == "pass"
                    and (not isinstance((row.get("tiers") or {}).get("L2"), dict) or not row.get("numeric"))
                    for row in observed
                ):
                    qualification, _, issued = _issued_qualification(
                        job["result"], response, source_sha256=job["source"]["sha256"], suite_sha256=job["suite_sha256"]
                    )
                    qualified = qualification["observed"]
                    if not set(observed_names) <= set(qualified) or any(
                        qualified[row["capsule"]].get("status") != row.get("status") for row in observed
                    ):
                        raise ValueError("public result differs from its issued qualification")
                    response["cases"] = [
                        dict(
                            row, **{key: qualified[row["capsule"]].get(key) for key in ("numeric", "tiers", "failure")}
                        )
                        for row in observed
                    ]
                    qualification_pin = issued["result"]
            except (OSError, ValueError, TypeError, KeyError) as exc:
                response, error = None, str(exc)
        observed = {row["capsule"]: row for row in (response or {}).get("cases", [])}
        for name in job["cases"]:
            row = observed.get(name)
            state = "invalid_result" if error else "missing"
            numeric, performance = None, None
            if row is not None:
                numeric = row.get("numeric") or {}
                l2 = (row.get("tiers") or {}).get("L2") or {}
                if not isinstance(l2, dict):
                    l2 = {}
                passed = _numeric_pass(row)
                state = (
                    "pass"
                    if passed
                    else (
                        "wrong"
                        if numeric.get("status") == "fail"
                        else "failed"
                        if row.get("status") == "fail"
                        else "ungraded"
                    )
                )
                failure = row.get("failure") or {}
                if any(
                    status in ("timeout", "timed_out")
                    for status in (row.get("status"), numeric.get("status"), l2.get("status"), failure.get("category"))
                ):
                    state = "timeout"
                performance = row.get("performance")
            cells.append(
                dict(
                    subject=job["subject"],
                    split=job["split"],
                    family=cases[name]["family"],
                    case=name,
                    treatment=subjects[job["subject"]]["treatment"],
                    replica=subjects[job["subject"]]["replica"],
                    role=subjects[job["subject"]].get(
                        "role", "initial_baseline" if job["subject"] == "baseline" else "final_compiler"
                    ),
                    baseline_subject=subjects[job["subject"]].get("baseline_subject", "baseline"),
                    shape_contract=cases[name]["contract"],
                    source=job["source"],
                    freeze=subjects[job["subject"]].get("freeze"),
                    state=state,
                    numeric=numeric,
                    performance=performance,
                    failure=(row or {}).get("failure"),
                    diagnostic=error,
                    result=response_pin,
                    qualification=qualification_pin,
                    model_comparison=None,
                    physical=dict(status="UNKNOWN", cycles=None, reason="no physical measurement selected"),
                )
            )
    baseline_ids = {
        subject["id"]
        for subject in subjects.values()
        if subject.get("role") == "initial_baseline" or subject["id"] == "baseline"
    }
    baselines = {(row["subject"], row["split"], row["case"]): row for row in cells if row["subject"] in baseline_ids}
    for cell in cells:
        if cell["subject"] in baseline_ids:
            continue
        key = (cell["baseline_subject"], cell["split"], cell["case"])
        if key not in baselines:
            raise ValueError("evaluation plan omits the selected initial baseline cell")
        baseline = baselines[key]
        cell["model_comparison"] = (
            compare(baseline.get("performance") or {}, cell.get("performance") or {})
            if baseline["state"] == cell["state"] == "pass"
            else dict(
                status="UNKNOWN", candidate_to_baseline_ratio=None, reason="both complete numerical cells must pass"
            )
        )
    summaries = []
    groups = sorted({(c["subject"], c["split"], c["family"]) for c in cells})
    for subject, split, family in groups:
        selected = [c for c in cells if (c["subject"], c["split"], c["family"]) == (subject, split, family)]
        ratios = [(c.get("model_comparison") or {}).get("candidate_to_baseline_ratio") for c in selected]
        complete = all(c["state"] == "pass" and _positive(r) for c, r in zip(selected, ratios, strict=True))
        summaries.append(
            dict(
                subject=subject,
                split=split,
                family=family,
                cells=len(selected),
                passed=sum(c["state"] == "pass" for c in selected),
                states={
                    state: sum(c["state"] == state for c in selected)
                    for state in sorted({c["state"] for c in selected})
                },
                unranked=sum(not _positive(r) for r in ratios),
                model_candidate_to_baseline_geomean=math.exp(sum(math.log(r) for r in ratios) / len(ratios))
                if complete
                else None,
                aggregate_status="PROVISIONAL" if complete else "UNKNOWN",
                generalization_claim_status=(
                    "separate_generalization_review_required"
                    if (subjects[subject].get("freeze") or {}).get("requirements_complete") is True
                    else "UNKNOWN_missing_RULES_settings_freeze"
                )
                if split == "heldout" and subject not in baseline_ids
                else "not_applicable",
                physical_status="UNKNOWN",
            )
        )
    collection = dict(
        schema="private.kernel_abstractions.evaluation_collection.v1",
        plan=_pin(plan_path),
        baseline_mode=plan.get("baseline_mode", "common"),
        baseline_bindings=plan.get("baseline_bindings", {}),
        migration=plan.get("migration"),
        cells=cells,
        summaries=summaries,
        physical_timing_qualified=False,
        interpretation=(
            "missing, failed, stale and timeout observations never become zero-cycle speedups; "
            "incomplete groups remain unranked"
        ),
    )

    if physical_collections:
        from physical_panel import apply_physical

        return apply_physical(collection, physical_collections)
    return collection


def _numeric_pass(row):
    l2 = (row.get("tiers") or {}).get("L2") or {}
    return (
        isinstance(l2, dict)
        and row.get("status") == "pass"
        and (row.get("numeric") or {}).get("status") == "pass"
        and l2.get("status") == "pass"
        and l2.get("engine") == "spike"
        and l2.get("fidelity") == "functional_model"
        and l2.get("derived_from_rtl") is False
    )


def _provider_usage(private):
    """Only terminal public usage counters; never retain message/tool/reasoning text."""
    files, observations = [], []
    for path in sorted(private.glob("author-*.jsonl")):
        before, first = _pin(path), len(observations)
        with path.open() as stream:
            for number, line in enumerate(stream, 1):
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(event, dict):
                    continue
                kind = event.get("type")
                if kind not in ("turn.completed", "result") or not isinstance(event.get("usage"), dict):
                    continue
                usage = event["usage"]
                counters = {
                    key: value
                    for key, value in usage.items()
                    if key
                    in (
                        "input_tokens",
                        "output_tokens",
                        "cached_input_tokens",
                        "cache_read_input_tokens",
                        "cache_creation_input_tokens",
                    )
                    and type(value) is int
                    and value >= 0
                }
                if counters:
                    observations.append(dict(file=path.name, line=number, format=kind, counters=counters))
        after = _pin(path)
        files.append(dict(after, stable_read=after == before))
        if before != after:
            del observations[first:]  # A growing live log has no exact usage snapshot yet.
    totals = {}
    for row in observations:
        group = totals.setdefault(row["format"], {})
        for key, value in row["counters"].items():
            group[key] = group.get(key, 0) + value
    return dict(
        status="reported" if observations else "UNKNOWN",
        files=files,
        observations=observations,
        totals_by_format=totals or None,
        convention="provider-reported counters; cache inclusion differs by provider; missing usage is not zero",
    )


def _progress_request(private, channel, rid, allowed):
    """Join existing private issued results to their public request publication."""
    job = private / "checks" / rid
    paths = {
        "request": channel / ("req_" + rid + ".json"),
        "response": channel / ("resp_" + rid + ".json"),
        "done": channel / ("done_" + rid),
        "result": job / "result.json",
        "worker_log": job / "worker.log",
    }
    files = {key: _pin(path) for key, path in paths.items() if path.is_file()}
    request = _read(paths["request"]) if "request" in files else {}
    response = _read(paths["response"]) if "response" in files else {}
    row = dict(
        request_id=rid,
        files=files,
        action=request.get("action", "selfcheck"),
        status=response.get("status", "pending"),
        source=None,
        requested_source_sha256=request.get("submission_sha256"),
        evaluation_wall_s=None,
        finished_unix=None,
        accepted_numeric_cases=[],
        cases=[],
        failure=response.get("error"),
        physical_timing="UNKNOWN",
    )
    if (job / "submission").is_dir():
        row["source"] = _tree(job / "submission")
    if "result" not in files:
        return row
    try:
        result = _read(paths["result"])
        snapshot = _tree(job / "submission")
        qualification, invocation, issued = _issued_qualification(
            paths["result"], result, source_sha256=snapshot["sha256"], suite_sha256=result.get("suite_sha256")
        )
        row.update(source=snapshot, issuer=issued["issuer"], qualification=issued["result"])
        started, finished = invocation.get("started_ns"), invocation.get("finished_ns")
        if type(started) is int and type(finished) is int and 0 < started <= finished:
            row.update(evaluation_wall_s=(finished - started) / 1e9, finished_unix=finished / 1e9)
        requested = request.get("submission_sha256")
        if (
            not requested
            or "done" not in files
            or response.get("selfcheck_request_id") != rid
            or response.get("requested_submission_sha256", response.get("submission_sha256")) != requested
            or response.get("submission_sha256") != requested
            or response.get("candidate_sha256") != snapshot["sha256"]
        ):
            raise ValueError("stale or incomplete public response/source join")
        names = request.get("capsules", "").split(",")
        observed = qualification.get("observed", {})
        if (
            not isinstance(observed, dict)
            or not names
            or len(set(names)) != len(names)
            or not set(names) <= set(allowed)
            or set(observed) != set(names)
            or set(result.get("requested_cases", [])) != set(names)
        ):
            raise ValueError("checked cases differ from this arm's requested training roster")
        row.update(
            status="verified_private_result",
            accepted_numeric_cases=[name for name in names if _numeric_pass(observed[name])],
            cases=[
                dict(
                    case=name,
                    numeric_pass=_numeric_pass(observed[name]),
                    status=observed[name].get("status"),
                    numeric=observed[name].get("numeric"),
                    failure=observed[name].get("failure"),
                    performance=observed[name].get("performance"),
                )
                for name in names
            ],
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        row.update(status="unverified_result", failure=str(error))
    return row


def progress(runs_path, *, baseline_collection=None, launch_audit=None):
    """Retrospective agent benefit from existing owner records; no evaluator runs."""
    runs = _read(runs_path)
    audit = _read(launch_audit) if launch_audit is not None else {"runs": []}
    if launch_audit is not None and audit.get("schema") != "private.kernel_abstractions.launch_audit.v1":
        raise ValueError("unsupported operator launch audit")
    starts = {str(Path(row["config"]).absolute()): row.get("started_unix") for row in audit["runs"]}
    if len(starts) != len(audit["runs"]):
        raise ValueError("ambiguous launch times for the same run")
    baseline = _read(baseline_collection) if baseline_collection is not None else {"cells": []}
    if baseline_collection is not None:
        if baseline.get("schema") != "private.kernel_abstractions.evaluation_collection.v1":
            raise ValueError("baseline must be an existing panel collection")
        _reopen(baseline["plan"])
    base = {}
    for cell in baseline["cells"]:
        if (cell.get("role") == "initial_baseline" or cell["subject"] == "baseline") and cell["state"] == "pass":
            _reopen(cell["result"])
            _reopen(cell["source"], tree=True)
            base[cell["subject"], cell["case"]] = cell
    reports = []
    for number, run in enumerate(runs["runs"]):
        config_path = str(Path(run["config"]).absolute())
        if baseline.get("baseline_mode") == "per_arm" and config_path not in baseline.get("baseline_bindings", {}):
            raise ValueError("per-arm baseline mapping omits the run configuration")
        baseline_id = baseline.get("baseline_bindings", {}).get(config_path, "baseline")
        config = _read(config_path, run["config_sha256"])
        private, channel = Path(config["private"]), Path(config["workspace"]) / ".qa_channel"
        status_path = private / "status.json"
        status = _read(status_path) if status_path.is_file() else {}
        start = starts.get(config_path, status.get("started_unix"))
        start = start if _positive(start) else None
        identities = {p.name[4:-5] for p in channel.glob("req_*.json")}
        identities.update(p.name[5:-5] for p in channel.glob("resp_*.json"))
        if (private / "checks").is_dir():
            identities.update(p.name for p in (private / "checks").iterdir() if p.is_dir())
        requests = []
        for rid in sorted(identities):
            try:
                requests.append(_progress_request(private, channel, rid, config["training_cases"]))
            except (OSError, ValueError, KeyError, TypeError) as error:
                requests.append(dict(request_id=rid, status="unreadable", failure=str(error), cases=[]))
        trajectories = []
        for name in config["training_cases"]:
            observations = []
            for request in requests:
                for case in request.get("cases", []):
                    baseline_case = base.get((baseline_id, name))
                    if case["case"] != name or not case["numeric_pass"] or baseline_case is None:
                        continue
                    comparison = compare(baseline_case.get("performance") or {}, case.get("performance") or {})
                    ratio = comparison.get("candidate_to_baseline_ratio")
                    if _positive(ratio):
                        finished = request.get("finished_unix")
                        observations.append(
                            dict(
                                request_id=request["request_id"],
                                source=request["source"],
                                finished_unix=finished,
                                elapsed_author_s=finished - start
                                if start is not None and finished is not None and finished >= start
                                else None,
                                candidate_to_baseline_ratio=ratio,
                                comparison=comparison,
                            )
                        )
            ordered = sorted(
                (
                    point
                    for point in observations
                    if point["finished_unix"] is not None and (start is None or point["finished_unix"] >= start)
                ),
                key=lambda point: (point["finished_unix"], point["request_id"]),
            )
            best, trajectory = 1.0, []
            for point in ordered:
                if point["candidate_to_baseline_ratio"] < best:
                    trajectory.append(point)
                    best = point["candidate_to_baseline_ratio"]
            first = trajectory[0] if trajectory else None
            trajectories.append(
                dict(
                    case=name,
                    accepted_model_observations=observations,
                    best_supported_model_trajectory=trajectory,
                    first_correct_improvement=first,
                    time_to_first_correct_improvement_s=first["elapsed_author_s"] if first else None,
                    timing_status="retrospective_provisional_model"
                    if first and first["elapsed_author_s"] is not None
                    else "UNKNOWN",
                    physical_confirmation="UNKNOWN",
                )
            )
        reports.append(
            dict(
                id=f"run-{number:02d}",
                replica=run.get("replica"),
                treatment=run.get("treatment", config.get("tooling")),
                config=_pin(config_path),
                baseline_subject=baseline_id,
                status=status,
                status_pin=_pin(status_path) if status_path.is_file() else None,
                started_unix=start,
                start_authority="operator_launch_audit"
                if config_path in starts and start is not None
                else "private_status"
                if start is not None
                else "UNKNOWN",
                requests=requests,
                trajectories=trajectories,
                usage=_provider_usage(private),
            )
        )
    return dict(
        schema="private.kernel_abstractions.agent_progress.v1",
        runs=_pin(runs_path),
        baseline_collection=_pin(baseline_collection) if baseline_collection else None,
        launch_audit=_pin(launch_audit) if launch_audit else None,
        reports=reports,
        interpretation=(
            "Correct implementation checks alone are not improvements. Ratios require an independent correct "
            "selected initial baseline and exact provisional model domain. Tool migration is separate from author "
            "improvement. Missing start/usage/physical evidence remains UNKNOWN."
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("plan")
    for key in ("runs", "panel", "inputs", "frozen", "out"):
        prepare.add_argument("--" + key, type=Path, required=True)
    prepare.add_argument("--timeout", type=int, default=900)
    finish = sub.add_parser("collect")
    finish.add_argument("--plan", type=Path, required=True)
    finish.add_argument("--out", type=Path, required=True)
    finish.add_argument("--physical-collection", type=Path, action="append", default=[],
                        help="private exact-custody physical collection; repeat for multiple images")
    agent = sub.add_parser("progress")
    agent.add_argument("--runs", type=Path, required=True)
    agent.add_argument("--baseline-collection", type=Path)
    agent.add_argument("--launch-audit", type=Path)
    agent.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "plan":
        result = plan(args.runs, args.panel, args.inputs, args.frozen, args.out, timeout_s=args.timeout)
        print(json.dumps(dict(jobs=len(result["jobs"]), executed=False, plan=str(args.out / "plan.json"))))
    else:
        if args.out.exists():
            raise ValueError("collection output already exists")
        _write(
            args.out,
            collect(args.plan, physical_collections=args.physical_collection)
            if args.action == "collect"
            else progress(args.runs, baseline_collection=args.baseline_collection, launch_audit=args.launch_audit),
        )


if __name__ == "__main__":
    main()
