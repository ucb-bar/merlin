"""Trusted sparse native-price calibration over original candidate ELF products.

Operator APIs only: no author-selected process, source rewrite, private baseline
or empirical multiplier is accepted. Physical observations use the existing
campaign ledger and selected runtime. Each fitted model is replayed, immutable,
and locally validated before a separate drained epoch can select it.
"""
from __future__ import annotations

import copy
import itertools
import json
import math
from pathlib import Path

from merlin.runtime import _native_files as F
from merlin_timing.fitting import design_check, quantile

from . import joint_control, joint_measurements
from .feedback import joint_performance as PERF
from ..phase2.contracts import document_sha256

SCHEMA = "merlin.joint_native_calibration_round.v1"


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return F.pin(path, max_bytes=16 << 20)


def _capsule(pin):
    import yaml
    path = Path(pin["path"])
    before = F.pin(path, max_bytes=16 << 20)
    if before["sha256"] != pin["sha256"]:
        raise ValueError("frozen capsule changed")
    value = yaml.safe_load(path.read_bytes())
    if F.pin(path, max_bytes=16 << 20) != before:
        raise ValueError("capsule changed during read")
    return value


def _at(document, keys):
    for key in keys:
        document = document[key]
    return document


def _set(document, keys, value):
    parent = _at(document, keys[:-1]) if len(keys) > 1 else document
    parent[keys[-1]] = value


def freeze_round(*, selection_pin, cases, parameters, output, question, stage="bootstrap"):
    """Freeze membership/price probes before physical labels are opened.

    Each case names an original completed grading root, its frozen compiler,
    capsule declaration, result pin, source-group identity and train/held split.
    The trusted caller selects variants from its own broker product registry.
    Bounds/prices are reviewed data; geometry, ISA and topology are never fitted.
    """
    selection = PERF._read(selection_pin)
    runtime = PERF._runtime(selection["runtime"], facts_pin=selection["facts"])
    model = json.loads(runtime.model_json)
    if stage not in {"bootstrap", "refinement"} or not question.strip():
        raise ValueError("calibration needs a bounded unresolved mechanism question")
    if not 1 <= len(parameters) <= 2 or math.prod(len(p["values"]) for p in parameters) > 16:
        raise ValueError("sparse native calibration permits at most two prices and sixteen probes")
    mapped = []
    spec = PERF._read(selection["runtime"])
    for alignment_pin in [spec["facts_alignment"], *spec.get("additional_fact_alignments", [])]:
        mapped.extend(tuple(r["model_path"]) for r in PERF._read(alignment_pin)["mappings"])
    seen = set()
    for parameter in parameters:
        path, values = parameter["path"], parameter["values"]
        if (not isinstance(path, list) or not path or any(type(p) is not str for p in path)
                or tuple(path) in seen or tuple(path) in mapped or len(values) not in {2, 3, 4}
                or any(type(x) not in {int, float} or not math.isfinite(x) or x < 0 for x in values)
                or len(set(values)) != len(values) or _at(model, path) not in values
                or not parameter.get("mechanism") or not parameter.get("reviewed_scope")):
            raise ValueError("price probe lacks reviewed mechanism, bounds or immutable-facts exclusion")
        # Only service prices may vary; the operator provides the exact model path.
        if not any(k in path for k in ("prices", "latency_cycles", "occupancy_cycles")):
            raise ValueError("calibration cannot vary model topology or geometry")
        seen.add(tuple(path))
    if not 4 <= len(cases) <= 16:
        raise ValueError("sparse round needs four to sixteen predeclared variants")
    groups, ids = {}, set()
    for case in cases:
        if (not isinstance(case["id"], str) or not case["id"]
                or any(not (c.isascii() and (c.isalnum() or c in "_-")) for c in case["id"])
                or case["id"] in ids or case["split"] not in {"train", "held"}):
            raise ValueError("case identity repeats or split is unsupported")
        ids.add(case["id"])
        group = case["source_group_sha256"]
        if group in groups and groups[group] != case["split"]:
            raise ValueError("one source group leaks across training and held validation")
        groups[group] = case["split"]
        cap = _capsule(case["capsule"])
        if cap.get("label") not in {"public", "dev"} or cap.get("kind") == "model":
            raise ValueError("calibration only accepts public small capsules, never full models")
        if cap["name"] not in joint_control.selected()["policy"]["gsim"]["witness_capsules"]:
            raise ValueError("calibration member is outside selected physical witness roster")
        grade = PERF._read(case["result"])
        if (grade.get("numeric", {}).get("status") != "pass"
                or grade.get("tiers", {}).get("L2", {}).get("status") != "pass"
                or grade["tiers"]["L2"].get("engine") != "spike"):
            raise ValueError("original candidate has no complete passing Spike numerical tier")
    if any(sum(split == name for split in groups.values()) < 2 for name in ("train", "held")):
        raise ValueError("training and held validation each need two independent source groups")
    value = dict(schema=SCHEMA, selection=selection_pin, runtime=selection["runtime"],
                 cases=cases, parameters=parameters, question=question, stage=stage,
                 gates=dict(max_p95=0.30, min_rank_agreement=0.90, min_separation=0.05,
                            minimum_train_improvement=0.02, max_condition=1000),
                 scope="local native-price calibration; broader model qualification remains separate")
    from .feedback.joint_timing_selection import choose
    choices = [choose(selection, PERF.capsule_identity(PERF.selected_capsule(selection, _capsule(c["capsule"])))) for c in cases]
    if len({document_sha256(c["runtime"]) for c in choices}) != 1:
        raise ValueError("calibration round mixes current evaluator epochs")
    value["runtime"] = choices[0]["runtime"]
    # The price grid must contain the actual current evaluator, including a tuned epoch.
    current = json.loads(PERF._runtime(value["runtime"], facts_pin=selection["facts"]).model_json)
    if any(_at(current, p["path"]) not in p["values"] for p in parameters):
        raise ValueError("price grid omits the actual incumbent model")
    return _write(output, value)


def _authorize_if_needed(name, plan):
    ledger = joint_control.gsim_ledger()
    state = json.loads(ledger.read_text()) if ledger.exists() else {}
    bounds = joint_measurements.limits(joint_control.selected()["policy"])
    if len(state.get("admissions", [])) < bounds["initial_diagnostics_max"]:
        return
    if any(name in row["remaining_capsules"] for row in state.get("authorizations", [])):
        return
    stage = plan["stage"]
    consumed = sum(r["stage"] in {"initial", "bootstrap"} for r in state.get("admissions", []))
    reserved = sum(len(r["remaining_capsules"]) for r in state.get("authorizations", []) if r["stage"] == "bootstrap")
    if stage == "bootstrap" and consumed + reserved >= bounds["bootstrap_observations_max"]:
        stage = "refinement"
    joint_measurements.authorize([name], question=plan["question"], stage=stage)


def _observe(case, *, runtime, context, directory, require_complete=True):
    """Use the original Phase 2 producer verifier and namespace runner."""
    from merlin.benchharness import hash_tree
    from merlin.targetgen.elf_lanes import PACKAGE_ELF_NAME
    from merlin.targetgen.target_experiment import load_target_experiment
    from .feedback.candidate_observations import observe_generated_artifacts
    from ..phase2 import archive_count_observation as OBS, campaign as PC
    root, compiler = Path(case["generated_root"]), Path(case["compiler"])
    if hash_tree(compiler)["sha256"] != case["compiler_sha256"]:
        raise ValueError("original candidate snapshot changed")
    grade = PERF._read(case["result"])
    own = observe_generated_artifacts(root, compiler=compiler, original_grade=grade)
    if own.get("product_collection", {}).get("consumption_join") not in {
        "reopened_original_native_products", "reopened_original_retained_functional_layer_products"
    }:
        raise ValueError("candidate product provenance could not be reopened")
    elf = PERF.original_executable(root)
    policy = PC.package_sandbox_policy(load_target_experiment(context.descriptor), root, compiler)
    value = OBS.observe(elf, runtime=runtime, policy=policy, directory=directory,
                        input_sha256=F.digest(elf), remaining_s=min(120, joint_control.remaining(drain=True)),
                        numerical_pass=True)
    if value.get("status") != "observed":
        raise ValueError(value.get("reason", "native replay unavailable"))
    report = PERF._read(value["report"])
    if require_complete and (not report.get("timing_complete") or report.get("unresolved")
                             or not report.get("estimated_cycles")):
        raise ValueError("native replay has unresolved timing; cannot fit missing costs as zero")
    return elf, value, report


def _measure(case, *, backend, elf, observation, model, output, diagnostic=False):
    """Measure the identical ELF and compare every original full output value."""
    import numpy as np
    from merlin.common import invocation_record as I
    from merlin.runtime.backends.chipyard_rocc_bound import BoundRoCCBackend
    from merlin.runtime.native_runtime import active_native_runtime
    if type(backend) is not BoundRoCCBackend or active_native_runtime() is None:
        raise ValueError("calibration needs the original selected backend in its native runtime namespace")
    engine = backend._engine_identity("gsim")
    if engine.get("binary", {}).get("sha256") != model["domain"]["label_engine_sha256"]:
        raise ValueError("physical engine differs from the selected native model domain")
    active_native_runtime().execution_identity(backend, elf, capture_bytes=True)
    pin = F.pin(elf, max_bytes=64 << 20)
    process = observation["processes"][0]
    I.verify(Path(process["path"]))
    console = Path(process["path"]).parent / "stdout.bin"
    if F.digest(console) != process["stdout_sha256"]:
        raise ValueError("original numerical console changed")
    expected, _ = backend.parse_output(console.read_bytes())
    with joint_control.physical_witness(_capsule(case["capsule"])["name"]) as timeout:
        raw = backend.run_elf(elf, simulator="gsim", timeout=timeout, capture_bytes=True)
    actual, metrics = backend.parse_output(raw)
    if (not expected or set(actual) != set(expected)
            or any(not np.array_equal(actual[k], expected[k], equal_nan=True) for k in expected)):
        raise ValueError("physical full outputs differ from the original checked ELF execution")
    cycles = metrics.get("cycles")
    if type(cycles) not in {int, float} or not math.isfinite(cycles) or cycles <= 0:
        raise ValueError("physical counter-window cycles absent")
    if F.pin(elf, max_bytes=64 << 20) != pin or backend._engine_identity("gsim") != engine:
        raise ValueError("ELF or physical runtime changed")
    output.mkdir(parents=True, exist_ok=False)
    (output / "console.bin").write_bytes(raw)
    return _write(output / "measurement.json", dict(
        schema=("merlin.joint_structure_diagnostic_measurement.v1" if diagnostic
                else "merlin.joint_calibration_measurement.v1"),
        **({"fit_eligible": False, "formal_certificate": False} if diagnostic else {}),
        elf=pin, engine=engine,
        case_id=case["id"], cycles=cycles, numeric="pass", scope_binding=observation["scope_binding"],
        console=F.pin(output / "console.bin", max_bytes=64 << 20),
        label_scope="original single kernel-call counter window, exact same ELF and embedded inputs"))


def _reopen_observation(record, runtime):
    from merlin.common import invocation_record as I
    observation = record["observation"]
    report = PERF._read(observation["report"])
    elf = Path(record["elf"])
    if (F.digest(elf) != observation["elf_sha256"]
            or report.get("elf_sha256") != observation["elf_sha256"]
            or report.get("model_sha256") != F.digest(runtime.model_path)
            or report.get("scope_binding") != observation["scope_binding"]
            or not report.get("timing_complete") or report.get("unresolved")):
        raise ValueError("retained native replay lost its original identity or complete scope")
    for process in observation["processes"]:
        if F.digest(Path(process["path"])) != process["sha256"]:
            raise ValueError("retained replay invocation changed")
        actual = I.verify(Path(process["path"]))
        if actual["stdout"]["sha256"] != process["stdout_sha256"]:
            raise ValueError("retained replay console changed")
    return elf, report


def _reopen_physical(receipt, record, backend):
    if (receipt.get("fit_eligible") is False
            or receipt.get("schema") == "merlin.joint_structure_diagnostic_measurement.v1"):
        raise ValueError("structural diagnostic observations are not admitted fit labels")
    observation = record["observation"]
    raw = Path(receipt["console"]["path"])
    if (F.pin(raw, max_bytes=64 << 20) != receipt["console"]
            or receipt["elf"]["sha256"] != observation["elf_sha256"]
            or receipt["engine"] != backend._engine_identity("gsim")
            or receipt["scope_binding"] != observation["scope_binding"]):
        raise ValueError("physical receipt lost exact execution/configuration identity")
    _, metrics = backend.parse_output(raw.read_bytes())
    if metrics.get("cycles") != receipt["cycles"] or receipt["numeric"] != "pass":
        raise ValueError("physical label differs from original counter output")
    return receipt["cycles"]


def local_validation(cases, labels, incumbent, candidate, *, gates):
    """Held-only local accuracy and within-workload variant ordering; no broad grant."""
    held = [c for c in cases if c["split"] == "held"]
    errors = lambda predictions: [abs(predictions[c["id"]] / labels[c["id"]] - 1) for c in held]
    before, after = errors(incumbent), errors(candidate)
    pairs = []
    for a, b in itertools.combinations(held, 2):
        if a["capsule"]["sha256"] != b["capsule"]["sha256"]:
            continue
        x, y = labels[a["id"]], labels[b["id"]]
        if abs(x-y) / max(x,y) <= gates["min_separation"]:
            continue
        px, py = candidate[a["id"]], candidate[b["id"]]
        pairs.append((x-y)*(px-py) > 0)
    p95, old_p95 = quantile(after, .95), quantile(before, .95)
    agreement = sum(pairs)/len(pairs) if pairs else None
    admitted = (p95 <= gates["max_p95"] and p95 <= old_p95
                and sum(after) <= sum(before) and agreement is not None
                and agreement >= gates["min_rank_agreement"])
    return dict(status="limited_calibrated_provisional" if admitted else "not_promoted",
                admitted=admitted, p95=p95, incumbent_p95=old_p95,
                materially_separated_pairs=len(pairs), rank_agreement=agreement,
                timing_qualified=False, broad_qualification=False,
                reason="local held-source accuracy and variant ordering only; full qualification gates unchanged")


def _collect_labels(cases, *, root, plan, backend, base_model, labels):
    for case in cases:
        record = json.loads((root / "probe-0" / (case["id"] + ".json")).read_text())
        measured = root / "physical" / case["id"] / "measurement.json"
        if not measured.exists():
            from .joint_calibration_admission import assert_allowed
            assert_allowed(plan)
            name = _capsule(case["capsule"])["name"]
            _authorize_if_needed(name, plan)
            _measure(case, backend=backend, elf=Path(record["elf"]),
                     observation=record["observation"], model=base_model, output=measured.parent)
        labels[case["id"]] = _reopen_physical(json.loads(measured.read_text()), record, backend)


def run_round(plan_pin, *, context, backend, output):
    """Execute a frozen round; retain every native replay and physical observation.

    Root's trusted service calls this after enough legal, numerically passing
    variants exist. Repeated calls reuse only this exact round's pinned completed
    receipts. Model promotion is a separate drained epoch, never an in-place edit.
    """
    plan = PERF._read(plan_pin)
    if plan.get("schema") != SCHEMA:
        raise ValueError("unsupported calibration plan")
    # Denial also covers retained labels and a previously completed selection.
    # Run before native sensitivity work or reopening any terminal result.
    from .joint_calibration_admission import assert_allowed
    assert_allowed(plan)
    root = Path(output);root.mkdir(parents=True, exist_ok=True)
    state = root / "plan.json"
    if state.exists() and json.loads(state.read_text()) != plan:
        raise ValueError("calibration round cannot change after observations")
    if not state.exists():_write(state, plan)
    for terminal in (root / "selection.json", root / "outcome.json"):
        if terminal.exists() and (terminal.name == "selection.json" or not json.loads(terminal.read_text()).get("admitted")):
            return F.pin(terminal, max_bytes=16 << 20)
    selection = PERF._read(plan["selection"])
    base_spec = PERF._read(plan["runtime"])
    base_runtime = PERF._runtime(plan["runtime"], facts_pin=selection["facts"])
    base_model = json.loads(base_runtime.model_json)
    cases = plan["cases"]
    paths = [p["path"] for p in plan["parameters"]]
    baseline = tuple(_at(base_model, p) for p in paths)
    grid = [baseline, *(v for v in itertools.product(*(p["values"] for p in plan["parameters"])) if v != baseline)]
    predictions, reports, runtime_pins, labels = [], [], [], {}
    for index, values in enumerate(grid):
        probe = root / f"probe-{index}"
        if not probe.exists():
            probe.mkdir()
            model = copy.deepcopy(base_model)
            for path,value in zip(paths,values):_set(model,path,value)
            mp = _write(probe / "model.json", model)
            spec = copy.deepcopy(base_spec);spec.update(model=mp["path"],model_sha256=mp["sha256"],review_id=f"sparse-native-{document_sha256(plan)}-{index}")
            _write(probe / "runtime.json", spec)
        reopened_model = json.loads((probe / "model.json").read_text())
        expected_model = copy.deepcopy(base_model)
        for path, value in zip(paths, values):
            _set(expected_model, path, value)
        if reopened_model != expected_model:
            raise ValueError("retained native probe changed its reviewed price point")
        rp = F.pin(probe / "runtime.json", max_bytes=16 << 20)
        runtime = PERF._runtime(rp, facts_pin=selection["facts"])
        pred, seen_reports = {}, {}
        # Held labels and predictions are opened only after selecting train winner below.
        for case in [c for c in cases if c["split"] == "train"] if index else cases:
            artifact = probe / (case["id"] + ".json")
            if artifact.exists():
                record=json.loads(artifact.read_text());elf,report=_reopen_observation(record,runtime)
            else:
                directory=Path(case["generated_root"])/"joint_calibration"/document_sha256(plan)/str(index)
                directory.parent.mkdir(parents=True,exist_ok=True)
                elf, observation, report=_observe(case,runtime=runtime,context=context,directory=directory)
                record=dict(elf=str(elf),observation=observation);_write(artifact,record)
            pred[case["id"]]=report["estimated_cycles"];seen_reports[case["id"]]=report
        predictions.append(pred);reports.append(seen_reports);runtime_pins.append(rp)
    train=[c["id"] for c in cases if c["split"]=="train"]
    sensitivity=[]
    for path_index in range(len(paths)):
        probes = [i for i,v in enumerate(grid)
                  if sum(a!=b for a,b in zip(v,baseline))==1 and v[path_index]!=baseline[path_index]]
        differences = [[(predictions[i][key]-predictions[0][key])/(grid[i][path_index]-baseline[path_index])
                        for key in train] for i in probes]
        # Resource overlap may hide one side of a price perturbation. Use the
        # strongest reviewed native-only sensitivity, without opening labels.
        sensitivity.append(max(differences, key=lambda column: sum(x*x for x in column)))
    design=design_check([list(row) for row in zip(*sensitivity)],maximum_condition=plan["gates"]["max_condition"])
    if not design["identifiable"]:
        return _write(root/"outcome.json",dict(status="not_identifiable",design=design,plan=plan_pin))
    _collect_labels([c for c in cases if c["split"]=="train"],root=root,plan=plan,
                    backend=backend,base_model=base_model,labels=labels)
    loss=lambda pred:sum((pred[k]/labels[k]-1)**2 for k in train)/len(train)
    best=min(range(len(predictions)),key=lambda i:(loss(predictions[i]),i))
    improvement=loss(predictions[0])-loss(predictions[best])
    if best==0 or improvement < plan["gates"]["minimum_train_improvement"]*max(loss(predictions[0]),1e-12):
        return _write(root/"outcome.json",dict(status="no_meaningful_training_improvement",design=design,plan=plan_pin))
    runtime=PERF._runtime(runtime_pins[best],facts_pin=selection["facts"])
    for case in [c for c in cases if c["split"]=="held"]:
        artifact = root/f"probe-{best}"/(case["id"]+".json")
        if artifact.exists():
            _,report=_reopen_observation(json.loads(artifact.read_text()),runtime)
        else:
            directory=Path(case["generated_root"])/"joint_calibration"/document_sha256(plan)/str(best)
            elf,observation,report=_observe(case,runtime=runtime,context=context,directory=directory)
            _write(artifact,dict(elf=str(elf),observation=observation))
        predictions[best][case["id"]]=report["estimated_cycles"];reports[best][case["id"]]=report
    _collect_labels([c for c in cases if c["split"]=="held"],root=root,plan=plan,
                    backend=backend,base_model=base_model,labels=labels)
    verdict=local_validation(cases,labels,predictions[0],predictions[best],gates=plan["gates"])
    outcome=dict(verdict,design=design,training_loss=loss(predictions[best]),incumbent_training_loss=loss(predictions[0]),
                 selected_runtime=runtime_pins[best],plan=plan_pin,parameters=dict(zip((str(p) for p in paths),grid[best])))
    evidence=(F.pin(root/"outcome.json", max_bytes=16 << 20)
              if (root/"outcome.json").exists() else _write(root/"outcome.json",outcome))
    if verdict["admitted"]:
        selected_reports=list(reports[best].values());features=set.intersection(*(set(r["feature_counts"]) for r in selected_reports))
        coverage=dict(scope_sha256=base_model["domain"]["scope_sha256"],memory_regime_sha256=base_model["domain"]["memory_regime_sha256"],
                      capsule_sha256=sorted({PERF.capsule_identity(PERF.selected_capsule(selection, _capsule(c["capsule"]))) for c in cases}),
                      feature_ranges={k:[min(r["feature_counts"][k] for r in selected_reports),max(r["feature_counts"][k] for r in selected_reports)] for k in sorted(features)})
        new=copy.deepcopy(selection);epochs=new.setdefault("timing_epochs",[])
        epochs.append(dict(epoch=1+max((e["epoch"] for e in epochs),default=0),mode="tuned",qualification="limited_calibrated_provisional",runtime=runtime_pins[best],calibration_evidence=evidence,coverage=coverage))
        return _write(root/"selection.json",new)
    return evidence


def request_promotion(outcome_pin, *, previous_run, review_id, output):
    """Request a native boundary drain only after this producer's accepted round."""
    selection = PERF._read(outcome_pin)
    epochs = selection.get("timing_epochs", [])
    if not epochs:
        raise ValueError("calibration produced no model selection")
    evidence = PERF._read(epochs[-1]["calibration_evidence"])
    if evidence.get("status") != "limited_calibrated_provisional" or evidence.get("admitted") is not True:
        raise ValueError("local calibration did not admit promotion")
    from .joint_epochs import request_model_epoch
    return request_model_epoch(previous_run, selection_path=outcome_pin["path"],
                               expected_sha256=outcome_pin["sha256"], review_id=review_id, output=output)


def select_variant_cases(rows, *, maximum=8):
    """Choose uncertain same-source variant pairs, keeping source groups disjoint.

    Input rows come from the trusted completed-request registry. Unknown timing,
    uncovered regimes and overlapping candidate intervals have priority. Selection
    never reads physical labels; deterministic group splitting precedes them.
    """
    if maximum != 8:
        raise ValueError("initial sparse selection is four source groups with two variants each")
    groups = {}
    for row in rows:
        groups.setdefault(row["source_group_sha256"], []).append(row)
    pairs = []
    for group, members in groups.items():
        choices = []
        for a, b in itertools.combinations(members, 2):
            if (a["capsule"]["sha256"] != b["capsule"]["sha256"]
                    or a["compiler_sha256"] == b["compiler_sha256"]):
                continue
            score = sum(r.get("estimate_status") != "PROVISIONAL" for r in (a,b))
            score += sum(r.get("outside_calibrated_coverage") is True for r in (a,b))
            intervals = [r.get("estimated_interval") for r in (a,b)]
            if all(isinstance(v,list) and len(v)==2 for v in intervals):
                score += max(v[0] for v in intervals) <= min(v[1] for v in intervals)
            choices.append((-score,a["id"],b["id"],a,b))
        if choices:
            chosen=min(choices,key=lambda r:r[:3]);pairs.append((chosen[0],group,chosen[-2:]))
    if len(pairs)<4:
        return dict(status="waiting_for_independent_variants",available_source_groups=len(pairs),cases=[])
    chosen=sorted(pairs,key=lambda r:r[:2])[:4]
    chosen.sort(key=lambda r:document_sha256(r[1]))
    cases=[]
    for index,(_,_,pair) in enumerate(chosen):
        for row in pair:
            case={k:v for k,v in row.items() if k not in {"estimate_status","outside_calibrated_coverage","estimated_interval"}}
            case["split"]="train" if index<2 else "held"
            cases.append(case)
    return dict(status="selected",cases=cases,physical_measurements="only after native identifiability, held only after training improvement")
