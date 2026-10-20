"""Issue new canonical static checks of unchanged completed compiler products.

Only the fixed trusted instruction-checker preparation may build code. All
candidate/model/device compilation stays disabled. Numerical execution is a
separate obligation; an interrupted historical producer is never completed here.
Run this operation in the ordinary owned process supervisor for descendant cleanup.
"""

from __future__ import annotations

import json
import math
import signal
import threading
import time
from pathlib import Path

import yaml

from merlin.common import invocation_record as IR
from merlin.compile.model_execution_inputs import file_sha256, strict_tree_sha256

from . import private_linked_source_preparation as preparation
from . import private_static_products as products_api

SCHEMA = "merlin.phase1.native_static_product_revalidation.v1"
SCOPE = "new_static_checks_of_retained_products"
STAGE = "native_static_product_revalidation"
TRUSTED_CHECKER_TIMEOUT_S = 120


def prepare_instructions(selected, *, target, output):
    """Bound the existing trusted issuers, never a candidate compiler command."""
    from . import private_instruction_coordinator as instructions

    if threading.current_thread() is not threading.main_thread() or signal.getitimer(signal.ITIMER_REAL)[0]:
        raise ValueError("trusted checker preparation needs a dedicated untimed main thread")
    prior = signal.getsignal(signal.SIGALRM)

    def expired(_signum, _frame):
        raise TimeoutError("trusted instruction checker preparation exceeded 120 seconds")

    signal.signal(signal.SIGALRM, expired)
    started = time.monotonic()
    signal.setitimer(signal.ITIMER_REAL, TRUSTED_CHECKER_TIMEOUT_S)
    try:
        result = instructions.prepare(selected, target=target, output=output)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior)
    elapsed = time.monotonic() - started
    if elapsed > TRUSTED_CHECKER_TIMEOUT_S:
        raise TimeoutError("trusted instruction checker preparation exceeded 120 seconds")
    return result, {"scope": "fresh trusted instruction tools only; no candidate/model compilation",
                    "budget_s": TRUSTED_CHECKER_TIMEOUT_S, "elapsed_s": elapsed}


def _revalidate(**arguments):
    from . import private_model_chain as chain

    with preparation.validation_scope():
        return chain._compile(**arguments)


def run(*, run_dir, descriptor, submission, original_partition, out):
    """Use the original run selection and full ordinary PFM verifier, with no build fallback."""
    from merlin.targetgen.target_experiment import load_target_experiment

    from . import compiler_profile_runtime as profile
    from . import private_full_models as models
    from . import private_instruction_declaration as declaration
    from . import private_model_chain as chain
    from .formal import _private_source_freeze_for_formal

    run_dir, descriptor, submission, original_partition, out = (
        Path(path).absolute() for path in (run_dir, descriptor, submission, original_partition, out)
    )
    original = json.loads(products_api.ordinary(original_partition).read_bytes())
    owner = profile.functional_service(run_dir)
    if owner is None:
        raise ValueError("static product revalidation has no original compiler-profile owner")
    target_experiment = load_target_experiment(descriptor)
    target = target_experiment.target
    environment = yaml.safe_load((run_dir / "environment.yaml").read_bytes())
    spec = products_api.ordinary(environment["private_full_model_spec"]["frozen_path"])
    freeze = _private_source_freeze_for_formal(run_dir, spec, target, workspace=owner.workspace,
                                             repo=owner.repo, descriptor=descriptor)
    name = original["selected_model"]
    programs = {key: list(stages) for key, stages in models.program_requirements_for(descriptor).items()}
    products = products_api.RetainedProducts(
        original_partition, run_dir=run_dir, name=name, candidate=submission, spec=spec,
        compiler_profile=profile.compiler_profile_binding(owner, run_dir=run_dir), programs=programs,
    )
    denied = (run_dir, submission, original_partition.parent,
              *(Path(pin["path"]).parent for pin in original["receipts"].values()))
    if (out.exists() or any(p.is_symlink() for p in (out, *out.parents))
            or any(out == path or out.is_relative_to(path) or path.is_relative_to(out) for path in denied)):
        raise ValueError("static product revalidation requires a fresh disjoint output owner")
    out.mkdir(mode=0o700, parents=True)
    selected = declaration.read(run_dir / "private_instruction_input/selection.json", target=target)
    arguments = {"run_dir": str(run_dir), "model": name,
                 "candidate_tree_sha256": strict_tree_sha256(submission)["sha256"],
                 "original_partition_sha256": file_sha256(original_partition)}
    with profile.original_compiler_execution(
        owner, target_experiment=target_experiment, candidate=submission, work_root=out / "profile"
    ):
        with IR.observe_call(
            out, stage=STAGE, function=_revalidate, arguments=arguments,
            inputs=(spec, descriptor, run_dir / "environment.yaml", run_dir / "input_bundle_manifest.yaml",
                    original_partition, *(Path(pin["path"]) for pin in original["receipts"].values())),
            outputs=(out / "native-static.json",),
            dependencies=(Path(__file__), Path(products_api.__file__), Path(chain.__file__), Path(models.__file__)),
        ) as observed:
            result = _revalidate(
                run_dir=run_dir, descriptor=descriptor, submission=submission, name=name, out=out,
                owner=owner, spec=spec, selected=selected, freeze=freeze, retained_products=products,
            )
            observed.returned()
    IR.verify(observed.path)
    result.invocation = observed.path
    result.verify()
    return result


def verify_record(record):
    """New typed receipt must retain its actual old inputs and fresh policy evidence."""
    from . import private_model_chain as chain

    retained = record.get("retained_product_revalidation")
    if record.get("schema") != SCHEMA or record.get("scope") != SCOPE or type(retained) is not dict:
        raise ValueError("static product revalidation has another schema/scope")
    tools = record.get("trusted_checker_preparation", {})
    if (tools.get("budget_s") != TRUSTED_CHECKER_TIMEOUT_S
            or type(tools.get("elapsed_s")) not in (int, float) or not math.isfinite(tools["elapsed_s"])
            or not 0 <= tools["elapsed_s"] <= TRUSTED_CHECKER_TIMEOUT_S
            or tools.get("scope") != "fresh trusted instruction tools only; no candidate/model compilation"):
        raise ValueError("static revalidation has no bounded fresh trusted checker preparation")
    old = json.loads(products_api.reopen(retained["original_partition"]).read_bytes())
    if (old.get("schema") != chain.SCHEMA or old.get("scope") != "canonical_static_only"
            or any(old.get(key) != record.get(key) for key in (
                *chain.partitions.BINDINGS, "target", "required_models", "required_programs", "selected_model"))
            or retained.get("receipts") != old.get("receipts") or record["receipts"] != old["receipts"]
            or retained.get("original_producer_completion") != "not_claimed"
            or set(retained.get("linked_policy_observations", {})) != set(record["receipts"])
            or set(retained.get("compiler_emission_observations", {})) != set(record["receipts"])):
        raise ValueError("static product revalidation changed original source/selection/products")
    for identity in (*old["inputs"], *old["receipts"].values()):
        products_api.reopen(identity)
    for program, result in retained["linked_policy_observations"].items():
        receipt = json.loads(products_api.reopen(record["receipts"][program]).read_bytes())
        emissions = retained["compiler_emission_observations"][program]
        if not isinstance(emissions, list):
            raise ValueError("retained program has no compiler-emission roster")
        if receipt["inputs"].get("device") is None:
            if result is not None or emissions:
                raise ValueError("host product gained device-policy credit")
            continue
        if not emissions:
            raise ValueError("retained device has no actual original compiler emissions")
        candidate = Path(receipt["inputs"]["device"]["package"])
        if (not candidate.is_absolute() or not candidate.is_dir()
                or any(path.is_symlink() for path in (candidate, *candidate.parents))
                or strict_tree_sha256(candidate)["sha256"] != record["candidate_tree_sha256"]):
            raise ValueError("retained compiler source no longer matches its selected candidate")
        if products_api._device_custody(
            receipt["output"], target=record["target"], candidate=candidate, profile=record["compiler_profile"]
        ) != emissions:
            raise ValueError("retained compiler emission roster differs from actual original device custody")
        _policy_observation(result, record, receipt)


def _policy_observation(result, record, receipt):
    """Reopen actual fresh policy issuance and its whole-linked audit observations."""
    from . import private_model_chain as chain

    if type(result) is not dict or result.get("status") != "accepted":
        raise ValueError("retained device has no accepted fresh linked-policy observation")
    policy_report = products_api.ordinary(result["report_path"])
    if (file_sha256(policy_report) != result.get("report_sha256")
            or result.get("elf_sha256") != receipt["output"]["elf_sha256"]):
        raise ValueError("fresh retained-product instruction proof changed")
    observed = policy_report.parent.parent / "invocation.json"
    call = IR.verify(observed)
    if (call.get("stage") != "device_final_linked_elf_policy"
            or call.get("callable") != "merlin.targetgen.contract.elf_admission.LinkedElfAdmissionService.evaluate"
            or call.get("arguments") != {"target": record["target"], "scope": "selected final linked-artifact policy"}
            or not any(pin.get("path") == str(policy_report) and pin.get("sha256") == result["report_sha256"]
                       for pin in call["outputs"])):
        raise ValueError("retained device instruction proof has another actual issuer")
    from merlin_experiments.phase2.contracts import document_sha256

    identity = document_sha256({"policy_sha256": result["policy_sha256"],
                               "accessor_sha256": result["accessor_intake_sha256"],
                               "selection": result["decoder_selection"]})
    inner, = policy_report.parent.glob("invocations/*/invocation.json")
    audit = IR.verify(inner)
    if (identity != record["instruction_selection"]["instruction_check_sha256"]
            or audit.get("stage") != "whole_linked_instruction_policy"
            or audit.get("callable") != "merlin_experiments.phase2.component_instruction_audit._scan"
            or audit.get("arguments") != {"decoder_selection": result["decoder_selection"],
                                         "policy_sha256": result["policy_sha256"],
                                         "accessor_intake_sha256": result["accessor_intake_sha256"]}
            or not any(pin.get("path") == receipt["output"]["elf"]
                       and pin.get("sha256") == receipt["output"]["elf_sha256"] for pin in audit["inputs"])
            or not any(pin.get("path") == str(policy_report) and pin.get("sha256") == result["report_sha256"]
                       for pin in audit["outputs"])
            or json.loads(policy_report.read_bytes()) != {
                key: value for key, value in result.items() if key not in {"report_path", "report_sha256"}}
            or result.get("prohibited_hits") != []):
        raise ValueError("retained device policy differs from the independently selected fresh checker")
    chain.partitions._selection(record, target=record["target"], instruction_path=Path(record["instruction_path"]))


def verify_producer(path, record, invocation):
    verify_record(record)
    call = IR.verify(invocation)
    if (call.get("kind") != "python_call" or call.get("stage") != STAGE
            or call.get("callable") != __name__ + "._revalidate"
            or call.get("arguments") != {
                "run_dir": record["run_dir"], "model": record["selected_model"],
                "candidate_tree_sha256": record["candidate_tree_sha256"],
                "original_partition_sha256": record["retained_product_revalidation"]["original_partition"]["sha256"]}
            or not any(pin.get("path") == str(path) and pin.get("sha256") == file_sha256(path)
                       for pin in call["outputs"])):
        raise ValueError("static revalidation has no matching completed independent producer")
