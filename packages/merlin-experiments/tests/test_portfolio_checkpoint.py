"""Installed portfolio admission checks actual files without native controller imports."""

import copy
import shutil
import socket
import subprocess
from pathlib import Path

import pytest
from merlin_experiments.phase2 import contracts as C
from merlin_experiments.phase2 import portfolio_checkpoint as PC

from merlin.benchharness import hash_tree
from merlin.perf import phase2_edit_contract


@pytest.fixture(autouse=True)
def refuse_processes_and_listeners(monkeypatch):
    def refused(*args, **kwargs):
        pytest.fail("checkpoint admission must not launch processes or listeners")

    monkeypatch.setattr(subprocess, "Popen", refused)
    monkeypatch.setattr(socket.socket, "bind", refused)


def _fixtures():
    import importlib.util
    import sys

    name = "portfolio_checkpoint_fixtures"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_FIXTURES = _fixtures()
write = _FIXTURES.write
digest = _FIXTURES.digest


@pytest.fixture(params=["ready", "blocked"])
def checkpoint(request, tmp_path):
    return _FIXTURES.build_checkpoint(tmp_path, status=request.param)


def test_real_ready_and_blocked_portfolio_roundtrip(checkpoint):
    assert checkpoint.consume() == checkpoint.document
    consumer = (
        PC.consume_global_candidate
        if checkpoint.document["schema"] == "global_perf_candidate_v1"
        else PC.consume_authoring_checkpoint
    )
    assert consumer(checkpoint.path, context=checkpoint.context) == checkpoint.document
    assert checkpoint.document["portfolio_members_ready"] in (1, 2)
    assert checkpoint.iteration["portfolio"]["full_model_simulation_allowed"] is False


@pytest.mark.parametrize("target", ["candidate", "dependencies", "iteration", "policy", "experiment"])
def test_external_mutation_refused(checkpoint, target):
    if target == "candidate":
        (checkpoint.candidate / "compiler.py").write_text("raise RuntimeError('must never execute')\n")
    elif target == "dependencies":
        (checkpoint.context.compiler_shared_source_root / "helper.py").write_text("VALUE = 2\n")
    elif target == "iteration":
        Path(checkpoint.document["iteration_record"]).write_text("{}")
    elif target == "policy":
        checkpoint.document["host_verification_policy"]["sha256"] = digest("other policy")
        checkpoint.save()
    else:
        checkpoint.experiment["portfolio_sha256"] = digest("other portfolio")
        checkpoint.save()
    with pytest.raises(ValueError):
        checkpoint.consume()


@pytest.mark.parametrize(
    "target", ["member_identity", "workload", "primary_alias", "coverage", "mechanism", "analysis"]
)
def test_rehashed_iteration_cannot_substitute_evidence(checkpoint, target):
    portfolio = checkpoint.iteration["portfolio"]
    if target == "member_identity":
        portfolio["members"][1]["identity"] = {"capsule": "substituted"}
    elif target == "workload":
        portfolio["members"][1]["analysis"]["workload"]["capsule_sha256"] = digest("other capsule")
    elif target == "primary_alias":
        portfolio["members"][0]["analysis_ref"] = "/some_other_analysis"
    elif target == "coverage":
        portfolio["members_total"] = 1
    elif target == "mechanism":
        checkpoint.iteration["round_mechanism_attribution"] = {"substituted": True}
    else:
        checkpoint.iteration["analysis"]["candidate_sha256"] = digest("other candidate")
    checkpoint.save(repin_iteration=True)
    with pytest.raises(ValueError):
        checkpoint.consume()


@pytest.mark.parametrize("target", ["candidate", "iteration"])
@pytest.mark.parametrize("kind", ["linked", "outside"])
def test_checkpoint_paths_must_be_unlinked_siblings(checkpoint, tmp_path, target, kind):
    key = "candidate_path" if target == "candidate" else "iteration_record"
    original = Path(checkpoint.document[key])
    outside = tmp_path / f"outside-{target}"
    if original.is_dir():
        shutil.copytree(original, outside)
    else:
        shutil.copyfile(original, outside)
    if kind == "linked":
        selected = checkpoint.path.parent / f"linked-{target}"
        selected.symlink_to(outside, target_is_directory=outside.is_dir())
    else:
        selected = outside
    checkpoint.document[key] = str(selected)
    checkpoint.save()
    with pytest.raises(ValueError):
        checkpoint.consume()


def test_edit_authority_seed_and_contract_are_bound(checkpoint):
    seed = checkpoint.path.parent / "edit_scope_seed"
    shutil.copytree(checkpoint.candidate, seed)
    contract = phase2_edit_contract.seal(
        {
            "schema": "compiler_edit_contract_v1",
            "existing_symbols": [{"surface_id": "schedule", "path": "compiler.py", "symbol": "schedule"}],
            "helper_extensions": [],
        }
    )
    authority = {
        "seed_path": str(seed),
        "initial_candidate_sha256": hash_tree(seed)["sha256"],
        "contract": contract,
        "contract_document_sha256": C.document_sha256(contract),
    }
    checkpoint.document["compiler_edit_authority"] = authority
    write(checkpoint.path.parent / "compiler_edit_authority.json", authority)
    checkpoint.save()
    assert checkpoint.consume() == checkpoint.document
    (seed / "compiler.py").write_text("def schedule():\n    return 999\n")
    with pytest.raises(ValueError, match="edit authority"):
        checkpoint.consume()


def test_unbacked_mechanism_catalog_refused(checkpoint):
    checkpoint.document["compiler_mechanism_catalog"] = {"schema": "substituted"}
    checkpoint.save()
    with pytest.raises(ValueError, match="mechanism catalog"):
        checkpoint.consume()


def test_unknown_round_schema_refused(checkpoint):
    checkpoint.document["schema"] = "unrecognized"
    checkpoint.save()
    with pytest.raises(ValueError, match="unsupported"):
        checkpoint.consume()


@pytest.mark.parametrize("kind", ["context", "probe", "source"])
@pytest.mark.parametrize("mutation", ["bytes", "scope", "binding"])
@pytest.mark.parametrize("checkpoint", ["ready"], indirect=True)
def test_optional_receipts_cannot_expand_scientific_scope(checkpoint, kind, mutation):
    analysis = checkpoint.iteration["analysis"]
    expected = {
        "compiler_digest": checkpoint.document["compiler_dependencies"]["compiler_implementation_sha256"],
        "target_digest": checkpoint.document["target_sha256"],
        "graph_digest": analysis["diagnostics"]["captured_logical_graph"]["logical_dispatch_digest"],
        "plan_digest": analysis["diagnostics"]["verified_global_plan_emission"]["plan_digest"],
    }
    receipt = {"binding": expected, "full_model_cycles": None}
    if kind == "context":
        field = "context_receipts"
        receipt.update(
            scope="controlled_source_prefix",
            global_cost_validated=False,
            global_speedup_proven=False,
            calibration_admissible=False,
            model_artifact_sha256=analysis["emission"]["candidate_lowered_sha256"],
        )
    elif kind == "probe":
        field = "probe_receipts"
        receipt.update(scope="mechanism_probe_only", warmup_runs=1, measured_runs=1)
    else:
        field = "source_pair_receipts"
        receipt.update(
            schema="global_source_contraction_preparation_receipt_v1",
            host_verifier_policy_sha256=checkpoint.document["host_verification_policy"]["sha256"],
            full_model_numerics_qualified=False,
            global_speedup_proven=False,
            numerical_pass=False,
            runtime_admitted=False,
        )
        preparation = checkpoint.path.parent / "preparation.json"
        write(preparation, receipt)
        preparation_hash = C.sha256_file(preparation)
        checkpoint.document["source_contraction_preparation_receipts"] = [
            {"path": str(preparation), "sha256": preparation_hash}
        ]
        receipt.update(schema="global_source_contraction_execution_receipt_v1", preparation_sha256=preparation_hash)
    receipt_path = checkpoint.path.parent / f"{kind}.json"
    write(receipt_path, receipt)
    checkpoint.document[field] = [{"path": str(receipt_path), "sha256": C.sha256_file(receipt_path)}]
    checkpoint.save()
    assert checkpoint.consume() == checkpoint.document
    if mutation == "bytes":
        receipt_path.write_text("{}")
    else:
        if mutation == "binding":
            receipt["binding"]["plan_digest"] = digest("another plan")
        elif kind == "source":
            receipt["full_model_numerics_qualified"] = True
        else:
            receipt["full_model_cycles"] = 17
        write(receipt_path, receipt)
        checkpoint.document[field][0]["sha256"] = C.sha256_file(receipt_path)
        checkpoint.save()
    with pytest.raises(ValueError):
        checkpoint.consume()


def test_context_copies_selected_policy(checkpoint):
    supplied = copy.deepcopy(checkpoint.context.host_policy)
    context = PC.CheckpointVerificationContext(supplied, checkpoint.context.compiler_shared_source_root)
    supplied["sha256"] = digest("mutated caller mapping")
    assert PC.consume_round_checkpoint(checkpoint.path, context=context) == checkpoint.document
