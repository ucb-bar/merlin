"""Retained resource custody with synthetic emitters, never hardware/model claims."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase0 import public_scale_sources
from merlin_experiments.phase1.brokers.common_transforms import CommonTransformService
from merlin_experiments.phase1.feedback import compile_only
from merlin_experiments.phase1.feedback import compiler_profile_runtime as R
from merlin_experiments.phase1.feedback import public_scalability_gate as G
from merlin_experiments.phase1.feedback.compiler_profile_child import _pin
from merlin_experiments.phase2.contracts import StageGateError, exact_tree_record
from test_public_codegen_resource_policy import policy
from test_public_scale_probe import member

from merlin.common import invocation_record as IR
from merlin.common.paths import repo_root
from merlin.common.tree_hash import hash_tree
from merlin.llvmlower.device_ir_scalability import DeviceIRBudget, analyze_device_ir
from merlin.targetgen import capsule_runner
from merlin.targetgen.sandbox import bwrap


def _save(path, data):
    path.write_text(json.dumps(data))


def _materialize(source, generated):
    (generated / "input.interface.mlir").write_bytes(source.read_bytes())
    for name in ("command_buffer.json", "lowered.target.mlir", "lowered.llvm.mlir"):
        (generated / name).write_text("synthetic emitter output\n")


@pytest.fixture
def retained(tmp_path, monkeypatch):
    run, base, candidate = (tmp_path / name for name in ("run", "probe", "frozen"))
    for path in (run, base, candidate):
        path.mkdir()
    (candidate / "tool.py").write_text("# synthetic source only\n")
    shutil.copytree(candidate, run / "submission")
    public, contract = run / "private_corpus_input/public", run / "private_corpus_input/contract"
    public.mkdir(parents=True)
    contract.mkdir()
    cap, _ = member(public)
    facts, descriptor = run / "facts.json", run / "descriptor.yaml"
    _save(facts, {"facts": {"target": "fixture", "arrays": [{"cols": 16}]}})
    descriptor.write_text("target: fixture\n")
    _save(run / "environment.yaml", {"run_config": {"launcher_argv": ["--descriptor", str(descriptor)]}})
    (run / "input_bundle_manifest.yaml").write_text("{}\n")
    chosen = policy()
    chosen["capsules"] = [cap["name"]]
    owner = CommonTransformService.__new__(CommonTransformService)
    owner.profile = {"public_codegen_resource_policy": chosen, "host_index_bits": 64, "sha256": "a" * 64}
    owner._verify = lambda: None
    owner.workspace, owner.bundle, owner.repo = tmp_path, {}, repo_root()
    monkeypatch.setattr(R, "functional_service", lambda _: owner)
    monkeypatch.setattr(bwrap, "frozen_selected_rtl_facts", lambda *args, **kwargs: facts)
    monkeypatch.setattr(capsule_runner, "discover_capsules", lambda *args, **kwargs: [cap])
    tool = base / "synthetic-translate"
    tool.write_text(f"#!{sys.executable}\nimport pathlib,sys\n"
                    "pathlib.Path(sys.argv[-1]).write_text('define void @kernel() {\\n  ret void\\n}\\n')\n")
    tool.chmod(0o700)
    config = base / "config.yaml"
    _save(config, {"runtime_pins": {"mlir_translate": _pin(tool)}})
    gate = base / "gate"
    (gate / "sources").mkdir(parents=True)
    cases, manifest = public_scale_sources.derive([cap], index_bits=64, tile_edge=16)
    rows = []
    for index, case in enumerate(cases):
        source = gate / "sources" / f"case-{index:03d}.mlir"
        source.write_text(case.source_ir)
        output = gate / f"case-{index:03d}"
        generated = output / "generated"
        generated.mkdir(parents=True)
        products = tuple(generated / name for name in (
            "input.interface.mlir", "command_buffer.json", "lowered.target.mlir", "lowered.llvm.mlir"))
        with IR.observe_call(output, stage="public_scale_source_lowering", function=_materialize,
                             arguments={}, inputs=(source,), outputs=products) as emission:
            with IR.observe_call(generated, stage="ordinary_source_emission", function=_materialize,
                                 arguments={}, inputs=(source,), outputs=products,
                                 dependencies=(candidate / "tool.py",)) as custody:
                _materialize(source, generated)
                custody.returned()
            emission.returned()
        llvm = generated / "lowered.ll"
        IR.run([str(tool), "--mlir-to-llvmir", str(products[-1]), "-o", str(llvm)], directory=output,
               stage="public_scale_llvm_translation", inputs=(products[-1],), outputs=(llvm,),
               capture_output=True, text=True)
        translation = next(p for p in output.glob("invocations/*/invocation.json")
                           if json.loads(p.read_bytes())["stage"] == "public_scale_llvm_translation")
        analysis = analyze_device_ir(llvm)
        row = compile_only._case_row(case)
        row.update(lowering_status="emitted", compilation_status="not_attempted",
                   code_growth={**analysis, "emitted_mlir_bytes": products[-1].stat().st_size,
                                "emitted_mlir_sha256": _pin(products[-1])["sha256"]},
                   resource_budget=DeviceIRBudget(**chosen["device_ir_budget"]).evaluate(analysis),
                   emission_invocation_sha256=_pin(emission.path)["sha256"],
                   translation_invocation_sha256=_pin(translation)["sha256"],
                   static_obligations={name: "UNKNOWN" for name in (
                       "semantic_coverage", "index_bounds", "resource_legality", "complete_output_coverage")})
        rows.append(row)
    report = {"mode": "scale_emission_only", "candidate_sha256": exact_tree_record(candidate)["sha256"],
              "compiler_tool_profile_sha256": "a" * 64, "facts_sha256": _pin(facts)["sha256"],
              "source_manifest": manifest, "cases": rows}
    report_path = gate / "public_compile_only.json"
    _save(report_path, report)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root(), text=True).strip()
    sources = []
    for module in (G, R):
        path = Path(module.__file__)
        raw = subprocess.check_output(["git", "show", f"{revision}:{path.relative_to(repo_root())}"], cwd=repo_root())
        sources.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
    selection = {"schema": "merlin.operator.resource_only_probe.v1", "original_run": str(run),
                 "implementation_revision": revision, "implementation_sources": sources,
                 "original_environment": _pin(run / "environment.yaml"),
                 "original_archive": _pin(run / "input_bundle_manifest.yaml"), "original_config": _pin(config),
                 "candidate": {"path": str(candidate), "hash_tree": hash_tree(candidate)},
                 "descriptor": _pin(descriptor), "facts": _pin(facts), "policy": chosen,
                 "public": str(public), "contract": str(contract)}
    selection_path, verdict_path = base / "selection.json", base / "verdict.json"
    _save(selection_path, selection)
    verdict = {"selection": _pin(selection_path), "candidate_unchanged": True, "result": {
        "status": "accepted", "required": True, "resource_complete": True, "selection": chosen,
        "candidate_sha256": report["candidate_sha256"], "compiler_tool_profile_sha256": "a" * 64,
        "report": _pin(report_path)}}
    _save(verdict_path, verdict)
    return SimpleNamespace(**locals())


def verify(case):
    return G.verify_retained_resource_observation(
        case.run, selection_path=case.selection_path, verdict_path=case.verdict_path)


def test_resource_reopen_rederives_all_cases_and_records_actual_products_without_execution(retained):
    result = verify(retained)
    assert result["result"]["resource_complete"] is True
    assert len(result["invocations"]) == 15
    assert len(result["products"]) == 30
    assert result["producer_sources"][0]["revision"] == retained.revision
    assert "no new execution" in result["scope"]


@pytest.mark.parametrize("changed", ["llvm", "mlir", "source", "native_candidate", "config", "descriptor", "record"])
def test_retained_product_or_source_mutation_is_never_accepted(retained, changed):
    paths = {
        "llvm": retained.gate / "case-000/generated/lowered.ll",
        "mlir": retained.gate / "case-000/generated/lowered.llvm.mlir",
        "source": retained.gate / "sources/case-000.mlir",
        "native_candidate": retained.run / "submission/tool.py", "config": retained.config,
        "descriptor": retained.descriptor,
        "record": next((retained.gate / "case-000").rglob("invocation.json")),
    }
    with paths[changed].open("a") as stream:
        stream.write(" changed\n")
    with pytest.raises((ValueError, KeyError, StageGateError)):
        verify(retained)


@pytest.mark.parametrize("changed", ["cost", "missing", "facts", "policy", "producer"])
def test_rehashing_summary_flags_cannot_replace_actual_products_or_selected_policy(retained, changed):
    if changed == "producer":
        retained.selection["implementation_sources"][0]["sha256"] = "0" * 64
        _save(retained.selection_path, retained.selection)
        retained.verdict["selection"] = _pin(retained.selection_path)
    else:
        if changed == "cost":
            retained.report["cases"][0]["code_growth"]["function_code_bytes"] += 1
        elif changed == "missing":
            retained.report["cases"].pop()
        elif changed == "facts":
            retained.report["facts_sha256"] = "0" * 64
        else:
            retained.report["cases"][0]["resource_budget"]["policy"]["max_code_bytes"] += 1
        _save(retained.report_path, retained.report)
        retained.verdict["result"]["report"] = _pin(retained.report_path)
    _save(retained.verdict_path, retained.verdict)
    with pytest.raises((ValueError, KeyError, StageGateError)):
        verify(retained)
