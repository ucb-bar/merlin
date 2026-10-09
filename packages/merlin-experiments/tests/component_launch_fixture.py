"""Diagnostic launch assembly fixture; no fresh author or target authority is issued.

Every experimental owner is deliberately unissued. Tests temporarily substitute
its verifier to exercise assembly joins. Removing that substitution must refuse
these objects through the unchanged production admission boundary.
"""
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.component_origin import FreshCompilerOrigin
from merlin_experiments.phase1.component_qualification import ComponentQualification
from merlin_experiments.phase2 import component_launch as CL
from merlin_experiments.phase2 import component_launch_inputs as L
from merlin_experiments.phase2.component_baseline import ComponentBaselineAdmission
from merlin_experiments.phase2.component_cca import ComponentCCAProvider
from merlin_experiments.phase2.component_execution import ComponentIndependentExecution
from merlin_experiments.phase2.component_experiment import ComponentView, RuntimeGrant
from merlin_experiments.phase2.component_measurement_qualification import IndependentMeasurementQualification
from merlin_experiments.phase2.component_runtime_authority import (
    IndependentComponentRuntime,
    IndependentRuntimeServices,
)
from merlin_experiments.phase2.component_variants import ComponentVariantSnapshot
from merlin_experiments.phase2.component_workflow import ComponentAnalyticalProvider, ComponentOnlyPolicy
from merlin_experiments.phase2.contracts import canonical_json, document_sha256, sha256_file
from test_component_launch import short_ipc_directory as short_ipc_directory
from test_edit_authority import package as package

from merlin.benchharness import hash_tree
from merlin.perf.component_cost import ComponentCostScope


def _unissued(cls, **selected):
    values = dict.fromkeys(row.name for row in fields(cls))
    values.update(selected)
    return cls(**values)


def _unqualified_feedback(**_kwargs):
    raise AssertionError("diagnostic assembly must not execute any compiler or observer")


@pytest.fixture
def diagnostic_launch(package, short_ipc_directory, monkeypatch):
    authority, candidate, contract, output = package
    authority.freeze(candidate, contract, has_iterations=False)
    root = output.parent
    descriptor = root / "descriptor.yaml"
    descriptor.write_text("target: synthetic\n")
    identity = document_sha256("diagnostic only, never experimental authority")
    corpus_root = root / "generated"
    corpus_root.mkdir()
    corpus = SimpleNamespace(root=corpus_root, manifest_sha256=identity, capsules_sha256=identity, capsules=())
    view_root = root / "public"
    view_root.mkdir()
    selections = {
        "compiler/library.py": b"# synthetic public library\n",
        "generated_input/source.mlir": b"module {}\n",
        "contract/performance_corpus_manifest.json": CL.public_component_manifest(corpus),
    }
    members = []
    for name, payload in sorted(selections.items()):
        path = view_root / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(payload)
        path.chmod(0o444)
        members.append({"path": name, "sha256": sha256_file(path), "n_bytes": len(payload),
                        "role": name.partition("/")[0]})
    manifest = view_root / "manifest.json"
    manifest.write_bytes(canonical_json({"schema": "merlin.component_agent_view.v1",
                                        "generation_sha256": identity, "library_sha256": identity, "members": members}))
    view = ComponentView(view_root, sha256_file(manifest), identity, identity)
    tools = root / "tools"
    tools.mkdir()
    grants = []
    for name in ("bwrap", "author-bwrap", "codex", "python3"):
        path = tools / name
        path.write_text("# diagnostic executable bytes\n")
        grants.append(RuntimeGrant(path, "/usr/bin/" + name, sha256_file(path)))
    grants = tuple(grants)
    auth = root / "control-auth.json"
    auth.write_text("{}")
    target = SimpleNamespace(path=descriptor, target="synthetic")
    hardware = SimpleNamespace(sha256=identity)
    functional = _unissued(IndependentComponentRuntime, hardware_intake=hardware, target_descriptor=descriptor,
                          services=IndependentRuntimeServices(_unqualified_feedback, _unqualified_feedback),
                          source_pins=(), qualified_roles=("grade", "stage_verifier"), qualification_sha256=identity,
                          callbacks=())
    fresh_inputs = SimpleNamespace(execution_support=functional, hardware=hardware, runtime=grants,
                                   control_runtime=grants, codex_binary=tools / "codex",
                                   codex_destination="/usr/bin/codex",
                                   author_sandbox=next(
                                       row for row in grants if row.destination == "/usr/bin/author-bwrap"
                                   ),
                                   auth_source=auth, target_experiment=target, output=root / "fresh-evidence")
    origin = _unissued(FreshCompilerOrigin, inputs=fresh_inputs, candidate=authority.seed,
                       receipt_sha256=identity, _issuer=object())
    receipt = root / "qualification" / "domain.json"
    receipt.parent.mkdir()
    source = root / "shared-source"
    source.mkdir()
    qualification = _unissued(ComponentQualification, candidate=authority.seed, corpus_root=corpus_root,
                             coverage_sha256=identity, receipt=receipt, receipt_sha256=identity,
                             target_descriptor=descriptor, contract_root=root / "contract", source_root=source,
                             runtime=grants, view=view, compiler_origin=origin, compiler_lineage=None,
                             runtime_authority=functional, runtime_authority_sha256=identity, _issuer=object())
    admission = ComponentBaselineAdmission(authority.seed, str(hash_tree(authority.seed)["sha256"]), qualification,
                                          identity, identity, corpus, descriptor, sha256_file(descriptor), object())
    scope = ComponentCostScope(identity, identity, identity)
    execution = ComponentIndependentExecution(admission, functional, candidate, authority,
                                             document_sha256(authority.binding),
                                             view, grants, qualification.contract_root, identity, scope,
                                             root / "execution-evidence", object())
    variant = _unissued(ComponentVariantSnapshot, execution=execution)
    adapter = root / "measurement" / "adapter.json"
    adapter.parent.mkdir()
    adapter.write_text("{}")
    reports = tuple((regime, adapter.parent / (regime + ".json"), identity) for regime in ("cold", "warm"))
    measured = _unissued(IndependentMeasurementQualification, functional_runtime=functional,
                        baseline_admission=admission, scope=scope, calibration_adapter=adapter,
                        variants=(variant,), receipt=adapter.parent / "qualification.json", screen_reports=reports)
    support = _unissued(IndependentComponentRuntime, qualification=measured,
                       services=IndependentRuntimeServices(_unqualified_feedback, _unqualified_feedback,
                                                           _unqualified_feedback, _unqualified_feedback,
                                                           _unqualified_feedback), hardware_intake=hardware,
                       target_descriptor=descriptor, source_pins=(), qualified_roles=(
                           "grade", "stage_verifier", "feature_provider", "cca_provider", "rtl_executor"),
                       qualification_sha256=identity, callbacks=())
    price = root / "prices.json"
    price.write_text("{}")
    stage = short_ipc_directory / "stage"
    data = {
        "schema": "merlin.component_launch_inputs.v1", "candidate": str(candidate), "view": str(view_root),
        "corpus": str(corpus_root), "descriptor": str(descriptor), "source_root": str(source),
        "contract_root": str(qualification.contract_root), "stage_root": str(stage),
        "qualification_root": str(receipt.parent), "edit_authority_root": str(authority.output),
        "edit_contract": str(authority.output / "compiler_edit_authority.json"),
        "runtime": L._grant_records(grants), "control_runtime": L._grant_records(grants),
        "readiness": [{"capability": name, "command": ["/usr/bin/python3", "--version"], "stdout_sha256": identity}
                      for name in ("compiler", "linker", "simulator", "isa", "cca")],
        "codex_binary": str(tools / "codex"), "codex_destination": "/usr/bin/codex", "auth_source": str(auth),
        "price_table": str(price), "analytical": {
            "calibration_adapter": str(adapter), "qualification": str(dict((r, p) for r, p, _ in reports)["warm"]),
            "objective": "warm", "max_workers": 1, "memory_per_worker_bytes": 1024, "engine_slots": 1,
            "output": str(stage / "analytical"), "lease_path": str(stage / "worker-leases.json"),
        },
    }
    declaration = root / "launch.json"
    calls = []
    monkeypatch.setattr(FreshCompilerOrigin, "verify", lambda self, **kwargs: calls.append(("origin", kwargs)))
    monkeypatch.setattr(ComponentQualification, "verify", lambda self, **kwargs: calls.append(("domain", kwargs)))
    monkeypatch.setattr(ComponentBaselineAdmission, "verify", lambda self, **kwargs: self.sha256)
    monkeypatch.setattr(ComponentIndependentExecution, "verify", lambda self: self.sha256)
    monkeypatch.setattr(ComponentVariantSnapshot, "verify", lambda self: identity)
    monkeypatch.setattr(IndependentMeasurementQualification, "verify", lambda self: identity)
    monkeypatch.setattr(IndependentMeasurementQualification, "verify_component_binding",
                        lambda self, value: calls.append(("cohort", value)))
    monkeypatch.setattr(IndependentMeasurementQualification, "verify_feedback_binding",
                        lambda self, **kwargs: calls.append(("feedback", kwargs)))
    def runtime_verify(self, *, required_roles=()):
        if not set(required_roles) <= set(self.qualified_roles):
            raise L.C.StageGateError("independent runtime lacks qualified requested roles")
        calls.append(("runtime", required_roles))
        return self.sha256
    monkeypatch.setattr(IndependentComponentRuntime, "verify", runtime_verify)
    analytical = ComponentAnalyticalProvider(_unqualified_feedback, Path(__file__), sha256_file(Path(__file__)),
                                            adapter, sha256_file(adapter), identity)
    def build_analytical(**kwargs):
        calls.append(("analytical", kwargs))
        return analytical
    monkeypatch.setattr(L, "build_independent_component_analytical_provider", build_analytical)
    cca = ComponentCCAProvider(_unqualified_feedback, Path(__file__), sha256_file(Path(__file__)),
                               authority.seed, admission.baseline_sha256, admission, support)
    def build_cca(**kwargs):
        calls.append(("cca", kwargs))
        return cca
    monkeypatch.setattr(L, "build_independent_component_cca_provider", build_cca)
    monkeypatch.setattr(ComponentOnlyPolicy, "_validate", lambda self: calls.append(("policy", self)))
    return SimpleNamespace(data=data, declaration=declaration, origin=origin, support=support, execution=execution,
                           admission=admission, calls=calls, qualification=qualification)
