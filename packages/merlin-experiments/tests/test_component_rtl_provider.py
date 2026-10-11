"""Synthetic constructor plumbing only; no issued runtime or engine execution."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import component_providers as P
from merlin_experiments.phase2 import development_feedback as DF
from merlin_experiments.phase2.contracts import StageGateError, sha256_file

from merlin.benchharness import hash_tree
from merlin.targetgen.target_experiment import TargetExperiment


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    baseline, corpus_root, contract = (tmp_path / name for name in ("baseline", "corpus", "contract"))
    for root in (baseline, corpus_root, contract):
        root.mkdir()
    (baseline / "compiler.py").write_text("# synthetic compiler; never executed\n")
    paths = {
        name: tmp_path / name for name in ("descriptor.yaml", "certificate.json", "facts.json", "circt.mlir", "engine")
    }
    for path in paths.values():
        path.write_text("synthetic constructor fixture; no engine execution\n")
    pins = {path: sha256_file(path) for path in paths.values()}
    target = TargetExperiment(
        target="synthetic",
        isa_headers=(),
        hwbringup_set=None,
        curated_harness=None,
        capsule_corpus=corpus_root,
        sim_via="synthetic",
        rtl_via="synthetic",
        rtl_repo=None,
        prior_backends=(),
        path=paths["descriptor.yaml"],
        descriptor_sha256=pins[paths["descriptor.yaml"]],
    )
    corpus = SimpleNamespace(
        root=corpus_root, capsules=(SimpleNamespace(family="family", capsule="member", descriptor={"label": "dev"}),)
    )
    admission = SimpleNamespace(
        baseline=baseline,
        baseline_sha256=hash_tree(baseline)["sha256"],
        corpus=corpus,
        target_descriptor=target.path,
        target_sha256=target.descriptor_sha256,
        qualification=SimpleNamespace(contract_root=contract),
        verify=lambda **kwargs: None,
    )

    def executor(**kwargs):
        pytest.fail("constructor must never execute an engine")

    owner = Path(__file__).resolve()
    pins[owner] = sha256_file(owner)
    runtime = SimpleNamespace(
        source_pins=tuple(pins.items()),
        services=SimpleNamespace(rtl_executor=executor),
        verify=lambda **kwargs: None,
    )
    # This fixture bypasses authority solely to test data binding. The independent
    # provider denial test below retains the real live-qualification requirement.
    monkeypatch.setattr(P, "_inputs", lambda *args: runtime)
    certificate = SimpleNamespace(
        path=paths["certificate.json"],
        sha256=pins[paths["certificate.json"]],
        target=target.target,
        pins={"selected_engine": {"path": str(paths["engine"]), "sha256": pins[paths["engine"]]}},
    )
    monkeypatch.setattr(DF.GATE, "load_certificate", lambda *args, **kwargs: certificate)
    monkeypatch.setattr(
        DF.GATE,
        "plan_evaluation",
        lambda *args, **kwargs: SimpleNamespace(
            admitted=True,
            eligible=True,
            selected_engine="gsim",
            use_gsim=True,
        ),
    )
    monkeypatch.setattr(DF.PME, "gsim_workload", lambda member: {})
    monkeypatch.setattr(
        DF.MS,
        "load_rtl_identity",
        lambda *args: {
            key: {"path": str(paths[name]), "sha256": pins[paths[name]]}
            for key, name in (("rtl_facts", "facts.json"), ("circt_core_hw", "circt.mlir"))
        },
    )
    monkeypatch.setattr(DF.FM, "derived_peak_macs_per_cycle", lambda *args: (None, "unavailable"))
    return dict(
        independent_runtime=runtime,
        baseline_admission=admission,
        target_experiment=target,
        certificate_path=paths["certificate.json"],
        certificate_sha256=pins[paths["certificate.json"]],
        rtl_facts_path=paths["facts.json"],
        rtl_facts_sha256=pins[paths["facts.json"]],
        work_root=tmp_path / "private-feedback",
    )


def test_constructor_selects_original_evaluator_executor_and_fresh_inputs(inputs):
    provider = P.build_independent_component_rtl_provider(**inputs)
    admission, runtime = inputs["baseline_admission"], inputs["independent_runtime"]
    assert type(provider) is DF.DevelopmentGsimFeedback
    assert provider.executor is runtime.services.rtl_executor
    assert provider.baseline == admission.baseline
    assert provider.baseline_sha256 == admission.baseline_sha256
    assert provider.corpus is admission.corpus
    assert provider.target_experiment is inputs["target_experiment"]
    assert provider.achievable_macs_per_cycle is None
    assert not inputs["work_root"].exists()


@pytest.mark.parametrize("changed", ["facts", "certificate", "engine", "descriptor", "work_root"])
def test_constructor_refuses_changed_or_transferred_selection(inputs, changed):
    if changed == "engine":
        (inputs["work_root"].parent / "engine").write_text("different engine")
    elif changed == "work_root":
        inputs["work_root"] = inputs["baseline_admission"].baseline / "new-work"
    elif changed == "descriptor":
        inputs["target_experiment"].path.write_text("different descriptor")
    else:
        inputs[changed.replace("facts", "rtl_facts") + "_sha256"] = "a" * 64
    with pytest.raises(StageGateError):
        P.build_independent_component_rtl_provider(**inputs)


def test_constructor_never_accepts_a_callback_or_saved_qualification():
    with pytest.raises(StageGateError, match="fresh Phase 1 baseline"):
        P.build_independent_component_rtl_provider(
            independent_runtime={"status": "PASS"},
            baseline_admission=None,
            target_experiment=None,
            certificate_path=None,
            certificate_sha256=None,
            rtl_facts_path=None,
            rtl_facts_sha256=None,
            work_root=Path("unused"),
        )
