"""Synthetic immutable-source routing; no compiler, engine or issuer is run."""

import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase2 import component_compiled_arms as A
from merlin_experiments.phase2.contracts import StageGateError, exact_tree_record


@pytest.fixture
def route(tmp_path, monkeypatch):
    paths = {name: tmp_path / name for name in ("retained", "baseline", "candidate", "member", "private")}
    for name, path in paths.items():
        path.mkdir()
        if name in {"retained", "baseline", "candidate"}:
            (path / "compiler.py").write_text("# distinct synthetic " + name + "\n")
    program = {
        "inputs": [{"name": "A", "role": "input", "shape": [1, 2], "dtype": "operand"}],
        "nodes": [{"name": "P", "op": "copy", "inputs": ["A"]}],
        "outputs": [{"name": "Y", "value": "P"}],
    }
    typed, source = A.component_program.render(program, operand_dtype="i8", accumulator_dtype="i32")
    descriptor = {
        "interface_mlir": "source.mlir",
        "component_program": typed,
        "operation": {"op": "component_program", "attributes": {"program": program}},
    }
    (paths["member"] / "capsule.yaml").write_text(yaml.safe_dump(descriptor))
    (paths["member"] / "source.mlir").write_text(source)
    member = SimpleNamespace(source_dir=paths["member"])
    corpus = SimpleNamespace(capsules=(member,))
    authority, observed = object(), []
    admission = SimpleNamespace(baseline=paths["baseline"], verify=lambda **kwargs: None)
    execution = SimpleNamespace(
        candidate=paths["candidate"],
        baseline_admission=admission,
        edit_authority=authority,
        output=paths["private"],
        view=None,
        runtime=None,
        container_transport=None,
        contract_root=tmp_path,
        scope=None,
        independent_runtime=SimpleNamespace(hardware_intake=SimpleNamespace(target="synthetic")),
    )
    owner = A.CompiledStaticArmFeatures(
        execution,
        SimpleNamespace(build_service=None, elf_admission=None, readelf=("unused", "a" * 64)),
        ("static-sites",),
        "b" * 64,
    )

    def verify(*, authority=None):
        if authority is not execution.edit_authority:
            raise StageGateError("retained revision lacks exact issued edit authority")

    retained = SimpleNamespace(
        source=paths["retained"], source_sha256=exact_tree_record(paths["retained"])["sha256"], verify=verify
    )
    # This fixed seam stub isolates routing. Existing live owner/qualification
    # denial controls remain responsible for authority; no fixture issues it.
    seam = ModuleType("merlin_experiments.phase2.retained_revisions")
    seam.executing_revision = lambda compiler: retained if compiler == retained.source else None
    monkeypatch.setitem(sys.modules, seam.__name__, seam)
    monkeypatch.setattr(A.CompiledStaticArmFeatures, "verify", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        A,
        "qualified_package_execution",
        lambda **kwargs: observed.append(("mount", kwargs["candidate"])) or nullcontext(),
    )

    def compile_only(**kwargs):
        observed.append(("compile", kwargs["package_dir"]))
        kwargs["output_root"].mkdir()
        (kwargs["output_root"] / "compile_only_result.json").write_text("{}\n")

    def extract(**kwargs):
        observed.append(("extract", kwargs["package_dir"]))
        return {"features": {"static_sites": 1}}

    monkeypatch.setattr(A.compile_only_execution, "compile_source_only", compile_only)
    monkeypatch.setattr(A, "observe_component_compiled_features", extract)
    return owner, paths, retained, member, corpus, observed


def test_retained_compiler_is_shared_directly_across_compile_and_extraction(route, monkeypatch):
    owner, paths, retained, member, corpus, observed = route
    monkeypatch.setattr(A.shutil, "copytree", lambda *args, **kwargs: pytest.fail("retained source must not be copied"))
    original_tree = A.exact_tree_record
    monkeypatch.setattr(
        A,
        "exact_tree_record",
        lambda path: (
            pytest.fail("mutable candidate must not be reopened") if path == paths["candidate"] else original_tree(path)
        ),
    )
    workspace = paths["private"] / "arm"
    owner.observe(compiler=retained.source, member=member, corpus=corpus, workspace=workspace, timeout_s=10)
    assert observed == [(stage, retained.source) for stage in ("mount", "compile", "extract")]
    assert not (workspace / "compiler").exists()


def test_legacy_source_keeps_its_private_copy(route):
    owner, paths, _retained, member, corpus, observed = route
    workspace = paths["private"] / "arm"
    owner.observe(compiler=paths["candidate"], member=member, corpus=corpus, workspace=workspace, timeout_s=10)
    assert observed == [(stage, workspace / "compiler") for stage in ("mount", "compile", "extract")]
    assert exact_tree_record(workspace / "compiler") == exact_tree_record(paths["candidate"])


def test_retained_revision_from_another_edit_authority_is_refused(route):
    owner, paths, retained, member, corpus, observed = route
    original_authority = owner.execution.edit_authority
    retained.verify = lambda *, authority=None: (_ for _ in ()).throw(StageGateError("foreign edit authority"))
    assert original_authority is not None
    with pytest.raises(StageGateError, match="foreign edit authority"):
        owner.compile(
            compiler=retained.source, member=member, corpus=corpus, workspace=paths["private"] / "arm", timeout_s=10
        )
    assert observed == []
