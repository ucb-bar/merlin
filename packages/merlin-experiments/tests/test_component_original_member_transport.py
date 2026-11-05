"""Private original-member masks and actual isolated import-order controls.

Owned mask files contain artificial canaries only. These controls grant no
author boundary, source-preparation, compiler or target-runtime authority.
"""

import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common import access as A
from merlin.common import invocation_record as I
from merlin.targetgen import native_component_execution as E
from merlin.targetgen import package_runtime as P

AS = importlib.import_module("merlin.targetgen.sandbox.answer_surfaces")
BW = importlib.import_module("merlin.targetgen.sandbox.bwrap")


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    import importlib.util
    import inspect

    spec = importlib.util.spec_from_file_location(
        "original_member_algorithm", Path(__file__).with_name("test_component_original_members.py")
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    return inspect.unwrap(fixture.isolated)(tmp_path, monkeypatch)


@pytest.mark.parametrize("first", ["native_component_execution", "native_component_inputs"])
def test_real_isolated_input_execution_import_orders_are_compatible(tmp_path, first):
    roots = [str(Path(A.__file__).parents[2]), str(Path(E.__file__).parents[2])]
    text = (
        "import sys,importlib;sys.path[:0]=" + repr(roots) + ";"
        "first=importlib.import_module('merlin.targetgen.'+sys.argv[1]);"
        "e=importlib.import_module('merlin.targetgen.native_component_execution');"
        "n=importlib.import_module('merlin.targetgen.native_component_inputs');"
        "assert e._bind is n._bind and e.NativeComponentExecutionError is n.NativeComponentExecutionError;"
        "print('both import orders preserve the shared error and binder')"
    )
    environment = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}
    (tmp_path / "environment.json").write_text(json.dumps(environment) + "\n")
    result = I.run(
        [sys.executable, "-I", "-B", "-c", text, first],
        directory=tmp_path,
        stage="original_member_import_order",
        cwd=tmp_path,
        env=environment,
        dependencies=(Path(E.__file__), Path(E.input_binding.__file__)),
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout == b"both import orders preserve the shared error and binder\n"
    I.require_environment(next(tmp_path.rglob("invocation.json")), environment=environment)


@pytest.mark.parametrize(
    "module",
    [
        "merlin_experiments.phase1.component_original_members",
        "merlin_experiments.phase1.component_qualification_members",
        "merlin.targetgen.native_component_inputs",
    ],
)
def test_original_member_replay_comparison_and_input_helper_source_install_bytecode_masks(
    tmp_path, monkeypatch, module
):
    relative = module.replace(".", "/")
    site = tmp_path / "python/lib/python3.12/site-packages"
    namespace = "packages/merlin-experiments/src/" if module.startswith("merlin_experiments") else "src/"
    source = tmp_path / namespace / (relative + ".py")
    installed = site / (relative + ".py")
    bytecode = installed.parent / "__pycache__" / (installed.stem + ".cpython-312.pyc")
    public = site / "merlin/targetgen/original_operator_sources.py"
    for path in (source, installed, bytecode, public):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"owned artificial mask canary\n")
    monkeypatch.setattr(A, "sys", SimpleNamespace(path=[str(site)], prefix=str(tmp_path / "python"), modules={}))
    monkeypatch.setattr(AS, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(AS, "artifacts_dir", lambda: tmp_path / "out/artifacts")
    monkeypatch.setattr(AS, "_evicted_oracle_modules", lambda: [])
    monkeypatch.setattr(AS, "_support_package_dirs", lambda: [])
    monkeypatch.setattr(AS, "experimenter_memory_dir", lambda: tmp_path / "absent-memory")
    policy = SimpleNamespace(
        target="owned_device",
        capsule_corpus=None,
        corpus_siblings=lambda: (),
        hidden_corpus=lambda: None,
        prior_backends=(),
        backend_package=None,
    )
    assert module in A.declared_modules("grader")
    surfaces = AS.answer_surfaces(policy)
    expected = {source, installed, bytecode}
    assert expected <= {item.path for item in surfaces}
    assert all(item.path != public and item.path not in public.parents for item in surfaces)
    unmasked = ["--ro-bind", str(tmp_path), str(tmp_path)]
    assert expected <= {item.path for item in BW.coverage_gap(unmasked, surfaces)}
    assert BW.coverage_gap(BW.apply_answer_masks(unmasked, surfaces), surfaces) == []


def test_actual_current_source_grant_excludes_owned_reference_envelope(tmp_path):
    """Only artificial owned siblings; this is a compiler namespace control."""
    from merlin_experiments.phase1.component_package_execution import ComponentPackageExecutor
    from merlin_experiments.phase2.component_runtime import inventory_runtime

    compiler = os.environ.get("MERLIN_TEST_CLANG") or os.environ.get("MERLIN_CLANG")
    sandbox = os.environ.get("MERLIN_TEST_BWRAP")
    if not compiler or not sandbox:
        pytest.skip("the actual source-only grant requires explicitly selected clang and bwrap")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    source = candidate / "probe.c"
    source.write_text(r"""
#include <stdio.h>
int main(int argc, char **argv) {
  if (argc != 5) return 11;
  char text[64]; FILE *input = fopen(argv[1], "r");
  if (!input || !fgets(text, sizeof(text), input)) return 12;
  fclose(input);
  for (int i=3; i<5; ++i) {
    FILE *private_input = fopen(argv[i], "r");
    if (private_input) { fclose(private_input); return 13; }
  }
  if (fopen("/evaluation-input/capsule.yaml", "r")) return 14;
  FILE *output=fopen(argv[2], "w"); if (!output) return 15;
  fputs("{\"source_only\":true}\n", output); fclose(output);
  puts("current source only; original envelope and reference excluded"); return 0;
}
""")
    tool = candidate / "probe"
    environment = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}
    build = I.run(
        [compiler, "-static", "-O2", str(source), "-o", str(tool)],
        directory=tmp_path / "build",
        stage="original_source_grant_probe_build",
        cwd=tmp_path,
        env=environment,
        inputs=(source,),
        outputs=(tool,),
        capture_output=True,
        timeout=60,
    )
    assert build.returncode == 0, build.stderr.decode()
    I.require_environment(next((tmp_path / "build").rglob("invocation.json")), environment=environment)
    private = tmp_path / "private"
    private.mkdir()
    interface, envelope, reference = (private / name for name in ("interface.mlir", "capsule.yaml", "reference.bin"))
    interface.write_text("module {}\n")
    envelope.write_text("OWNED ARTIFICIAL ENVELOPE\n")
    reference.write_bytes(b"OWNED ARTIFICIAL REFERENCE")
    # Reuse the ordinary independent synthetic view fixture, not a new grant.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "original_source_view", Path(__file__).with_name("test_component_experiment.py")
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    view = fixture._view(tmp_path)
    runtime = inventory_runtime(files=(), trees=(), executables=((Path(sandbox), "/usr/bin/bwrap"),))
    package = P.Package(
        candidate,
        {
            "language": "c",
            "commands": {"parse": {"argv": ["{tool}", "{input_mlir}", "{output_json}", str(envelope), str(reference)]}},
        },
        tool,
    )
    output = private / "command_buffer.json"
    executor = ComponentPackageExecutor(candidate, view, runtime, private)
    result = executor.run_entrypoint(package, "parse", interface, output, invocation_directory=private)
    if result.returncode and "Operation not permitted" in result.stderr:
        assert not output.exists()
        (tmp_path / "namespace-unavailable.json").write_text(
            json.dumps(
                {
                    "scope": "namespace not constructed; no isolation proof",
                    "returncode": result.returncode,
                    "stderr": result.stderr,
                }
            )
            + "\n"
        )
        pytest.skip("selected native bwrap cannot create this namespace; no isolation qualification")
    assert result.returncode == 0, result.stderr
    assert output.read_bytes() == b'{"source_only":true}\n'
    assert envelope.read_text() == "OWNED ARTIFICIAL ENVELOPE\n"
    assert reference.read_bytes() == b"OWNED ARTIFICIAL REFERENCE"
    record = I.verify(next(private.rglob("invocation.json")))
    arguments = record["argv"]
    assert any(
        arguments[index : index + 3] == ["--ro-bind", str(interface), "/evaluation-input/interface.mlir"]
        for index in range(len(arguments))
    )
    assert str(private) not in arguments


def test_context_forwards_exact_original_member_without_minting_a_grade(isolated, tmp_path, monkeypatch):
    """Synthetic context bookkeeping only; the supplied pass has no products."""
    import importlib.util
    import inspect

    from merlin_experiments.phase1 import component_original_members as M
    from merlin_experiments.phase2 import component_runtime_support as support
    from merlin_experiments.phase2.contracts import StageGateError

    spec = importlib.util.spec_from_file_location(
        "original_member_context", Path(__file__).with_name("test_component_runtime_support.py")
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    context_root = tmp_path / "context"
    context_root.mkdir()
    context = inspect.unwrap(fixture.prepared)(context_root, monkeypatch)
    standard = isolated[0]
    standard.references.schema_intake = SimpleNamespace(software=SimpleNamespace(hardware=context.hardware_intake))
    owner = M.prepare(standard_ir=standard, destination=tmp_path / "members")
    member = owner.require_complete()["members"][0]
    package = tmp_path / "compiler"
    package.mkdir()
    (package / "tool.py").write_text("inert diagnostic compiler\n")
    calls = []

    def capture(**arguments):
        calls.append(arguments)
        return {"numeric_report": {"status": "pass"}}

    monkeypatch.setattr(P, "active_package_executor", lambda: SimpleNamespace(container_transport=None))
    monkeypatch.setattr(support, "execute_component", capture)
    with pytest.raises(StageGateError, match="canonical admitted membership"):
        context.grade(
            package,
            capsules_root=[Path(member["capsule_root"])],
            runs_root=tmp_path / "grade",
            target=context.build_service.target,
            contract=context.contract_root,
            timeout=30,
            original_members=owner,
        )
    assert len(calls) == 1 and calls[0]["original_member"].owner is owner
    assert calls[0]["original_member"].source_slot == 0
    assert calls[0]["source_verifier"].__self__ is context
    assert calls[0]["build_service"].recipe is context.build_service.recipe
    with pytest.raises(StageGateError, match="remain UNKNOWN"):
        context.stage_verifier(result_path=tmp_path / "grade/result.json")
