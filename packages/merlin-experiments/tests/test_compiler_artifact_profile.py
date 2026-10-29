"""Actual process/protocol controls; synthetic owner/box, no runtime qualification."""

from __future__ import annotations

import json
import os
import subprocess
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1.brokers import compiler_call_channel
from merlin_experiments.phase1.feedback import compiler_profile_runtime as R
from merlin_experiments.phase2 import campaign as C

from merlin.common import invocation_record
from merlin.targetgen import package_runtime as P
from merlin.targetgen.artifact_bundle import BundleError, SelectedArtifactProfile


@pytest.fixture
def structural(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    profile_path = tmp_path / "selected-profile.yaml"
    profile_path.write_text("name: independent-artifact-protocol\n")
    profile = SelectedArtifactProfile.capture(profile_path)
    manifest = {
        "abi_version": "0.2",
        "artifact_type": "mlir_oot_target_backend",
        "target": "independent",
        "language": "python",
        "integrity_exempt": False,
        "authoring": {"mode": "hand_curated"},
        "entrypoints": {"tool": "compiler.py"},
        "commands": {
            name: {"argv": ["{tool}", name, "{input_mlir}", "{output_json}"]}
            for name in ("parse", "lower_interface_to_target", "emit_command_buffer", "emit_target_artifact")
        },
    }
    (candidate / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    script = candidate / "compiler.py"
    script.write_text(
        "from pathlib import Path\nimport sys\n"
        "assert Path(sys.argv[2]).read_text() == 'module {}\\n'\n"
        "Path(sys.argv[3]).write_bytes(b'\\0opaque artifact\\xff')\n"
        'print(\'{"protocol":"structural-only"}\')\n'
    )
    source = tmp_path / "input.mlir"
    source.write_text("module {}\n")
    output = tmp_path / "output.bin"
    output.write_bytes(b"previous output")
    owner = R.CommonTransformService.__new__(R.CommonTransformService)
    monkeypatch.setattr(R.CommonTransformService, "compiler_imports", lambda self: nullcontext())
    calls = []

    def policy(target, workspace, package, *, inputs):
        return C.PackageSandboxPolicy(
            ("synthetic-box", "--bind", str(workspace), str(workspace), "--ro-bind", str(package), str(package)),
            (),
            (),
            Path(workspace),
            Path(package),
            target,
        )

    @contextmanager
    def transport(*args, **kwargs):
        yield SimpleNamespace(env_prefix=lambda prefix: prefix, remaining_timeout=lambda: kwargs["timeout_s"])

    def invoke(self, selected_policy, package, argv, **kwargs):
        # This protocol control invokes an ordinary harmless Python subprocess.
        # The synthetic policy assertion is not a native namespace qualification.
        assert str(profile_path) not in selected_policy.argv
        assert str(source.parent) not in selected_policy.argv
        staged_input, staged_output = map(Path, argv[-2:])
        assert staged_input.parent == staged_output.parent != source.parent
        assert selected_policy.argv[-3:] == ("--ro-bind", str(staged_input), str(staged_input))
        calls.append((tuple(argv), kwargs))
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        if kwargs["evidence"] is None:
            return subprocess.run(argv, cwd=package, env=environment, capture_output=True, text=True, timeout=5)
        return invocation_record.run(
            argv,
            directory=kwargs["evidence"],
            stage="structural-protocol-control",
            inputs=kwargs["inputs"],
            dependencies=(Path(R.__file__), script, *kwargs["dependencies"]),
            cwd=package,
            env=environment,
            capture_output=True,
            text=True,
            timeout=5,
        )

    monkeypatch.setattr(C, "package_sandbox_policy", policy)
    monkeypatch.setattr(compiler_call_channel, "compiler_call_transport", transport)
    monkeypatch.setattr(R._Executor, "_invoke", invoke)
    package = P.load_package(candidate, artifact_profile=profile)
    return SimpleNamespace(
        candidate=candidate,
        owner=owner,
        profile=profile,
        source=source,
        output=output,
        package=package,
        work=tmp_path / "private-calls",
        calls=calls,
        script=script,
    )


def _scope(value):
    return R.original_compiler_execution(
        value.owner,
        target_experiment=None,
        candidate=value.candidate,
        work_root=value.work,
    )


@pytest.mark.parametrize("phase2_box", [False, True])
def test_original_caller_preserves_exact_structural_profile_and_opaque_output(structural, phase2_box):
    value = structural
    box = C.PackageSandboxPolicy(("not-invoked",), (), (), value.source.parent, value.candidate, None, "")
    evidence = value.source.parent / "observations"
    with _scope(value), C.boxed_entrypoints(box) if phase2_box else nullcontext():
        result = P.run_entrypoint(
            value.package,
            "emit_target_artifact",
            value.source,
            value.output,
            artifact_profile=value.profile,
            invocation_directory=evidence,
        )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"protocol": "structural-only"}
    assert value.output.read_bytes() == b"\0opaque artifact\xff"
    assert value.calls[0][0][2] == "emit_target_artifact"
    assert value.calls[0][1]["dependencies"] == (value.profile.path,)
    assert len(value.calls) == 1 and list(value.work.iterdir()) == []
    record_path = next(evidence.rglob("invocation.json"))
    record = invocation_record.verify(record_path)
    assert {pin["path"] for pin in record["dependencies"]} >= {str(value.profile.path), str(value.script)}
    assert Path(record["stdout"]["path"]).read_text() == result.stdout
    value.profile.path.write_text("name: changed-after-call\n")
    with pytest.raises(ValueError, match="changed"):
        invocation_record.verify(record_path)


@pytest.mark.parametrize("selection", [None, "dict", "stale"])
def test_missing_forged_or_stale_profile_refuses_before_invocation(structural, selection):
    value = structural
    profile = value.profile
    if selection is None:
        profile = None
    elif selection == "dict":
        profile = {"path": str(profile.path), "sha256": profile.sha256, "name": profile.name}
    else:
        profile.path.write_text("name: changed-before-call\n")
    with _scope(value), pytest.raises(P.CertFailure, match="structural emission only"):
        P.run_entrypoint(value.package, "emit_target_artifact", value.source, value.output, artifact_profile=profile)
    assert not value.calls and value.output.read_bytes() == b"previous output"


def test_structural_profile_cannot_select_legacy_llvm_execution(structural):
    value = structural
    with _scope(value), pytest.raises(P.CertFailure, match="legacy LLVM"):
        P.run_entrypoint(
            value.package, "lower_target_to_llvm", value.source, value.output, artifact_profile=value.profile
        )
    assert not value.calls and value.output.read_bytes() == b"previous output"


@pytest.mark.parametrize("mutation", ["profile", "source"])
def test_mutation_during_process_refuses_before_output_publication(structural, mutation):
    value = structural
    changed = value.profile.path if mutation == "profile" else value.script
    value.script.write_text(
        "from pathlib import Path\nimport sys\n"
        f"Path({str(changed)!r}).write_text('name: changed-during-call\\n')\n"
        "Path(sys.argv[3]).write_bytes(b'untrusted result')\nprint('untrusted stdout')\n"
    )
    expected = P.CertFailure if mutation == "profile" else ValueError
    with _scope(value), pytest.raises(expected, match="changed|another source"):
        P.run_entrypoint(
            value.package, "emit_target_artifact", value.source, value.output, artifact_profile=value.profile
        )
    assert len(value.calls) == 1 and value.output.read_bytes() == b"previous output"


def test_immutable_profile_owner_refuses_python_cache_opt_in(structural):
    value = structural
    with _scope(value), pytest.raises(ValueError, match="Python bytecode.*immutable"):
        P.run_entrypoint(
            value.package,
            "emit_target_artifact",
            value.source,
            value.output,
            write_bytecode=True,
            artifact_profile=value.profile,
        )
    assert not value.calls and not list(value.candidate.rglob("*.pyc"))


def test_legacy_protocol_and_structural_permission_remain_distinct(structural):
    value = structural
    document = yaml.safe_load((value.candidate / "manifest.yaml").read_bytes())
    document["abi_version"] = "0.1"
    document["commands"]["lower_target_to_llvm"] = document["commands"].pop("emit_target_artifact")
    document["commands"]["lower_target_to_llvm"]["argv"][1] = "lower_target_to_llvm"
    (value.candidate / "manifest.yaml").write_text(yaml.safe_dump(document))
    legacy = P.load_package(value.candidate)
    with _scope(value):
        result = P.run_entrypoint(legacy, "emit_target_artifact", value.source, value.output)
    assert result.returncode == 0
    assert value.calls[0][0][2] == "lower_target_to_llvm" and value.calls[0][1]["dependencies"] == ()
    with pytest.raises(P.CertFailure, match="requires artifact ABI 0.2"):
        P.load_package(value.candidate, artifact_profile=value.profile)
    with pytest.raises(BundleError, match="invalid"):
        SelectedArtifactProfile(value.profile.path, "", value.profile.sha256, value.profile.bytes).validate()
