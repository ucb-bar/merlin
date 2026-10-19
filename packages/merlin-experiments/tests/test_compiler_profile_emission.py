"""Transport adversaries; synthetic profile fixture grants no target capability."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.feedback import compiler_profile_emission as E
from merlin_experiments.phase1.feedback import compiler_profile_runtime as R
from test_compiler_profile_execution import execution  # noqa: F401 -- shared synthetic transport fixture

from merlin.common import invocation_record
from merlin.common.cpu_pool import current_thread_owns_cpu_lease
from merlin.targetgen.compiler_pass_selection import _pin


@pytest.fixture
def emitted(request, monkeypatch):
    owner, package, _ = request.getfixturevalue("execution")
    owner.bundle_manifest = package.parent / "manifest.yaml"
    owner.bundle_manifest.write_text("synthetic transport fixture\n")
    owner.profile = {"sha256": "a" * 64}
    owner.snapshot_digest = "b" * 64
    def verify(self):
        assert current_thread_owns_cpu_lease()

    monkeypatch.setattr(R.CommonTransformService, "_verify", verify)
    pkg = SimpleNamespace(directory=package, integrity_exempt=False, manifest={})
    monkeypatch.setattr(E.P, "load_package", lambda path: pkg)
    monkeypatch.setattr(E.P, "integrity_scan", lambda pkg: None)
    code = "import pathlib,sys; pathlib.Path(sys.argv[2]).write_text('{}'); print('module {}')"
    monkeypatch.setattr(
        E.P, "_resolve_argv", lambda pkg, name, inp, out: [sys.executable, "-c", code, str(inp), str(out)]
    )
    monkeypatch.setattr(E.P, "_needs_interpreter", lambda *args: False)

    def invoke(self, policy, package, argv, **kwargs):
        return invocation_record.run(
            argv,
            directory=kwargs["evidence"],
            stage="original_profile_package",
            inputs=kwargs["inputs"],
            dependencies=(Path(R.__file__),),
            capture_output=True,
            text=True,
        )

    monkeypatch.setattr(R._Executor, "_invoke", invoke)
    capsule = package.parent / "capsule"
    capsule.mkdir()
    source = capsule / "capsule.interface.mlir"
    source.write_text("module {}")
    (capsule / "capsule.yaml").write_text("fixture: true")
    result, receipt = E.emit_analysis(
        service=owner,
        target_experiment=None,
        package=package,
        input_mlir=source,
        command_buffer=package.parent / "command.json",
        lowered=package.parent / "lowered.mlir",
        work_root=package.parent / "observed",
    )
    return owner, package, source, result, receipt


def test_actual_original_scope_retains_temporary_products(emitted):
    _, _, source, result, receipt = emitted
    record = receipt.record()
    assert result.returncode == 0
    assert record["schema"] == E.SCHEMA
    assert Path(record["copied_input"]["path"]).read_bytes() == source.read_bytes()
    assert Path(record["copied_output"]["path"]).read_bytes() == Path(record["command_buffer"]["path"]).read_bytes()
    assert record["copied_input"]["path"] != record["source"]["path"]
    assert Path(record["lowered"]["path"]).read_text() == result.stdout
    assert R._EMISSION_OBSERVATION.get() is None
    assert not current_thread_owns_cpu_lease()


@pytest.mark.parametrize("key", ["source", "copied_input", "copied_output", "command_buffer", "lowered", "invocation"])
def test_mutated_product_or_original_source_refuses(emitted, key):
    *_, receipt = emitted
    record = receipt.record()
    Path(record[key]["path"]).write_text("changed original bytes")
    with pytest.raises(ValueError, match="artifact changed"):
        receipt.revalidate()


def test_saved_json_and_unissued_objects_cannot_mint_receipt(emitted):
    *_, receipt = emitted
    with pytest.raises(TypeError, match="actual original executor"):
        E.OriginalProfileEmission(receipt.record())
    forged = E.OriginalProfileEmission.__new__(E.OriginalProfileEmission)
    forged._record = receipt.record()
    with pytest.raises(ValueError, match="live issuer"):
        forged.revalidate()


def test_changed_compiler_and_changed_implementation_refuse(emitted, monkeypatch):
    _, package, _, _, receipt = emitted
    original = (package / "compiler.py").read_bytes()
    (package / "compiler.py").write_text("changed compiler")
    with pytest.raises(ValueError, match="compiler changed"):
        receipt.revalidate()
    (package / "compiler.py").write_bytes(original)
    monkeypatch.setattr(E, "_sources", lambda: [])
    with pytest.raises(ValueError, match="implementation changed"):
        receipt.revalidate()


def test_complete_native_source_binding_and_foreign_arm_refusal(emitted):
    _, _, source, _, receipt = emitted
    record = receipt.record()
    copied = source.parent / "bundle-copy.mlir"
    copied.write_bytes(source.read_bytes())
    native = source.parent / "native.json"
    native.write_text(
        json.dumps(
            {
                "source": {
                    "capsule_declaration": _pin(source.parent / "capsule.yaml"),
                    "interface": _pin(copied),
                }
            }
        )
    )
    arm = dict(
        compiler_package=record["compiler_package"],
        command_buffer_artifact=record["command_buffer"],
        lowered=record["lowered"],
        native_result=_pin(native),
    )

    class Owner:
        def check_compiler_binding(self, package):
            assert current_thread_owns_cpu_lease()
            assert package == Path(record["compiler_package"]["path"])

        def revalidate(self):
            assert current_thread_owns_cpu_lease()

    assert receipt.check_native_artifacts(Owner(), arm)["schema"] == E.SCHEMA
    wrong = {**arm, "lowered": _pin(source)}
    with pytest.raises(ValueError, match="another native input"):
        receipt.check_native_artifacts(Owner(), wrong)
    copied.write_text("changed runtime source")
    with pytest.raises(ValueError, match="source custody"):
        receipt.check_native_artifacts(Owner(), arm)


def test_foreign_or_nested_observation_refuses(emitted):
    owner, package, source, _, receipt = emitted
    with pytest.raises(ValueError, match="original selected invocation"):
        E.retain_original_products(object(), None, None, "emit_analysis_bundle", source, source, source, None, source)
    token = R._EMISSION_OBSERVATION.set(receipt)
    try:
        with pytest.raises(ValueError, match="no nested observation"):
            E.emit_analysis(
                service=owner,
                target_experiment=None,
                package=package,
                input_mlir=source,
                command_buffer=source.parent / "again.json",
                lowered=source.parent / "again.mlir",
                work_root=source.parent / "again",
            )
    finally:
        R._EMISSION_OBSERVATION.reset(token)


def test_replaced_runtime_method_refuses_live_receipt(emitted, monkeypatch):
    *_, receipt = emitted
    monkeypatch.setattr(R._Executor, "_verify", lambda *args: None)
    with pytest.raises(ValueError, match="implementation changed"):
        receipt.revalidate()


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt])
def test_failed_final_issuance_revokes_observation(emitted, monkeypatch, error_type):
    owner, package, source, _, _ = emitted
    caught = []

    def fail(self):
        caught.append(self)
        raise error_type("final original custody refusal")

    monkeypatch.setattr(E.OriginalProfileEmission, "revalidate", fail)
    with pytest.raises(error_type, match="final original custody refusal"):
        E.emit_analysis(
            service=owner,
            target_experiment=None,
            package=package,
            input_mlir=source,
            command_buffer=source.parent / "retry.json",
            lowered=source.parent / "retry.mlir",
            work_root=source.parent / "retry",
        )
    assert caught and caught[-1] not in E._ISSUED and caught[-1] not in E._PENDING
    assert R._EMISSION_OBSERVATION.get() is None
