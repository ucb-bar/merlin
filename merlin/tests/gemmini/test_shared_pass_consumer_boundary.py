"""Public package transport boundaries, without compiler or execution qualification.

Only an explicit MERLIN_TEST_GEMMINI_V23_PACKAGE selects the published package.
Injected invocations never execute its compiler or import a support provider.
Neutral manifests keep the transport regressions runnable without that checkout.
"""

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin.common.digest import sha256_file
from merlin.targetgen import capsule_common as CC
from merlin.targetgen import package_runtime as P
from merlin.targetgen.contract.interface_emit import emit_interface_mlir, parse_interface_mlir

PACKAGE_ENV = "MERLIN_TEST_GEMMINI_V23_PACKAGE"


@dataclass(frozen=True)
class PublicPackage:
    package: P.Package
    interface: str
    pins: tuple[tuple[Path, str], ...] = ()

    def revalidate(self):
        for path, expected in self.pins:
            assert sha256_file(path) == expected


def neutral_package(tmp_path):
    manifest = dict(
        artifact_type="mlir_oot_target_backend",
        target="neutral",
        package_id="neutral_legacy_transport",
        language="python",
        authoring=dict(mode="agent_generated_from_rtl_facts"),
        integrity_exempt=False,
        entrypoints=dict(tool="compiler.py"),
        commands={
            "parse": dict(argv=["{tool}", "--verify-diagnostics", "{input_mlir}"]),
            "lower_interface_to_target": dict(argv=["{tool}", "--lower-interface", "{input_mlir}"]),
            "emit_command_buffer": dict(argv=["{tool}", "--buffer={output_json}", "{input_mlir}"]),
            "lower_target_to_llvm": dict(argv=["{tool}", "--emit-target-artifact", "{input_mlir}"]),
            "emit_analysis_bundle": dict(
                argv=["{tool}", "--buffer={output_json}", "--emit-target-artifact", "{input_mlir}"]
            ),
        },
    )
    buffer = dict(
        abi_version="0.1",
        target="neutral",
        tensors=dict(
            A=dict(shape=[1, 4], dtype="i8", role="input"),
            Y=dict(shape=[1, 4], dtype="i8", role="output"),
        ),
        commands=[dict(opcode="MOVEMENT", operands=dict(src="A", dst="Y"))],
    )
    return PublicPackage(P.Package(tmp_path, manifest, tmp_path / "compiler.py"), emit_interface_mlir(buffer))


@pytest.fixture(params=("neutral", "published-v23"))
def public_package(request, tmp_path, record_property):
    if request.param == "neutral":
        return neutral_package(tmp_path)
    import os

    selected = os.environ.get(PACKAGE_ENV)
    if not selected:
        pytest.skip(f"set {PACKAGE_ENV} to the public V23 snapshot; absence establishes no target qualification")
    directory = Path(selected).resolve()
    required = [directory / name for name in ("manifest.yaml", "checkpoint.json", "package-files.json")]
    required.append(directory / "examples/scalar_readout.mlir")
    absent = [str(path) for path in required if not path.is_file()]
    if absent:
        pytest.skip("selected public V23 package files are absent: " + ", ".join(absent))
    manifest = yaml.safe_load(required[0].read_text())
    checkpoint = json.loads(required[1].read_text())
    inventory = json.loads(required[2].read_text())
    assert manifest["target"] == "gemmini"
    assert checkpoint["schema"] == "gemmini.phase1.provisional_checkpoint.v1"
    assert inventory["package_sha256"] == checkpoint["package_sha256"]
    assert inventory["n_files"] == len(inventory["files"])
    pins = [(path, sha256_file(path)) for path in required]
    for relative, identity in inventory["files"].items():
        path = (directory / "package" / relative).resolve()
        assert path.is_relative_to((directory / "package").resolve()) and path.is_file()
        assert path.stat().st_size == identity["bytes"] and sha256_file(path) == identity["sha256"]
        pins.append((path, identity["sha256"]))
    assert "emit_analysis_bundle" in manifest["commands"]
    assert "target_ir_consumer" not in manifest and "emit_target_bundle" not in manifest["commands"]
    tool = directory / manifest["entrypoints"]["tool"]
    assert tool.is_file()
    result = PublicPackage(P.Package(directory, manifest, tool), required[-1].read_text(), tuple(pins))
    record_property(
        "public_transport_scope",
        json.dumps(
            dict(
                manifest_sha256=sha256_file(required[0]),
                checkpoint_sha256=sha256_file(required[1]),
                package_sha256=inventory["package_sha256"],
                selected_hardware=checkpoint["selected_hardware"],
                scope="metadata identities only; no tools, compiler, hardware facts or runtime qualified",
            ),
            sort_keys=True,
        ),
    )
    return result


class UnreadSelection:
    """No semantics can authorize an absent consuming transport declaration."""

    def __getattribute__(self, name):
        raise AssertionError("consumer refusal must precede inspecting selection " + name)


def test_selected_rewrite_refuses_missing_consumer_before_invocation(public_package, tmp_path):
    invocation_names = []

    def forbidden(_package, name, *_args, **_kwargs):
        invocation_names.append(name)
        pytest.fail("missing consumer reached a package invocation")

    generated = tmp_path / "selected"
    with pytest.raises(P.CertFailure, match="explicit.*merlin.target_ir_consumer.v1"):
        CC.lower_interface(
            public_package.package,
            public_package.interface,
            generated,
            contract=None,
            timeout=1,
            invoke=forbidden,
            target_configuration=UnreadSelection(),
        )
    assert invocation_names == [] and not generated.exists()
    public_package.revalidate()


@pytest.mark.parametrize("bundled", [True, False], ids=["bundle", "split"])
def test_legacy_emission_consumes_original_interface_not_saved_target(public_package, tmp_path, bundled):
    # Removing the optional bundle exercises the legacy split protocol, never a
    # modification to the external package or an invented consuming contract.
    package = copy.copy(public_package.package)
    package.manifest = copy.deepcopy(package.manifest)
    if not bundled:
        package.manifest["commands"].pop("emit_analysis_bundle")
    source = tmp_path / "original.interface.mlir"
    source.write_text(public_package.interface)
    original = source.read_bytes()
    original_sha256 = sha256_file(source)
    generated = tmp_path / "legacy"
    diagnostic = 'module { "uninterpreted.diagnostic"() : () -> () }\n'
    changed = 'module { "uninterpreted.changed_diagnostic"() : () -> () }\n'
    artifact = "; injected transport reply for original interface " + original_sha256 + "\nmodule {}\n"
    buffer = parse_interface_mlir(public_package.interface)
    calls = []

    def invoke(_package, name, inp, output=None, *, timeout):
        assert timeout == 1
        inp = Path(inp)
        argv = P._resolve_argv(_package, name, inp, output)
        calls.append(dict(name=name, input=str(inp), input_sha256=sha256_file(inp), argv=argv))
        assert str(inp.resolve()) in argv
        assert inp.read_bytes() == original
        if name == "parse":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if name == "lower_interface_to_target":
            return SimpleNamespace(returncode=0, stdout=diagnostic, stderr="")
        saved = generated / "lowered.target.mlir"
        assert saved.is_file()
        if name in ("emit_analysis_bundle", "emit_command_buffer"):
            assert saved.read_text() == diagnostic
            saved.write_text(changed)
            output.write_text(json.dumps(buffer))
            return SimpleNamespace(returncode=0, stdout=artifact if bundled else "", stderr="")
        assert name == "emit_target_artifact" and saved.read_text() == changed
        return SimpleNamespace(returncode=0, stdout=artifact, stderr="")

    actual_buffer, actual_artifact = CC.lower_interface(
        package, source, generated, contract=None, timeout=1, invoke=invoke, overlap=False
    )
    expected = ["parse", "lower_interface_to_target"]
    expected += ["emit_analysis_bundle"] if bundled else ["emit_command_buffer", "emit_target_artifact"]
    assert [call["name"] for call in calls] == expected
    assert {call["input_sha256"] for call in calls} == {original_sha256}
    retained = generated / "input.interface.mlir"
    assert {call["input"] for call in calls} == {str(retained)}
    assert retained.read_bytes() == source.read_bytes() == original
    assert (generated / "lowered.target.mlir").read_text() == changed
    assert actual_buffer == buffer and actual_artifact == artifact
    assert (generated / "lowered.llvm.mlir").read_text() == artifact
    assert not (generated / "selected.target.mlir").exists()
    assert not (generated / "target_configuration.json").exists()
    public_package.revalidate()
