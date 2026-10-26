"""Certificate failures affect admission only when no independent device candidate wins."""

import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from mlir_fixtures import pointwise

from merlin.common.paths import repo_root
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen import core_aten_components as components
from merlin.targetgen import core_aten_device as device
from merlin.targetgen import core_aten_exact_dtype as exact
from merlin.targetgen import core_aten_numeric as numeric
from merlin.targetgen import core_aten_spec_int8 as spec
from merlin.targetgen import native_ieee_certificate as native
from merlin.targetgen.native_cpu_errors import NativeCPUCertificateUnavailable, require_certificate_available
from merlin.targetgen.native_dtype_domain import component_semantics
from merlin.targetgen.software_spec import load_software_spec


@pytest.fixture(params=["missing", "mismatched"])
def unavailable_certificate(request, tmp_path, monkeypatch):
    certificate = json.loads(Path(native.__file__).with_suffix(".json").read_text())
    library = tmp_path / "libtorch_cpu.so"
    if request.param == "mismatched":
        library.write_bytes(b"not the certified library")
    certificate["host"]["library"] = str(library)
    content = dict(certificate)
    content.pop("content_sha256")
    certificate["content_sha256"] = native.digest(content)
    (tmp_path / "native_ieee_certificate.json").write_text(json.dumps(certificate))
    monkeypatch.setattr(native, "__file__", str(tmp_path / "native_ieee_certificate.py"))
    return library


def arithmetic():
    return pointwise(["tensor<4xf16>"], "tensor<4xf16>", "%v=arith.addf %a,%a:f16")


def declined_components(directory, report):
    return components.prepare_integer_components(
        directory, report, {}, SimpleNamespace(admit_numeric_schedule=lambda *a, **kw: None),
        None, arithmetic(), limb_attribute="test.limb",
    )


def test_shared_prover_preserves_optional_refusal(unavailable_certificate):
    proof = component_semantics(arithmetic())
    assert proof["status"] == "refused"
    assert proof["code"] == NativeCPUCertificateUnavailable.code
    assert proof["reason"]


@pytest.mark.parametrize("available", [False, True])
@pytest.mark.parametrize("source_dtype", ["f64", "f32"])
def test_binary64_host_refusal_is_independent_of_certificate(tmp_path, monkeypatch, available, source_dtype):
    body = "%v=arith.addf %a,%a:f64" if source_dtype == "f64" else (
        "%wide=arith.extf %a:f32 to f64\n%v=arith.addf %wide,%wide:f64"
    )
    module = pointwise([f"tensor<4x{source_dtype}>"], "tensor<4xf64>", body)
    stored = json.loads(Path(native.__file__).with_suffix(".json").read_text())

    def certificate():
        if not available:
            raise FileNotFoundError("missing certificate library")
        return stored

    monkeypatch.setattr(native, "certificate", certificate)
    assert component_semantics(module)["code"] == "native_binary64_arithmetic_form_unproved"
    if source_dtype == "f32":
        return
    software_spec = repo_root() / "examples/gemmini/target/software-spec.yaml"
    selection = dict(spec=load_software_spec(software_spec), host={}, identities={}, capability_contract={})
    monkeypatch.setattr(spec, "select_spec", lambda *args: selection)
    (tmp_path / "model.mlir").write_text(str(module))
    report = dict(cases=[dict(status="bundled", input_abi=[dict(dtype=source_dtype, shape=[4])],
                             output_abi=[dict(dtype="f64", shape=[4])], output_indices=[0])])
    result = numeric.prepare_spec_int8(
        tmp_path, report, target=selection["spec"]["target"], facts={}, software_spec=software_spec,
        host_capabilities=None, provider=SimpleNamespace(admit_numeric_schedule=lambda *a, **kw: None),
    )
    assert result["native_admission"]["status"] == "host"
    assert result["refusal_details"]["code"] == "native_binary64_arithmetic_form_unproved"


@pytest.mark.parametrize("control", [
    '%ok=arith.constant true\ncf.assert %ok, "bounds"',
    '%lo=arith.constant 0:index\n%hi=arith.constant 1:index\n'
    'scf.for %i=%lo to %hi step %hi {scf.yield}',
])
def test_independent_control_flow_refusal_does_not_require_certificate(
    unavailable_certificate, control,
):
    source = str(arithmetic()).replace("func.return", control + "\nfunc.return")
    module = parse_mlir_text(source)
    module.verify()
    proof = component_semantics(module)
    assert proof["code"] == "native_primitive_semantics_unproved"
    assert "pointwise theorem" in proof["reason"]


def test_native_refusal_does_not_block_later_device_candidate(
    unavailable_certificate, tmp_path, monkeypatch,
):
    from merlin.targetgen import native_byte_activation, native_quantized_activation, native_wide_activation

    (tmp_path / "model.mlir").write_text(str(arithmetic()))
    report = dict(cases=[dict(
        status="bundled", input_abi=[dict(dtype="f16", shape=[4])], output_abi=[dict(dtype="f16", shape=[4])],
    )])
    monkeypatch.setattr(spec, "select_spec", lambda *a: {})
    for module, name in (
        (exact, "prepare_bitwise"), (native_byte_activation, "prepare"), (native_wide_activation, "prepare"),
    ):
        monkeypatch.setattr(module, name, lambda *a: None)
    admitted = dict(status="admitted", exact_dtype_sites=1)

    def independent_candidate(*args):
        assert report["native_certificate"] == "unavailable"
        return admitted

    monkeypatch.setattr(native_quantized_activation, "prepare", independent_candidate)
    result = numeric._prepare_spec_int8(
        tmp_path, report, target="test", facts={}, software_spec=None, host_capabilities=None,
        provider=SimpleNamespace(admit_numeric_schedule=lambda *a, **kw: None),
    )
    assert result is admitted


@pytest.mark.parametrize("disposition", ["host", "device", "policy_host"])
def test_certificate_is_decisive_only_at_final_admission(
    unavailable_certificate, disposition, tmp_path, monkeypatch,
):
    report = {"cases": []}
    source = tmp_path / "model.mlir"
    source.write_text(str(arithmetic()))
    original = source.read_bytes()

    def candidates(directory, report, **kwargs):
        assert declined_components(directory, report) is None
        # Candidate preparation may rewrite sidecars; a host result must restore them.
        source.write_text("candidate source")
        refusal = json.loads((directory / "exact-dtype-refusal.json").read_text())
        return dict(
            native_admission=dict(
                status="device" if disposition == "device" else "host",
                reason_code="non_native_unit_construction" if disposition == "policy_host"
                else "native_dtype_exact_components_unavailable",
            ),
            refusal_details=refusal,
        )

    monkeypatch.setattr(numeric, "_prepare_spec_int8_candidates", candidates)
    if disposition == "host":
        with pytest.raises(NativeCPUCertificateUnavailable):
            numeric.prepare_spec_int8(tmp_path, report)
        assert source.read_bytes() == original
    else:
        result = numeric.prepare_spec_int8(tmp_path, report)
        assert result["native_certificate"] == "unavailable"
        assert result["native_admission"]["status"] == ("device" if disposition == "device" else "host")
        assert (source.read_bytes() == original) == (disposition == "policy_host")


@pytest.mark.parametrize("consumer", ["provider", "covering", "coverage", "catalog"])
def test_required_provider_proof_raises(unavailable_certificate, consumer, tmp_path):
    proof = component_semantics(arithmetic())
    provider = SimpleNamespace(admit_numeric_schedule=lambda *a, **kw: proof)
    calls = {
        "provider": lambda: spec.provider_proof(provider, "", package=None, policy={}),
        "covering": lambda: spec.covering_proof(provider, "", [], package=None, policy={}),
        "coverage": lambda: spec.provider_coverage(None, proof, None),
        "catalog": lambda: exact.bitwise_catalog_coverage(None, proof),
    }
    with pytest.raises(NativeCPUCertificateUnavailable):
        calls[consumer]()


def test_admitted_proof_and_ordinary_refusal_are_unchanged():
    for proof in (
        dict(status="proven", certificate_sha256="supplied"),
        dict(status="refused", code="native_cpu_arithmetic_certificate_refused", reason="counterexample"),
    ):
        assert require_certificate_available(proof) is proof
        provider = SimpleNamespace(admit_numeric_schedule=lambda *a, **kw: proof)
        assert spec.provider_proof(provider, "", package=None, policy={}) == (proof, None)
        assert components.Refusal(**proof).args == (proof,)


def test_bounded_runner_records_harness_error_without_host(
    unavailable_certificate, tmp_path, monkeypatch,
):
    from merlin.targetgen import core_aten_batch as batch

    script = repo_root() / "build_tools/scripts/run_bounded_core_aten_suite.py"
    module_spec = importlib.util.spec_from_file_location("certnarrow_runner", script)
    runner = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(runner)
    case = {"case_id": "fixture", "overload": "aten.add.Tensor"}
    captured = tmp_path / "captures" / runner.case_capture_name(case)
    captured.mkdir(parents=True)
    (captured / "capture.json").write_text(json.dumps({"status": "captured_exact"}))
    (captured / "capsule.linalg.mlir").write_text("module {}")
    monkeypatch.setattr(runner, "bundle_admission_reason", lambda *a: None)
    monkeypatch.setattr(batch, "build_core_aten_batch", lambda *a: {"cases": [{"status": "bundled"}]})
    monkeypatch.setattr(device, "selected_facts", lambda *a: nullcontext({}))
    monkeypatch.setattr(device, "load_execution_provider", lambda *a: SimpleNamespace(__file__=__file__))

    def prepare(directory, report, **kwargs):
        assert declined_components(directory, report) is None
        refusal = json.loads((directory / "exact-dtype-refusal.json").read_text())
        monkeypatch.setattr(numeric, "_prepare_spec_int8_candidates", lambda *a, **kw: dict(
            native_admission=dict(status="host", reason_code="native_dtype_exact_components_unavailable"),
            refusal_details=refusal,
        ))
        return original_prepare(directory, report, **kwargs)

    original_prepare = numeric.prepare_spec_int8
    monkeypatch.setattr(numeric, "prepare_spec_int8", prepare)
    monkeypatch.setattr(device, "select_source_routing", lambda *a: pytest.fail("host routing reached"))
    args = SimpleNamespace(
        output=tmp_path, label="test", arena_mb=1, dram_mb=None, execution_timeout=1,
        target="test", numeric_policy="spec_int8", device_package=tmp_path,
        execution_provider=None, facts=None, software_spec=tmp_path / "software.yaml",
        host_capabilities=tmp_path / "host.yaml", evaluate_only=False,
    )
    runner.execute(args, case, {})
    result = tmp_path / args.label / "shards" / runner.case_capture_name(case) / "result.json"
    observation = json.loads(result.read_text())["observation"]
    assert observation["status"] == "harness_error"
    assert observation["code"] == NativeCPUCertificateUnavailable.code
    assert observation["evidence"] == dict(lane="unassigned", native_certificate="unavailable")
    assert runner.response(args, {"selected_cases": [case]}, {})["results"][case["case_id"]] == observation
