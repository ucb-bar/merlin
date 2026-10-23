"""Real private grading/source admission; hardware transport is a unit-test substitute.

The target verifier has separate actual artifact controls. These tests never
launch a simulator or create a real corpus/release.
"""

import copy
import hashlib
import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from xdsl.dialects.builtin import FunctionType, StringAttr

from merlin.common.paths import repo_root
from merlin.common.yaml import load_yaml
from merlin.frontends.linalg_mlir import parse_mlir_file
from merlin.kernels.shapes import observe_contractions
from merlin.llvmlower.result_layout import ATTRIBUTE
from merlin.targetgen import core_aten_capsules as C
from merlin.targetgen import core_aten_device as D
from merlin.targetgen import core_aten_spec_int8 as S
from merlin.targetgen import spike_full_call as F
from merlin.targetgen.capsule_common import load_capsule, score_lanes
from merlin.targetgen.core_aten_capture import case_capture_name
from merlin.targetgen.golden_store import FILES, write_golden
from merlin.targetgen.host_capabilities import validate_host_capabilities
from merlin.targetgen.software_spec import load_software_spec
from merlin.targetgen.spec_fact_drift import resolve_spec


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def captured_source(tmp_path_factory):
    python = os.environ["MERLIN_M2M_PYTHON"]
    destination = tmp_path_factory.mktemp("qualified-score-source")
    script = r"""
import json,sys,torch,m2m
import torch.ao.quantization.fx._decomposed
class Model(torch.nn.Module):
 def forward(self,a,b):
  qa=torch.ops.quantized_decomposed.quantize_per_tensor.default(a,.125,0,-128,127,torch.int8)
  qb=torch.ops.quantized_decomposed.quantize_per_tensor.default(b,.125,0,-128,127,torch.int8)
  raw=torch.bmm(qa.int(),qb.int())
  return ((raw.reshape(1,1).float()*.125)*.125).reshape(1,1,1)
inputs=(torch.full((1,1,1024),-16.),torch.full((1,1024,1),-16.))
result=m2m.convert(Model(),inputs,backend="fx_importer",capture_trace=True)
assert result.ok and result.capture_trace["status"]=="complete"
open(sys.argv[1]+"/source.mlir","w").write(result.mlir_text)
open(sys.argv[1]+"/layout.json","w").write(result.module.attributes["m2m.result_layout"].data)
"""
    run = subprocess.run([python, "-c", script, str(destination)], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr
    return destination


@pytest.fixture
def prepared(tmp_path, monkeypatch, captured_source):
    base = repo_root() / "examples/gemmini/target"
    host = repo_root() / "examples/gemmini/platform/host-capabilities-core-aten.yaml"
    spec, _ = resolve_spec(load_software_spec(base / "software-spec.yaml"), {})
    selection = dict(
        spec=spec,
        host=validate_host_capabilities(load_yaml(host)),
        identities={
            key: dict(path=str(path), sha256=sha(path))
            for key, path in (("software_spec", base / "software-spec.yaml"), ("host_capabilities", host))
        },
    )
    arrays = [np.full((1, 1, 1024), -16, np.float32), np.full((1, 1024, 1), -16, np.float32)]
    result = np.full((1, 1, 1), 2**18, np.float32)

    def document(array):
        return dict(
            kind="tensor",
            dtype="float32",
            shape=list(array.shape),
            stride=[s // array.itemsize for s in array.strides],
            storage_offset=0,
            requires_grad=False,
            values=array.tolist(),
        )

    arguments = dict(args=dict(kind="tuple", items=[document(a) for a in arrays]), kwargs={})
    case = dict(
        overload="aten.bmm.default",
        case_id="synthetic-quantized-midpoint",
        arguments=arguments,
        post_arguments=copy.deepcopy(arguments),
        expected=document(result),
        comparison="torch_close",
        comparison_parameters=dict(rtol=0, atol=0),
        mutated_arguments=[],
        output_input_aliases=[],
    )
    source = tmp_path / "capture" / case_capture_name(case)
    source.mkdir(parents=True)
    layout = json.loads((captured_source / "layout.json").read_text())
    (source / "capture.json").write_text(
        json.dumps(
            dict(
                overload=case["overload"],
                status="captured_exact",
                capture_meta=dict(
                    ok=True,
                    opaque=0,
                    input_abi=[dict(dtype="f32", shape=list(a.shape)) for a in arrays],
                    output_abi=[dict(dtype="f32", shape=[1, 1, 1])],
                    result_contract=layout,
                ),
            )
        )
    )
    (source / "capsule.linalg.mlir").write_bytes((captured_source / "source.mlir").read_bytes())
    (source / "inputs.json").write_text(json.dumps([a.tolist() for a in arrays]))
    (source / "golden.json").write_text("{}")
    (source / "capsule.pytorch.py").write_text("# synthetic unit source\n")

    def numeric_fixture(bundle, report, flow):
        module = parse_mlir_file(bundle / "model.mlir")
        plan = S.admission_plan(module, selection, captured=True)
        assert plan["status"] == "admitted", plan
        (site,) = plan["sites"]
        ((mac, _),) = observe_contractions(module)
        _, restored, _ = S.host_readout_chain(mac, with_bridge=True)
        function = next(op for op in module.walk() if op.name == "func.func" and op.sym_name.data == "forward")
        returned = next(op for op in function.body.block.ops if op.name == "func.return")
        index = len(returned.operands)
        returned.operands = [*returned.operands, mac.results[0], restored]
        function.properties["function_type"] = FunctionType.from_lists(
            [a.type for a in function.body.block.args], [v.type for v in returned.operands]
        )
        if ATTRIBUTE in module.attributes:
            contract = json.loads(module.attributes[ATTRIBUTE].data)
            contract["results"].extend([None, None])
            module.attributes[ATTRIBUTE] = StringAttr(json.dumps(contract))
        module.verify()
        (bundle / "model.mlir").write_text(str(module) + "\n")
        raw = [
            result.tobytes(),
            *[a.tobytes() for a in arrays],
            np.array([[[2**24]]], np.int32).tobytes(),
            result.tobytes(),
        ]
        assert index == 3
        site.update(accumulator_output_index=index, readout_output_index=index + 1)
        plan.update(
            quantized_contractions=1,
            calibration=dict(performed=True, source="synthetic_unit_qparams"),
            integer_reference=dict(
                status="prepared",
                engine="independent_unit_integer_math",
                expected_output_bytes={str(i): value.hex() for i, value in enumerate(raw)},
            ),
            source_declaration_continuity=dict(
                status="preserved", initial_source_admissions=[step for step in site["steps"] if step["step"] == 3]
            ),
        )
        report["output_count"] += 2
        report["cases"][0].update(numeric_policy="spec_int8", numeric_admission=copy.deepcopy(plan))
        report["numeric_preparation"] = dict(selection=selection, **plan)

    monkeypatch.setattr(C, "prepare_numeric_fixture", numeric_fixture)
    public, private = tmp_path / "public", tmp_path / "_private/public"
    manifest = C.write_capsules(
        {"cases": [case]}, source.parent, public, private, numeric_flow=dict(policy="spec_int8")
    )
    capsule = load_capsule(public / manifest["capsules"][0]["id"])
    capsule.update(lane_expectation="any", scored=True)
    bundle = tmp_path / "runtime"
    C.runtime_bundle(capsule, bundle)
    provider_path = tmp_path / "trusted_provider.py"
    provider_path.write_text("# selected trusted unit transport\n")
    elf, trace = bundle / "unit.elf", bundle / "unit.trace"
    elf.write_bytes(b"unit ELF transport")
    trace.write_bytes(b"unit trace transport")
    evidence = dict(
        elf=str(elf),
        trace=str(trace),
        elf_sha256=sha(elf),
        trace_sha256=sha(trace),
        case_id=case["case_id"],
        target="gemmini",
        status="measured",
        attribution="single_case_shard",
        executed_instructions=1,
    )
    monkeypatch.setattr(
        D,
        "verify_execution_evidence",
        lambda e: (
            isinstance(e, dict)
            and sha(Path(e["elf"])) == e["elf_sha256"]
            and sha(Path(e["trace"])) == e["trace_sha256"]
        ),
    )
    metadata = [{k: v for k, v in document(a).items() if k not in {"kind", "values"}} for a in (result, *arrays)]
    raw = [
        result.tobytes(),
        *[a.tobytes() for a in arrays],
        np.array([[[2**24]]], np.int32).tobytes(),
        result.tobytes(),
    ]
    execution = dict(
        lane="device",
        executed_instructions=1,
        execution_error=None,
        output_bytes=raw,
        semantic_readback=dict(
            metadata=metadata,
            aliases=[[], [1], [2]],
            pre_bytes={str(i): a.tobytes().hex() for i, a in enumerate(arrays)},
        ),
        execution_evidence=evidence,
        routing=dict(
            lane="device",
            routed=True,
            target="gemmini",
            routed_operations=["unit-source-bound"],
            execution_evidence=evidence,
        ),
        provenance=dict(execution_provider_sha256=sha(provider_path)),
        engine="unit_transport",
        derived_from_rtl=False,
    )
    calls = []

    def verify_artifacts(actual_bundle, **kwargs):
        calls.append(kwargs)
        return dict(
            schema="merlin.numeric_execution_artifacts.v1",
            status="verified",
            bindings=dict(
                source_sha256=sha(actual_bundle / "model.mlir"),
                catalog_sha256="a" * 64,
                target_ir_sha256="b" * 64,
                object_sha256="c" * 64,
                elf_sha256=evidence["elf_sha256"],
                trace_sha256=evidence["trace_sha256"],
            ),
            checks=dict(schedule=True, partial_sums=True, device_binding=True),
        )

    provider = SimpleNamespace(verify_numeric_execution_artifacts=verify_artifacts)
    monkeypatch.setattr(D, "load_execution_provider", lambda *a, **k: provider)

    @contextmanager
    def facts(*args):
        yield {}

    monkeypatch.setattr(D, "selected_facts", facts)
    monkeypatch.setattr(S, "select_spec", lambda *a: selection)
    adapter = F.full_call_adapter("gemmini", isa="unit", execution_provider=provider_path)
    adapter.run_full_call = lambda **kwargs: copy.deepcopy(execution)
    return SimpleNamespace(
        capsule=capsule,
        bundle=bundle,
        execution=execution,
        provider=provider,
        adapter=adapter,
        selection=selection,
        calls=calls,
        private=private,
        raw=raw,
        tmp=tmp_path,
    )


def grade(ctx):
    run = ctx.execution
    return C.grade_readback(
        ctx.capsule,
        run["output_bytes"],
        semantic_readback=run["semantic_readback"],
        routing=run["routing"],
        provenance=run["provenance"],
    )


def qualify(ctx):
    return ctx.adapter.qualify_full_call(
        ctx.capsule, bundle=ctx.bundle, execution=ctx.execution, verdict=grade(ctx), package_dir=ctx.tmp, timeout=30
    )


def test_repeated_qualification_retains_distinct_private_proof_attempts(prepared):
    original = prepared.provider.verify_numeric_execution_artifacts

    def verifier(bundle, **kwargs):
        assert not kwargs["workdir"].exists()
        kwargs["workdir"].mkdir()
        return original(bundle, **kwargs)

    prepared.provider.verify_numeric_execution_artifacts = verifier
    assert qualify(prepared)["status"] == "qualified"
    assert qualify(prepared)["status"] == "qualified"
    assert prepared.calls[-1]["workdir"] != prepared.calls[-2]["workdir"]


def test_normalized_catalog_source_requires_complete_trusted_reproduction(prepared):
    from merlin.runtime.backends.zephyr_model import _prepare_model_mlir

    build = prepared.bundle / "spike-build"
    catalog_dir = build / "catalog"
    catalog_dir.mkdir(parents=True)
    scratch = prepared.tmp / "trusted-normalization"
    scratch.mkdir()
    normalized = _prepare_model_mlir(prepared.bundle / "model.mlir", scratch, bundle_dir=prepared.bundle)
    source = catalog_dir / "source.mlir"
    source.write_bytes(normalized.read_bytes())
    manifest = catalog_dir / "catalog.json"
    manifest.write_text(json.dumps({"source_snapshot": str(source)}))
    (build / "device_signatures.json").write_text(json.dumps({"catalog_manifest": str(manifest)}))
    original = prepared.provider.verify_numeric_execution_artifacts

    def verifier(bundle, **kwargs):
        proof = original(bundle, **kwargs)
        proof["bindings"]["source_sha256"] = sha(source)
        return proof

    prepared.provider.verify_numeric_execution_artifacts = verifier
    assert sha(source) != sha(prepared.bundle / "model.mlir")
    receipt = qualify(prepared)
    assert receipt["status"] == "qualified"
    assert receipt["bindings"]["source_sha256"] == sha(prepared.bundle / "model.mlir")
    # A self-rehashed changed source still cannot pass the independent byte join.
    source.write_text(source.read_text() + "\n")
    refused = qualify(prepared)
    assert refused["status"] == "unqualified"
    assert refused["reasons"] == ["catalog_source_normalization_differs"]


def repack(ctx, mutate):
    golden = C.load_private(ctx.capsule)
    mutate(golden["batch_map"])
    plan = golden["batch_map"]["numeric_preparation"]
    golden["batch_map"]["cases"][0]["numeric_admission"] = {
        k: copy.deepcopy(v) for k, v in plan.items() if k != "selection"
    }
    answers = ctx.private / ctx.capsule["name"]
    write_golden(answers, golden)
    ctx.capsule["full_call_golden"] = {name: sha(answers / name) for name in FILES if (answers / name).exists()}
    C.runtime_bundle(ctx.capsule, ctx.bundle)


def test_real_private_grade_and_source_gate_issue_answer_free_receipt(prepared):
    receipt = qualify(prepared)
    assert receipt["status"] == "qualified", receipt
    assert len(receipt["bindings"]) == 9 and all(v is True for v in receipt["checks"].values())
    assert receipt["bindings"]["case_sha256"] == prepared.capsule["full_call_golden"]["golden.yaml"]
    assert len(prepared.calls) == 1
    assert "expected_output_bytes" not in json.dumps(receipt)
    assert str(prepared.tmp) not in json.dumps(receipt)


def test_identical_f32_readout_cannot_hide_wrong_raw_accumulator(prepared):
    assert np.float32(16777217).tobytes() == np.float32(16777216).tobytes()
    prepared.execution["output_bytes"][3] = np.array([[[16777217]]], np.int32).tobytes()
    row = next(iter(grade(prepared)["cases"].values()))
    assert row["status"] == "mismatch" and row["integer_reference"]["status"] == "mismatch"
    assert qualify(prepared)["status"] == "unqualified" and not prepared.calls


@pytest.mark.parametrize(
    "defect",
    ["no_sites", "no_quantizers", "empty_reference", "bool_count", "unadmitted", "native_narrowing", "wrong_raw_index"],
)
def test_missing_malformed_or_native_caveat_never_qualifies(prepared, defect):
    def mutate(report):
        plan = report["numeric_preparation"]
        if defect == "no_sites":
            plan["sites"] = []
        elif defect == "no_quantizers":
            plan["quantization_sites"] = []
        elif defect == "empty_reference":
            plan["integer_reference"]["expected_output_bytes"] = {}
        elif defect == "bool_count":
            plan["quantized_contractions"] = True
        elif defect == "unadmitted":
            plan["status"] = "refused"
        elif defect == "native_narrowing":
            plan["host_int8_narrowing"] = True
        else:
            plan["sites"][0]["accumulator_output_index"] = 0

    repack(prepared, mutate)
    assert qualify(prepared)["status"] == "unqualified" and not prepared.calls


def test_correct_host_with_nonnull_routing_retains_functional_grade_but_no_device_credit(prepared):
    prepared.execution.update(lane="host", executed_instructions=0, routing=dict(lane="host", routed=False))
    prepared.execution.pop("execution_evidence")
    row = next(iter(grade(prepared)["cases"].values()))
    assert row["status"] == "pass" and row["numeric_execution_policy"] == "original_host"
    assert qualify(prepared)["status"] == "unqualified" and not prepared.calls


@pytest.mark.parametrize("policy", [None, "int8_w8a8", "fp32", {"policy": "spec_int8"}])
def test_unprepared_and_diagnostic_flows_never_get_spec_credit(prepared, policy):
    prepared.capsule["operation"]["attributes"]["numeric_flow"] = policy
    assert qualify(prepared)["status"] == "unqualified" and not prepared.calls


@pytest.mark.parametrize("defect", ["source", "provider", "elf", "forged_hook"])
def test_changed_actual_execution_bindings_never_qualify(prepared, defect):
    if defect == "source":
        (prepared.bundle / "model.mlir").write_text((prepared.bundle / "model.mlir").read_text() + "\n")
    elif defect == "provider":
        prepared.execution["provenance"]["execution_provider_sha256"] = "d" * 64
    elif defect == "elf":
        Path(prepared.execution["execution_evidence"]["elf"]).write_bytes(b"changed")
    else:
        prepared.provider.verify_numeric_execution_artifacts = lambda *a, **k: dict(
            status="verified", checks=dict(schedule=True)
        )
    assert qualify(prepared)["status"] == "unqualified"


def run(ctx, adapters=None):
    config = SimpleNamespace(target="gemmini", rtl_tiers=())
    return C.run_capsule(
        ctx.capsule,
        ctx.tmp,
        SimpleNamespace(run_path=ctx.tmp / "run"),
        config,
        adapters or {"L2": ctx.adapter},
        None,
        {},
        30,
        False,
    )


def test_trusted_full_call_result_overwrites_candidate_claims(prepared):
    prepared.execution.update(
        spec_qualified_device=dict(status="qualified"), full_call_execution_identity=dict(case_sha256="f" * 64)
    )
    result = run(prepared)
    assert result["status"] == "pass", result
    assert result["spec_qualified_device"]["status"] == "qualified", result
    assert result["full_call_execution_identity"]["case_sha256"] == prepared.capsule["full_call_golden"]["golden.yaml"]
    score = {}
    score_lanes([result], score)
    assert score["device_lane_scored_pass"] == 1
    assert score["spec_qualified_device_pass"] == 1 and score["device_unqualified_pass"] == 0


def test_missing_trusted_producer_ignores_candidate_qualification_json(prepared):
    prepared.execution["spec_qualified_device"] = qualify(prepared)
    del prepared.adapter.qualify_full_call
    result = run(prepared)
    assert result["status"] == "pass"
    assert result["spec_qualified_device"]["status"] == "unqualified"


def test_later_qualified_tier_cannot_erase_earlier_unqualified_tier(prepared):
    prepared.capsule["required_oracle_tiers"] = ["L2", "L3"]
    earlier = SimpleNamespace(run_full_call=prepared.adapter.run_full_call, compiles_source_bundle=True)
    result = run(prepared, {"L2": earlier, "L3": prepared.adapter})
    assert result["status"] == "pass"
    assert result["tiers"]["L3"]["spec_qualified_device"]["status"] == "qualified"
    assert result["spec_qualified_device"]["status"] == "unqualified"
