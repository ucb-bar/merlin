"""Core ATen spec validation is sealed Phase 0 evidence with two-directional mismatches."""

from __future__ import annotations

import copy
import hashlib
import json
import runpy
from argparse import Namespace

import pytest
import yaml
from merlin_experiments.phase0 import spec_validation as SV

from merlin.common.paths import repo_root
from merlin.targetgen.core_aten_capture import case_capture_name

LABEL = "spec-validation"


def _tensor(shape):
    return {"kind": "tensor", "dtype": "float32", "shape": shape, "stride": [1] * len(shape), "storage_offset": 0}


def _case(case_id, overload="aten.mm.default", *, empty=False):
    evidence = json.loads((repo_root() / "merlin/tests/data/exactness/core_aten_193_lane_evidence.json").read_bytes())
    case = copy.deepcopy(next(row for row in evidence["cases"] if row["overload"] == overload))
    case["case_id"] = case_id
    if empty:
        case["expected"] = _tensor([0])
    return case


def _step(status, declaration="contraction", line=109):
    decision = {"status": status, "reason": "" if status == "admitted" else "not declared"}
    if declaration:
        decision.update(declaration=declaration, source={"path": "/spec.yaml", "line": line, "restriction_line": 112})
    return {"step": 3, "op": "contraction", "software": decision}


def _plan(admitted, spec_sha, *, steps=None):
    identities = {"selection": {"identities": {"software_spec": {"sha256": spec_sha}}}}
    if admitted:
        return {
            **identities,
            "status": "admitted",
            "native_admission": {"status": "device"},
            "quantized_contractions": 1,
            "integer_reference": {"status": "prepared"},
            "sites": [{"status": "admitted", "steps": steps or [_step("admitted")]}],
        }
    return {
        **identities,
        "status": "host",
        "native_admission": {"status": "host", "reason_code": "no_admitted_device_candidate"},
        "quantized_contractions": 0,
        "sites": [],
        "non_contraction_sites": [{"steps": steps or [_step("unsupported", declaration=None)]}],
    }


def _result(case, stamp, *, lane, count=0, status="pass", observation_status="executed"):
    evidence = {"executed_instructions": count, "trace_sha256": "a" * 64, "elf_sha256": "b" * 64} if count else None
    routing = {"lane": lane, **({"execution_evidence": evidence} if evidence else {})}
    return {
        "case_id": case["case_id"],
        "case_sha256": SV.case_sha256(case),
        "numeric_policy": "spec_int8",
        "provenance": stamp,
        "observation": {
            "status": observation_status,
            "reason": None if status == "pass" else "bits differ",
            "evidence": {
                "lane": lane,
                "semantic_verdict": {
                    "status": status,
                    "semantic_scope": "full" if status == "pass" else "numeric_only",
                    "numeric_status": status,
                    "routing": routing,
                },
            },
        },
    }


@pytest.fixture
def spec(tmp_path):
    path = tmp_path / "software-spec.yaml"
    document = yaml.safe_load((repo_root() / "examples/gemmini/target/software-spec.yaml").read_bytes())
    document["operations"]["contraction"]["status"] = "reviewed"
    path.write_text(yaml.safe_dump(document))
    return path


def _write_selection(tmp_path, cases):
    writer = runpy.run_path(str(repo_root() / "build_tools/scripts/make_spec_validation_selection.py"))
    suites = {}
    for source, rows in cases.items():
        path = tmp_path / f"{source}.json"
        path.write_text(json.dumps({writer["SOURCES"][source]: rows}))
        suites[source] = path
    document = writer["build"](suites)
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(document))
    return path, suites, document


class Run:
    def __init__(self, tmp_path, spec):
        self.root = tmp_path / "run"
        self.spec = spec
        self.spec_sha = hashlib.sha256(spec.read_bytes()).hexdigest()
        self.descriptor_host = repo_root() / "examples/gemmini/platform/host-capabilities-portable-v3.yaml"
        self.component_host = tmp_path / "component-host.yaml"
        self.component_host.write_bytes(
            (repo_root() / "examples/gemmini/platform/host-capabilities-core-aten.yaml").read_bytes()
        )
        self.stamp = {
            "software_spec_sha256": self.spec_sha,
            "host_capabilities_sha256": hashlib.sha256(self.component_host.read_bytes()).hexdigest(),
            "suite_sha256": "c" * 64,
        }
        (self.root / LABEL).mkdir(parents=True)
        (self.root / LABEL / "provenance.json").write_text(json.dumps(self.stamp))
        self.cases = {}
        self.selection_path = tmp_path / "selection.json"

    def add(self, case, *, admitted, result=None, source="bounded", **kwargs):
        self.cases.setdefault(source, []).append(case)
        executed_case = SV.runner_case(source, case)
        shard = self.root / LABEL / "shards" / case_capture_name(executed_case)
        shard.mkdir(parents=True)
        if admitted is not None:
            plan = _plan(admitted, kwargs.pop("plan_sha", self.spec_sha), steps=kwargs.pop("steps", None))
            plan["selection"]["identities"]["host_capabilities"] = {
                "sha256": kwargs.pop("plan_host_sha", self.stamp["host_capabilities_sha256"])
            }
            (shard / "spec-int8-admission.json").write_text(json.dumps(plan))
        if result is not False:
            (shard / "result.json").write_text(json.dumps(result or _result(executed_case, self.stamp, **kwargs)))
        return case

    def selection(self):
        path, suites, _ = _write_selection(self.selection_path.parent, self.cases)
        return SV.load_selection(path, suites)

    def prepare(self, **kwargs):
        kwargs.setdefault("verify_evidence", lambda evidence: True)
        kwargs.setdefault("operator_review", {"switch": "fixture", "enabled": True})
        kwargs.setdefault("software_spec", self.spec)
        kwargs.setdefault("target", "gemmini")
        kwargs.setdefault("rtl_facts", repo_root() / "merlin/tests/fixtures/core_aten_device/facts.json")
        kwargs.setdefault("host_capabilities", self.descriptor_host)
        kwargs.setdefault("component_host_capabilities", self.component_host)
        return SV.prepare(self.selection(), self.root, label=LABEL, **kwargs)


@pytest.fixture
def run(tmp_path, spec):
    return Run(tmp_path, spec)


def _verdicts(record):
    return {row["case_id"]: row["verdict"] for row in record["cases"]}


def test_every_verdict_is_derived_from_spec_expectation_and_observation(run):
    run.add(_case("dev"), admitted=True, lane="device", count=17)
    run.add(_case("host", "aten.sort.default"), admitted=False, lane="host")
    run.add(_case("empty", "aten.sort.default", empty=True), admitted=False, lane="host")
    run.add(_case("admitted-host"), admitted=True, lane="host")
    run.add(_case("escaped", "aten.sort.default"), admitted=False, lane="device", count=4)
    run.add(_case("wrong", "aten.sort.default"), admitted=False, lane="host", status="mismatch")
    capture = _case("nocap", "aten.sort.default")
    run.add(
        capture,
        admitted=None,
        result={
            "case_sha256": SV.case_sha256(SV.runner_case("bounded", capture)),
            "numeric_policy": "spec_int8",
            "provenance": run.stamp,
            "observation": {"status": "capture_unavailable", "reason": "frontend", "evidence": {"lane": "unassigned"}},
        },
    )
    record = run.prepare()
    assert _verdicts(record) == {
        "dev": SV.AGREE_DEVICE,
        "host": SV.AGREE_HOST,
        "empty": SV.NO_WORK,
        "admitted-host": SV.ADMITTED_NOT_DEVICE,
        "escaped": SV.DEVICE_NOT_ADMITTED,
        "wrong": SV.NUMERIC_FAIL,
        "nocap": SV.CAPTURE_UNAVAILABLE,
    }
    assert record["verdict"] == "failed"
    rows = {row["case_id"]: row for row in record["cases"]}
    assert rows["dev"]["observed"]["executed_instructions"] == 17
    assert rows["dev"]["spec_expectation"]["placement"] == "accelerator"
    assert rows["dev"]["spec_expectation"]["steps"][0]["source"]["path"] == str(run.spec)
    assert record["unmodeled"] == ["aten.sort.default"]
    assert record["operator_review"]["declarations"] == {"contraction": "reviewed"}


def test_mismatches_name_the_admitting_spec_row_in_both_directions(run):
    run.add(_case("admitted-host"), admitted=True, lane="host")
    run.add(_case("escaped", "aten.sort.default"), admitted=False, lane="device", count=4)
    record = run.prepare()
    by_case = {item["case_id"]: item for item in record["mismatches"]}
    label = by_case["admitted-host"]["spec_rows"][0]
    assert label.startswith(f"contraction @ {run.spec}:")
    assert by_case["escaped"]["verdict"] == SV.DEVICE_NOT_ADMITTED
    assert by_case["escaped"]["spec_rows"] == ["unsupported @ unsourced"]
    assert record["by_spec_row"][label] == {SV.ADMITTED_NOT_DEVICE: 1}
    assert record["by_overload"]["aten.sort.default"] == {SV.DEVICE_NOT_ADMITTED: 1}


@pytest.mark.parametrize(
    "ranks,plan_admitted,expected", [([2], False, SV.AGREE_DEVICE), ([5], True, SV.DEVICE_NOT_ADMITTED)]
)
def test_selected_spec_admission_overrides_cached_plan(tmp_path, spec, ranks, plan_admitted, expected):
    document = yaml.safe_load(spec.read_bytes())
    document["operations"]["contraction"]["ranks"] = ranks
    spec.write_text(yaml.safe_dump(document))
    run = Run(tmp_path, spec)
    run.add(_case("dev"), admitted=plan_admitted, lane="device", count=3)
    record = run.prepare()
    assert _verdicts(record)["dev"] == expected
    assert record["cases"][0]["spec_expectation"]["steps"][0]["status"] == (
        "admitted" if ranks == [2] else "unsupported"
    )


def test_validation_uses_the_shared_fact_resolved_selection(run, monkeypatch):
    from merlin.targetgen import functional_truth_admission as admission
    from merlin.targetgen import software_spec

    select = admission.select_admission
    admit = software_spec.admit_operation
    selected = []
    screened = []

    def select_spec(*args, **kwargs):
        result = select(*args, **kwargs)
        selected.append(result["spec"])
        assert all("hardware" not in row for row in result["spec"]["operations"])
        return result

    def screen(spec, *args):
        screened.append(spec)
        return admit(spec, *args)

    monkeypatch.setattr(admission, "select_admission", select_spec)
    monkeypatch.setattr(software_spec, "admit_operation", screen)
    run.add(_case("dev"), admitted=False, lane="device", count=3)
    assert _verdicts(run.prepare())["dev"] == SV.AGREE_DEVICE
    assert len(selected) == len(screened) == 1
    assert screened[0] is selected[0]


def test_complete_passing_record_is_validated_deterministic_and_sealed(run, tmp_path):
    run.add(_case("dev"), admitted=True, lane="device", count=3)
    run.add(_case("host", "aten.sort.default"), admitted=False, lane="host")
    first = run.prepare()
    second = run.prepare()
    assert first["verdict"] == "validated" and first["mismatches"] == []
    assert first["software_spec_sha256"] == run.spec_sha
    assert first["sha256"] == SV.digest(first) == second["sha256"]
    path = SV.write(first, tmp_path / "record.json")
    assert SV.verify(path)["sha256"] == first["sha256"]
    expected = hashlib.sha256(
        json.dumps({k: v for k, v in first.items() if k != "sha256"}, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert first["sha256"] == expected
    tampered = json.loads(path.read_bytes())
    tampered["cases"][0]["verdict"] = SV.AGREE_HOST
    path.write_bytes(SV.canonical_json(tampered) + b"\n")
    with pytest.raises(ValueError, match="sealed digest"):
        SV.verify(path)


def test_subset_and_missing_results_never_validate(run):
    run.add(_case("dev"), admitted=True, lane="device", count=3)
    run.add(_case("pending"), admitted=None, result=False)
    record = run.prepare()
    assert _verdicts(record)["pending"] == SV.NOT_RUN
    assert record["verdict"] == "incomplete"
    subset = run.prepare(case_keys=[("bounded", "dev")])
    assert subset["scope"] == {"validated_cases": 1, "complete": False, "subset": True}
    assert subset["verdict"] == "incomplete"


def test_stale_result_is_not_run(run):
    case = _case("dev")
    stale = _result(case, {"software_spec_sha256": "0" * 64}, lane="device", count=3)
    run.add(case, admitted=True, result=stale)
    assert _verdicts(run.prepare())["dev"] == SV.NOT_RUN


def test_stale_spec_identity_refuses(run):
    run.add(_case("dev"), admitted=True, lane="device", count=3, plan_sha="f" * 64)
    with pytest.raises(ValueError, match="different software spec"):
        run.prepare()
    run.stamp["software_spec_sha256"] = "e" * 64
    (run.root / LABEL / "provenance.json").write_text(json.dumps(run.stamp))
    with pytest.raises(ValueError, match="runner executed a different software spec"):
        run.prepare()


def test_unverified_device_evidence_or_missing_receipt_is_not_device(run):
    run.add(_case("dev"), admitted=True, lane="device", count=3)
    assert _verdicts(run.prepare(verify_evidence=lambda evidence: False))["dev"] == SV.ADMITTED_NOT_DEVICE
    record = run.prepare(receipt_policy="spec_qualified")
    assert _verdicts(record)["dev"] == SV.ADMITTED_NOT_DEVICE
    assert record["cases"][0]["reason"] == "no spec-qualified device receipt"


def test_truth_disagreement_is_reported_but_never_gates(run, monkeypatch):
    from merlin.targetgen import functional_truth

    run.add(_case("host", "aten.sort.default"), admitted=False, lane="host")
    monkeypatch.setattr(functional_truth, "expectation", lambda case: {"lane": "device", "family": "N4"})
    record = run.prepare(with_truth_audit=True)
    assert record["verdict"] == "validated"
    assert record["truth_disagreements"] == [
        {
            "case_id": "host",
            "source": "bounded",
            "overload": "aten.sort.default",
            "truth_lane": "device",
            "spec_placement": "host",
        }
    ]


def test_validation_names_an_existing_scheme_and_never_adds_one(run):
    run.add(_case("dev"), admitted=True, lane="device", count=3)
    assert run.prepare()["quantization_scheme"] == {"id": "q_pow2_tensor", "status": "declared"}
    with pytest.raises(ValueError, match="not declared"):
        run.prepare(quantization_scheme="core_aten_validation")


def test_selection_reader_consumes_writer_format_and_preserves_suite_identity(tmp_path):
    canonical = {"overload": "aten.mm.default"}
    bounded = _case("aten.mm.default")  # Same identity is valid in different suites.
    path, suites, document = _write_selection(tmp_path, {"corpus_193": [canonical], "bounded": [bounded]})
    selection = SV.load_selection(path, suites)
    assert [(r["source"], SV.case_key(r["case"])) for r in selection["cases"]] == [
        ("bounded", "aten.mm.default"),
        ("corpus_193", "aten.mm.default"),
    ]
    assert selection["selection_sha256"] == document["selection_sha256"]
    executed = SV.runner_suite(selection)["selected_cases"]
    assert executed == [SV.runner_case("bounded", bounded), SV.runner_case("corpus_193", canonical)]
    assert len({case_capture_name(case) for case in executed}) == 2


def test_selection_reader_checks_file_and_selection_digests(tmp_path):
    path, suites, document = _write_selection(tmp_path, {"bounded": [_case("dev")]})
    document["case_count"] += 1
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="selection_sha256 mismatch"):
        SV.load_selection(path, suites)
    path, suites, _ = _write_selection(tmp_path, {"bounded": [_case("dev")]})
    suites["bounded"].write_bytes(suites["bounded"].read_bytes() + b"\n")
    with pytest.raises(ValueError, match="file_sha256 mismatch"):
        SV.load_selection(path, suites)
    with pytest.raises(ValueError, match="suite file is required"):
        SV.load_selection(path, {})


def test_selection_reader_resolves_only_selected_suite_ids(tmp_path):
    path, suites, document = _write_selection(tmp_path, {"bounded": [_case("a"), _case("b")]})
    writer = runpy.run_path(str(repo_root() / "build_tools/scripts/make_spec_validation_selection.py"))
    document["suites"]["bounded"].update(case_ids=["b"], case_count=1)
    document["case_count"] = 1
    document["selection_sha256"] = writer["selection_digest"](document)
    path.write_text(json.dumps(document))
    assert [SV.case_key(row["case"]) for row in SV.load_selection(path, suites)["cases"]] == ["b"]
    document["suites"]["bounded"]["case_ids"] = ["unknown"]
    document["selection_sha256"] = writer["selection_digest"](document)
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="known case identities"):
        SV.load_selection(path, suites)


@pytest.mark.parametrize("field", ["case_count", "case_ids", "denominator_sha256", "suite_sha256"])
def test_selection_reader_refuses_inconsistent_suite_metadata(tmp_path, field):
    path, suites, document = _write_selection(tmp_path, {"bounded": [_case("a")]})
    writer = runpy.run_path(str(repo_root() / "build_tools/scripts/make_spec_validation_selection.py"))
    document["suites"]["bounded"][field] = ["a", "a"] if field == "case_ids" else "wrong"
    document["selection_sha256"] = writer["selection_digest"](document)
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="case_ids|mismatch"):
        SV.load_selection(path, suites)


def test_selection_reader_refuses_assumed_inline_format(tmp_path):
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"cases": [_case("a")]}))
    with pytest.raises(ValueError, match="not a .* selection"):
        SV.load_selection(path, {})


def test_prepare_keeps_same_case_id_from_different_suites(run):
    run.add(_case("same"), source="corpus_193", admitted=True, lane="device", count=3)
    run.add(_case("same", "aten.sort.default"), source="bounded", admitted=False, lane="host")
    record = run.prepare()
    assert [(row["source"], row["case_id"], row["verdict"]) for row in record["cases"]] == [
        ("bounded", "same", SV.AGREE_HOST),
        ("corpus_193", "same", SV.AGREE_DEVICE),
    ]
    assert record["verdict"] == "validated"
    subset = run.prepare(case_keys=[("corpus_193", "same")])
    assert [(row["source"], row["case_id"]) for row in subset["cases"]] == [("corpus_193", "same")]
    assert subset["verdict"] == "incomplete"


def test_cli_resolves_suite_files_and_passes_namespaced_subset_to_runner(run, tmp_path, monkeypatch):
    run.add(_case("same"), source="corpus_193", admitted=True, lane="device", count=3)
    run.add(_case("same", "aten.sort.default"), source="bounded", admitted=False, lane="host")
    path, suites, _ = _write_selection(tmp_path, run.cases)
    host = run.descriptor_host
    cli = runpy.run_path(str(repo_root() / "build_tools/scripts/run_spec_validation.py"))
    commands = []
    monkeypatch.setattr(cli["subprocess"], "run", lambda command, **kwargs: commands.append(command))
    assert (
        cli["main"](
            [
                "--selection",
                str(path),
                "--corpus-193",
                str(suites["corpus_193"]),
                "--bounded",
                str(suites["bounded"]),
                "--output",
                str(run.root),
                "--software-spec",
                str(run.spec),
                "--host-capabilities",
                str(host),
                "--component-host-capabilities",
                str(run.component_host),
                "--source",
                "bounded",
                "--case-id",
                "same",
                "--no-reverify-evidence",
                "--",
                "--target",
                "gemmini",
                "--facts",
                str(repo_root() / "merlin/tests/fixtures/core_aten_device/facts.json"),
            ]
        )
        == 1
    )
    command = commands[0]
    assert "--case-ids" in command
    assert command.count("--host-capabilities") == 1
    assert command[command.index("--host-capabilities") + 1] == str(run.component_host)
    assert "--component-host-capabilities" not in command
    assert json.loads((run.root / LABEL / "selected-case-ids.json").read_text()) == ["bounded:same"]
    record = SV.verify(run.root / LABEL / "record.json")
    assert record["selection"]["sha256"] == SV.load_selection(path, suites)["selection_sha256"]
    assert [(row["source"], row["case_id"]) for row in record["cases"]] == [("bounded", "same")]
    assert record["verdict"] == "incomplete"
    assert record["descriptor_host_capabilities"]["path"] == str(host.resolve())
    assert record["component_host_capabilities"]["path"] == str(run.component_host.resolve())
    selection = SV.load_selection(path, suites)
    assert cli["_subset"](selection, Namespace(source=None, case_id=["same"], limit=None)) == [
        ("bounded", "same"),
        ("corpus_193", "same"),
    ]


def _record(run, tmp_path, *, passing=True, **kwargs):
    run.add(_case("dev"), admitted=True, lane="device" if passing else "host", count=3 if passing else 0)
    return SV.write(run.prepare(**kwargs), tmp_path / "record.json")


def test_gate_is_off_by_default_and_explicit_when_required():
    assert SV.gate_required(None) is False
    assert SV.gate_required({"phase0_spec_validation": "off"}) is False
    assert SV.gate_required({"phase0_spec_validation": "required"}) is True
    with pytest.raises(ValueError, match="must be one of"):
        SV.gate_required({"phase0_spec_validation": True})


def test_derivation_binding_is_keyed_to_the_spec_digest(run, tmp_path):
    required = {"phase0_spec_validation": "required"}
    off = SV.bind(None, software_spec=run.spec, workload_spec={})
    assert off == {"gate": "off", "status": "not_bound", "software_spec_sha256": run.spec_sha}
    with pytest.raises(ValueError, match="no spec-validation record was supplied"):
        SV.bind(None, software_spec=run.spec, workload_spec=required)
    path = _record(run, tmp_path)
    inputs = SV.verify(path)["input_identity"]
    bound = SV.bind(path, software_spec=run.spec, workload_spec=required, input_identity=inputs)
    record = SV.verify(path)
    assert bound["status"] == "bound" and bound["sha256"] == record["sha256"] and bound["verdict"] == "validated"
    SV.verify_binding(bound, software_spec_sha256=run.spec_sha, workload_spec=required, input_identity=inputs)
    edited = tmp_path / "edited-spec.yaml"
    edited.write_bytes(run.spec.read_bytes() + b"# edit\n")
    with pytest.raises(ValueError, match="not the selected"):
        SV.bind(path, software_spec=edited, workload_spec=required)
    with pytest.raises(ValueError, match="differs from the selected software spec"):
        SV.verify_binding(bound, software_spec_sha256="0" * 64, workload_spec=required)
    with pytest.raises(ValueError, match="binds no spec-validation record"):
        SV.verify_binding(off, software_spec_sha256=run.spec_sha, workload_spec=required, input_identity=inputs)
    SV.verify_binding(off, software_spec_sha256=run.spec_sha, workload_spec={})
    path.unlink()
    with pytest.raises(ValueError, match="unreadable"):
        SV.verify_binding(bound, software_spec_sha256=run.spec_sha, workload_spec=required, input_identity=inputs)


def test_required_gate_refuses_a_failed_record(run, tmp_path):
    path = _record(run, tmp_path, passing=False)
    assert SV.verify(path)["verdict"] == "failed"
    with pytest.raises(ValueError, match="is failed"):
        SV.bind(
            path,
            software_spec=run.spec,
            workload_spec={"phase0_spec_validation": "required"},
            input_identity=SV.verify(path)["input_identity"],
        )
    assert (
        SV.bind(path, software_spec=run.spec, workload_spec={}, input_identity=SV.verify(path)["input_identity"])[
            "verdict"
        ]
        == "failed"
    )


def test_selected_synthesis_fails_closed_without_a_bound_record(tmp_path, run, monkeypatch):
    from types import SimpleNamespace

    from merlin_experiments.phase0 import evidence, profiles

    from merlin.targetgen import software_spec as SS

    spec = tmp_path / "selected-software-spec.yaml"
    spec.write_text(
        yaml.safe_dump(
            {
                "schema": SS.SCHEMA,
                "target": "test_device",
                "capability_contract": {"name": "test_device", "compute_units": []},
                "numerical_semantics": {
                    "model": {"engine": "specir_fp_reduce"},
                    "operand_dtype": "fp8_e4m3",
                    "accumulator_dtype": "bf16",
                    "readout_dtype": "bf16",
                    "product_rounding": "accumulator_format",
                    "rounding": "rne",
                    "reduction_order": "index_sequential",
                    "reduction_cadence": "per_step",
                    "subnormal_operand_flush": False,
                },
                "operations": [{"id": "matrix", "placement": "accelerator", "signature": {"ranks": [2]}}],
            }
        )
    )
    record = _record(run, tmp_path)
    document = SV.verify(record)
    current = SimpleNamespace(
        source_snapshots=(),
        software_spec={"resolved": "selected"},
        raw_facts=b'{"facts": "selected"}',
        contract={"name": "selected"},
        host_capabilities={"native": {"capability_spec": {"host": "selected"}}},
    )
    monkeypatch.setattr(evidence, "select_evidence", lambda *args, **kwargs: current)
    document["input_identity"] = SV.evidence_identity(current)
    document["software_spec_sha256"] = hashlib.sha256(spec.read_bytes()).hexdigest()
    document["sha256"] = SV.digest(document)
    SV.write(document, record)
    recipe, conformance, descriptor = (tmp_path / n for n in ("recipe.yaml", "conformance.yaml", "descriptor.yaml"))
    recipe.write_text("software_spec: selected-software-spec.yaml\n")
    inputs = dict(recipe=recipe, descriptor=descriptor, conformance_spec=conformance)

    def select(requirement, workload_spec):
        conformance.write_text(yaml.safe_dump({"name": "test_device", **requirement}))
        descriptor.write_text(yaml.safe_dump({"workload_spec": workload_spec}))
        synth = tmp_path / "synth.yaml"
        identity = profiles.synthesis_input_identity(**inputs)
        synth.write_text(yaml.safe_dump({"provenance": {"selected_inputs": identity}}))
        return profiles.verify_selected_synthesis(synth, rtl_facts=tmp_path / "facts.json", **inputs)

    required = {"phase0_spec_validation": "required"}
    assert select({}, {})["status"] == "verified"
    with pytest.raises(ValueError, match="binds no spec-validation record"):
        select({}, required)
    binding = SV.bind(record, software_spec=spec, workload_spec=required, input_identity=SV.evidence_identity(current))
    assert select({"spec_validation": binding}, required)["status"] == "verified"
    for attribute, value, field in (
        ("raw_facts", b'{"facts": "changed"}', "rtl_facts_sha256"),
        ("software_spec", {"resolved": "changed"}, "resolved_software_spec_sha256"),
        ("contract", {"name": "changed"}, "capability_contracts_sha256"),
        ("host_capabilities", {"native": {"capability_spec": {"host": "changed"}}}, "host_capabilities_sha256"),
    ):
        previous = getattr(current, attribute)
        setattr(current, attribute, value)
        with pytest.raises(ValueError, match=field):
            select({"spec_validation": binding}, required)
        setattr(current, attribute, previous)
    spec.write_bytes(spec.read_bytes() + b"# edited after validation\n")
    with pytest.raises(ValueError, match="differs from the selected software spec"):
        select({"spec_validation": binding}, required)


@pytest.mark.parametrize("lane,count", [("device", 3), ("device", 0), ("host", 3), ("unassigned", 0)])
@pytest.mark.parametrize("empty", [False, True])
def test_declined_case_requires_explicit_host_lane_without_device_work(run, lane, count, empty):
    run.add(_case("declined", "aten.sort.default", empty=empty), admitted=False, lane=lane, count=count)
    record = run.prepare(verify_evidence=lambda evidence: False)
    assert _verdicts(record)["declined"] == SV.DEVICE_NOT_ADMITTED
    assert record["verdict"] == "failed"


@pytest.mark.parametrize(
    "field",
    ["resolved_software_spec_sha256", "rtl_facts_sha256", "capability_contracts_sha256", "host_capabilities_sha256"],
)
@pytest.mark.parametrize("stage", ["derivation", "verification"])
def test_validation_gate_rejects_changed_selection(run, tmp_path, field, stage):
    path = _record(run, tmp_path)
    record = SV.verify(path)
    assert field in record.get("input_identity", {})
    current = dict(record["input_identity"])
    required = {"phase0_spec_validation": "required"}
    binding = SV.bind(path, software_spec=run.spec, workload_spec=required, input_identity=current)
    current[field] = "0" * 64
    with pytest.raises(ValueError, match=field):
        if stage == "derivation":
            SV.bind(path, software_spec=run.spec, workload_spec=required, input_identity=current)
        else:
            SV.verify_binding(
                binding, software_spec_sha256=run.spec_sha, workload_spec=required, input_identity=current
            )


def test_gate_rejects_missing_input_identities(run, tmp_path):
    path = _record(run, tmp_path)
    current = SV.verify(path)["input_identity"]
    required = {"phase0_spec_validation": "required"}
    with pytest.raises(ValueError, match="current selection"):
        SV.bind(path, software_spec=run.spec, workload_spec=required)
    binding = SV.bind(path, software_spec=run.spec, workload_spec=required, input_identity=current)
    with pytest.raises(ValueError, match="current selection"):
        SV.verify_binding(binding, software_spec_sha256=run.spec_sha, workload_spec=required)
    record = SV.verify(path)
    del record["input_identity"]
    record["sha256"] = SV.digest(record)
    SV.write(record, path)
    with pytest.raises(ValueError, match="current selection"):
        SV.bind(path, software_spec=run.spec, workload_spec=required, input_identity=current)


def test_distinct_host_lanes_bind_the_descriptor_and_screen_the_component(run, tmp_path, monkeypatch):
    from merlin.targetgen import functional_truth_admission as admission

    select = admission.select_admission
    descriptor = select(
        "gemmini",
        rtl_facts=repo_root() / "merlin/tests/fixtures/core_aten_device/facts.json",
        software_spec=run.spec,
        host_capabilities=run.descriptor_host,
    )
    inputs = SV.selection_identity(
        resolved_spec=descriptor["spec"],
        rtl_facts=descriptor["rtl_facts"],
        capability_contracts=[descriptor["capability_contract"]],
        host_capabilities=[descriptor["host_capabilities"]],
    )
    original_expectation = SV.spec_expectation

    def expectation(directory, case, selected):
        assert selected["identities"]["host_capabilities"]["path"] == str(run.component_host.resolve())
        assert selected["host_capabilities"] == yaml.safe_load(run.component_host.read_bytes())
        return original_expectation(directory, case, selected)

    monkeypatch.setattr(SV, "spec_expectation", expectation)
    path = _record(run, tmp_path)
    record = SV.verify(path)
    for field, host in (
        ("descriptor_host_capabilities", run.descriptor_host),
        ("component_host_capabilities", run.component_host),
    ):
        assert record[field] == {"path": str(host.resolve()), "sha256": hashlib.sha256(host.read_bytes()).hexdigest()}
    assert record["descriptor_host_capabilities"]["sha256"] != record["component_host_capabilities"]["sha256"]
    assert record["input_identity"] == inputs
    assert (
        record["cases"][0]["spec_expectation"]["host_capabilities_sha256"]
        == record["component_host_capabilities"]["sha256"]
    )
    required = {"phase0_spec_validation": "required"}
    binding = SV.bind(path, software_spec=run.spec, workload_spec=required, input_identity=inputs)
    SV.verify_binding(binding, software_spec_sha256=run.spec_sha, workload_spec=required, input_identity=inputs)
    component_inputs = SV.selection_identity(
        resolved_spec=descriptor["spec"],
        rtl_facts=descriptor["rtl_facts"],
        capability_contracts=[descriptor["capability_contract"]],
        host_capabilities=[yaml.safe_load(run.component_host.read_bytes())],
    )
    with pytest.raises(ValueError, match="host_capabilities_sha256"):
        SV.bind(path, software_spec=run.spec, workload_spec=required, input_identity=component_inputs)
    with pytest.raises(ValueError, match="host_capabilities_sha256"):
        SV.verify_binding(
            binding, software_spec_sha256=run.spec_sha, workload_spec=required, input_identity=component_inputs
        )
    run.component_host.write_bytes(run.component_host.read_bytes() + b"# changed after binding\n")
    with pytest.raises(ValueError, match="component host capabilities differ"):
        SV.verify(path)
    with pytest.raises(ValueError, match="component host capabilities differ"):
        SV.bind(path, software_spec=run.spec, workload_spec=required, input_identity=inputs)
    with pytest.raises(ValueError, match="component host capabilities differ"):
        SV.verify_binding(binding, software_spec_sha256=run.spec_sha, workload_spec=required, input_identity=inputs)


def test_prepare_refuses_substituted_component_document(run):
    run.add(_case("dev"), admitted=True, lane="device", count=3)
    with pytest.raises(ValueError, match="runner executed different component host capabilities"):
        run.prepare(component_host_capabilities=run.descriptor_host)


def test_prepare_refuses_mismatched_component_admission_plan(run):
    run.add(_case("dev"), admitted=True, lane="device", count=3, plan_host_sha="f" * 64)
    with pytest.raises(ValueError, match="admission plan selected different component host capabilities"):
        run.prepare()


@pytest.mark.parametrize(
    "extra,message",
    [
        ([], "--component-host-capabilities"),
        (
            ["--component-host-capabilities", "component.yaml", "--", "--host-capabilities", "override.yaml"],
            "before --",
        ),
        (["--component-host-capabilities", "component.yaml", "--", "--host-capabilities=override.yaml"], "before --"),
    ],
)
def test_cli_requires_component_and_refuses_runner_host_override(extra, message, capsys):
    cli = runpy.run_path(str(repo_root() / "build_tools/scripts/run_spec_validation.py"))
    with pytest.raises(SystemExit) as error:
        cli["main"](
            [
                "--selection",
                "selection.json",
                "--output",
                "run",
                "--software-spec",
                "spec.yaml",
                "--host-capabilities",
                "descriptor.yaml",
                *extra,
            ]
        )
    assert error.value.code == 2
    assert message in capsys.readouterr().err
