"""Canonical per-model static builds reused by the selected quantized chain.

Only the original native run's verified profile, authored-source freeze and
independent instruction issuer are used. The returned image provider transports
those freshly verified images; it never upgrades historical structural builds.
Numerical equivalence remains the separate exact-kernel/full-chain obligation.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from merlin.common import invocation_record as IR
from merlin.common.paths import build_dir
from merlin.compile.model_execution_inputs import file_sha256, strict_tree_sha256
from merlin_experiments.phase1.feedback import compiler_profile_runtime as profile
from merlin_experiments.phase1.feedback import private_full_model_execution as execution
from merlin_experiments.phase1.feedback import private_full_models as models
from merlin_experiments.phase1.feedback import private_instruction_coordinator as instructions
from merlin_experiments.phase1.feedback import private_instruction_declaration as declaration
from merlin_experiments.phase1.feedback import private_linked_source_preparation as preparation
from merlin_experiments.phase1.feedback import private_model_partitions as partitions

SCHEMA = "merlin.phase1.native_static_chain_partition.v1"
PROGRESS = "private_model_chain_progress.json"


def _progress(out):
    """Private stage observations grant no source, build or numerical credit."""
    from .private_source_preflight import _observer

    return _observer(
        lambda event: print(json.dumps(event, sort_keys=True), file=sys.stderr, flush=True), out / PROGRESS
    )


def _read(path):
    return json.loads(Path(path).read_bytes())


def _pin(path):
    path = Path(path).absolute()
    return {"path": str(path), "sha256": file_sha256(path)}


def _partition_complete(record, name, candidate):
    """Use the existing full native validator on this partition's exact row.

    This local projection is never published as a full-roster gate result.
    ``join`` below validates the original complete roster before publishing one.
    """
    required, programs = record.get("required_models"), record.get("required_programs")
    if (record.get("mode") != "canonical_model_partition" or record.get("selected_model") != name
            or record.get("passed") is not False or record.get("partition_passed") is not True
            or not isinstance(required, list) or name not in required or not isinstance(programs, dict)
            or set(programs) != set(required) or len(record.get("models", [])) != 1
            or record["models"][0].get("model") != name):
        return False
    projection = {**record, "required_models": [name], "required_programs": {name: programs[name]}, "passed": True}
    projection.pop("mode")
    return models.complete(projection, required_models=[name], required_programs={name: programs[name]},
                           candidate_sha256=candidate)


@dataclass
class NativePartition:
    path: Path
    record: dict
    selection: object
    invocation: Path | None = None
    _source_preparation: object = field(default=None, init=False, repr=False)

    def _verify_static_record(self):
        if self._source_preparation is None:
            self._source_preparation = preparation.Preparation(self)
        with self._source_preparation.activate(self):
            return _verify_record(self.record)

    def verify(self):
        self.selection.verify()
        if self.invocation is None:
            raise ValueError("native partition producer has not completed")
        _producer(self.path, self.record, self.invocation)
        if _read(self.path) != self.record:
            raise ValueError("native static partition record changed")
        self._verify_static_record()

    def build(self, stage, *, target, label, submission, submission_sha256, settings, directory, admission=None):
        """The full-chain build-provider API; no compiler or alternate ELF is invoked."""
        from merlin.model_check.full_build_native import adopt

        self.verify()
        model, separator, program = label.partition(":")
        if not separator or model != self.record["selected_model"] or target != self.record["target"]:
            raise ValueError("native image provider selected a different model/target")
        receipt_pin = self.record["receipts"][program]
        receipt = _read(receipt_pin["path"])
        result = adopt(receipt, stage=stage, target=target, label=label, submission=submission,
                       submission_sha256=submission_sha256, settings=settings,
                       qualification={"partition": _pin(self.path), "program": program,
                                      "instruction_selection": self.record["instruction_selection"]},
                       admission=self.selection.service)
        self.verify()
        directory = Path(directory)
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        with (directory / "build.json").open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        return result


def _verify_record(record):
    from merlin_experiments.phase1.feedback.private_source_freeze import valid_binding

    from . import private_static_revalidation as revalidation

    name, candidate = record["selected_model"], record["candidate_tree_sha256"]
    retained = record.get("schema") == revalidation.SCHEMA
    if retained:
        revalidation.verify_record(record)
    if ((not retained and (record.get("schema") != SCHEMA or record.get("scope") != "canonical_static_only"))
            or not valid_binding(record["authored_source_freeze"], record["private_spec_sha256"])):
        raise ValueError("native static partition lost its original source binding")
    for pin in record["inputs"]:
        if file_sha256(Path(pin["path"])) != pin["sha256"]:
            raise ValueError("native static partition selected input changed")
    static = _read(record["static_partition_report"])
    if (file_sha256(Path(record["static_partition_report"])) != record["static_partition_report_sha256"]
            or not _partition_complete(static, name, candidate)
            or any(static.get(key) != record[key] for key in (
                "candidate_tree_sha256", "private_spec_sha256", "authored_source_freeze"))):
        raise ValueError("native static partition is incomplete or changed")
    partitions._selection(record, target=record["target"], instruction_path=Path(record["instruction_path"]))
    if set(record["receipts"]) != set(static["required_programs"][name]):
        raise ValueError("native partition omitted a program receipt")
    entries = static["models"][0]["checks"]["build"]["programs"]
    for entry in entries:
        pin = record["receipts"][entry["program"]]
        if file_sha256(Path(pin["path"])) != pin["sha256"]:
            raise ValueError("native compiled receipt changed")
        receipt = _read(pin["path"])
        if (receipt.get("status") != "compiled" or receipt["output"]["elf_sha256"] != entry["elf_sha256"]
                or receipt["output"]["compilation_recipe"]["sha256"] != entry["compilation_recipe"]["recipe_sha256"]):
            raise ValueError("native compiled receipt is not its verified linked program")
    return static


def _compile(*, run_dir, descriptor, submission, name, out, owner, spec, selected, freeze, retained_products=None):
    """Fixed producer invoked only while the original native profile is active."""
    from merlin.targetgen.target_experiment import load_target_experiment
    from merlin_experiments.phase1.feedback.private_facts import selected_input_facts
    from merlin_experiments.phase1.feedback.private_result_format_selection import from_run

    target = load_target_experiment(descriptor).target
    roster, programs = models.requirements_for(descriptor), models.program_requirements_for(descriptor)
    gate = execution.gate_for(descriptor, required_programs=programs)
    if name not in roster or selected is None:
        raise ValueError("native chain requires a declared model and independent instruction selection")
    options = _build_options(gate, spec=spec, name=name, programs=programs)
    static_root = build_dir() / "phase1-native-chain" / run_dir.name / name / "static"
    if static_root.exists() or static_root.is_symlink():
        raise ValueError("native static build requires fresh selected output below MERLIN_OUT_ROOT")
    observe = _progress(out)
    if retained_products is not None:
        from . import private_static_revalidation as revalidation

        observe({"stage": "retained_source_selection", "model": name})
        revalidation.require_source_binding(
            retained_products, run_dir=run_dir, spec=spec, freeze=freeze, target=target)
    observe({"stage": "instruction_selection", "model": name})
    trusted_tools = None
    if retained_products is None:
        prepared = instructions.prepare(selected, target=target, output=out / "instruction-selection")
    else:
        prepared, trusted_tools = revalidation.prepare_instructions(
            selected, target=target, output=out / "instruction-selection")
    with selected_input_facts(spec, target=target, required_models=roster) as facts:
        prepared.require_facts(facts)
        if retained_products is not None:
            original = retained_products.verify()[0]
            if original["fact_reader_binding"] != facts:
                raise ValueError("retained products changed original facts")
        result = models.run(
            submission, spec, target=target, required_models=roster, required_programs=programs,
            loader_env_requirements=models.loader_env_requirements_for(descriptor), out=static_root,
            selected_model=name, source_freeze=freeze,
            source_freeze_root=run_dir / "private_full_model_input/sources",
            build_options=options, linked_elf_admission=prepared.service,
            native_result_format_selection=from_run(run_dir, target=target),
            progress=observe,
            **({"retained_products": retained_products} if retained_products is not None else {}),
        )
        prepared.require_facts(facts)
    if not _partition_complete(result, name, strict_tree_sha256(submission)["sha256"]):
        observe({"stage": "static_partition_refused", "model": name, "status": "fail"})
        raise ValueError("native static model partition did not pass: " + str(result.get("models")))
    static_path, = static_root.glob("private_full_model_partition-*.json")
    issuance, = (out / "instruction-selection/invocations").glob("*/invocation.json")
    record = {
        "schema": SCHEMA, "scope": "canonical_static_only", "run_dir": str(run_dir),
        "target": target, "selected_model": name, "required_models": list(roster),
        "required_programs": {model: list(stages) for model, stages in programs.items()},
        "candidate_tree_sha256": result["candidate_tree_sha256"],
        "private_spec_sha256": result["private_spec_sha256"],
        "authored_source_freeze": result["authored_source_freeze"],
        "compiler_profile": profile.compiler_profile_binding(owner, run_dir=run_dir),
        "fact_reader_binding": facts, "instruction_declaration_sha256": file_sha256(selected.path),
        "instruction_path": str(selected.path), "instruction_selection": prepared.record(),
        "instruction_selection_receipt": str(out / "instruction-selection/selection.json"),
        "instruction_selection_invocation": str(issuance),
        "instruction_source_pins": [[str(p), digest] for p, digest in selected.pins],
        "static_partition_report": str(static_path), "static_partition_report_sha256": file_sha256(static_path),
        "receipts": (retained_products.verify()[0]["receipts"] if retained_products is not None else {
            program: _pin(static_root / name / program / "baremetal_model.json") for program in programs[name]}),
        "inputs": [_pin(path) for path in (spec, descriptor, run_dir / "environment.yaml",
                                           run_dir / "input_bundle_manifest.yaml", Path(__file__))],
    }
    if retained_products is not None:
        record.update(schema=revalidation.SCHEMA, scope=revalidation.SCOPE,
                      retained_product_revalidation=result["retained_product_revalidation"],
                      trusted_checker_preparation=trusted_tools)
        record["inputs"].extend([_pin(Path(revalidation.__file__)),
                                 *retained_products.verify()[1]])
    path = out / "native-static.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    partition = NativePartition(path, record, prepared)
    partition._verify_static_record()
    observe({"stage": "static_checks_complete", "model": name, "status": "verified"})
    return partition


def _build_options(gate, *, spec, name, programs):
    from . import private_numerical_selection

    if gate is None and private_numerical_selection.read(spec) is None:
        raise ValueError("native chain requires an explicitly selected full-chain criterion or execution gate")
    options = execution.build_options(gate) if gate is not None else {}
    for program in programs[name]:
        options.setdefault(name, {}).setdefault(program, {}).update(readback="full", host_llvm_bitcode=True)
        options[name][program].setdefault("group_profile", True)
    return options


def run_static(*, run_dir, descriptor, submission, name, out):
    """Use one already prepared, live native run; never prepare an alternate owner."""
    from merlin.targetgen.target_experiment import load_target_experiment
    from merlin_experiments.phase1.feedback.formal import _private_source_freeze_for_formal

    run_dir, descriptor, submission, out = map(lambda p: Path(p).absolute(), (run_dir, descriptor, submission, out))
    if out.exists():
        raise ValueError("native static partition requires a fresh output root")
    out.mkdir(parents=True, mode=0o700)
    owner = profile.functional_service(run_dir)
    if owner is None:
        raise ValueError("native chain has no verified original compiler-profile owner")
    target = load_target_experiment(descriptor).target
    environment = yaml.safe_load((run_dir / "environment.yaml").read_bytes())
    spec = Path(environment["private_full_model_spec"]["frozen_path"])
    freeze = _private_source_freeze_for_formal(run_dir, spec, target, workspace=owner.workspace,
                                              repo=owner.repo, descriptor=descriptor)
    selected = declaration.read(run_dir / "private_instruction_input/selection.json", target=target)
    with profile.original_compiler_execution(owner, target_experiment=load_target_experiment(descriptor),
                                              candidate=submission, work_root=out / "profile"):
        with IR.observe_call(
            out, stage="native_static_chain_partition", function=_compile,
            arguments={"run_dir": str(run_dir), "model": name,
                       "candidate_tree_sha256": strict_tree_sha256(submission)["sha256"]},
            inputs=(spec, descriptor, run_dir / "environment.yaml", run_dir / "input_bundle_manifest.yaml"),
            outputs=(out / "native-static.json",),
            dependencies=(Path(__file__), Path(models.__file__), Path(instructions.__file__)),
        ) as observed:
            result = _compile(run_dir=run_dir, descriptor=descriptor, submission=submission, name=name, out=out,
                              owner=owner, spec=spec, selected=selected, freeze=freeze)
            observed.returned()
    IR.verify(observed.path)
    result.invocation = observed.path
    result.verify()
    return result


def _producer(path, record, invocation):
    from . import private_static_revalidation as revalidation

    if record.get("schema") == revalidation.SCHEMA:
        revalidation.verify_producer(path, record, invocation)
        return
    call = IR.verify(invocation)
    if (call.get("kind") != "python_call" or call.get("stage") != "native_static_chain_partition"
            or call.get("callable") != __name__ + "._compile"
            or call.get("arguments") != {"run_dir": record["run_dir"], "model": record["selected_model"],
                                         "candidate_tree_sha256": record["candidate_tree_sha256"]}
            or not any(pin.get("path") == str(path) and pin.get("sha256") == file_sha256(path)
                       for pin in call["outputs"])):
        raise ValueError("native static partition has no matching completed original producer")


def join(records, *, run_dir, descriptor, candidate_sha256):
    """Join every original static partition; no numerical gate is synthesized."""
    roster, programs = models.requirements_for(descriptor), models.program_requirements_for(descriptor)
    if set(records) != set(roster):
        raise ValueError("native static join requires the complete declared model roster")
    run_dir = Path(run_dir).absolute()
    owner = profile.functional_service(run_dir)
    if owner is None:
        raise ValueError("native static join has no original compiler-profile owner")
    expected_profile = profile.compiler_profile_binding(owner, run_dir=run_dir)
    rows, first = [], None
    for name in roster:
        path = Path(records[name])
        record = _read(path)
        if (record["run_dir"] != str(run_dir) or record["selected_model"] != name
                or record["candidate_tree_sha256"] != candidate_sha256
                or record["compiler_profile"] != expected_profile
                or record["instruction_path"] != str(run_dir / "private_instruction_input/selection.json")):
            raise ValueError("native static partition belongs to another original run/model")
        actual = _verify_record(record)
        issued, = path.parent.glob("invocations/*/invocation.json")
        _producer(path, record, issued)
        if first is not None and any(record[key] != first[key] for key in partitions.BINDINGS):
            raise ValueError("native static partitions changed original source/profile/facts/policy")
        first = record
        rows.extend(actual["models"])
    result = {**actual, "models": rows, "passed": True, "required_models": list(roster),
              "required_programs": {name: list(programs[name]) for name in roster},
              "worker_records": {name: _pin(path) for name, path in records.items()},
              "compiler_profile": expected_profile, "fact_reader_binding": first["fact_reader_binding"],
              "instruction_selections_by_model": {name: _read(path)["instruction_selection"]
                                                  for name, path in records.items()}}
    for key in ("mode", "selected_model", "partition_passed"):
        result.pop(key, None)
    result["passed"] = models.complete(result, required_models=roster, required_programs=programs,
                                        candidate_sha256=candidate_sha256)
    if not result["passed"]:
        raise ValueError("native static joined roster did not satisfy the original full validator")
    return result
