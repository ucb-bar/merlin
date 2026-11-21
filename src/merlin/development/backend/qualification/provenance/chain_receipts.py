"""Reopen the explicitly selected quantized full-chain proof, never a CPU-twin claim."""

from __future__ import annotations

import json
from pathlib import Path

from merlin.common.digest import hash_tree, sha256_file
from merlin.common.process import invocation_records as IR
from merlin.development.backend.optimization.contracts import document_sha256
from merlin.development.backend.session import launch_config as L
from merlin.evaluation.model.bundles.models import load as load_models
from merlin.evaluation.model.cli import tree_sha256
from merlin.evaluation.model.whole_model import output_audit as audit
from merlin.evaluation.model.whole_model.output_audit import Files
from merlin.evaluation.model.whole_model.quantized_comparison import report_complete
from merlin.evaluation.model.whole_model.relink import verified_recipe

# ---- private_chain_receipt ---------------------------------------------------------------------

SCHEMA = "merlin.phase1.selected_quantized_full_chain.v1"

_CACHE = {}


def _require(value, message):
    if not value:
        raise ValueError(message)


class _Files:
    def __init__(self):
        self.pins = {}

    def pin(self, path, expected=None):
        path = Path(path)
        _require(
            path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)),
            "full-chain proof path is relative or indirect",
        )
        actual = sha256_file(path)
        _require(expected is None or actual == expected, "full-chain proof input bytes changed")
        _require(
            str(path) not in self.pins or self.pins[str(path)] == actual, "conflicting full-chain proof identities"
        )
        self.pins[str(path)] = actual
        return path

    def selected(self, pin):
        _require(isinstance(pin, dict) and set(pin) == {"path", "sha256"}, "invalid full-chain file pin")
        return self.pin(pin["path"], pin["sha256"])

    def read(self, path):
        path = self.pin(path)
        value = json.loads(path.read_bytes())
        self.pin(path, self.pins[str(path)])
        return value

    def unchanged(self):
        for path, digest in self.pins.copy().items():
            self.pin(path, digest)


def _native_images(files, *, partition_path, model, report, work):
    """Tie each numerical build to the original verified static producer's image."""
    partition = files.read(partition_path)
    builds = {}
    for program in model.programs:
        build = files.read(work.parent / "programs" / program.name / "build.json")
        qualification = {
            "partition": {"path": str(partition_path), "sha256": sha256_file(partition_path)},
            "program": program.name,
            "instruction_selection": partition["instruction_selection"],
        }
        original = files.read(files.selected(partition["receipts"][program.name]))
        reported = next(p for p in report["programs"] if p["program"] == program.name)["build"]
        _require(
            build.get("native_qualification")
            == build["key"].get("native_qualification")
            == reported.get("native_qualification")
            == qualification
            and build["elf_sha256"] == original["output"]["elf_sha256"]
            and build["build_hash"] == original["output"]["build_hash"],
            "executed chain did not use its independently admitted native model image",
        )
        builds[program.name] = build
    return partition, builds


def _native_spec(files, *, spec, frozen, run_dir):
    """The canonical source and sandbox copy are distinct ordinary files."""
    digest = sha256_file(spec)
    _require(
        frozen["sha256"] == frozen["source_sha256"] == frozen["frozen_sha256"] == digest
        and files.pin(frozen["path"], digest) == run_dir / "private_full_model_input/spec.yaml"
        and files.pin(frozen["frozen_path"], digest) == spec,
        "numerical proof is not bound to the original native frozen specification",
    )


def verify(receipt_path, *, run_dir, private_spec_path, candidate_sha256):
    """Re-evaluate full-chain gates and every original readback/reference array.

    A process-local successful audit may be reused only after reopening all of its
    original artifacts. Static source, supplier and instruction admission remains
    a separate required AND in the native caller.
    """
    from merlin.development.backend.qualification.provenance import private_numerical_selection as selection

    files = _Files()
    selected = selection.read(private_spec_path)
    _require(selected is not None, "private full-chain criterion was not selected")
    spec = files.pin(private_spec_path)
    receipt = files.read(receipt_path)
    _require(
        Path(receipt_path) == Path(selected["static_root"]) / "full-chain-qualification.json",
        "full-chain proof is outside the selected producer directory",
    )
    _require(
        receipt.get("schema") == SCHEMA
        and receipt.get("passed") is True
        and receipt.get("run_dir") == str(Path(run_dir).absolute())
        and receipt.get("private_spec_sha256") == sha256_file(spec)
        and receipt.get("candidate_tree_sha256") == candidate_sha256,
        "full-chain proof does not belong to the selected native run/compiler/specification",
    )
    config_path = files.selected(receipt["config"])
    _require(str(config_path) == selected["config"], "full-chain config differs from private selection")
    resolution = receipt["resolution"]
    _require(set(resolution) == {"task_root", "prep", "run_id"}, "invalid full-chain config resolution")
    launch = L.build(L.load(config_path), **resolution)
    _require(
        Path(launch.variables["RUN"]) == Path(run_dir).absolute()
        and sha256_file(files.pin(launch.path(launch.document["inputs"]["private_full_model_spec"])))
        == sha256_file(spec),
        "full-chain configuration selects another native owner/private specification",
    )
    import yaml

    environment = yaml.safe_load(files.pin(Path(run_dir) / "environment.yaml").read_bytes())
    frozen = environment["private_full_model_spec"]
    _native_spec(files, spec=spec, frozen=frozen, run_dir=Path(run_dir))
    identity = receipt["selected_identity"]
    bundle = Path(launch.environment["MERLIN_MODEL_CHECK_BUNDLE"])
    expected = {
        "source_revision": launch.variables["SOURCE_REVISION"],
        "private_spec_sha256": sha256_file(spec),
        "bundle_manifest_sha256": sha256_file(files.pin(bundle / "MANIFEST.json")),
        "bundle_sums_sha256": sha256_file(files.pin(bundle / "SHA256SUMS")),
        "models_config_sha256": sha256_file(files.pin(bundle / "configuration/models.yaml")),
        "runtime_pins": {
            name: sha256_file(files.pin(launch.path(pin["path"]), pin["sha256"]))
            for name, pin in launch.document["runtime_pins"].items()
        },
    }
    _require(identity == expected, "full-chain source/input/runtime identity differs")
    for key, path in (
        ("models", bundle / "configuration/models.yaml"),
        ("reference_manifest", bundle / "MANIFEST.json"),
        ("reference_sums", bundle / "SHA256SUMS"),
    ):
        _require(files.selected(selected[key]) == path, "private numerical bundle selection differs")
    policy = launch.document["full_model_verification"]
    _require(
        policy["comparison"] == selected["comparison"]
        and policy["output_scope"] == "all_declared"
        and {"atol": policy["atol"], "rtol": policy["rtol"], "equal_nan": False, "finite_required": True}
        == audit.FLOAT_POLICY
        == selected["float_policy"],
        "full-chain comparison policy differs",
    )
    plan_path = files.selected(
        {**policy["all_output_plan"], "path": str(launch.path(policy["all_output_plan"]["path"]))}
    )
    plan = files.read(plan_path)
    _require(
        plan.get("schema") == "merlin.operator.all_output_plan.v1"
        and plan["selected_identity"] == identity
        and files.read(files.selected(plan["expected_identity"])) == identity
        and plan["comparison"] == {"float": audit.FLOAT_POLICY, "integer_bool": "exact"},
        "all-output plan selected identity/comparison differs",
    )
    _require(
        files.selected(plan["helper"]) == Path(audit.__file__).resolve(),
        "all-output plan does not select the canonical auditor",
    )
    requirement = files.selected(plan["requirement"])
    package = receipt["submission"]
    _require(
        package["hash_tree_sha256"] == candidate_sha256
        and hash_tree(Path(package["path"]))["sha256"] == candidate_sha256
        and tree_sha256(Path(package["path"])) == package["model_check_tree_sha256"],
        "full-chain compiler bytes differ",
    )
    models = load_models(bundle / "configuration/models.yaml")
    _require(
        set(receipt["models"]) == set(models.models) == set(plan["references"]),
        "full-chain proof has an incomplete model roster",
    )
    index = files.read(Path(selected["static_root"]) / "static-index.json")
    _require(
        index.get("schema") == "merlin.phase1.native_static_chain_index.v1"
        and index.get("run_dir") == str(Path(run_dir).absolute())
        and set(index.get("records", {})) == set(models.models),
        "full-chain static producer index differs from the native model roster",
    )
    results = {}
    for name, row in receipt["models"].items():
        _require(row.get("exit_code") == 0, "full-chain child did not complete successfully")
        report_path, job_path = files.selected(row["report"]), files.selected(row["job"])
        report, job = files.read(report_path), files.read(job_path)
        _require(
            job["submission"] == package
            and job["selected_identity"] == identity
            and report_complete(report, models.model(name), policy, package),
            "full-chain report lacks complete selected execution/kernel/reference gates",
        )
        reference_path = files.selected(plan["references"][name])
        reference = files.read(reference_path)
        work = Path(job["work"]) / package["model_check_tree_sha256"][:16] / name / "runs"
        roots = [p for p in work.iterdir() if p.is_dir() and (p / policy["input"]).is_dir()]
        _require(len(roots) == 1, "full-chain execution has no unambiguous original run root")
        partition_path = files.pin(index["records"][name])
        partition, builds = _native_images(
            files, partition_path=partition_path, model=models.model(name), report=report, work=work
        )

        for program in models.model(name).programs:
            relinks = verify_relinks(
                builds[program.name],
                program=program.name,
                run_root=roots[0] / policy["input"],
                steps=program.steps,
                selection_record=partition,
            )
            _require(
                relinks["status"] == "verified" and relinks["steps"] == program.steps,
                "executed step lacks complete selected instruction/supplier qualification",
            )
            for pin in relinks["artifacts"]:
                files.selected(pin)
        cache_key = (str(report_path), sha256_file(report_path), sha256_file(job_path), sha256_file(plan_path))
        observed = _CACHE.get(cache_key)
        if observed is None:
            observed = audit.verify(
                reference_path=reference_path,
                expected_reference_sha256=plan["references"][name]["sha256"],
                job_pin_path=job_path,
                report_path=report_path,
                models_path=models.path,
                run_root=roots[0],
                expected_identity=identity,
                expected_identity_path=Path(plan["expected_identity"]["path"]),
                requirement_path=requirement,
                extension_runtime_key=plan["extension_runtime_key"],
            )
        expected_steps = sum(p.steps for p in models.model(name).programs)
        expected_outputs = sum(p["steps_declared"] * len(p["declared_outputs"]) for p in reference["programs"])
        _require(
            observed.get("status") == "pass"
            and observed.get("unchanged") is True
            and observed.get("steps_compared") == observed.get("steps_expected") == expected_steps
            and observed.get("outputs_compared") == observed.get("n_declared_outputs") == expected_outputs,
            "full-chain all-output evidence is incomplete or mismatched",
        )
        for pin in observed["artifacts"]:
            files.selected(pin)
        _CACHE[cache_key] = observed
        results[name] = {
            "passed": True,
            "steps": expected_steps,
            "arrays": expected_outputs,
            "kernel_exact": report["gates"]["kernel_exact"],
            "report": row["report"],
            "job": row["job"],
        }
    files.unchanged()
    _require(selection.read(spec) == selected, "private full-chain criterion changed during verification")
    return {
        "schema": SCHEMA,
        "mode": selected["mode"],
        "passed": True,
        "run_dir": str(Path(run_dir).absolute()),
        "private_spec_sha256": sha256_file(spec),
        "candidate_tree_sha256": candidate_sha256,
        "models": results,
        "host_execution": {"selected": False, "status": "not_selected"},
        "reference_scope": "selected sealed quantized full chain plus exact accelerator kernels",
        "receipt": {"path": str(receipt_path), "sha256": sha256_file(receipt_path)},
    }


# ---- private_chain_relinks ---------------------------------------------------------------------
# Reopen actual per-input instruction and supplier observations after execution.
#
# Transient ELFs may have been deliberately discarded. Their successful original
# observations are joined by exact digest, never described as current ELF bytes.
# The caller separately admits the native partition and numerical/readback proof.


def _require(value, message):
    if not value:
        raise ValueError(message)


def _historical_issuer(files, path, *, report_path, report, elf, expected_check):
    """Only the original ELF input may be unavailable after intentional cleanup."""
    from merlin.development.backend.optimization.controls import component_instruction_audit as audit

    call = files.json(path)
    _require(
        call.get("schema") == IR.SCHEMA
        and call.get("kind") == "python_call"
        and call.get("stage") == "whole_linked_instruction_policy"
        and call.get("callable") == audit.__name__ + "._scan"
        and call.get("status") == "completed"
        and call.get("outcome") == "returned"
        and all(call.get(k) is True for k in ("inputs_unchanged", "dependencies_unchanged", "executable_unchanged")),
        "relink has no completed unchanged original instruction issuer",
    )
    check = {
        "policy_sha256": report.get("policy_sha256"),
        "accessor_sha256": report.get("accessor_intake_sha256"),
        "selection": report.get("decoder_selection"),
    }
    _require(
        document_sha256(check) == expected_check
        and call.get("arguments")
        == {
            "policy_sha256": check["policy_sha256"],
            "accessor_intake_sha256": check["accessor_sha256"],
            "decoder_selection": check["selection"],
        },
        "relink instruction issuer selected another original policy/accessor/decoder",
    )
    for pin in (call["executable"], call["stdout"], call["stderr"], *call["dependencies"]):
        files.selected(pin)
    source = files.pin(Path(audit.__file__).resolve())
    _require(source in call["dependencies"], "relink instruction issuer lost its actual source producer")
    expected_elf = {"path": elf["path"], "sha256": elf["sha256"]}
    _require(
        call["inputs"].count(expected_elf) == 1 and call["outputs"] == [files.pin(report_path)],
        "instruction issuer does not bind the actual relink ELF",
    )
    for pin in call["inputs"]:
        if pin != expected_elf or Path(pin["path"]).is_file():
            files.selected(pin)


def verify_relinks(build, *, program, run_root, steps, selection_record):
    """Verify one native program's actual step policy/supplier receipts; return pins.

    ``run_root`` is the selected input directory containing ``program.0`` etc.
    The fixed native partition/producer is independently verified by the caller.
    """
    from merlin.compiler.lower.model.link_supplier_trace import verify_link_suppliers

    files = Files()
    base = verified_recipe(build)
    expected_check = selection_record["instruction_selection"]["instruction_check_sha256"]
    _require(type(steps) is int and steps > 0, "relink verification needs the declared positive step count")
    native = build.get("native_qualification") or {}
    _require(
        native.get("program") == program
        and native.get("instruction_selection") == selection_record["instruction_selection"],
        "relink base build lost its original native instruction selection",
    )
    files.selected(native["partition"])
    original_suppliers = base.get("link_suppliers")
    expected_suppliers = (
        {symbol: Path(item["selected_input"]["path"]) for symbol, item in original_suppliers["symbols"].items()}
        if original_suppliers
        else None
    )
    rows = []
    for step in range(steps):
        directory = Path(run_root) / f"{program}.{step}"
        recipe = files.json(directory / "compilation_recipe.json")
        elf = recipe["executable"]
        _require(
            recipe.get("status") == "completed"
            and recipe.get("commands")
            and all(row.get("status") == "returned" for row in recipe["commands"])
            and recipe["commands"][-1].get("output") == elf
            and elf["path"] == str(directory / "model.elf"),
            "step has no completed original input-specific link",
        )
        admission_path = directory / "native_linked_elf.json"
        admission = files.json(admission_path)
        report_path = Path(admission["report_path"])
        files.pin(report_path, admission["report_sha256"])
        report = files.json(report_path)
        _require(
            report.get("schema") == "merlin.independent_linked_instruction_audit.v1"
            and report.get("status") == admission.get("status") == "accepted"
            and report.get("elf_sha256") == admission.get("elf_sha256") == elf["sha256"]
            and report.get("prohibited_hits") == []
            and report.get("sections"),
            "step instruction policy did not accept the actual input-specific ELF",
        )
        invocations = list((report_path.parent / "invocations").glob("*/invocation.json"))
        _require(len(invocations) == 1, "relink instruction audit lacks one original issuer observation")
        _historical_issuer(
            files, invocations[0], report_path=report_path, report=report, elf=elf, expected_check=expected_check
        )
        for pin in report["decoder_receipts"]:
            files.selected(pin)
        link = recipe["commands"][-1]
        if expected_suppliers:
            verify_link_suppliers(
                recipe.get("link_suppliers"),
                argv=link["argv"],
                input_identities=link["inputs"],
                expected_suppliers=expected_suppliers,
            )
            for path in expected_suppliers.values():
                files.pin(path)
        execution = [files.json(path) for path in (directory / "invocations").glob("*/invocation.json")]
        execution = [row for row in execution if row.get("stage") == "selected_rocc_process"]
        _require(
            len(execution) == 1
            and execution[0].get("status") == "completed"
            and execution[0].get("returncode") == 0
            and execution[0].get("inputs") == [{"path": elf["path"], "sha256": elf["sha256"]}],
            "instruction-admitted ELF differs from the actually executed step",
        )
        retained = Path(elf["path"]).is_file()
        if retained:
            files.selected(elf)
        rows.append(
            {
                "step": step,
                "elf_sha256": elf["sha256"],
                "transient_elf_current_reopen": "retained" if retained else "unavailable_deleted",
            }
        )
    files.unchanged()
    return {
        "status": "verified",
        "program": program,
        "steps": len(rows),
        "rows": rows,
        "artifacts": [{"path": str(path), "sha256": digest} for path, digest in files.pins.items()],
        "scope": "original completed policy/supplier/execution correspondence; deleted ELF bytes unavailable",
    }
