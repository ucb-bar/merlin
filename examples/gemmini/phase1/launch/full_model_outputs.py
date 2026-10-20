"""Pinned private all-output intake and one offline audit per completed model child."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from merlin.common.digest import sha256_file


def _pinned(pin):
    if not isinstance(pin, dict) or set(pin) != {"path", "sha256"}:
        raise ValueError("all-output artifact has no exact identity")
    path = Path(pin["path"])
    if not path.is_absolute() or not path.is_file() or sha256_file(path) != pin["sha256"]:
        raise ValueError(f"all-output artifact differs: {path}")
    return path


def _read(pin):
    path = _pinned(pin)
    value = json.loads(path.read_text())
    _pinned(pin)
    return value


def _helper(pin):
    path = _pinned(pin)
    spec = importlib.util.spec_from_file_location("phase1_pinned_all_output_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _pinned(pin)
    if not callable(getattr(module, "verify", None)):
        raise ValueError("all-output helper has no offline verification API")
    return module


def validate(launch, *, identity, models):
    """Admit only metadata here: the offline audit later reopens every array byte."""
    policy = launch.document.get("full_model_verification") or {}
    scope = policy.get("output_scope", "judged_only")
    if scope != "all_declared":
        if scope != "judged_only" or "all_output_plan" in policy:
            raise ValueError("invalid full-model output scope/plan")
        return None
    pin = policy.get("all_output_plan")
    if not isinstance(pin, dict):
        raise ValueError("all-declared output scope requires a pinned plan")
    pin = {**pin, "path": str(launch.path(pin["path"]))}
    if any(
        Path(launch.path(row["path"])).resolve() == Path(pin["path"]).resolve()
        for row in launch.document.get("runtime_pins", {}).values()
    ):
        raise ValueError("all-output plan cannot recursively pin itself through selected runtime identity")
    plan = _read(pin)
    if (
        plan.get("schema") != "merlin.operator.all_output_plan.v1"
        or plan.get("selected_identity") != identity
        or plan.get("run_id", launch.run_id) != launch.run_id
        or _read(plan["expected_identity"]) != identity
    ):
        raise ValueError("all-output plan selected source/input/runtime identity differs")
    float_policy = {"atol": policy["atol"], "rtol": policy["rtol"], "equal_nan": False, "finite_required": True}
    module = _helper(plan["helper"])
    selected_comparison = (module.comparison_policy(models.path) if hasattr(module, "comparison_policy")
                           else getattr(module, "FLOAT_POLICY", None))
    if (
        plan.get("comparison") != {"float": float_policy, "integer_bool": "exact"}
        or selected_comparison != float_policy
        or plan.get("extension_runtime_key") not in identity["runtime_pins"]
        or set(plan.get("references", {})) != set(models.models)
    ):
        raise ValueError("all-output plan comparison/runtime/model roster differs")
    _pinned(plan["requirement"])
    bundle_keys = ("bundle_manifest_sha256", "bundle_sums_sha256", "models_config_sha256")
    expected = {}
    for name, record in plan["references"].items():
        reference = _read(record)
        model = models.model(name)
        if (
            reference.get("schema") != "merlin.operator.full_declared_output_reference.v1"
            or reference.get("status") != "complete"
            or reference.get("model") != name
            or reference.get("input_id") != policy["input"]
            or reference.get("representation") != "declared_storage"
            or reference.get("selected_identity") != {key: identity[key] for key in bundle_keys}
            or reference.get("judged_crossjoin", {}).get("bit_exact") is not True
        ):
            raise ValueError(f"all-output reference selection is incomplete: {name}")
        programs = reference["programs"]
        if [row["program"] for row in programs] != [p.name for p in model.programs]:
            raise ValueError(f"all-output reference program roster differs: {name}")
        for program, row in zip(model.programs, programs, strict=True):
            if (
                row["steps_declared"] != program.steps
                or not row["declared_outputs"]
                or [step["step"] for step in row["steps"]] != list(range(program.steps))
                or [output["output"] for output in row["declared_outputs"]] != list(range(len(row["declared_outputs"])))
            ):
                raise ValueError(f"all-output reference step/output roster differs: {name}")
        expected[name] = {
            "steps": sum(p.steps for p in model.programs),
            "outputs": sum(
                p.steps * len(row["declared_outputs"]) for p, row in zip(model.programs, programs, strict=True)
            ),
        }
    _pinned(pin)
    return {"pin": pin, "plan": plan, "helper": module, "expected": expected, "identity": identity}


def audit(context, *, name, child, package, models_path):
    """Cache even an unavailable result on this child; polling never repeats IR parsing."""

    def admitted():
        _pinned(context["pin"])
        for key in ("helper", "expected_identity", "requirement"):
            _pinned(context["plan"][key])
        _pinned(context["plan"]["references"][name])

    admitted()
    if "all_outputs" in child:
        try:
            _pinned({key: child["all_outputs"][key] for key in ("path", "sha256")})
            if (
                sha256_file(child["report"]) != child["all_outputs_report_sha256"]
                or sha256_file(child["pin"]) != child["all_outputs_job_sha256"]
            ):
                raise ValueError("completed report/job changed after audit")
        except (OSError, ValueError) as error:
            return {**child["all_outputs"], "status": "error", "detail": str(error)}
        return child["all_outputs"]
    child["all_outputs_report_sha256"] = sha256_file(child["report"])
    child["all_outputs_job_sha256"] = sha256_file(child["pin"])
    path = child["report"].with_name(name + "-all-outputs.json")
    plan, identity = context["plan"], context["identity"]
    try:
        admitted()
        job = json.loads(child["pin"].read_text())
        work = Path(job["work"]) / package["model_check_tree_sha256"][:16] / name / "runs"
        roots = [root for root in work.iterdir() if root.is_dir() and (root / child["input_id"]).is_dir()]
        if len(roots) != 1:
            raise ValueError("full-model job lacks one unambiguous retained execution root")
        result = context["helper"].verify(
            reference_path=Path(plan["references"][name]["path"]),
            expected_reference_sha256=plan["references"][name]["sha256"],
            job_pin_path=child["pin"],
            report_path=child["report"],
            models_path=models_path,
            run_root=roots[0],
            expected_identity=identity,
            expected_identity_path=Path(plan["expected_identity"]["path"]),
            requirement_path=Path(plan["requirement"]["path"]),
            extension_runtime_key=plan["extension_runtime_key"],
        )
        admitted()
        expected = context["expected"][name]
        if (
            result.get("schema") != "merlin.operator.full_declared_output_audit.v1"
            or result.get("status") != "pass"
            or result.get("unchanged") is not True
            or result.get("model") != name
            or result.get("input_id") != child["input_id"]
            or result.get("selected_identity") != identity
            or result.get("submission") != package
            or result.get("steps_compared") != expected["steps"]
            or result.get("steps_expected") != expected["steps"]
            or result.get("outputs_compared") != expected["outputs"]
            or result.get("n_declared_outputs") != expected["outputs"]
        ):
            result = {**result, "status": "fail", "operator_detail": "all-output evidence is incomplete or failed"}
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = {
            "schema": "merlin.operator.full_declared_output_audit.v1",
            "status": "error",
            "detail": f"{type(error).__name__}: {error}",
        }
    result["admitted_plan"] = context["pin"]
    temporary = path.with_suffix(".json.next")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    child["all_outputs"] = {
        "status": result["status"],
        "path": str(path),
        "sha256": sha256_file(path),
        "outputs_compared": result.get("outputs_compared", 0),
        "steps_compared": result.get("steps_compared", 0),
    }
    return child["all_outputs"]


def finalize(context, children):
    """Reopen every original artifact once at aggregate completion; never reparse IR."""
    path = next(iter(children.values()))["report"].parent / "all-outputs-finalization.json"
    pins = {}

    def admit(pin):
        if pin["path"] in pins and pins[pin["path"]] != pin["sha256"]:
            raise ValueError("conflicting all-output final artifact identities")
        if pin["path"] not in pins:
            _pinned(pin)
            pins[pin["path"]] = pin["sha256"]

    try:
        admit(context["pin"])
        if set(children) != set(context["expected"]):
            raise ValueError("all-output final model roster is incomplete")
        for child in children.values():
            row = child["all_outputs"]
            pin = {key: row[key] for key in ("path", "sha256")}
            admit(pin)
            value = _read(pin)
            if row["status"] != "pass" or value["status"] != "pass" or not value.get("artifacts"):
                raise ValueError("all-output final model artifact evidence is incomplete")
            for artifact in value["artifacts"]:
                admit(artifact)
        value = {"status": "pass", "artifacts_verified": len(pins), "artifacts": pins}
    except (OSError, ValueError, KeyError, TypeError) as error:
        value = {"status": "error", "detail": f"{type(error).__name__}: {error}"}
    path.write_text(json.dumps(value, indent=2) + "\n")
    return {"status": value["status"], "path": str(path), "sha256": sha256_file(path)}
