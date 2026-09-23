"""Read-only discovery of stored phase orchestrations, not a second run database."""

from __future__ import annotations

from pathlib import Path

from .spec import SpecError

# This is a navigation projection of frozen records, not another evidence store.
# Keep private profiles, holdout catalogs, and implementation inventories out of
# the public view even though their fingerprints remain in resolved-plan.json.
_VISIBLE_INPUTS = {
    "0": ("descriptor", "recipe", "performance_template", "synth_profile", "smt_profile"),
    "1": ("descriptor", "corpus_seal", "bundle_manifest", "oracle_timing"),
    "2": (
        "descriptor",
        "rtl_facts",
        "perf_profile",
        "gsim_certificate",
        "functional_gsim_certificate",
        "campaign_config",
        "deployment",
    ),
}


def lineage(run_dir: Path) -> dict:
    """Project a run's recorded handoffs without consulting mutable live inputs.

    The orchestration record binds the frozen plan's bytes. An input is called
    pinned only when that plan contains its fingerprint; historical receipts with
    weaker evidence remain visible as such. Engine output paths and process exit
    states are navigation aids, not certification or publication verdicts.
    """
    from .runner import _read_json, status

    record = status(run_dir)  # Checks orchestration -> frozen-plan SHA-256 binding.
    plan = _read_json(Path(record["run_dir"]) / "resolved-plan.json")
    pins = plan.get("inputs", {})

    def selected(name: str, path: str) -> dict:
        pin = pins.get(name)
        if isinstance(pin, dict) and pin.get("path") == path and isinstance(pin.get("sha256"), str):
            return {"path": path, "sha256": pin["sha256"], "identity": "frozen_input"}
        return {"path": path, "sha256": None, "identity": "historical_unverified"}

    phases = {}
    for number, command in plan.get("phases", {}).items():
        inputs = {}
        for name in _VISIBLE_INPUTS.get(number, ()):
            path = command.get("inputs", {}).get(name)
            if path is not None:
                key = f"phase{number}:{name}"
                if number == "0" and name in {"recipe", "performance_template", "synth_profile", "smt_profile"}:
                    key = f"phase0:operator:{name}"
                member = plan.get("phase0_operator_inputs", {}).get(key) if number == "0" else None
                inputs[name] = (
                    {"path": path, "sha256": None, "identity": "declared_absent_at_freeze"}
                    if member is not None and member.get("present") is False
                    else selected(key, path)
                )
        latest = next(
            (entry for entry in reversed(record["attempts"]) if entry.get("phase") == number),
            {},
        )
        handoff = {}
        if number == "0":
            operator = plan.get("phase0_operator_inputs", {})
            hidden = operator.get("phase0:operator:hidden_profile")
            handoff["private_profile_present_at_freeze"] = bool(hidden and hidden.get("present"))
            if latest.get("output_sha256"):
                handoff["generated_capsules_sha256"] = latest["output_sha256"]
        elif number == "1" and "corpus_seal" in inputs:
            handoff["selected_corpus_release"] = str(Path(inputs["corpus_seal"]["path"]).parent.parent)
        elif number == "2":
            config = plan.get("spec", {}).get("phases", {}).get("2", {}).get("config", {})
            for name in (
                "functional_run_id",
                "functional_submission_sha256",
                "gsim_certificate_sha256",
                "functional_gsim_certificate_sha256",
            ):
                if name in config:
                    handoff[name] = config[name]
        phases[number] = {
            "adapter": command["adapter"],
            "inputs": inputs,
            "handoff": handoff,
            "output": {
                "path": latest.get("engine_output", command.get("engine_output")),
                "state": latest.get("state", "not_started"),
                "sha256": latest.get("output_sha256"),
            },
        }
    definition = plan.get("definition")
    return {
        "scope": "frozen_orchestration_lineage",
        "run_dir": record["run_dir"],
        "experiment": record["experiment"],
        "target": record["target"],
        "state": record["state"],
        "frozen_at": plan.get("frozen_at"),
        "definition": selected("definition", definition) if definition is not None else None,
        "phases": phases,
        "evidence_authority": record["evidence_authority"],
        "qualification": "not_assessed_by_lineage",
    }


def runs(*, root: Path | None = None, target: str | None = None, experiment: str | None = None) -> dict:
    """Read canonical target/experiment/run records without following directory aliases.

    Native engine runs remain owned by AET. Explicit noncanonical run directories
    are inspected with ``status``; discovery does not search arbitrary artifacts or
    rewrite historical records. Corrupt/incomplete orchestrations remain visible as
    problems, never silently presented as successful runs.
    """
    from merlin.common.paths import runs_dir

    from .runner import status

    base = (runs_dir() if root is None else root).expanduser().resolve()
    result = {"root": str(base), "scope": "phase_orchestrations", "runs": [], "problems": []}
    if not base.exists():
        return result
    if not base.is_dir():
        raise SpecError(f"run discovery root is not a directory: {base}")

    def directories(parent: Path):
        try:
            return sorted(path for path in parent.iterdir() if not path.is_symlink() and path.is_dir())
        except OSError as exc:
            result["problems"].append({"run_dir": str(parent), "error": type(exc).__name__})
            return []

    for target_dir in directories(base):
        if target is not None and target_dir.name != target:
            continue
        for experiment_dir in directories(target_dir):
            if experiment is not None and experiment_dir.name != experiment:
                continue
            for run_dir in directories(experiment_dir):
                records = [run_dir / "resolved-plan.json", run_dir / "orchestration.json"]
                if not any(path.exists() or path.is_symlink() for path in records):
                    continue
                try:
                    if any(path.is_symlink() for path in records):
                        raise SpecError("orchestration records may not be symlinks")
                    if any(not path.is_file() for path in records):
                        raise SpecError("orchestration records must be present regular files")
                    record = status(run_dir)
                    if record["target"] != target_dir.name or record["experiment"] != experiment_dir.name:
                        raise SpecError("record identity differs from its target/experiment directory")
                except (OSError, SpecError, KeyError, TypeError, ValueError) as exc:
                    result["problems"].append({"run_dir": str(run_dir), "error": str(exc)})
                    continue
                result["runs"].append(record)
    return result
