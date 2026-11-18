"""Stored run discovery, lineage, and compiler-history export."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path

from merlin.experiments.definition import SpecError

# ---- history -----------------------------------------------------------------------------------
# Read-only discovery of stored phase orchestrations, not a second run database.

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
    from merlin.experiments.runs.plan_sources import _read_json
    from merlin.experiments.runs.plans import status

    record = status(run_dir)  # Checks orchestration -> frozen-plan SHA-256 binding.
    plan = _read_json(Path(record["run_dir"]) / "resolved-plan.json")
    pins = plan.get("inputs", {})

    def phase0_descriptor_snapshot(path: str, pin: dict) -> dict | None:
        """Recognize the evidence copy of a pinned descriptor by its actual bytes."""
        digest = pin.get("sha256")
        if not isinstance(digest, str):
            return None
        phase0 = Path(record["run_dir"]) / "phase0"
        software = phase0 / "software"
        snapshots = software / "source-snapshots"
        selected_path = Path(path)
        manifest_path = phase0 / "evidence-manifest.json"
        if selected_path.parent != snapshots:
            return None
        try:
            if any(member.is_symlink() for member in (phase0, software, snapshots, selected_path, manifest_path)):
                return None
            if not selected_path.is_file() or not manifest_path.is_file():
                return None
            manifest = _read_json(manifest_path)
            sources = manifest.get("sources") if isinstance(manifest, dict) else None
            if not isinstance(sources, list):
                return None
            relative = f"software/source-snapshots/{selected_path.name}"
            if not any(
                isinstance(source, dict)
                and source.get("role") == "descriptor"
                and source.get("path") == relative
                and source.get("sha256") == digest
                for source in sources
            ):
                return None
            with selected_path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    return None
        except (OSError, SpecError, TypeError, ValueError):
            return None
        return {"path": path, "sha256": digest, "identity": "frozen_equivalent_snapshot"}

    def selected(name: str, path: str) -> dict:
        pin = pins.get(name)
        if isinstance(pin, dict) and pin.get("path") == path and isinstance(pin.get("sha256"), str):
            return {"path": path, "sha256": pin["sha256"], "identity": "frozen_input"}
        if name == "phase0:descriptor" and isinstance(pin, dict):
            snapshot = phase0_descriptor_snapshot(path, pin)
            if snapshot is not None:
                return snapshot
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
    definition_pin = pins.get("definition")
    if isinstance(definition_pin, dict) and isinstance(definition_pin.get("path"), str):
        # Fresh Phase 0 freezes the authored definition under its own input
        # snapshot. The plan's `definition` field remains the original source
        # location for navigation; comparing that mutable path to the frozen
        # pin incorrectly labels a new run as historical/unverified.
        definition_record = selected("definition", definition_pin["path"])
        if definition != definition_pin["path"]:
            definition_record["source_path"] = definition
    else:
        definition_record = selected("definition", definition) if definition is not None else None
    return {
        "scope": "frozen_orchestration_lineage",
        "run_dir": record["run_dir"],
        "experiment": record["experiment"],
        "target": record["target"],
        "state": record["state"],
        "frozen_at": plan.get("frozen_at"),
        "definition": definition_record,
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
    from merlin.experiments.definition import ENGINE_STAGES
    from merlin.experiments.runs.plans import status

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

    phase_suites = {"phase0", "phase1", "phase2", *ENGINE_STAGES.values()}
    for target_dir in directories(base):
        if target is not None and target_dir.name != target:
            continue
        owners = directories(target_dir)
        optimization = target_dir / ENGINE_STAGES["2"]
        if optimization.is_dir() and not optimization.is_symlink():
            owners.append(optimization)
        for experiment_dir in owners:
            # A single-phase orchestration lives under its phase (runs/<target>/phase<N>/), named
            # <TS>_<experiment>_<sha7>; every other one under runs/<target>/<experiment>/.
            by_phase = experiment_dir.name in phase_suites or experiment_dir == optimization
            if experiment is not None and not by_phase and experiment_dir.name != experiment:
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
                    placed = (
                        f"_{record['experiment']}_" in f"{run_dir.name}_"
                        if by_phase
                        else record["experiment"] == experiment_dir.name
                    )
                    if record["target"] != target_dir.name or not placed:
                        raise SpecError("record identity differs from its target/experiment directory")
                    if experiment is not None and record["experiment"] != experiment:
                        continue
                except (OSError, SpecError, KeyError, TypeError, ValueError) as exc:
                    result["problems"].append({"run_dir": str(run_dir), "error": str(exc)})
                    continue
                result["runs"].append(record)
    return result


# ---- backend_export ----------------------------------------------------------------------------
# Export an existing phase champion and inspect its preserved service metadata.
#
# The champion owner retains evidence validation, source custody and publication.
# This command neither creates a compiler adapter nor installs or publishes code.


def service_declarations(root: Path) -> dict:
    """Read ordinary compiler declarations without importing or validating them."""
    result = {
        "scope": "preserved_project_metadata_only",
        "compatibility": "not_checked",
        "installed_by_export": False,
        "code_loaded_by_inspection": False,
        "providers": [],
    }
    path = root / "pyproject.toml"
    if not path.exists():
        return {**result, "status": "missing_metadata"}
    try:
        with path.open("rb") as stream:
            document = tomllib.load(stream)
        if "project" not in document:
            return {**result, "status": "missing_project_metadata"}
        project = document.get("project", {})
        if not isinstance(project, dict):
            raise ValueError("project must be a table")
        dynamic = project.get("dynamic", [])
        if not isinstance(dynamic, list) or any(not isinstance(value, str) for value in dynamic):
            raise ValueError("project.dynamic must be a list of field names")
        groups = project.get("entry-points", {})
        if not isinstance(groups, dict):
            raise ValueError("project.entry-points must be a table")
        if "entry-points" in dynamic:
            if "entry-points" in project:
                raise ValueError("entry points cannot be both static and dynamic")
            return {**result, "status": "dynamic"}
        providers = groups.get("merlin.backends", {})
        if not isinstance(providers, dict) or any(
            not isinstance(name, str) or not name.strip() or not isinstance(value, str) or not value.strip()
            for name, value in providers.items()
        ):
            raise ValueError("merlin.backends must map nonempty names to nonempty entry-point strings")
        return {
            **result,
            "status": "declared" if providers else "absent",
            "providers": [{"name": name, "entry_point": providers[name]} for name in sorted(providers)],
        }
    except (OSError, ValueError):
        return {**result, "status": "invalid_metadata"}


def _object(pairs):
    document = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate evidence key: {key}")
        document[key] = value
    return document


def _record(path: Path) -> dict:
    def invalid_constant(value):
        raise ValueError(f"invalid JSON numeric constant: {value}")

    document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object, parse_constant=invalid_constant)
    if not isinstance(document, dict):
        raise ValueError(f"evidence must be a JSON object: {path}")
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="merlin experiment export",
        description="Export the selected frozen/best compiler history using its existing champion evidence.",
        epilog="Preserves existing packaging; does not synthesize adapters, install code or publish remotely.",
    )
    parser.add_argument("--target", required=True)
    parser.add_argument("--repo", type=Path, required=True, help="the run's existing OOT Git history")
    parser.add_argument("--phase", type=int, choices=(1, 2), required=True)
    parser.add_argument("--package-id", required=True)
    for name in ("provenance", "certification", "measurements", "isa-prohibition"):
        parser.add_argument("--" + name, type=Path, required=True, help="existing scoped evidence JSON record")
    parser.add_argument("--rev", help="selected revision; defaults to the phase's frozen/best tag")
    parser.add_argument("--run-dir", type=Path, help="owning run directory; defaults to the OOT history's parent")
    parser.add_argument("--stage-root", type=Path, help="explicit retained export staging parent")
    parser.add_argument("--artifacts-root", type=Path, help="artifact root containing targets/<target>/champions")
    args = parser.parse_args(argv)

    from merlin.development.targetgen.packaging.champions import ChampionError, export_champion

    try:
        records = {
            name: _record(getattr(args, name))
            for name in ("provenance", "certification", "measurements", "isa_prohibition")
        }
        destination = export_champion(
            args.target,
            args.repo,
            package_id=args.package_id,
            phase=args.phase,
            rev=args.rev,
            phase2_run=args.run_dir,
            stage_root=args.stage_root,
            artifacts_root=args.artifacts_root,
            **records,
        )
    except (ChampionError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"directory": str(destination), "services": service_declarations(destination)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
