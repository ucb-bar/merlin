"""Stage Phase 0 with the existing private snapshot and guarded Python owners."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import yaml

from merlin.common.paths import checkout_root, module_source_path, python_import_roots
from merlin_experiments import frozen_python, source_snapshot
from merlin_experiments.measured_launch import FROZEN_PHASE0_ENV_POLICY
from merlin_experiments.phase1.source_inputs import fingerprint

from .evidence import export_evidence, select_evidence


def _input_owner(path: Path) -> str:
    return hashlib.sha256(str(path.absolute()).encode()).hexdigest()


def _model_root(evidence, software_path: str) -> Path | None:
    """Resolve the declared model owner only while observing the live selection."""
    from merlin.targetgen.software_spec import software_spec_references

    sources = [source for source in evidence.source_snapshots if source.role == "software-reference:numerical_model"]
    if not sources:
        return None
    references = software_spec_references(software_path, document=evidence.software_spec)
    root = references.get("numerical_model")
    if root is None or any(not source.path.is_relative_to(root) for source in sources):
        raise ValueError("captured numerical source escapes its selected model owner")
    return root


def selection(command: dict, target: str):
    inputs = command["inputs"]
    capture = command.get("phase0_m2m_selection") or {}
    return select_evidence(
        target,
        descriptor=inputs["descriptor"],
        capture_python=capture.get("python"),
        capability_contract_path=inputs.get("capability_contract"),
        facts_path=inputs.get("rtl_facts"),
        hardware_spec=inputs.get("hardware_spec"),
        software_spec=inputs.get("software_spec"),
        conformance_spec=inputs.get("conformance_spec"),
        prohibited_roles=(command.get("instruction_policy") or {}).get("prohibited_instruction_roles") or (),
    )


def selected_inputs(command: dict, target: str) -> tuple[dict[str, str], dict]:
    from .component_semantic_basis import ComponentSemanticBasis

    observed = selection(command, target)
    paths = {f"phase0:evidence:{i:04d}": str(path) for i, path in enumerate(observed.source_paths)}
    paths.update({f"phase0:materialized:{i:04d}": str(path) for i, path in enumerate(_materialized_inputs(command))})
    basis = ComponentSemanticBasis.from_recipe(command.get("inputs", {}).get("recipe"))
    if basis is not None:
        paths.update({f"phase0:semantic_basis:{i:04d}": source["path"] for i, source in enumerate(basis.sources())})
    return paths, {
        "status": observed.status,
        "raw_facts_sha256": observed.raw_facts_sha256,
        "views_sha256": hashlib.sha256(observed.views_json).hexdigest(),
        "diagnostics": observed.diagnostics,
    }


def _materialized_inputs(command: dict) -> dict[Path, tuple[Path, str]]:
    """Own only receipt-bound iteration payloads selected by the synthesis profile."""
    from merlin.targetgen.capsule_source import materialized_model_artifacts

    value = command.get("inputs", {}).get("synth_profile")
    if not value or not Path(value).is_file():
        return {}
    profile = Path(value).absolute()
    document = yaml.safe_load(profile.read_bytes())
    result = {}
    for entry in (document or {}).get("capsules", []):
        binding = entry.get("materialized_capture")
        if not binding:
            continue
        relative = Path(binding["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("frozen materialized capture must be contained beside its synthesis profile")
        source = profile.parent / relative
        resolved = {**binding, "path": str(source)}
        loader = None
        if binding.get("loader_path"):
            relative_loader = Path(binding["loader_path"])
            if relative_loader.is_absolute() or ".." in relative_loader.parts:
                raise ValueError("frozen materialized loader must be contained beside its synthesis profile")
            loader = profile.parent / relative_loader
            resolved["loader_path"] = str(loader)
        artifact = materialized_model_artifacts(resolved)
        receipt_path = source.parent / "capture_receipt.json"
        receipt = json.loads(receipt_path.read_bytes())
        if hashlib.sha256(receipt_path.read_bytes()).hexdigest() != binding["receipt_sha256"]:
            raise ValueError("selected materialized capture receipt changed before freezing")
        members = set(receipt["artifacts"]) | {"capture_receipt.json"}
        if artifact.meta.get("framework_catalog"):
            members.add("pytorch-opset.json")
        for member in sorted(members):
            path = source.parent / member
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"selected materialized member is not an ordinary file: {path}")
            result[path] = (profile.parent, path.relative_to(profile.parent).as_posix())
        if loader is not None:
            if loader.is_symlink() or not loader.is_file():
                raise ValueError(f"selected materialized loader is not an ordinary file: {loader}")
            if hashlib.sha256(loader.read_bytes()).hexdigest() != binding.get("loader_sha256"):
                raise ValueError("selected materialized loader changed before freezing")
            result[loader] = (profile.parent, loader.relative_to(profile.parent).as_posix())
    return result


def _source_layout():
    root = checkout_root()
    if root is None:
        # Give an installed import root a real contained directory name: the
        # snapshot import-layout owner deliberately rejects the ambiguous '.'.
        root = module_source_path("merlin").parent.parent.parent
    source_roots, import_roots = [], []
    for parent in python_import_roots():
        for name in ("merlin", "merlin_experiments", "merlin_analysis", "merlin_dse", "merlin_mining"):
            path = parent / name
            if not path.is_dir() or not path.resolve().is_relative_to(root):
                continue
            relative = path.resolve().relative_to(root).as_posix()
            if relative not in source_roots:
                source_roots.append(relative)
                import_roots.append(path.resolve().parent.relative_to(root).as_posix())
    for relative in ("merlin/schemas", "merlin/contract"):
        if (root / relative).is_dir() and not any((root / relative).is_relative_to(root / p) for p in source_roots):
            source_roots.append(relative)
    return root, tuple(source_roots), tuple(dict.fromkeys(import_roots))


def stage(plan: dict) -> dict:
    """Freeze selected evidence before executing; historical plans are unchanged."""
    frozen = copy.deepcopy(plan)
    command = frozen["phases"].get("0")
    if command is None or not command.get("inputs", {}).get("software_spec"):
        return frozen
    if not command.get("phase0_evidence"):
        raise ValueError("explicit software selection lacks planned Phase 0 evidence")
    target, run_root = frozen["target"], Path(frozen["run_dir"])
    evidence = selection(command, target)
    from .component_semantic_basis import ComponentSemanticBasis

    basis = ComponentSemanticBasis.from_recipe(command["inputs"].get("recipe"))
    basis_sources = basis.sources() if basis is not None else []
    basis_pins = [pin for name, pin in frozen["inputs"].items() if name.startswith("phase0:semantic_basis:")]
    if basis_pins != [{"path": row["path"], "sha256": row["sha256"]} for row in basis_sources]:
        raise ValueError("selected semantic basis membership or bytes changed before freezing")
    selected_digest = command["phase0_evidence"].get("views_sha256")
    if selected_digest is not None and selected_digest != hashlib.sha256(evidence.views_json).hexdigest():
        raise ValueError("Phase 0 resolved evidence changed before staging; create a new experiment plan")
    expected = {name: pin for name, pin in frozen["inputs"].items() if name.startswith("phase0:evidence:")}
    if [pin["path"] for pin in expected.values()] != [str(path) for path in evidence.source_paths]:
        raise ValueError("Phase 0 evidence membership changed before staging")
    for pin in expected.values():
        if fingerprint(pin["path"]) != pin["sha256"]:
            raise ValueError("Phase 0 evidence bytes changed before staging")
    artifact_root = run_root / "phase0"
    manifest = export_evidence(evidence, artifact_root)
    root, source_roots, import_roots = _source_layout()
    provider = source_snapshot.selected_provider(target)
    # The selected support is an independent, contained private owner, even
    # when its original location was a checkout reference shelf.
    provider["kind"] = "external"
    declared, names, layout = {}, {}, {}
    materialized = _materialized_inputs(command)
    materialized_pins = [pin for name, pin in frozen["inputs"].items() if name.startswith("phase0:materialized:")]
    if [pin["path"] for pin in materialized_pins] != [str(path) for path in materialized]:
        raise ValueError("selected materialized iteration membership changed before freezing")
    if any(fingerprint(pin["path"]) != pin["sha256"] for pin in materialized_pins):
        raise ValueError("selected materialized iteration bytes changed before freezing")
    for i, (name, value) in enumerate(frozen["input_paths"].items()):
        if name.startswith("phase0:evidence:"):
            continue  # Exact source bytes already live in the evidence bundle.
        path = Path(value)
        if path.name == ".env" or path.name.startswith(".env."):
            raise ValueError("environment configuration is not a Phase 0 source input")
        if not path.is_file():
            raise ValueError(f"Phase 0 snapshot requires ordinary selected files: {path}")
        key = f"input_{i:04d}"
        declared[key], names[name] = path, key
        parent, relative = materialized.get(path, (path.parent, path.name))
        owner = _input_owner(parent)
        layout[key] = f"{source_snapshot.INPUT_ROOT}/{owner}/{relative}"
    # Evidence bytes remain in their archival bundle, but executable references
    # need their original relative tree. Copy only the observed source inventory,
    # never an entire numerical checkout or its environment configuration.
    model_root = _model_root(evidence, command["inputs"]["software_spec"])
    source_names = {}
    for i, source in enumerate(evidence.source_snapshots):
        key = f"evidence_{i:04d}"
        declared[key] = source.path
        source_names[str(source.path)] = key
        if source.role == "software-reference:numerical_model":
            relative = source.path.relative_to(model_root).as_posix()
            owner = _input_owner(model_root)
        else:
            relative = source.path.name
            owner = _input_owner(source.path.parent)
        layout[key] = f"{source_snapshot.INPUT_ROOT}/{owner}/{relative}"
    snapshot = artifact_root / "private" / "source"
    excluded = {"merlin/contract/capsules"}
    for relative in source_roots:
        for path in (root / relative).rglob(".env*"):
            if path.name == ".env" or path.name.startswith(".env."):
                excluded.add(path.relative_to(root).as_posix())
    aliases = {}
    core = module_source_path("merlin").parent.resolve().relative_to(root).as_posix()
    if core != "merlin" and not any(Path(p).parts[0] == "merlin" for p in source_roots):
        # Resource selectors use root/merlin in both layouts. This alias points
        # only inside the sealed tree; it is never a link back to an installation.
        aliases["merlin"] = core
    source_snapshot.create(
        root,
        snapshot,
        output_root=Path(frozen["storage_root"]),
        source_roots=source_roots,
        python_roots=import_roots,
        legacy_roots=(),
        target_name=target,
        provider=provider,
        declared_inputs=declared,
        declared_input_layout=layout,
        root_files=(),
        runtime_links=False,
        exclude_paths=tuple(sorted(excluded)),
        internal_aliases=aliases,
    )
    receipt = source_snapshot.verify(snapshot)
    captured_paths = {}
    for source in evidence.source_snapshots:
        captured = source_snapshot.remap_input(snapshot, receipt, source.path, name=source_names[str(source.path)])
        if captured.read_bytes() != source.content:
            raise ValueError("Phase 0 evidence bytes changed during source staging")
        captured_paths[str(source.path)] = str(captured)
    original_inputs = copy.deepcopy(frozen["inputs"])
    original_paths = dict(frozen["input_paths"])
    for name, key in names.items():
        frozen["input_paths"][name] = str(source_snapshot.remap_input(snapshot, receipt, declared[key], name=key))
    for name, source in zip(expected, manifest["sources"], strict=True):
        frozen["input_paths"][name] = str(artifact_root / source["path"])
    path_map = {value: frozen["input_paths"][name] for name, value in original_paths.items()}
    if basis is not None:
        recipe_path = str(Path(command["inputs"]["recipe"]).resolve())
        recipe_pin = {"path": recipe_path, "sha256": fingerprint(recipe_path)}
        for source in [*basis_sources, recipe_pin]:
            captured = path_map[source["path"]]
            if fingerprint(captured) != source["sha256"]:
                raise ValueError("selected semantic basis bytes changed during source staging")
            captured_paths[source["path"]] = captured
        frozen["phase0_semantic_basis"] = {**basis.record(), "recipe": recipe_pin}
    command["argv"] = [path_map.get(arg, arg) for arg in command["argv"]]
    command["inputs"] = {name: path_map.get(value, value) for name, value in command["inputs"].items()}
    # Absent sidecars remain absent frozen selections, never live-path probes.
    for name, value in list(command["inputs"].items()):
        if value not in path_map.values() and not Path(value).exists():
            owner = _input_owner(Path(value).parent)
            staged = str(snapshot / source_snapshot.INPUT_ROOT / owner / Path(value).name)
            command["argv"] = [staged if arg == value else arg for arg in command["argv"]]
            command["inputs"][name] = staged
    command["argv"] += ["--evidence-input", str(artifact_root)]
    command["entrypoint"] = path_map[command["entrypoint"]]
    command["cwd"] = str(snapshot)
    command["source_snapshot"] = str(snapshot)
    command["phase0_environment_policy"] = FROZEN_PHASE0_ENV_POLICY
    command["env"].update(source_snapshot.provider_environment(snapshot, receipt))
    command["env"].update(
        MERLIN_REPO_ROOT=str(snapshot),
        PYTHONPATH=os.pathsep.join(str(snapshot / p) for p in import_roots),
        PYTHONSAFEPATH="1",
        PYTHONNOUSERSITE="1",
    )
    command["env"]["MERLIN_TARGET_EXPERIMENT"] = command["inputs"]["descriptor"]
    command["env"]["MERLIN_PHASE0_FROZEN_SOURCE_MAP"] = json.dumps(captured_paths, sort_keys=True)
    if "--evidence-mode" in command["argv"]:
        command["env"]["MERLIN_PHASE0_EVIDENCE_MODE"] = command["argv"][command["argv"].index("--evidence-mode") + 1]
    selected_m2m = command.get("phase0_m2m_selection")
    if selected_m2m is not None:
        from . import m2m_runtime

        selected_m2m = m2m_runtime.stage(selected_m2m, artifact_root / "private" / "m2m-source")
        m2m_runtime.verify(selected_m2m)
        command["phase0_m2m_selection"] = selected_m2m
        command["env"].update(m2m_runtime.environment(selected_m2m))
        from .sealed_generation import CONFIG_ENV

        m2m_receipt = artifact_root / "private" / "m2m-runtime.json"
        with m2m_receipt.open("xb") as stream:
            stream.write(m2m_runtime.receipt(selected_m2m))
        m2m_receipt.chmod(0o444)
        if CONFIG_ENV in command["env"]:
            command["env"][CONFIG_ENV] = json.dumps(
                m2m_runtime.sealed_capture_config(
                    selected_m2m,
                    artifact_root,
                    execution_timeout_seconds=command.get("phase0_capture_timeout_seconds"),
                    bwrap=command.get("phase0_bwrap"),
                ),
                sort_keys=True,
            )
        frozen["phase0_m2m_runtime_receipt"] = str(m2m_receipt)
        frozen["input_paths"]["phase0:m2m_runtime_receipt"] = str(m2m_receipt)
    model = (evidence.software_spec.get("numerical_semantics") or {}).get("model") or {}
    env_name = model.get("source_root_env")
    if env_name in {"HOME", "home", "CODEX_HOME"} or (env_name and env_name in command["env"]):
        raise ValueError("numerical model environment collides with process ownership")
    # An unavailable model stays unavailable: inherited environment or an
    # installed package must not silently fill an uncaptured source selection.
    model_env = {"MERLIN_PHASE0_NUMERICAL_MODEL_ROOT": str(snapshot / "_missing_numerical_model")}
    if env_name:
        model_env[env_name] = ""
    if model_root is not None:
        # The numerical files may already belong to a sealed source/provider root.
        first = next(
            source for source in evidence.source_snapshots if source.role == "software-reference:numerical_model"
        )
        captured = Path(captured_paths[str(first.path)])
        relative = first.path.relative_to(model_root)
        staged_root = captured.parents[len(relative.parts) - 1]
        model_env["MERLIN_PHASE0_NUMERICAL_MODEL_ROOT"] = str(staged_root)
        if env_name:
            model_env[env_name] = str(staged_root)
    command["env"].update(model_env)
    command["frozen_launch"] = frozen_python.python_command(
        snapshot,
        command["argv"],
        verifier_source=Path(source_snapshot.__file__),
        instrumentation_from_snapshot=True,
    )
    from merlin_experiments.runner import _phase0_operator_inputs

    frozen["phase0_operator_inputs"] = _phase0_operator_inputs(command)
    for name, record in frozen["phase0_operator_inputs"].items():
        if record is not None and record["present"]:
            frozen["input_paths"][name] = record["path"]
    frozen["inputs"] = {
        name: {"path": value, "sha256": fingerprint(value)} for name, value in frozen["input_paths"].items()
    }
    frozen["phase0_original_inputs"] = original_inputs
    frozen["phase0_evidence_bundle"] = str(artifact_root)
    frozen["phase0_source_snapshot"] = str(snapshot)
    frozen["phase0_frozen_source_paths"] = captured_paths
    frozen["phase0_numerical_environment"] = model_env
    return frozen


def _verify_frozen_sources(plan: dict) -> dict:
    """Verify archived source and command bindings without consulting live owners."""
    snapshot = Path(plan["phase0_source_snapshot"])
    receipt = source_snapshot.verify(snapshot)
    command = plan["phases"]["0"]
    if command.get("source_snapshot") != str(snapshot) or command.get("cwd") != str(snapshot):
        raise ValueError("frozen Phase 0 source selection changed")
    if command.get("phase0_environment_policy") != FROZEN_PHASE0_ENV_POLICY:
        raise ValueError("frozen Phase 0 launch predates selected-only environment; freeze a new run")
    if command["env"].get("MERLIN_REPO_ROOT") != str(snapshot):
        raise ValueError("frozen Phase 0 repository ownership changed")
    for key, value in source_snapshot.provider_environment(snapshot, receipt).items():
        if command["env"].get(key) != value:
            raise ValueError(f"frozen Phase 0 provider selection changed: {key}")
    from .evidence import load_exported_evidence

    evidence = load_exported_evidence(plan["phase0_evidence_bundle"])
    if evidence.target != plan["target"]:
        raise ValueError("frozen Phase 0 evidence identity changed")
    paths = plan["phase0_frozen_source_paths"]
    if command["env"].get("MERLIN_PHASE0_FROZEN_SOURCE_MAP") != json.dumps(paths, sort_keys=True):
        raise ValueError("frozen Phase 0 semantic source routing changed")
    basis_receipt = plan.get("phase0_semantic_basis")
    basis_sources = [*basis_receipt["selected_sources"], basis_receipt["recipe"]] if basis_receipt is not None else []
    if set(paths) != {str(source.path) for source in evidence.source_snapshots} | {
        row["path"] for row in basis_sources
    }:
        raise ValueError("frozen Phase 0 semantic source membership changed")
    for source in evidence.source_snapshots:
        path = Path(paths[str(source.path)])
        if not path.is_relative_to(snapshot) or path.read_bytes() != source.content:
            raise ValueError("frozen Phase 0 semantic source bytes changed")
    if basis_receipt is not None:
        from .component_semantic_basis import ComponentSemanticBasis

        for source in basis_sources:
            path = Path(paths[source["path"]])
            if not path.is_relative_to(snapshot) or fingerprint(path) != source["sha256"]:
                raise ValueError("frozen Phase 0 semantic basis source bytes changed")
        basis = ComponentSemanticBasis.from_recipe(command["inputs"]["recipe"], routing=paths)
        expected = {key: basis_receipt[key] for key in ("schema", "sha256", "semantics")}
        if basis is None or basis.reviewed_semantics() != expected:
            raise ValueError("frozen Phase 0 reviewed semantic basis changed")
    for key, value in plan["phase0_numerical_environment"].items():
        if command["env"].get(key) != value:
            raise ValueError("frozen Phase 0 numerical model routing changed")
    selected_m2m = command.get("phase0_m2m_selection")
    from .sealed_generation import CONFIG_ENV

    if selected_m2m is None:
        if any(
            key in command["env"] for key in ("MERLIN_M2M_DIR", "MERLIN_MODEL2MLIR", "MERLIN_M2M_PYTHON", CONFIG_ENV)
        ):
            raise ValueError("unselected Model2MLIR runtime entered frozen Phase 0")
    else:
        from . import m2m_runtime

        if any(command["env"].get(key) != value for key, value in m2m_runtime.environment(selected_m2m).items()):
            raise ValueError("frozen Phase 0 Model2MLIR runtime routing changed")
        if CONFIG_ENV in command["env"] or command["env"].get("MERLIN_PHASE0_EVIDENCE_MODE") == "verified":
            expected = json.dumps(
                m2m_runtime.sealed_capture_config(
                    selected_m2m,
                    Path(plan["phase0_evidence_bundle"]),
                    execution_timeout_seconds=command.get("phase0_capture_timeout_seconds"),
                    bwrap=command.get("phase0_bwrap"),
                ),
                sort_keys=True,
            )
            if command["env"].get(CONFIG_ENV) != expected:
                raise ValueError("frozen Phase 0 sealed Model2MLIR source routing changed; freeze a new run")
        receipt_path = Path(plan["phase0_m2m_runtime_receipt"])
        if receipt_path.read_bytes() != m2m_runtime.receipt(selected_m2m):
            raise ValueError("frozen Phase 0 Model2MLIR runtime receipt changed")
    return receipt


def verify_completed_artifact(plan: dict) -> None:
    """Verify a finished run's copied producer, never its historical host venv."""
    receipt = _verify_frozen_sources(plan)
    if receipt["external_links"]:
        raise ValueError("completed Phase 0 source snapshot has live external dependency links")
    selected_m2m = plan["phases"]["0"].get("phase0_m2m_selection")
    if selected_m2m is not None:
        from . import m2m_runtime

        copied = Path(selected_m2m["frozen_root"])
        run = Path(plan["run_dir"])
        if copied != copied.resolve(strict=True) or not copied.is_relative_to(run):
            raise ValueError("frozen Model2MLIR source is not an ordinary run-owned copy")
        m2m_runtime.verify_frozen_copy(selected_m2m)


def verify(plan: dict) -> None:
    """Verify private copies AND the selected live host runtime before execution."""
    _verify_frozen_sources(plan)
    selected_m2m = plan["phases"]["0"].get("phase0_m2m_selection")
    if selected_m2m is not None:
        from . import m2m_runtime

        m2m_runtime.verify(selected_m2m)
