"""Run-owned public grading and descriptor-policy views, sealed by the native snapshot.

The authored bundle and its candidate grants are unchanged. A private derived input is
declared in the run's effective bundle before freezing. Resume consumes that frozen input,
not the source tree or a shared publication link. Records belong in host-only environment
evidence; they are references into the existing native snapshot, not another seal.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from merlin.targetgen.capsule_common import discover_capsules
from merlin.targetgen.sandbox import bwrap

from ..corpus.admission import public_capsules_for
from ..corpus.preparation import copy_input, ordinary_tree, private_json
from .run_inputs import bundle_manifest_identity
from .source_inputs import fingerprint


@dataclass(frozen=True)
class CorpusView:
    public: Path
    policy: Path
    contract: Path
    workload_coverage: dict | None = None


@dataclass(frozen=True)
class PreparedBundle:
    bundle: dict
    corpus_record: dict
    authored_sha256: str
    effective_sha256: str
    private_full_model_record: dict | None = None


def require_reviewed_bundle(te, authored_path: Path, bundle: dict) -> None:
    """A reviewed run may grant only the bundle staged with its sealed descriptor.

    Otherwise a caller could pair a new corpus seal with an older bundle whose
    allowed paths still expose retired public claim capsules.  Seal verification
    later binds the entire release payload, including this exact manifest.
    """
    if not isinstance(bundle, dict):
        raise ValueError("reviewed Phase 1 bundle is not a manifest mapping")
    bundle_id = bundle.get("bundle_id")
    if not isinstance(bundle_id, str) or not bundle_id or Path(bundle_id).name != bundle_id:
        raise ValueError("reviewed Phase 1 bundle has an invalid identity")
    expected = te.path.parent / "input_bundles" / bundle_id / "input_bundle_manifest.yaml"
    if (
        authored_path.absolute() != expected.absolute()
        or expected.is_symlink()
        or expected.parent.is_symlink()
        or expected.parent.parent.is_symlink()
    ):
        raise ValueError("reviewed Phase 1 requires the selected release's generated bundle manifest")
    from ..phase0.coverage_commitment import read_inputs

    coverage_inputs = read_inputs(te.capsule_corpus.parent)
    selected = (coverage_inputs or {}).get("capability_contract")
    if isinstance(selected, dict) and selected:
        path = te.declared_contract_path()
        if path is None or not path.resolve().is_relative_to(te.path.parent.resolve()):
            raise ValueError(
                "reviewed Phase 1 lacks a release-local Phase 0 capability contract; "
                "freeze a new run and prepare a new release"
            )
        released = yaml.safe_load(path.read_bytes())
        operand_selection = released.get("rocc_operand_selection") if isinstance(released, dict) else None
        if operand_selection is not None:
            from merlin.common.digest import sha256_bytes
            from merlin.targetgen.rocc.selected_operand_protocol import _json, _read_pin, _sha, selected_inputs

            if (type(operand_selection) is not dict
                    or set(operand_selection) != {"original_facts", "sidecar"}
                    or type(operand_selection["original_facts"]) is not dict
                    or type(operand_selection["sidecar"]) is not dict
                    or operand_selection["original_facts"].get("path") != str(te.path.parent / "rtl_facts/facts.json")
                    or operand_selection["sidecar"].get("path") != str(
                        te.path.parent / "contracts/operand-protocol.json")):
                raise ValueError("reviewed operand selection must name its exact release-local facts and sidecar")
            raw_facts = (coverage_inputs or {}).get("raw_facts_utf8")
            raw_pin = ((coverage_inputs or {}).get("evidence") or {}).get("raw_facts_sha256")
            if type(raw_facts) is not str or sha256_bytes(raw_facts.encode("utf-8")) != _sha(raw_pin):
                raise ValueError("reviewed operand selection lacks its exact original raw Phase 0 facts pin")
            # The selected evidence exporter canonically serializes its loaded
            # facts view. It preserves the complete original JSON document,
            # though that export's byte pin differs from the intake byte pin.
            # Parse both without duplicates/nonfinite numbers and compare their
            # typed canonical forms, so true/1 or 1/1.0 cannot compare as equal.
            raw_document = _json(raw_facts.encode("utf-8"), 16 * 1024 * 1024)
            selected_document = _json(_read_pin(operand_selection["original_facts"], 16 * 1024 * 1024),
                                      16 * 1024 * 1024)
            canonical = {"sort_keys": True, "separators": (",", ":"), "allow_nan": False}
            if json.dumps(raw_document, **canonical) != json.dumps(selected_document, **canonical):
                raise ValueError("reviewed operand selection differs from unchanged original Phase 0 facts")
            original_only = {row.get("path") for row in bundle.get("host_inputs", []) if isinstance(row, dict)}
            if not set(selected_inputs(operand_selection)) <= original_only:
                raise ValueError("reviewed operand selection lacks its complete original-only frozen source roster")
        if released != selected:
            from merlin.runtime.backends import chipyard_rocc_bound as runtime
            from merlin.runtime.backends import rocc_selection

            try:
                block = released["runner"]["chipyard_rocc"]
                runtime._require(type(block) is dict, "released execution selection must be a mapping")
                expected_contract = selected
                if "execution_resources" in block:
                    timeout = runtime._execution_timeout_ceiling({"execution_resources": block["execution_resources"]})
                    expected_contract = rocc_selection.select_native_timeout(expected_contract, timeout)
                engines = block["engines"]
                runtime._require(type(engines) is dict, "released engines must be a mapping")
                gsim = engines.get("gsim", {})
                runtime._require(type(gsim) is dict, "released gSIM selection must be a mapping")
                if "readback_transport" in gsim:
                    expected_contract = rocc_selection.select_gsim_readback(
                        expected_contract, gsim["readback_transport"]
                    )
                if operand_selection is not None:
                    expected_contract = rocc_selection.select_operand_protocol(expected_contract, operand_selection)
            except (KeyError, TypeError, ValueError, runtime.RoCCExecutionError) as error:
                raise ValueError(
                    "released capability contract differs from selected Phase 0 coverage inputs"
                ) from error
            if released != expected_contract:
                raise ValueError("released capability contract differs from selected Phase 0 coverage inputs")
        grants = {
            row.get("path") for row in bundle.get("allowed", []) if isinstance(row, dict) and row.get("mode") == "ro"
        }
        if str(path) not in grants:
            raise ValueError("reviewed Phase 1 bundle does not grant the selected capability contract read-only")


def _reconcile_public_software_denial(te, bundle, paths, *, spec, source_freeze, source_root):
    """Keep a reviewed common-tool semantics declaration public, never model data."""
    from merlin.compile.model_execution_inputs import file_sha256
    from merlin.targetgen.software_spec import software_spec_path_for_recipe, validate_software_spec
    from merlin.targetgen.tool_profile import COMMON_SUBSTRATE_V2, COMMON_SUBSTRATE_V3, verify_tool_profile

    from ..corpus.numeric_policy import numeric_profile_path
    from .feedback import private_source_freeze

    profile = verify_tool_profile(bundle)
    if profile is None or profile.get("common_substrate") not in {COMMON_SUBSTRATE_V2, COMMON_SUBSTRATE_V3}:
        return paths
    root = te._source_root().absolute()
    recipe = numeric_profile_path(te.numeric_profile, repo=root)
    if recipe is None:
        return paths
    selected = software_spec_path_for_recipe(recipe)
    if selected is None:
        return paths
    grants = [row for row in bundle.get("allowed", [])
              if row.get("mode") == "ro" and row.get("note") == "admitted software semantics"
              and bwrap.resolve_grant(row["path"], root).absolute() == selected]
    if not grants:
        return paths
    if len(grants) != 1:
        raise ValueError("public software declaration has ambiguous grants")
    resolved = private_source_freeze.verify(spec, source_freeze, root=source_root, target=te.target)
    copies = [copy for (original, field), copy in resolved.items()
              if original == str(selected) and field == "software_spec"]
    if not copies:
        return paths
    if any(original == str(selected) and field != "software_spec" for original, field in resolved):
        raise ValueError("public software declaration has a private non-software role")
    if selected.is_symlink() or any(parent.is_symlink() for parent in selected.parents):
        raise ValueError("public software declaration is indirect")
    digest = file_sha256(selected)
    if any(file_sha256(copy) != digest for copy in copies):
        raise ValueError("public software declaration differs from authenticated private source")
    document = yaml.safe_load(selected.read_bytes())
    checked = validate_software_spec(document, target=te.target, source=selected)
    if checked.get("status") != "reviewed":
        raise ValueError("public software declaration is not reviewed")
    # The exact already-reviewed grant authorizes these semantics bytes. Only
    # its original software-role spelling is removed; frozen copies and all
    # capture/weight/golden/archive directories retain their private denials.
    return [row for row in paths if not (row.get("path") == str(selected) and row.get("kind") == "file")]


def prepare_bundle(
    run_dir: Path,
    te,
    authored_path: Path,
    bundle: dict,
    *,
    contract: Path,
    capsules_root: Path | None = None,
    environment: dict | None = None,
    private_full_model_spec: Path | None = None,
) -> PreparedBundle:
    """Archive fresh declarations or verify existing ones; resume never derives from live data."""
    authored = run_dir / "authored_input_bundle_manifest.yaml"
    effective = run_dir / "input_bundle_manifest.yaml"
    private_source = None
    private_paths = None
    frozen_authored = None
    frozen_authored_root = run_dir.absolute() / "private_full_model_input" / "sources"
    if environment is None and private_full_model_spec is not None:
        from merlin_experiments.phase1.feedback.private_full_models import (
            loader_env_requirements_for,
            private_input_paths,
            requirements_for,
        )

        private_source = Path(private_full_model_spec).absolute()
        if (
            private_source.is_symlink()
            or not private_source.is_file()
            or private_source.is_relative_to(run_dir.absolute())
        ):
            raise ValueError("operator-private full-model specification is absent, indirect or inside the run")
        from .feedback import private_source_freeze

        frozen_authored = private_source_freeze.stage(private_source, frozen_authored_root, target=te.target)
        private_paths = private_input_paths(
            private_source,
            target=te.target,
            required_models=requirements_for(te.path),
            scope_requirements=loader_env_requirements_for(te.path),
            source_freeze=frozen_authored,
            source_freeze_root=frozen_authored_root,
        )
        private_paths = _reconcile_public_software_denial(
            te, bundle, private_paths, spec=private_source,
            source_freeze=frozen_authored, source_root=frozen_authored_root,
        )
        # The host writes these only after authoring.  A resumed agent must not
        # see private model evidence or its paths through a broad run grant.
        private_paths.extend(
            {"path": str(run_dir.absolute() / name), "kind": kind}
            for name, kind in (
                ("grading_private_full_models", "dir"),
                ("run_manifest.yaml", "file"),
                ("environment.yaml", "file"),
            )
        )
    if environment is None:
        with os.fdopen(os.open(authored, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
            stream.write(authored_path.read_bytes())
        authored_sha = bundle_manifest_identity(authored, bundle)
        document, record = stage(run_dir, te, bundle, contract=contract, capsules_root=capsules_root)
        private_record = None
        if private_source is not None:
            from merlin.compile.model_execution_inputs import file_sha256

            staged = run_dir.absolute() / "private_full_model_input" / "spec.yaml"
            copy_input(private_source, staged, private=True)
            staged.chmod(0o400)
            staged.parent.chmod(0o500)
            document.setdefault("host_inputs", []).append(
                {"path": str(staged), "note": "operator-only frozen complete-network validation specification"}
            )
            document["host_inputs"].append(
                {"path": str(frozen_authored_root), "note": "operator-only frozen authored source ownership"}
            )
            document["private_validation_paths"] = private_paths
            private_record = {
                "source": str(private_source),
                "source_sha256": file_sha256(private_source),
                "path": str(staged),
                "sha256": file_sha256(staged),
                "masked_paths": private_paths,
                "source_freeze": frozen_authored,
            }
        with os.fdopen(os.open(effective, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
            yaml.safe_dump(document, stream, sort_keys=False)
    else:
        authored_sha = bundle_manifest_identity(authored, bundle)
        if environment.get("authored_bundle_manifest_sha256") != authored_sha:
            raise RuntimeError("resume refused: authored input bundle identity changed")
        record = environment.get("public_corpus_input")
        if not isinstance(record, dict) or record.get("descriptor_sha256") != te.descriptor_sha256:
            raise RuntimeError("resume refused: frozen public corpus descriptor identity is absent or changed")
        expected_source = str(run_dir.absolute() / "private_corpus_input")
        if record.get("staging_path") != expected_source:
            raise RuntimeError("resume refused: public corpus input belongs to a different run")
        if effective.is_symlink() or not effective.is_file():
            raise RuntimeError("resume refused: effective input bundle is absent or indirect")
        document = yaml.safe_load(effective.read_text())
        expected_bundle = copy.deepcopy(bundle)
        expected_bundle.setdefault("host_inputs", []).append(
            {"path": expected_source, "note": "host-only run corpus view"}
        )
        private_record = environment.get("private_full_model_spec")
        if private_record is not None:
            if not isinstance(private_record, dict):
                raise RuntimeError("resume refused: operator-private validation record is malformed")
            expected_bundle.setdefault("host_inputs", []).append(
                {
                    "path": private_record.get("path"),
                    "note": "operator-only frozen complete-network validation specification",
                }
            )
            frozen_authored = private_record.get("source_freeze")
            if frozen_authored is not None:
                from .feedback import private_source_freeze

                private_source_copy = Path(str(private_record.get("path") or ""))
                if private_source_copy != run_dir.absolute() / "private_full_model_input" / "spec.yaml":
                    raise RuntimeError("resume refused: private specification belongs to another run")
                private_source_freeze.verify(
                    private_source_copy, frozen_authored, root=frozen_authored_root, target=te.target
                )
                expected_bundle["host_inputs"].append(
                    {"path": str(frozen_authored_root), "note": "operator-only frozen authored source ownership"}
                )
            expected_bundle["private_validation_paths"] = private_record.get("masked_paths")
        if document != expected_bundle:
            raise RuntimeError("resume refused: effective bundle changed candidate grants or declared inputs")
    effective_sha = bundle_manifest_identity(effective, document)
    if environment is not None and environment.get("bundle_manifest_sha256") != effective_sha:
        raise RuntimeError("resume refused: effective input bundle bytes changed")
    return PreparedBundle(document, record, authored_sha, effective_sha, private_record)


def _resources(capsules: list[dict]) -> None:
    """Require declared file resources to travel with their capsule, without rewriting refs."""
    for capsule in capsules:
        directory = Path(capsule["__dir__"]).resolve()
        attributes = (capsule.get("operation") or {}).get("attributes") or {}
        references = {
            "interface_mlir": capsule.get("interface_mlir"),
            "linalg_mlir": capsule.get("linalg_mlir"),
            "pytorch_ref.loader": (capsule.get("pytorch_ref") or {}).get("loader"),
            **{key: attributes.get(key) for key in (
                "weights", "weights_manifest", "loader_dependencies", "loader_profile")},
        }
        for field, value in references.items():
            if value is None:
                continue
            relative = Path(str(value))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"run corpus cannot freeze non-capsule-local resource: {field}")
            member = directory / relative
            if not member.exists() or member.is_symlink() or not member.resolve().is_relative_to(directory):
                raise ValueError(f"run corpus has an absent or indirect resource: {field}")


def stage(run_dir: Path, te, bundle: dict, *, contract: Path, capsules_root: Path | None = None) -> tuple[dict, dict]:
    """Prepare a fresh private input from either the descriptor or one explicit corpus."""
    stage_root = Path(run_dir).absolute() / "private_corpus_input"
    stage_root.mkdir(mode=0o700, exist_ok=False)
    schema_source = Path(contract).absolute() / "schemas"
    copy_input(schema_source, stage_root / "contract/schemas", private=True)
    contract = stage_root / "contract"
    source_root = Path(capsules_root).absolute() if capsules_root is not None else None
    source_before = None
    if source_root is None:
        sources = list(te.graded_roots())
        coverage_root = te.capsule_corpus.parent
    else:
        ordinary_tree(source_root)
        source_before = fingerprint(source_root)
        categories = [
            child
            for child in sorted(source_root.iterdir())
            if child.is_dir()
            and not child.name.startswith(("_", "."))
            and child.name != "hidden"
            and next(child.glob("*/capsule.yaml"), None) is not None
        ]
        sources = categories or [source_root]
        coverage_root = source_root.parent if source_root == te.capsule_corpus else source_root
        if not discover_capsules(sources, labels={"public", "dev"}, contract=contract):
            raise ValueError("explicit public/dev corpus contains no capsules")
    staged_sources = []
    commitments = [{"original": str(schema_source), "staged": "contract/schemas", "role": "host_schema"}]
    for index, source in enumerate(sources):
        destination = stage_root / "policy" / str(index)
        copy_input(Path(source), destination, private=True)
        staged_sources.append(destination)
        commitments.append({"original": str(Path(source).absolute()), "staged": f"policy/{index}", "role": "corpus"})
    if source_root is None and list(te.graded_roots()) != sources:
        raise RuntimeError("descriptor corpus root membership changed during preparation")
    _resources(discover_capsules(staged_sources, labels={"public", "dev"}, contract=contract))
    public = stage_root / "public"
    if capsules_root is None:
        public_capsules_for(te, corpus_roots=staged_sources, destination=public)
        mode = "descriptor_cohort"
    else:
        selected = discover_capsules(staged_sources, labels={"public", "dev"}, contract=contract)
        _resources(selected)
        if not selected:
            raise ValueError("public/dev override contains no capsules")
        public.mkdir(mode=0o700)
        for capsule in selected:
            source = Path(capsule["__dir__"])
            # The copied policy view is the authority for both grading and the
            # raw override. Never re-read live source capsule bytes here.
            policy_relative = source.relative_to(stage_root / "policy")
            original = sources[int(policy_relative.parts[0])] / Path(*policy_relative.parts[1:])
            relative = original.relative_to(source_root)
            copy_input(source, public / relative, private=True)
            commitments.append(
                {
                    "original": str(original.absolute()),
                    "staged": (Path("public") / relative).as_posix(),
                    "role": "corpus",
                }
            )
        if fingerprint(source_root) != source_before:
            raise RuntimeError("public/dev override changed during preparation")
        mode = "public_dev_override"
    _resources(discover_capsules(public, labels={"public", "dev"}, contract=contract))
    from ..phase0.coverage_commitment import (
        INPUT_PATH,
        _digest,
        admitted_source_roots,
        observe_cohort,
        read_inputs,
        requires_workload_coverage,
    )

    selected_inputs = read_inputs(coverage_root)
    if selected_inputs is not None:
        if selected_inputs.get("target") != te.target:
            raise ValueError("selected coverage inputs belong to a different target")
        copy_input(coverage_root / INPUT_PATH.parent, stage_root / "coverage", private=True)
        commitments.append(
            {
                "original": str((coverage_root / "_phase0").absolute()),
                "staged": "coverage",
                "role": "coverage_inputs",
            }
        )
    completeness = observe_cohort(
        selected_inputs,
        admitted_source_roots(staged_sources, public, contract=contract),
        target=te.target,
        contract=contract,
    )
    private_json(stage_root / "coverage-report.json", completeness)
    private_json(stage_root / "source_commitments.json", {"version": 1, "mode": mode, "sources": commitments})
    record = {
        "version": 1,
        "staging_path": str(stage_root),
        "content_sha256": fingerprint(stage_root),
        "mode": mode,
        "authority": "private_preparation_bound_to_native_snapshot_not_operator_approval",
        "descriptor_sha256": te.descriptor_sha256,
        "workload_coverage": {
            "required": requires_workload_coverage(te, selected_inputs),
            "status": completeness["status"],
            "report_sha256": _digest(completeness),
            "inputs_sha256": completeness["inputs_sha256"],
            "cohort_sha256": completeness["cohort"]["sha256"],
        },
    }
    # The native snapshot copies private bytes with independent inodes, never public CAS.
    # Seal the staging copy too; setup never modifies it after recording its identity.
    for member in [*stage_root.rglob("*"), stage_root]:
        member.chmod(member.stat().st_mode & ~0o222)
    effective = copy.deepcopy(bundle)
    effective.setdefault("host_inputs", []).append({"path": str(stage_root), "note": "host-only run corpus view"})
    return effective, record


def resolve(
    ws: Path,
    bundle: dict,
    record: dict,
    *,
    repo: Path,
    reviewed_roots: tuple[Path, ...] | None = None,
) -> CorpusView:
    """Return only the already-verified native snapshot view; never read live staging."""
    if not isinstance(record, dict) or record.get("version") != 1:
        raise RuntimeError("run corpus input record is missing or unsupported")
    source = record.get("staging_path")
    if not isinstance(source, str) or not Path(source).is_absolute():
        raise RuntimeError("run corpus input record has no absolute staging identity")
    [frozen] = bwrap.snapshot_input_paths(ws, bundle, [Path(source)], repo=repo)
    if fingerprint(frozen) != record.get("content_sha256"):
        raise RuntimeError("run corpus input differs from its prepared native snapshot identity")
    commitment = json.loads((frozen / "source_commitments.json").read_text())
    if reviewed_roots is not None and (
        record.get("mode") != "descriptor_cohort" or commitment.get("mode") != "descriptor_cohort"
    ):
        raise RuntimeError("reviewed run corpus requires the complete descriptor cohort; raw overrides are unverified")
    manifest = bwrap.verify_bundle_snapshot(ws, bundle, repo=repo)
    declared = [Path(row["destination"]) for row in [*manifest["grants"], *manifest.get("host_records", [])]]
    covered = []
    for row in commitment["sources"]:
        original = Path(row["original"])
        relative = Path(row["staged"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("run corpus source commitment escapes its frozen input")
        reviewed_corpus = reviewed_roots is not None and row["role"] == "corpus"
        if reviewed_corpus and not any(original == root or original.is_relative_to(root) for root in reviewed_roots):
            raise RuntimeError("reviewed run corpus has a source outside its reviewed original roots")
        if any(original == path or original.is_relative_to(path) for path in declared):
            covered.append((original, frozen / relative))
        elif reviewed_corpus:
            raise RuntimeError("reviewed run corpus has a source outside its original declared snapshot")
    if covered:
        originals = bwrap.snapshot_input_paths(ws, bundle, [source for source, _ in covered], repo=repo)
        for original, (_, staged) in zip(originals, covered, strict=True):
            if fingerprint(original) != fingerprint(staged):
                raise RuntimeError("prepared corpus differs from original frozen source; cannot borrow its review")
    public, policy, contract = frozen / "public", frozen / "policy", frozen / "contract"
    if not public.is_dir() or not policy.is_dir() or not (contract / "schemas").is_dir():
        raise RuntimeError("run corpus snapshot is missing a declared view")
    from ..phase0.coverage_commitment import _digest, admitted_source_roots, verify_cohort_binding

    completeness = None
    commitment_record = record.get("workload_coverage")
    if commitment_record is not None:
        selected_path = frozen / "coverage/coverage-inputs.json"
        selected_inputs = json.loads(selected_path.read_bytes()) if selected_path.is_file() else None
        saved = json.loads((frozen / "coverage-report.json").read_bytes())
        verify_cohort_binding(
            saved,
            selected_inputs,
            admitted_source_roots(policy, public, contract=contract),
            contract=contract,
        )
        completeness = saved
        if (
            _digest(saved) != commitment_record.get("report_sha256")
            or saved.get("inputs_sha256") != commitment_record.get("inputs_sha256")
            or saved["cohort"]["sha256"] != commitment_record.get("cohort_sha256")
        ):
            raise RuntimeError("frozen whole-workload coverage differs from its actual admitted-cohort commitment")
    return CorpusView(public=public, policy=policy, contract=contract, workload_coverage=completeness)
