"""Host-owned Phase 0 generation implementation."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import yaml

from merlin.targetgen import corpus_spec as CS  # noqa: E402
from merlin.targetgen.target_experiment import load_target_experiment  # noqa: E402

from .claim_boundary import assert_no_claim_capsules, held_out_models
from .instruction_roles import enforcement_problems, validate_roles
from .profiles import load_profile, validate_profile_inputs
from .program_admission import entry_refusal_is_final, screen_written
from .provenance import _capture_failure_reason, _scrub_capsule_dir, update_provenance_manifest
from .sealed_generation import verified_capture_failure
from .software_screen import diagnostic_entry, screen_entry
from .staging import write_staged
from .sweeps import _performance_facts, _resolve_flat_extents, expand_sweeps
from .writer import SYNTH_ROLE, UnprovableForbid, _write_capsule


def _selected_capture_recipe(evidence_root: Path, *, target: str, operand_dtype: str, accumulator_dtype: str) -> dict:
    """Read exactly one frozen SW-scoped recipe for a live integer model capture.

    Hardware-only recipe derivation is intentionally not a fallback: it can quantize families
    the selected software spec explicitly left on the host.
    """
    from merlin.common import quant_formats
    from merlin.targetgen import quant_recipe

    index_path = evidence_root / "software" / "quantization-recipes.json"
    index = json.loads(index_path.read_bytes())
    if index.get("schema") != "merlin.phase0.capture_recipes.v1" or index.get("target") != target:
        raise ValueError("frozen capture-recipe index has wrong schema or target")
    wanted_operand = quant_formats.get(operand_dtype).name
    wanted_accumulator = quant_formats.get(accumulator_dtype).name
    matches = []
    for row in index.get("recipes") or []:
        rel = Path(str(row.get("path") or ""))
        if rel.is_absolute() or not rel.parts or ".." in rel.parts or rel.parts[0] != "software":
            raise ValueError("frozen capture recipe has an unsafe relative path")
        path = evidence_root / rel
        if path.is_symlink() or not path.is_file():
            raise ValueError("frozen capture recipe is absent or symlinked")
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != row.get("sha256") or path.stem != actual:
            raise ValueError("frozen capture recipe bytes differ from their index")
        recipe = json.loads(raw)
        if recipe.get("target") != target or recipe.get("status") != quant_recipe.DERIVED:
            raise ValueError("frozen capture recipe has wrong target or unresolved status")
        digest = quant_recipe.digest(recipe)
        if recipe.get("recipe_sha256") != digest or row.get("recipe_sha256") != digest:
            raise ValueError("frozen capture recipe semantic digest differs from its index")
        if (
            quant_formats.get(recipe["weight"]["dtype"]).name == wanted_operand
            and quant_formats.get(recipe["activation"]["dtype"]).name == wanted_operand
            and quant_formats.get(recipe["accumulator_dtype"]).name == wanted_accumulator
        ):
            matches.append(recipe)
    if len(matches) != 1:
        raise ValueError(
            f"frozen evidence has {len(matches)} unambiguous SW-scoped recipes for "
            f"{wanted_operand}/{wanted_accumulator}; refusing a hardware-only default"
        )
    return matches[0]


def _descriptor_for(target: str) -> Path:
    from merlin.common.paths import repo_root

    return repo_root() / "merlin" / "experiments" / "capsule_bench" / "targets" / target / "target_experiment.yaml"


def _require_distinct_corpus_destinations(te, *, output_root: str | Path, evidence_root: str | Path | None) -> None:
    """Keep generated Phase 0 bytes out of retained and descriptor-selected input corpora.

    The legacy corpus is still addressable by frozen benchmark grants. Rewriting it
    under a new derivation would silently change those experiments, even when the
    caller explicitly supplied ``--output-root``. Resolve paths before comparing
    them so an alias cannot bypass this source-ownership check.
    """
    from merlin.common.paths import checkout_root
    from merlin.targetgen.corpora import capsule_corpus_roots

    # In source mode MERLIN_REPO_ROOT may select an external experiment
    # workspace that has no copy of this checkout's historical registry. The
    # implementation checkout still owns those retained paths; an installed
    # wheel instead reads its bundled registry. Both modes keep malformed or
    # missing selected metadata fail-closed.
    sources = [*capsule_corpus_roots(owner_root=checkout_root())]
    selected = getattr(te, "capsule_corpus", None)
    if selected:
        sources.append(Path(selected))
    for method_name in ("graded_roots", "perf_roots", "hidden_roots"):
        method = getattr(te, method_name, None)
        if callable(method):
            sources.extend(method())
    sources = sorted({Path(source).expanduser().resolve() for source in sources})
    # A frozen run executes from its own sealed source snapshot below the evidence root, and that
    # snapshot deliberately EXCLUDES the capsule corpus. A corpus path resolved inside it names
    # nothing: it is the run's own read-only copy of the checkout, not a retained input to protect.
    frozen = Path(evidence_root).expanduser().resolve() / "private" / "source" if evidence_root is not None else None
    if frozen is not None:
        sources = [
            source for source in sources if not ((source == frozen or frozen in source.parents) and not source.exists())
        ]
    for field, raw in (("output_root", output_root), ("evidence_root", evidence_root)):
        if raw is None:
            continue
        destination = Path(raw).expanduser().resolve()
        for source in sources:
            if destination == source or destination in source.parents or source in destination.parents:
                raise ValueError(
                    f"Phase 0 {field} {destination} overlaps source capsule corpus {source}; "
                    "select a separate run artifact destination"
                )


def _ensure_contract_on_path(descriptor: Path) -> None:
    """If the descriptor names an out-of-tree ``target_contract`` (e.g. radiance's contract lives under
    the ``radiance`` target package), prepend its package root to ``MERLIN_TARGET_PATH`` so the registry
    resolves the manifest. Read from the descriptor, so it stays target-agnostic."""
    from merlin.common.paths import repo_root

    raw = yaml.safe_load(descriptor.read_text())
    tc = (raw.get("hardware_spec") or {}).get("target_contract")
    if not tc:
        return
    pkg = (repo_root() / tc).resolve().parent.parent  # .../contracts/target_contract.yaml -> package root
    cur = os.environ.get("MERLIN_TARGET_PATH", "")
    if str(pkg) not in cur.split(os.pathsep):
        os.environ["MERLIN_TARGET_PATH"] = os.pathsep.join([str(pkg), cur]) if cur else str(pkg)


def _verify_derivation_evidence(conformance_spec: str | Path, evidence) -> None:
    """Refuse a derived corpus when its selected provider/facts changed."""
    requirement = yaml.safe_load(Path(conformance_spec).read_text(encoding="utf-8")) or {}
    if not isinstance(requirement, dict):
        raise ValueError("derived conformance requirement must be a mapping")
    execution = (requirement.get("derivation") or {}).get("phase0_execution") or {}
    expected = evidence.derivation_identity
    missing = sorted(key for key in expected if key not in execution)
    if missing:
        raise ValueError(
            "derived requirement lacks selected capability identity "
            f"({', '.join(missing)}); rerun corpus derive with the selected provider"
        )
    changed = sorted(key for key, value in expected.items() if execution[key] != value)
    if changed:
        raise ValueError(
            "stale capability evidence: "
            f"{', '.join(changed)} changed since corpus derivation; "
            "rerun corpus derive with the selected provider and facts"
        )


def _frozen_application_captures(root: Path, manifest: dict) -> dict[str, Path]:
    """Resolve saved capture bytes from this run's exported evidence only."""
    captures = {}
    for source in manifest.get("sources") or []:
        role = source.get("role")
        if not isinstance(role, str) or not role.startswith("application-capture:"):
            continue
        label = role.partition(":")[2]
        member = Path(source["path"])
        if not label or label in captures or member.is_absolute() or ".." in member.parts:
            raise ValueError("invalid or duplicate frozen application capture")
        path = root / member
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError(f"frozen application capture differs from exported evidence: {label}")
        captures[label] = path
    return captures


def _with_selected_model_recipe(
    entry: dict,
    *,
    evidence_root: Path | None,
    target: str,
    binding,
) -> dict:
    """Use the frozen SW-scoped recipe when an integer model has no explicit choice."""
    if (
        evidence_root is None
        or (entry.get("kind") != "model" and entry.get("op") != "model")
        or entry.get("materialized_capture")
        or entry.get("quant_recipe")
        or entry.get("quant_scheme")
        or not binding.integer
    ):
        return entry
    return {
        **entry,
        "quant_recipe": _selected_capture_recipe(
            evidence_root,
            target=target,
            operand_dtype=str(entry.get("operand_dtype") or binding.operand_dtype),
            accumulator_dtype=str(binding.accum_dtype),
        ),
    }


def _prepare_model_capture_entry(
    entry: dict,
    *,
    evidence_root: Path | None,
    target: str,
    binding,
    capture=None,
) -> dict:
    """Check a selected static model capture tool before any capsule writer starts."""
    if entry.get("kind") != "model" and entry.get("op") != "model":
        return entry
    if entry.get("materialized_capture"):
        return entry

    from merlin.targetgen import capsule_source as source

    try:
        selected = _with_selected_model_recipe(entry, evidence_root=evidence_root, target=target, binding=binding)
    except Exception:  # noqa: BLE001 -- the writer records this entry's recipe failure as before
        return entry
    recipe = selected.get("quant_recipe") or source.derived_recipe(
        getattr(binding, "target", None), str(selected.get("operand_dtype") or binding.operand_dtype)
    )
    if source._static_pt2e_model(
        "model",
        scheme=None if recipe is not None else selected.get("quant_scheme"),
        recipe=recipe,
        already_quantized=selected.get("capture_quantization") == "already_materialized",
    ):
        capture = capture if capture is not None else source.PytorchRefSource()
        integerizer = capture.m2m_dir / "m2m/capture/pt2e_integerize.py"
        if capture.available() and not integerizer.is_file():
            raise ValueError(
                f"selected model2MLIR checkout {capture.m2m_dir} lacks {integerizer}; "
                "static int8 model capture requires m2m.capture.pt2e_integerize"
            )
        if capture.available() and not entry.get("micro_model"):
            # The PT2E quantizer imports these from the selected checkout only
            # after source capsules start writing. The derived micro-model is
            # the sole exception: its graph is checked at capture time for
            # complete ATen provenance and absence of BatchNorm before Merlin
            # permits a missing fold API. Every other static model must have
            # the API before any writer runs.
            probe = subprocess.run(
                [
                    str(capture.python),
                    "-B",
                    "-c",
                    "import sys; sys.path.insert(0, sys.argv[1]); "
                    "from m2m.capture.trace import "
                    "pt2e_conv_bn_fold_candidates, attach_pt2e_conv_bn_folds",
                    str(capture.m2m_dir),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if probe.returncode:
                raise ValueError(
                    f"selected model2MLIR checkout {capture.m2m_dir} lacks the required "
                    "m2m.capture.trace PT2E fold-provenance API "
                    "(pt2e_conv_bn_fold_candidates, attach_pt2e_conv_bn_folds); "
                    "select a compatible checkout and interpreter before generating capsules"
                )
    return selected


def generate_target(
    target: str,
    *,
    descriptor: str | Path | None = None,
    output_root: str | Path | None = None,
    profiles_root: str | Path | None = None,
    recipe: str | Path | None = None,
    software_spec: str | Path | None = None,
    capability_contract: str | Path | None = None,
    hardware_spec: str | Path | None = None,
    rtl_facts: str | Path | None = None,
    evidence_root: str | Path | None = None,
    evidence_input: str | Path | None = None,
    evidence_mode: str | None = None,
    performance_template: str | Path | None = None,
    conformance_spec: str | Path | None = None,
    synth_profile: str | Path | None = None,
    smt_profile: str | Path | None = None,
    hidden_profile: str | Path | None = None,
    prohibited_instruction_roles: list[str] | None = None,
) -> list[Path]:
    from merlin.common.paths import checkout_root

    if output_root is None:
        raise ValueError("Phase 0 requires an explicit output_root; generated capsules must not default to source data")
    profile_inputs = dict(
        profiles_root=profiles_root,
        recipe=recipe,
        performance_template=performance_template,
        conformance_spec=conformance_spec,
        synth_profile=synth_profile,
        smt_profile=smt_profile,
        hidden_profile=hidden_profile,
    )
    if software_spec is not None:
        profile_inputs["software_spec"] = software_spec
    validate_profile_inputs(**profile_inputs)
    if profiles_root is None and recipe is None:
        raise ValueError("Phase 0 requires explicit recipe inputs or profiles_root before descriptor setup")
    if checkout_root() is None and (descriptor is None or (profiles_root is None and recipe is None)):
        raise ValueError("installed Phase 0 requires explicit profiles_root or recipe, descriptor and output_root")
    # The profile selector and the hardware descriptor are separate identities:
    # a profile can be reused with an explicitly supplied out-of-tree target.
    explicit_descriptor = descriptor is not None
    descriptor = Path(descriptor).expanduser().resolve() if descriptor is not None else _descriptor_for(target)
    if evidence_input is None:
        _ensure_contract_on_path(descriptor)
    te = load_target_experiment(descriptor)
    _require_distinct_corpus_destinations(te, output_root=output_root, evidence_root=evidence_root)
    hardware_target = te.target if explicit_descriptor else target
    profile = load_profile(
        target,
        descriptor=descriptor,
        diagnostic=evidence_mode == "diagnostic",
        **{key: value for key, value in profile_inputs.items() if value is not None},
    )
    if profile.get("capsule_policy") == "derived_only":
        # Never substitute the old authored/reference corpus for a missing
        # derivation. Functional membership must come from new byte-bound inputs.
        accepted_synthesis = {"verified"}
        if evidence_mode == "diagnostic":
            accepted_synthesis.add("incomplete_diagnostic")
        if conformance_spec is None or profile.get("_synth_verification", {}).get("status") not in accepted_synthesis:
            raise ValueError(
                "derived-only Phase 0 requires fresh digest-bound conformance and synthesis inputs; "
                "run merlin experiment corpus derive first"
            )
        authored = [
            e.get("name")
            for e in profile.get("capsules", [])
            if e.get("source_role")
            not in {"derived_sweep", "model_derived", "solver_derived", "materialized_iteration_capture"}
        ]
        if authored:
            raise ValueError(f"derived-only Phase 0 refuses authored capsule membership: {authored}")
    if evidence_mode not in (None, "diagnostic", "verified"):
        raise ValueError("evidence_mode must be diagnostic or verified")
    if (
        evidence_mode == "verified"
        and evidence_input is None
        and software_spec is None
        and not profile.get("_software_spec_path")
    ):
        raise ValueError("verified Phase 0 requires selected software and hardware evidence")
    evidence = None
    evidence_manifest = None
    if evidence_input is not None or software_spec is not None or profile.get("_software_spec_path"):
        from .evidence import export_evidence, load_exported_evidence, select_evidence

        if evidence_input is not None:
            evidence = load_exported_evidence(Path(evidence_input))
            if evidence.target != hardware_target:
                raise ValueError("frozen hardware evidence target differs from the descriptor")
            captured_specs = {source.sha256 for source in evidence.source_snapshots if source.role == "software-spec"}
            if profile.get("_software_spec_identity", {}).get("sha256") not in captured_specs:
                raise ValueError("selected software spec differs from the exact captured evidence")
        else:
            evidence = select_evidence(
                hardware_target,
                descriptor=descriptor,
                capability_contract_path=capability_contract,
                facts_path=rtl_facts,
                software_spec=software_spec or profile.get("_software_spec_path"),
                hardware_spec=hardware_spec,
                conformance_spec=conformance_spec,
                prohibited_roles=tuple(prohibited_instruction_roles or ()),
            )
        if evidence_mode != "diagnostic" and evidence.status != "verified":
            raise ValueError(
                "verified Phase 0 requires reviewed, coherent evidence: "
                + "; ".join(
                    str(row.get("reason", row)) if isinstance(row, dict) else str(row) for row in evidence.diagnostics
                )
            )
        if profile.get("capsule_policy") == "derived_only":
            _verify_derivation_evidence(conformance_spec, evidence)
        artifact_root = Path(evidence_root) if evidence_root is not None else Path(output_root) / "_evidence"
        evidence_manifest = export_evidence(evidence, artifact_root)
    declared_claims = [str(model) for model in (getattr(te, "workload_spec", None) or {}).get("models") or ()]
    claim_plan = profile.get("_claim_model_evaluation")
    if profile.get("_synth_verification", {}).get("status") == "verified":
        if (
            not isinstance(claim_plan, dict)
            or claim_plan.get("schema") != "claim_model_evaluation_v1"
            or claim_plan.get("model_count") != len(declared_claims)
            or claim_plan.get("visibility") != "owner_only_after_phase1_freeze"
            or claim_plan.get("public_capsules_emitted") != 0
        ):
            raise ValueError("verified synthesis lacks an owner-only claim-model evaluation obligation")
    if (
        profile.get("_synth_verification", {"status": "absent"})["status"] == "unverified_legacy"
        and evidence_mode != "diagnostic"
    ):
        raise ValueError(
            "selected synthesized profile has no digest-bound conformance/recipe/workload inputs; "
            "regenerate, review, and select a new sidecar before verified Phase 0 execution"
        )
    selected = (
        {"contract": evidence.contract, "facts": evidence.loaded_facts, "taxonomy": evidence.isa_taxonomy}
        if evidence is not None
        else {}
    )
    binding = CS.derive_binding(te, profile.get("datapath", {}), **selected)
    declared_roles = validate_roles(prohibited_instruction_roles)
    instruction_policy = _instruction_policy(hardware_target, declared_roles, evidence, rtl_facts)
    if declared_roles:
        require_enforceable_policy(evidence_mode, declared_roles, instruction_policy)
    out_root = Path(output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    # `sweeps:` (if any) expand into the same flat entries `capsules:` holds, so
    # everything downstream — builders, goldens, coverage — is unchanged.
    facts = (
        _performance_facts(hardware_target, evidence=evidence)
        if evidence is not None
        else _performance_facts(hardware_target)
    )
    _sweep_skips: list = []
    _runtime_blocked: list = []
    _performance_errors: list = []
    requirement_bytes = Path(conformance_spec).read_bytes() if conformance_spec is not None else None
    selected_requirement = yaml.safe_load(requirement_bytes) if requirement_bytes is not None else None
    if selected_requirement is not None and not isinstance(selected_requirement, dict):
        raise ValueError("selected conformance requirement must be a mapping")
    entries = expand_sweeps(
        profile,
        binding,
        trait_facts=facts,
        skipped=_sweep_skips,
        blocked_unimplemented=_runtime_blocked,
        errors=_performance_errors,
        selected_requirement=selected_requirement,
        requirement_sha256=hashlib.sha256(requirement_bytes).hexdigest() if requirement_bytes is not None else None,
        **({"evidence": evidence} if evidence is not None else {}),
    )
    assert_no_claim_capsules(entries, held_out_models(te))
    entries = [_resolve_flat_extents(e, binding) for e in entries]
    entries = [_with_candidate_policy(e, instruction_policy) for e in entries]
    entries = [_with_reference_gate(e, te, descriptor) for e in entries]
    semantics = (profile.get("datapath") or {}).get("numerical_semantics")
    if semantics is not None:
        semantics = copy.deepcopy(semantics)
        if evidence is not None:
            model_sources = [
                (str(source.path), source.sha256)
                for source in evidence.source_snapshots
                if source.role == "software-reference:numerical_model"
            ]
            semantics["model"]["source_bundle_sha256"] = hashlib.sha256(
                json.dumps(sorted(model_sources), separators=(",", ":")).encode()
            ).hexdigest()
        entries = [{**entry, "numerical_semantics": copy.deepcopy(semantics)} for entry in entries]
    if evidence is not None and evidence.software_spec:
        screened = []
        for entry in entries:
            decision = screen_entry(
                evidence.software_spec,
                entry,
                defaults={
                    "operand_dtype": binding.operand_dtype,
                    "accumulator_dtype": binding.accum_dtype,
                },
                host_capabilities=evidence.host_capabilities,
            )
            screened.append(diagnostic_entry(entry, decision) if entry_refusal_is_final(entry, decision) else entry)
        entries = screened
    from .sealed_generation import bind_source

    selected_capture = bind_source(entries, verified=evidence is not None and evidence_mode != "diagnostic")
    capture_option = {"capture": selected_capture} if selected_capture is not None else {}
    entries = [
        _prepare_model_capture_entry(
            entry,
            evidence_root=artifact_root if evidence is not None else None,
            target=hardware_target,
            binding=binding,
            capture=selected_capture,
        )
        for entry in entries
    ]
    for _s in _sweep_skips:
        _why = _s.get("reason") or f"gate {(_s.get('gate') or {}).get('outcome')}"
        _what = f"class {_s['label']} of family {_s['family']}" if _s.get("label") else f"family {_s['family']}"
        print(f"  [skip] performance {_what}: {_why}")
    template = copy.deepcopy(profile.get("_performance_template") or {})
    declared_families = [dict(row) for row in (template.get("families") or [])]
    # A requirement-selected pattern may derive several exact claim cohorts.
    # Keep both identities: the shared template's digest-bound pattern and each
    # concrete family with the frozen requirement row that produced it.
    derived_families: dict[str, dict] = {}
    for entry in entries:
        performance = entry.get("performance") or {}
        basis = performance.get("requirement_basis") or {}
        family = performance.get("family")
        if basis.get("axis") == "scope.performance.required" and family:
            record = {
                "family": family,
                "claim": performance.get("claim"),
                "derived_from_pattern": basis.get("pattern_family"),
                "requirement_basis": copy.deepcopy(basis),
                "fit_axes": ["K"],
                "comparison_roles": ["prediction", "measurement"],
            }
            if family in derived_families and derived_families[family] != record:
                raise ValueError(f"derived performance family {family!r} has divergent requirement provenance")
            derived_families[family] = record
    template["derived_families"] = [derived_families[name] for name in sorted(derived_families)]
    declared_families.extend(template["derived_families"])
    family_counts = {row["family"]: {"admitted_members": 0, "written_members": 0} for row in declared_families}
    for entry in entries:
        family = (entry.get("performance") or {}).get("family")
        if family and entry.get("cat") == "_perf":
            family_counts.setdefault(family, {"admitted_members": 0, "written_members": 0})
            family_counts[family]["admitted_members"] += 1
    # SCRUB EACH CAPSULE AS IT IS WRITTEN, not after the whole corpus succeeds. Scrubbing at the end
    # means one unrelated failure -- a capture that needs an external exporter, say -- aborts the run
    # with every capsule written so far still carrying its absolute `prov.weights_file` path. Measured:
    # a run that died on the last entry left `/scratch/.../capsule_m2m_<rand>/weights.safetensors` in
    # tracked MLIR across six capsules, in a repo that is published. Hygiene that only holds on the
    # happy path is not hygiene.
    #
    # ONE FAILING CAPSULE MUST NOT DESTROY THE WHOLE CORPUS. Letting the exception propagate meant a
    # single entry that needs an external exporter took every entry after it down with it: measured, a
    # capture that torch.export refuses (an LSTM whose `_flat_weights` are assigned rather than
    # registered) aborted the run before any of the tail-path sweep capsules were written, so a coverage
    # gap stayed open for a reason that had nothing to do with it. Failures are COLLECTED, reported by
    # name, and re-raised at the end -- the run still fails, it just fails after doing the work it could.
    written, failures, unbuilt_roster, unprovable_forbids, omitted, admission = [], [], [], [], [], []
    refused_generated: list[Path] = []
    for e in entries:
        family = (e.get("performance") or {}).get("family")
        try:
            if evidence is not None and e.get("micro_model"):
                captures = _frozen_application_captures(artifact_root, evidence_manifest)
                declared = set((evidence.application_inventory or {}).get("applications") or {})
                if set(captures) != declared:
                    raise ValueError("micro-model capture roster differs from frozen application inventory")
                e = {
                    **e,
                    "_frozen_application_captures": captures,
                    "_frozen_software_spec": copy.deepcopy(evidence.software_spec),
                }
            e = _with_selected_model_recipe(
                e,
                evidence_root=artifact_root if evidence is not None else None,
                target=hardware_target,
                binding=binding,
            )
            if evidence is not None and evidence.software_spec:
                decision = screen_entry(
                    evidence.software_spec,
                    e,
                    defaults={
                        "operand_dtype": binding.operand_dtype,
                        "accumulator_dtype": binding.accum_dtype,
                    },
                    host_capabilities=evidence.host_capabilities,
                )
                admission.append({"capsule": e.get("name"), **decision})
                # An entry carries no rank/layout/tail/alias/composition observation, so before it is
                # written only a refusal is final; the written program's own screen below decides the
                # rest, and verified mode requires that screen to admit.
                final = entry_refusal_is_final(e, decision)
                if evidence_mode != "diagnostic" and final:
                    raise ValueError("SW operation admission: " + decision["reason"])
                if final:
                    e = diagnostic_entry(e, decision)
            if evidence is None:
                w = write_staged(
                    lambda root, e=e: _write_capsule(e, binding, root, facts.get("sha256", ""), **capture_option),
                    out_root,
                    member=_member_path(e),
                )
            else:
                from merlin.targetgen.rtl.facts import observed_facts
                from merlin.targetgen.target_registry import observed_contract

                captured_path = artifact_root / "hardware" / "circt" / "facts.json"
                with (
                    observed_contract(hardware_target, evidence.contract),
                    observed_facts(
                        hardware_target, evidence.refreshed_facts, captured_path if captured_path.is_file() else None
                    ),
                ):
                    w = write_staged(
                        lambda root, e=e: _write_capsule(e, binding, root, facts.get("sha256", ""), **capture_option),
                        out_root,
                        member=_member_path(e),
                    )
        except Exception as exc:  # noqa: BLE001 — reported, never swallowed
            detail = _capture_failure_reason(exc)
            if isinstance(exc, UnprovableForbid) and str(e.get("source_role") or "") == SYNTH_ROLE:
                # See `UnprovableForbid`. The capsule is removed rather than left on disk: a directory
                # the corpus does not list is exactly the kind of half-written state the seal cannot
                # see, and a stale one would be picked up by the next glob as though it had been built.
                shutil.rmtree(out_root / str(e.get("cat") or "") / str(e.get("name") or ""), ignore_errors=True)
                unprovable_forbids.append(
                    {"capsule": e.get("name", "?"), "family": e.get("op"), "reason": str(exc)[:400]}
                )
                print(f"  [lane] {e.get('name')}: NOT BUILT — its forbid is not provable on this target")
                continue
            if _is_roster_capsule(e):
                # A ROSTER MODEL IS A DECLARED INPUT, NOT A COMPILER RESULT. The roster axis emits one
                # whole-model capsule per model the target's `workload_spec` names, at the format the
                # target admits -- and whether a given network can be CAPTURED at that format depends on
                # things outside this repo: whether the loader's dataset is present, whether the m2m venv
                # has the model's package, whether torchAO's scheme runs on this host, whether
                # torch.export accepts the module under quantization. Measured on the four declared
                # models: one captures, one needs an ImageNet npz, one needs a package the venv lacks,
                # one is refused by torch.export under W8A8 while its weight-only capture succeeds.
                #
                # Aborting the corpus for those would make a fact about the ROSTER read as a broken
                # generator, and would take every other capsule down with it. Recording it keeps the rule
                # that matters -- a requirement that produced no capsule is never indistinguishable from
                # one that is met -- because the manifest names the model and the reason, and no capsule
                # exists to be graded, so nothing can pass in its place.
                why = _capture_failure_reason(exc)
                unbuilt_roster.append(
                    {
                        "capsule": e.get("name", "?"),
                        "model": e.get("model"),
                        "operand_dtype": e.get("operand_dtype"),
                        "quant_scheme": e.get("quant_scheme"),
                        "status": "not_built",
                        "reason": why,
                    }
                )
                print(f"  [roster] {e.get('name')}: NOT BUILT — {why}")
                continue
            failures.append((e.get("name", "?"), detail))
            if family:
                _performance_errors.append(
                    {
                        "family": family,
                        "member": e.get("name", "?"),
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "detail": str(exc)[:500],
                    }
                )
            continue
        if w:
            if evidence is not None and evidence.software_spec:
                actual = yaml.safe_load((Path(w) / "capsule.yaml").read_bytes())
                observed = screen_written(actual, Path(w), target=hardware_target, evidence=evidence)
                if observed is None:
                    observed = screen_entry(
                        evidence.software_spec,
                        e,
                        defaults={
                            "operand_dtype": binding.operand_dtype,
                            "accumulator_dtype": binding.accum_dtype,
                        },
                        capsule=actual,
                        host_capabilities=evidence.host_capabilities,
                    )
                admission.append({"capsule": e.get("name"), "observation": "emitted_capsule", **observed})
                if observed["status"] == "unsupported" and Path(w).parent.name != "_diagnostic":
                    diagnostic = out_root / "_diagnostic" / Path(w).name
                    diagnostic.parent.mkdir(parents=True, exist_ok=True)
                    if diagnostic.exists():
                        raise ValueError("diagnostic capsule destination already exists")
                    Path(w).rename(diagnostic)
                    w = diagnostic
                actual["software_screen"] = observed
                if observed["status"] == "unsupported":
                    actual["source_reference"] = diagnostic_entry(actual, observed)["source_reference"]
                (Path(w) / "capsule.yaml").write_text(yaml.safe_dump(actual, sort_keys=False))
                if evidence_mode != "diagnostic" and observed["status"] != "admitted":
                    failures.append((e.get("name", "?"), "emitted SW operation admission: " + observed["reason"]))
                # Verified evidence admits a generation-time capture only from the sealed runner.
                capture_refusal = verified_capture_failure(actual)
                if evidence_mode != "diagnostic" and capture_refusal is not None:
                    failures.append((e.get("name", "?"), "generation-time capture: " + capture_refusal))
            _scrub_capsule_dir(w)
            written.append(w)
            # A qualified model's per-group capsules are corpus members of their own: Phase 1 grades
            # the very programs that qualified it.
            for extra in _qualifying_group_capsules(Path(w), out_root):
                if extra in written:
                    continue
                if evidence is not None and evidence.software_spec:
                    member = yaml.safe_load((extra / "capsule.yaml").read_bytes())
                    observed = screen_written(member, extra, target=hardware_target, evidence=evidence)
                    if observed is None:
                        observed = screen_entry(
                            evidence.software_spec,
                            {"name": member.get("name"), "kind": member.get("kind")},
                            defaults={"operand_dtype": binding.operand_dtype, "accumulator_dtype": binding.accum_dtype},
                            capsule=member,
                            host_capabilities=evidence.host_capabilities,
                        )
                    admission.append({"capsule": member.get("name"), "observation": "qualifying_group", **observed})
                    # The same rule every emitted member is held to: an operation the software spec
                    # refuses cannot qualify a model, and verified mode admits only reviewed ones.
                    if observed["status"] == "unsupported" or (
                        evidence_mode != "diagnostic" and observed["status"] != "admitted"
                    ):
                        # The writer already created this group as numerical evidence. Keep its
                        # linked bytes inspectable, but never classify it as hand-authored or put
                        # it in either selected experiment corpus.
                        member["software_screen"] = observed
                        (extra / "capsule.yaml").write_text(yaml.safe_dump(member, sort_keys=False))
                        _scrub_capsule_dir(extra)
                        refused_generated.append(extra)
                        failures.append((e.get("name", "?"), f"qualifying group {extra.name}: {observed['reason']}"))
                        continue
                    member["software_screen"] = observed
                    (extra / "capsule.yaml").write_text(yaml.safe_dump(member, sort_keys=False))
                _scrub_capsule_dir(extra)
                written.append(extra)
            if family and Path(w).parent.name == "_perf":
                family_counts[family]["written_members"] += 1
        elif family:
            _performance_errors.append(
                {
                    "family": family,
                    "member": e.get("name", "?"),
                    "status": "error",
                    "error_type": "NoOutput",
                    "detail": "capsule writer returned no output",
                }
            )
        else:
            omitted.append(
                {
                    "capsule": e.get("name", "?"),
                    "source": e.get("source"),
                    "status": "not_built",
                    "reason": "required writer source or lowering input unavailable",
                }
            )
            if evidence_mode == "verified":
                failures.append((e.get("name", "?"), "required capsule writer produced no artifact"))
    # A performance sweep can fail before it becomes an entry (for example,
    # while resolving a selected oracle). Such failures are recorded in the
    # manifest, but must also fail the run after the other capsules are written;
    # otherwise the controller reports success while a required claim cohort
    # has zero members. Writer failures already appear in both lists.
    failed_names = {name for name, _ in failures}
    for error in _performance_errors:
        name = str(error.get("member") or error.get("family") or "<performance sweep>")
        if name not in failed_names:
            kind = str(error.get("error_type") or "unknown error")
            detail = str(error.get("detail") or "no further detail recorded").replace("\n", " ")[:240]
            failures.append((name, f"performance materialization failed ({kind}): {detail}; inspect MANIFEST.yaml"))
            failed_names.add(name)
    if failures:
        print(f"  [FAIL] {len(failures)} capsule(s) could not be written:")
        for name, why in failures:
            print(f"    - {name}: {why}")
    # Record provenance for what we just emitted. The MANIFEST header has always CLAIMED the generator
    # rewrites it, but no writer existed, so it drifted silently as soon as the corpus grew.
    declared_blocked = [
        {
            "family": row["family"],
            "status": "blocked_unimplemented",
            "reason": row["reason"],
            "emitter": copy.deepcopy(row["performance"]["emitter"]),
            "fit_axes": list(row.get("fit_axes") or []),
            "comparison_roles": list(row.get("comparison_roles") or []),
        }
        for row in (template.get("blocked_unimplemented") or [])
    ]
    generated_members = sum(row["written_members"] for row in family_counts.values())
    performance_record = {
        "shared_template": {"path": template.get("path"), "sha256": template.get("sha256")},
        "facts": {
            "target": hardware_target,
            "sha256": facts["sha256"],
            "digest_kind": "derived_performance_document",
            **({"raw_facts_sha256": evidence.raw_facts_sha256} if evidence is not None else {}),
        },
        "phase": {
            "category": "_perf",
            "label": "dev",
            "included_in_functional_grade": False,
            "exclusion": "TargetExperiment.corpus_siblings excludes underscore-prefixed categories",
        },
        "families": declared_families,
        "counts": {
            "declared_families": len(declared_families),
            "generated_families": sum(1 for row in family_counts.values() if row["written_members"] > 0),
            "generated_members": generated_members,
            "by_family": family_counts,
        },
        "skipped_inapplicable": _sweep_skips,
        "blocked_unimplemented": declared_blocked + _runtime_blocked,
        "errors": _performance_errors,
    }
    if unbuilt_roster:
        print(
            f"  [roster] {len(unbuilt_roster)} declared roster model(s) could not be captured at this "
            f"target's derived format; they are recorded as not_built, never as covered"
        )
    if unprovable_forbids:
        print(
            f"  [lane] {len(unprovable_forbids)} synthesized host-lane capsule(s) were dropped: their "
            f"program contains regions this target admits, so the forbid they assert is not provable"
        )
    from merlin_experiments.phase0.hidden_disjointness import check as hidden_disjointness

    disjointness = hidden_disjointness(written)
    if disjointness["status"] != "disjoint":
        failures.append(
            (
                "hidden cohort",
                f"{disjointness['overlapping_hidden_capsules']} hidden capsule(s) repeat a public program",
            )
        )
    superseded = _prune_superseded_synth(entries, written, target=target)
    if superseded:
        print(
            f"  [prune] {len(superseded)} synthesized capsule(s) whose requirement cell no longer "
            f"exists were removed: {', '.join(superseded)}"
        )
    update_provenance_manifest(
        written,
        cap_root=out_root,
        target=hardware_target,
        refused_generated=refused_generated,
        performance_record=performance_record,
        unbuilt_roster=unbuilt_roster,
        claim_model_evaluation=claim_plan,
        unprovable_forbids=unprovable_forbids,
        superseded=superseded,
    )
    if evidence is not None:
        from merlin_experiments.phase1.source_inputs import fingerprint

        from .coverage_commitment import observe_cohort, selected_inputs, write_inputs

        coverage_root = artifact_root / "coverage"
        coverage_root.mkdir(parents=True, exist_ok=True)
        accounting = json.loads((coverage_root / "operation-accounting.json").read_bytes())
        coverage_inputs = selected_inputs(evidence, accounting=accounting, prohibited_roles=declared_roles)
        coverage_input_record = write_inputs(out_root, coverage_inputs)
        # Absent only when a caller supplies its own inputs; the coverage report then blocks on it.
        drift = coverage_inputs.get("spec_fact_drift") or {"findings": []}
        (coverage_root / "spec-fact-drift.json").write_bytes(
            (json.dumps(drift, sort_keys=True, indent=2) + "\n").encode()
        )
        for row in drift["findings"]:
            if row["classification"] != "ok":
                print(f"  [spec-drift] {row['classification']}: {row['family']}.{row['field']} {row.get('values')}")
        generated_manifest = yaml.safe_load((out_root / "MANIFEST.yaml").read_text()) or {}
        selections = (generated_manifest.get("phase_corpora") or {}).get(hardware_target) or {}
        cohort_reports = {}
        selected_reports = {}
        for phase in ("phase1", "phase2"):
            members = (selections.get(phase) or {}).get("generated_members") or []
            report = observe_cohort(
                coverage_inputs, [out_root / member for member in members], target=hardware_target, phase=phase
            )
            if phase == "phase2":
                from .form_perf import attach_to_phase2_report

                form_coverage = _form_perf_coverage(profile, selected_requirement, out_root, members)
                attach_to_phase2_report(report, form_coverage)
            selected_reports[phase] = report
            report_path = coverage_root / f"{phase}-capsule-coverage.json"
            raw = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode()
            report_path.write_bytes(raw)
            cohort_reports[phase] = {
                "path": str(report_path),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "status": report["status"],
                "n_capsules": report["cohort"]["n_capsules"],
            }
        from .phase2_guards import build_guard_link

        guard_link = build_guard_link(out_root, coverage_inputs, selected_reports["phase1"], selected_reports["phase2"])
        guard_path = coverage_root / "phase2-functional-guards.json"
        guard_raw = (json.dumps(guard_link, sort_keys=True, indent=2) + "\n").encode()
        guard_path.write_bytes(guard_raw)
        guard_record = {
            "path": str(guard_path),
            "sha256": hashlib.sha256(guard_raw).hexdigest(),
            "status": guard_link["status"],
            "n_guards": len(guard_link["guards"]),
        }
        receipt = {
            "schema": "merlin.phase0_generation.v1",
            "target": hardware_target,
            "evidence_status": evidence.status,
            "mode": evidence_mode or "verified",
            "software_spec": profile.get("_software_spec_identity"),
            "synthesis": profile.get("_synth_verification"),
            "operation_accounting": str(artifact_root / "coverage/operation-accounting.json"),
            "quantization_contract": str(artifact_root / "software/quantization-contract.json"),
            "capsules_written": len(written),
            "omitted": omitted,
            "unbuilt_roster": unbuilt_roster,
            "operation_admission": admission,
            "unprovable_forbids": unprovable_forbids,
            "failures": [{"capsule": n, "reason": why} for n, why in failures],
            "performance_materialization_errors": len(_performance_errors),
            "qualification": "not_established",
            "corpus_manifest": str(out_root / "MANIFEST.yaml"),
            "coverage_inputs": coverage_input_record,
            "cohort_coverage": cohort_reports,
            "phase2_functional_guards": guard_record,
            "instruction_policy": copy.deepcopy(instruction_policy),
            "hidden_disjointness": disjointness,
            "capsule_commitments": [
                {"member": path.relative_to(out_root).as_posix(), "sha256": fingerprint(path)}
                for path in sorted(written)
            ],
        }
        (coverage_root / "generation.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        manifest_path = out_root / "MANIFEST.yaml"
        manifest = yaml.safe_load(manifest_path.read_text()) or {}
        manifest["coverage_inputs"] = coverage_input_record
        manifest["instruction_policy"] = copy.deepcopy(instruction_policy)
        manifest["phase0_evidence"] = {
            "status": evidence.status,
            "mode": evidence_mode or "verified",
            "raw_facts_sha256": evidence.raw_facts_sha256,
            "software_spec": profile.get("_software_spec_identity"),
            "manifest": str(artifact_root / "evidence-manifest.json"),
            "generation_receipt": str(coverage_root / "generation.json"),
        }
        manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    if failures:
        raise RuntimeError(
            f"{len(failures)} capsule(s) failed to generate: {', '.join(n for n, _ in failures)}; the "
            f"rest of the corpus was written, so re-running after fixing them is cheap"
        )
    return written


def _prune_superseded_synth(entries, written, *, target: str) -> list:
    """Remove synthesized capsule directories the profile no longer asks for, and name them.

    THE PROFILE IS THE WHOLE AUTHORITY for `derived_sweep` capsules -- it is regenerated from the
    requirement, so a directory of that role which is not in it is evidence for a cell that is no
    longer required. Nothing removed one, and they do not merely sit there: a superseded
    `SY_kdepth_certified` inflated a sealed cohort count by one and broke the MANIFEST, and twelve
    superseded MX capsules survived the alignment classes their own datapath cannot express, each one
    still offering itself to the cover as evidence for a cell the requirement had dropped.

    Deliberately narrow: only directories whose capsule declares this generator's synthesized role, and
    only under the roots this target just wrote to. A hand-authored capsule is never touched, and a
    target whose profile carries no synthesized entries at all prunes nothing (a profile that failed to
    synthesize must not be read as "every cell was dropped").
    """
    keep = {str(e.get("name")) for e in entries if str(e.get("source_role") or "") == SYNTH_ROLE}
    if not keep:
        return []
    roots = {d.parent for d in written if isinstance(d, Path)}
    removed = []
    for root in sorted(roots):
        for cy in sorted(root.glob("*/capsule.yaml")):
            try:
                cap = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            name = str(cap.get("name") or cy.parent.name)
            if str(cap.get("source_role") or "") != SYNTH_ROLE or name in keep:
                continue
            shutil.rmtree(cy.parent, ignore_errors=True)
            removed.append(name)
    return sorted(removed)


def _is_roster_capsule(entry: dict) -> bool:
    """A whole-model capsule the ROSTER axis emitted, as opposed to any other model capsule.

    Read off the entry's own generalization axis rather than its name, so a renamed prefix does not
    silently reclassify a capsule into the lane that is allowed not to build.
    """
    return (
        str(entry.get("kind")) == "model"
        and str((entry.get("generalization") or {}).get("generalization_axis") or "") == "roster"
    )


def _form_perf_coverage(profile: dict, requirement: dict | None, out_root: Path, members: list) -> dict:
    """The perf-coverage section of the Phase 2 report: each derived form class above the shared
    template's declared share needs a form-perf member and a vendor bar in the emitted cohort."""
    from .form_perf import form_perf_coverage

    declared = [
        sweep for sweep in profile.get("sweeps") or [] if isinstance(sweep, dict) and sweep.get("requires_form_scope")
    ]
    threshold = (declared[0]["requires_form_scope"] or {}).get("min_predicted_cycle_share") if declared else None
    capsules = []
    for member in members:
        path = out_root / member / "capsule.yaml"
        if path.is_file():
            capsules.append(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    return form_perf_coverage(requirement, capsules, threshold=threshold)


def require_enforceable_policy(evidence_mode: str | None, roles: list[str], policy: dict) -> None:
    """Verified Phase 0 refuses to seal a declared prohibition it cannot enforce: an underivable
    taxonomy, or a role that matches none of the target's instructions (a rule that forbids nothing)."""
    if evidence_mode != "verified" or not roles:
        return
    refused = enforcement_problems(policy, roles)
    if refused:
        raise ValueError(
            "verified Phase 0 cannot resolve the declared prohibited instruction roles against this "
            f"target's instruction taxonomy: {'; '.join(refused)}"
        )


def _instruction_policy(target: str, roles: list[str], evidence, rtl_facts) -> dict:
    """The declared prohibition resolved against the target's own derived instruction taxonomy."""
    from .instruction_roles import derive_role_taxonomy, resolve_policy

    if not roles:
        return resolve_policy([], {"status": "not_derived"})
    if evidence is None:
        return resolve_policy(roles, derive_role_taxonomy(target))
    from merlin.targetgen.rtl.facts import observed_facts
    from merlin.targetgen.target_registry import observed_contract

    with (
        observed_contract(target, evidence.contract),
        observed_facts(target, evidence.refreshed_facts, Path(rtl_facts) if rtl_facts is not None else None),
    ):
        taxonomy = derive_role_taxonomy(target)
    policy = resolve_policy(roles, taxonomy)
    policy["taxonomy"] = taxonomy
    return policy


def _with_candidate_policy(entry: dict, policy: dict) -> dict:
    """A form-perf member's candidate arm carries the declared roles by value; the vendor arm is exempt."""
    from .instruction_roles import candidate_arm_policy

    arms = (entry.get("performance") or {}).get("arms")
    if not isinstance(arms, dict) or "candidate" not in arms:
        return entry
    out = copy.deepcopy(entry)
    out["performance"]["arms"]["candidate"]["instruction_policy"] = candidate_arm_policy(policy)
    return out


def _with_reference_gate(entry: dict, te, descriptor: Path) -> dict:
    """Attach a model's declared whole-model reference gate result (``workload_spec.reference_gates``)."""
    if entry.get("kind") != "model" and entry.get("op") != "model":
        return entry
    declared = ((getattr(te, "workload_spec", None) or {}).get("reference_gates")) or {}
    if not isinstance(declared, dict):
        raise ValueError("workload_spec.reference_gates must map a model to its gate result")
    path = declared.get(str(entry.get("model") or "")) or declared.get(str(entry.get("name") or ""))
    if path is None:
        return entry
    resolved = Path(str(path))
    if not resolved.is_absolute():
        resolved = Path(descriptor).parent / resolved
    return {**entry, "reference_gate": str(resolved)}


def _member_path(entry: dict) -> str | None:
    """``<category>/<name>`` of the member an entry writes, or None when it does not say."""
    category, name = entry.get("cat"), entry.get("name")
    return f"{category}/{name}" if category and name else None


def _qualifying_group_capsules(written: Path, out_root: Path) -> list[Path]:
    """The per-group capsule directories a qualified model capsule names as its evidence."""
    capsule_file = written / "capsule.yaml"
    if not capsule_file.is_file():
        return []
    capsule = yaml.safe_load(capsule_file.read_text(encoding="utf-8")) or {}
    rows = (capsule.get("model_qualification") or {}).get("group_capsules") or []
    out = []
    for row in rows:
        member = out_root / str(row.get("capsule") or "")
        if not (member / "capsule.yaml").is_file():
            raise ValueError(
                f"model {capsule.get('name')!r} names qualifying group capsule {row.get('capsule')!r} that is absent"
            )
        out.append(member)
    return out
