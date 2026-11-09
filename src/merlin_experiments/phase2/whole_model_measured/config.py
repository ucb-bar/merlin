"""The objective config a ``whole_model_measured`` launch declares, and the objective it builds.

Everything a measurement is attributed through is named here, as data the run keeps (read-only)
beside its record: the builder (and its pin), each machine, each machine's build options, each
machine's reference result, where the store lives, and the experiment's instruction POLICY.  A
config that is underdetermined is refused AT LAUNCH -- not at the first comparison an hour in.

THE POLICY IS CHECKED, NOT TRUSTED.  ``prohibited_instruction_roles`` is the experiment's declared
rule (``policy.prohibited_instruction_roles`` in the experiment definition).  Every CANDIDATE section
(the screen, the certifier and every held-out model) must carry exactly those roles in its build
options, because the build option is what the builder routes by and what the whole-ELF gate checks:
a relaunch that silently dropped them once ran a no-FSM campaign without its prohibition.  The
reference arms are measured without the rule by construction (they are the bar).

THE RULE IS THE ONE PHASE 0 SEALED.  Declared roles alone say nothing about which instructions they
forbid; the sealed Phase 0 corpus's ``instruction_policy`` says that, for this target.  A config that
declares roles carries that policy by value (``instruction_policy``, stamped at prepare by
:func:`seal_policy` from ``--phase0-manifest``, the config's ``phase0_manifest``, or the selected
descriptor's corpus manifest), and is refused unless it resolved, declares every role, and prohibits at
least one instruction per role: a run whose rule forbids nothing measures a program nobody checked.

A section's ``machine`` is either a full spec or a registry reference::

    machine: {registry: /abs/whole-model-machines.yaml, name: <machine>, overrides: {...}}

resolved here, once, so the spec a job records -- and the store's key -- is the resolved data.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from merlin.perf import whole_model_builder
from merlin_experiments.phase0 import instruction_roles as IR

from . import capabilities as CAP
from . import gates as G
from . import registry as R
from .identity import builder_identity, store_root_for
from .objective import WholeModelObjective
from .service import MeasurementService

CONFIG_SCHEMA = "merlin_whole_model_objective_config_v1"
DERIVED_MECHANISMS = "derive_from_capture"
#: The production builder (the core's service builder over the whole-model build); a config may name
#: another (a test double, a second target's).  Imported, not only named, so the dependency is real.
DEFAULT_BUILDER = f"{whole_model_builder.__name__}:{whole_model_builder.build.__name__}"
DEFAULT_REFERENCE_BUILDER = f"{whole_model_builder.__name__}:{whole_model_builder.build_reference.__name__}"
CANDIDATE_SECTIONS = ("screen", "certifier")
#: The sealed Phase 0 instruction policy, carried by value in a config that declares roles.
SEALED_POLICY = "instruction_policy"
#: Optional config key naming the sealed Phase 0 corpus manifest the policy is read from.
SEALED_POLICY_MANIFEST = "phase0_manifest"


class ConfigError(ValueError):
    """The objective config is underdetermined or inconsistent."""


def resolve_machine(section: Mapping[str, Any], *, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    """A section's machine as a full spec (a registry reference resolved; a full spec returned as is)."""
    machine = dict(section.get("machine") or {})
    if "registry" in machine:
        if set(machine) - {"registry", "name", "overrides"}:
            raise ConfigError(f"a registry machine reference takes registry, name and overrides; got {sorted(machine)}")
        return R.resolve(
            machine["registry"],
            str(machine.get("name") or ""),
            environment=environment,
            overrides=machine.get("overrides"),
        )
    return machine


def declared_roles(document: Mapping[str, Any]) -> list[str]:
    roles = document.get("prohibited_instruction_roles")
    if roles is None:
        return []
    if not isinstance(roles, list) or any(not isinstance(r, str) or not r for r in roles):
        raise ConfigError("prohibited_instruction_roles must be a list of role names")
    return list(roles)


def check_policy(document: Mapping[str, Any]) -> list[str]:
    """The declared roles, after refusing any candidate section whose build options disagree, and a
    declaration the sealed Phase 0 instruction policy cannot enforce."""
    roles = declared_roles(document)
    sections = [(name, document.get(name)) for name in CANDIDATE_SECTIONS if document.get(name)]
    sections += [(f"held_out.{name}", body) for name, body in (document.get("held_out") or {}).items()]
    for name, section in sections:
        carried = list(((section or {}).get("build_options") or {}).get(G.PROHIBITED_ROLES) or ())
        if sorted(carried) != sorted(roles):
            raise ConfigError(
                f"the {name} section's build options carry prohibited roles {carried!r} but the experiment "
                f"declares {roles!r}; the rule a build is shaped by and the rule it is judged by must agree"
            )
    if roles:
        problems = IR.enforcement_problems(document.get(SEALED_POLICY), roles)
        if problems:
            raise ConfigError(
                f"the config declares prohibited roles {roles!r} but carries no enforceable sealed Phase 0 "
                f"instruction policy ({'; '.join(problems)}); seal one with --phase0-manifest"
            )
    return roles


def sealed_policy_source(document: Mapping[str, Any], *, target: str, manifest: Path | None = None) -> Path:
    """Where the sealed Phase 0 instruction policy is read from: ``manifest`` (``--phase0-manifest``),
    the config's ``phase0_manifest``, else the selected descriptor's corpus ``MANIFEST.yaml``."""
    if manifest is not None:
        return Path(manifest)
    if document.get(SEALED_POLICY_MANIFEST):
        return Path(str(document[SEALED_POLICY_MANIFEST]))
    from merlin.targetgen.target_experiment import descriptor_for, load_target_experiment

    descriptor = document.get("descriptor") or descriptor_for(target)
    if descriptor is None:
        raise ConfigError(f"no sealed Phase 0 manifest was named and {target!r} has no descriptor to find one")
    corpus = getattr(load_target_experiment(Path(str(descriptor))), "capsule_corpus", None)
    if corpus is None:
        raise ConfigError(f"the descriptor {descriptor} names no capsule corpus, so no sealed policy")
    return Path(corpus).parent / "MANIFEST.yaml"


def read_sealed_policy(path: Path) -> dict[str, Any]:
    """The ``instruction_policy`` a sealed Phase 0 corpus manifest carries (or a bare policy document)."""
    import hashlib

    import yaml

    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"the sealed Phase 0 manifest {path} does not exist")
    raw = path.read_bytes()
    document = yaml.safe_load(raw)
    if not isinstance(document, Mapping):
        raise ConfigError(f"{path} is not a mapping")
    policy = document if document.get("schema") == IR.POLICY_SCHEMA else document.get("instruction_policy")
    if not isinstance(policy, Mapping):
        raise ConfigError(f"{path} carries no Phase 0 instruction_policy")
    return {**dict(policy), "sealed_source": {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}}


def seal_policy(document: Mapping[str, Any], *, target: str, manifest: Path | None = None) -> dict[str, Any]:
    """``document`` with the sealed Phase 0 instruction policy stamped in by value, when it declares roles.

    A config that already carries one keeps it (a relaunch is held to the policy it was launched under)
    unless ``manifest`` names another, which then must be the same policy."""
    out = json.loads(json.dumps(dict(document), default=str))
    if not declared_roles(out):
        return out
    if out.get(SEALED_POLICY) is not None and manifest is None:
        return out
    sealed = read_sealed_policy(sealed_policy_source(out, target=target, manifest=manifest))
    kept = out.get(SEALED_POLICY)
    if kept is not None:
        strip = lambda p: {k: v for k, v in dict(p).items() if k != "sealed_source"}  # noqa: E731
        if strip(kept) != strip(sealed):
            raise ConfigError(
                "the named Phase 0 manifest's instruction policy differs from the one this run was sealed under"
            )
    out[SEALED_POLICY] = sealed
    return out


def with_policy(document: Mapping[str, Any], roles: list[str]) -> dict[str, Any]:
    """``document`` with the declared ``roles`` written into every candidate section's build options."""
    out = json.loads(json.dumps(dict(document), default=str))
    out["prohibited_instruction_roles"] = list(roles)
    for name in CANDIDATE_SECTIONS:
        if out.get(name):
            options = dict(out[name].get("build_options") or {})
            if roles:
                options[G.PROHIBITED_ROLES] = list(roles)
            else:
                options.pop(G.PROHIBITED_ROLES, None)
            out[name]["build_options"] = options
    for body in (out.get("held_out") or {}).values():
        options = dict(body.get("build_options") or {})
        if roles:
            options[G.PROHIBITED_ROLES] = list(roles)
        body["build_options"] = options
    return out


#: The operator's opt-out of fused regions (``fused_regions: false`` in the objective config, or the
#: CLI's ``--no-fused-regions``); a section's own ``build_options.allow_regions: false`` opts it out too.
FUSED_REGIONS = "fused_regions"


def prepare_document(document: Mapping[str, Any], *, target: str) -> dict[str, Any]:
    """Resolve output location and candidate mechanisms from selected model inputs.

    The model's host/accelerator closure is computed by the same target-neutral
    analysis used by the builder. A closed model can exercise package-declared
    passes and regions; an open model cannot. The decision is frozen into the
    run config, never inferred from a target name or a hand-authored kernel.

    FUSED REGIONS ARE ALLOWED BY DEFAULT: a package that opts in may answer adjacent groups as one
    kernel (whole-model and cell programs alike) unless the operator opts out (:data:`FUSED_REGIONS`)
    or the model is open.  Passes stay what ``mechanism_policy`` derives.  Every decision, and why, is
    the run's ``fused_region_decision`` receipt.
    """
    from merlin.common.paths import artifacts_dir

    out = json.loads(json.dumps(dict(document), default=str))
    if not out.get("store"):
        if not target or target in (".", "..") or "/" in target or "\\" in target:
            raise ConfigError(f"invalid target name for a generated store: {target!r}")
        out["store"] = str((artifacts_dir() / "perf-studies" / "whole-model" / target).resolve())
    if "mechanism_derivation" in out and out.get("mechanism_policy") is None:
        raise ConfigError("mechanism_derivation is a preparation receipt, not an operator declaration")
    policy = out.get("mechanism_policy")
    if policy not in (None, DERIVED_MECHANISMS):
        raise ConfigError(f"unknown mechanism_policy {policy!r}")
    opted_out = out.get(FUSED_REGIONS) is False
    if out.get(FUSED_REGIONS) not in (None, True, False):
        raise ConfigError(f"{FUSED_REGIONS} is true or false, not {out.get(FUSED_REGIONS)!r}")

    from merlin.perf.whole_model_open import is_open_model

    decisions, regions = {}, {}
    sections = [(name, out.get(name)) for name in CANDIDATE_SECTIONS]
    sections += [(f"held_out.{name}", body) for name, body in (out.get("held_out") or {}).items()]
    for name, section in sections:
        if not section:
            continue
        options = dict(section.get("build_options") or {})
        capsule = options.get("model_capsule")
        has_capsule = isinstance(capsule, str) and Path(capsule).is_dir()
        if policy == DERIVED_MECHANISMS and not has_capsule:
            raise ConfigError(f"{name} needs a frozen model_capsule directory to derive mechanisms")
        open_model, unknown_why = None, "no frozen model capsule to derive the model's closure from"
        if has_capsule and policy == DERIVED_MECHANISMS:
            open_model = is_open_model(capsule, target)
        elif has_capsule:
            try:
                from merlin.perf.whole_model_capsule import load_model_capsule

                load_model_capsule(capsule)  # a directory that is not a whole model capsule is refused first
                open_model = is_open_model(capsule, target)
            except Exception as exc:  # noqa: BLE001 -- an underivable closure leaves regions off, said why
                unknown_why = f"the model's closure could not be derived: {type(exc).__name__}: {str(exc)[:200]}"
        if policy == DERIVED_MECHANISMS:
            if "allow_passes" in options and options["allow_passes"] is not (not open_model):
                raise ConfigError(f"{name}.allow_passes contradicts the selected model's derived closure")
            options["allow_passes"] = not open_model
        section_out = options.get("allow_regions") is False
        if options.get("allow_regions") is True and open_model:
            raise ConfigError(f"{name}.allow_regions contradicts the selected model's derived closure (open)")
        if opted_out or section_out:
            options["allow_regions"] = False
            regions[name] = {"allowed": False, "why": "opted out by the operator"}
        elif open_model is None:
            # NOT DERIVABLE: the model's closure is unknown -- left to the builder's own default (off) unless
            # the operator declared it, and said so, never assumed closed.
            declared = options.get("allow_regions") is True
            regions[name] = {
                "allowed": declared,
                "why": f"declared by the operator; {unknown_why}" if declared else unknown_why,
            }
        else:
            options["allow_regions"] = not open_model
            regions[name] = {
                "allowed": not open_model,
                "why": "the model is open: its host regions compute between groups"
                if open_model
                else "default: a closed model's package may claim fused regions",
            }
        section["build_options"] = options
        if policy == DERIVED_MECHANISMS:
            decisions[name] = {
                "model_closure": "open" if open_model else "closed",
                "allow_passes": options["allow_passes"],
                "allow_regions": options["allow_regions"],
            }
    if policy == DERIVED_MECHANISMS:
        out["mechanism_derivation"] = {"source": "merlin.perf.whole_model_open.is_open_model", "sections": decisions}
    out["fused_region_decision"] = {"default": "allowed", "opted_out": opted_out, "sections": regions}
    return out


#: The objective config's exactness contract: a path to the target's reviewed contract when authored,
#: carried BY VALUE once a run is prepared (:func:`seal_exactness`), so a run is graded under the contract
#: it was launched with however the file changes afterwards.
EXACTNESS = "exactness"


def exactness_contract(document: Mapping[str, Any], *, target: str):
    """The :class:`merlin.perf.exactness.Contract` a config declares: by value, by path, or the default
    (every form exact) when it declares none.  A declared contract that cannot be read is an error."""
    from merlin.perf import exactness as EX

    declared = document.get(EXACTNESS)
    try:
        if isinstance(declared, str) and declared:
            return EX.load(declared)
        return EX.Contract.from_value(declared if isinstance(declared, Mapping) else None, target=target)
    except EX.ExactnessError as exc:
        raise ConfigError(f"the objective config's exactness contract: {exc}") from exc


def seal_exactness(document: Mapping[str, Any], *, target: str) -> dict[str, Any]:
    """``document`` with its exactness contract carried by value (the default's, when it declared none)."""
    out = json.loads(json.dumps(dict(document), default=str))
    out[EXACTNESS] = exactness_contract(document, target=target).to_document()
    return out


def store_roots(document: Mapping[str, Any], *, environment: Mapping[str, str] | None = None) -> dict[str, Path]:
    """``{section: store root}`` a config's sections own, computed exactly as :func:`from_config` does."""
    builder = dict(document.get("builder") or {"spec": DEFAULT_BUILDER, "sha256": None})
    identity = builder_identity(str(builder["spec"]), builder.get("sha256"))
    base = Path(str(document.get("store") or ""))
    roots = {}
    for name in CANDIDATE_SECTIONS:
        section = document.get(name)
        if section:
            roots[name] = store_root_for(
                base,
                builder=builder,
                machine=resolve_machine(section, environment=environment),
                build_options=dict(section.get("build_options") or {}),
                builder_sha256=identity["sha256"],
            )
    return roots


def from_config(
    document: Mapping[str, Any], *, target: str, environment: Mapping[str, str] | None = None
) -> WholeModelObjective:
    """Build the objective a launch declares, refusing anything underdetermined AT LAUNCH."""
    if not isinstance(document, Mapping) or document.get("schema") != CONFIG_SCHEMA:
        raise ConfigError(f"a whole-model objective config must declare schema {CONFIG_SCHEMA}")
    builder = dict(document.get("builder") or {"spec": DEFAULT_BUILDER, "sha256": None})
    if not builder.get("spec"):
        raise ConfigError("the objective config names no builder")
    base = Path(str(document.get("store") or ""))
    if not base.is_absolute():
        raise ConfigError("the objective config's store must be an absolute path")
    if not document.get("screen"):
        raise ConfigError("the objective config declares no screen section")
    check_policy(document)
    identity = builder_identity(str(builder["spec"]), builder.get("sha256"))
    env = {str(k): str(v) for k, v in (document.get("environment") or {}).items()}
    exactness = {
        "contract": exactness_contract(document, target=target).to_document(),
        "forms": dict(document.get("group_forms") or {}),
    }

    def root_of(section: Mapping[str, Any], machine: Mapping[str, Any]) -> Path:
        return store_root_for(
            base,
            builder=builder,
            machine=machine,
            build_options=dict(section.get("build_options") or {}),
            builder_sha256=identity["sha256"],
        )

    certifier_root = None
    if document.get("certifier"):
        certifier_root = root_of(document["certifier"], resolve_machine(document["certifier"], environment=environment))

    def service(section: Mapping[str, Any], *, is_screen: bool) -> tuple[MeasurementService, Path | None]:
        machine = resolve_machine(section, environment=environment)
        if machine.get("target") != target:
            raise ConfigError(f"machine target {machine.get('target')!r} is not this run's {target!r}")
        R.validate_limits((machine.get("timing") or machine).get("cannot_express"), where="machine")
        reference = Path(str(section["reference"])) if section.get("reference") else None
        return (
            MeasurementService(
                root_of(section, machine),
                target=target,
                builder=str(builder["spec"]),
                builder_sha256=builder.get("sha256"),
                machine=machine,
                slots=int(section.get("slots") or 1),
                max_pending=int(section.get("max_pending") or 3),
                timeout_seconds=float(section.get("timeout_seconds") or 6 * 3600),
                python=document.get("python"),
                environment=env,
                build_options=dict(section.get("build_options") or {}),
                reference=reference,
                excuse_reference_failures=bool(section.get("excuse_reference_failures")),
                pre_measure_check=document.get("pre_measure_check"),
                retain=document.get("retain"),
                certifier_root=certifier_root if is_screen else None,
                instruction_policy=document.get(SEALED_POLICY),
                min_build_free_bytes=section.get("min_build_free_bytes"),
                machine_capabilities=CAP.compact(CAP.section_report(section, environment=environment)),
                exactness=exactness,
            ),
            reference,
        )

    screen, screen_reference = service(document["screen"], is_screen=True)
    certifier, certifier_reference = (None, None)
    if document.get("certifier"):
        certifier, certifier_reference = service(document["certifier"], is_screen=False)
    # TRANSFER CHECK, OPT-IN: one measurement service per held-out model; absent, nothing changes.
    held_out = {name: service(section, is_screen=False) for name, section in (document.get("held_out") or {}).items()}
    objective = WholeModelObjective(
        screen=screen,
        screen_reference=screen_reference,
        certifier=certifier,
        certifier_reference=certifier_reference,
        repeats_on_best=int(document.get("repeats_on_best") or 2),
        primary_name=str(document.get("primary_name") or ""),
        held_out=held_out,
    )
    objective.config = json.loads(json.dumps(dict(document), default=str))
    return objective


__all__ = [
    "CONFIG_SCHEMA",
    "ConfigError",
    "DEFAULT_BUILDER",
    "DEFAULT_REFERENCE_BUILDER",
    "EXACTNESS",
    "FUSED_REGIONS",
    "SEALED_POLICY",
    "SEALED_POLICY_MANIFEST",
    "check_policy",
    "declared_roles",
    "exactness_contract",
    "seal_exactness",
    "from_config",
    "read_sealed_policy",
    "resolve_machine",
    "seal_policy",
    "sealed_policy_source",
    "store_roots",
    "with_policy",
]
