"""The machine registry: how to RUN a whole-model program on each machine a target declares.

WHICH DEVICE a machine is stays the pin registry's (``merlin/contract/hardware_pins.yaml``), resolved
by content in :mod:`.machines` before anything runs.  HOW to run a program there -- which queue,
which workload slot, which host preparation, which functional model grades it locally, what the
machine cannot express -- is the TARGET's data, declared once in a registry file the target's example
owns (for instance ``examples/<target>/phase2/whole-model-machines.yaml``) and read here.  Nothing in
this module names a target, a board or a simulator binary.

A registry entry is one machine::

    schema: merlin.phase2.whole_model_measured.machines.v1
    target: <target>
    machines:
      <name>:
        kind: firesim | gsim | spike | contract | paired | batched | cell
        ...            # the kind's own fields (see machines.machine_from_spec)
        cannot_express:  # optional: failures this machine cannot avoid, as a named class
          - {output_element_bytes: 4, compare: exact, reason: "..."}
        adjudicates: [cycle_count]  # optional: the claim kinds this machine's numbers settle

A value may be ``{env: NAME}`` (the host's location, refused when unset) or ``{env: NAME, join:
"relative/path"}``.  A ``paired``/``batched`` entry names its halves by registry name (``timing: <name>``,
``local: <name>``); they are expanded into full specs, because a detached worker receives only data.
"""

from __future__ import annotations

import copy
import getpass
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = "merlin.phase2.whole_model_measured.machines.v1"

#: Machine kinds whose spec is run directly by :func:`.machines.machine_from_spec`.
RUN_KINDS = ("firesim", "gsim", "spike")
#: Kinds that compose or delegate.
COMPOSITE_KINDS = ("paired", "batched", "contract", "cell")
#: The comparisons a ``cannot_express`` limit may name (the verdict's own vocabulary).
LIMIT_COMPARES = ("exact", "bounded_int")
#: The claim kinds a machine may declare it adjudicates.
CLAIM_KINDS = ("cycle_count", "correctness", "structure")


class RegistryError(ValueError):
    """The machine registry is missing, malformed, or cannot resolve a machine on this host."""


def load(path: str | Path) -> dict[str, Any]:
    """The registry document at ``path``, schema-checked."""
    import yaml

    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise RegistryError(f"no machine registry at {path}")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise RegistryError(f"{path} is not YAML: {exc}") from exc
    if not isinstance(document, Mapping) or document.get("schema") != SCHEMA:
        raise RegistryError(f"{path} does not declare schema {SCHEMA}")
    if not str(document.get("target") or "").strip():
        raise RegistryError(f"{path} declares no target")
    machines = document.get("machines")
    if not isinstance(machines, Mapping) or not machines:
        raise RegistryError(f"{path} declares no machines")
    for name, entry in machines.items():
        if not isinstance(entry, Mapping):
            raise RegistryError(f"machine {name!r} is not a mapping")
        kind = entry.get("kind")
        if kind not in (*RUN_KINDS, *COMPOSITE_KINDS):
            raise RegistryError(f"machine {name!r} has unknown kind {kind!r}")
        validate_limits(entry.get("cannot_express"), where=str(name))
        adjudicates = entry.get("adjudicates")
        if adjudicates is not None and (
            not isinstance(adjudicates, list) or any(value not in CLAIM_KINDS for value in adjudicates)
        ):
            raise RegistryError(f"machine {name!r} adjudicates an unknown claim kind: {adjudicates!r}")
    return dict(document)


def validate_limits(limits: Any, *, where: str) -> None:
    """A ``cannot_express`` list: each limit names a width, a comparison and a REASON.

    A limit with no reason would excuse a failure nobody can account for, which is the silent excuse
    the class exists to replace."""
    if limits is None:
        return
    if not isinstance(limits, list):
        raise RegistryError(f"{where}: cannot_express must be a list")
    for limit in limits:
        if not isinstance(limit, Mapping):
            raise RegistryError(f"{where}: a cannot_express limit is not a mapping")
        width = limit.get("output_element_bytes")
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise RegistryError(f"{where}: a cannot_express limit needs a positive output_element_bytes")
        if limit.get("compare") not in LIMIT_COMPARES:
            raise RegistryError(f"{where}: a cannot_express limit compares one of {LIMIT_COMPARES}")
        if not str(limit.get("reason") or "").strip():
            raise RegistryError(f"{where}: a cannot_express limit states its reason")


def _env_value(value: Mapping[str, Any], *, name: str, environment: Mapping[str, str] | None) -> str:
    variable = str(value.get("env") or "")
    if environment is not None and variable in environment:
        resolved = environment[variable]
    else:
        from merlin.common.paths import env

        resolved = env(variable)
    if not resolved:
        raise RegistryError(f"{name} names ${variable}, which is unset on this host")
    join = value.get("join")
    return str(Path(resolved) / str(join)) if join else str(resolved)


def _resolve(value: Any, *, name: str, environment: Mapping[str, str] | None) -> Any:
    if isinstance(value, Mapping) and set(value) <= {"env", "join"} and "env" in value:
        return _env_value(value, name=name, environment=environment)
    if isinstance(value, Mapping):
        return {k: _resolve(v, name=f"{name}.{k}", environment=environment) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, name=f"{name}[{i}]", environment=environment) for i, v in enumerate(value)]
    return value


def resolve(
    registry: Mapping[str, Any] | str | Path,
    name: str,
    *,
    environment: Mapping[str, str] | None = None,
    submitter: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    _seen: tuple[str, ...] = (),
) -> dict[str, Any]:
    """The full run spec for machine ``name``: host locations resolved, halves expanded, ready for a
    detached worker.  ``overrides`` replaces top-level fields a launch sets for this run (a batch's
    control, its size), never the device's identity fields."""
    document = load(registry) if isinstance(registry, (str, Path)) else dict(registry)
    machines = document.get("machines") or {}
    if name in _seen:
        raise RegistryError(f"machine {name!r} refers to itself through {list(_seen)}")
    entry = machines.get(name)
    if not isinstance(entry, Mapping):
        raise RegistryError(f"the registry declares no machine {name!r} ({sorted(machines)})")
    target = str(document["target"])
    spec: dict[str, Any] = {"target": target, "registry_name": name}
    for key, value in entry.items():
        spec[key] = _resolve(copy.deepcopy(value), name=f"{name}.{key}", environment=environment)
    for key, value in dict(overrides or {}).items():
        if key in ("kind", "target", "hw_config", "registry_name"):
            raise RegistryError(f"a launch may not override machine {name!r}'s {key}")
        spec[key] = copy.deepcopy(value)
    kind = spec["kind"]
    if kind in ("paired", "batched"):
        for half in ("timing", "local"):
            reference = spec.get(half)
            if isinstance(reference, str):
                spec[half] = resolve(
                    document, reference, environment=environment, submitter=submitter, _seen=(*_seen, name)
                )
            if not isinstance(spec.get(half), Mapping):
                raise RegistryError(f"{kind} machine {name!r} names no {half} machine")
    if kind == "firesim":
        chipyard = str(spec.get("chipyard") or "")
        if not chipyard:
            raise RegistryError(f"FireSim machine {name!r} names no chipyard root")
        spec["prepare_command"] = [
            [str(token).replace("{chipyard}", chipyard) for token in argv]
            for argv in _argv_list(spec.get("prepare_command"))
        ]
        arguments = [str(a) for a in spec.get("submit_arguments") or ()]
        if "--user" not in arguments:
            arguments += ["--user", submitter or getpass.getuser()]
        spec["submit_arguments"] = arguments
        workload = Path(chipyard) / "sims" / "firesim" / "deploy" / "workloads" / f"{spec.get('workload')}.json"
        if not workload.is_file():
            raise RegistryError(f"workload {spec.get('workload')!r} is not declared on this host ({workload})")
    return spec


def _argv_list(value: Any) -> list[list[str]]:
    steps = list(value or ())
    if steps and all(isinstance(token, str) for token in steps):
        return [list(steps)]
    return [list(step) for step in steps]


def adjudicates(spec: Mapping[str, Any], claim: str = "cycle_count") -> dict[str, Any]:
    """Whether the machine ``spec`` DECLARES that its numbers settle ``claim``.

    Three states, never two: ``ADJUDICATED`` when the registry says so, ``UNADJUDICATED`` when the
    machine declares its claims and this is not one, ``UNKNOWN`` when it declares none -- a number from a
    machine nobody described is not the same as a number from one described as a ranking signal."""
    timing = spec.get("timing") if spec.get("kind") in ("paired", "batched") else spec
    declared = (timing or {}).get("adjudicates") if isinstance(timing, Mapping) else None
    if declared is None:
        return {"status": "UNKNOWN", "claim": claim, "reason": "the machine declares no adjudicated claims"}
    status = "ADJUDICATED" if claim in declared else "UNADJUDICATED"
    return {
        "status": status,
        "claim": claim,
        "machine": (timing or {}).get("registry_name"),
        "declared": list(declared),
    }


__all__ = ["SCHEMA", "RegistryError", "adjudicates", "load", "resolve", "validate_limits"]
