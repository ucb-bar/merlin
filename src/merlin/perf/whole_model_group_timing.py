"""Tier B of the phase-2 fast feedback: the groups an edit CHANGED, timed on the elaborated-RTL emulator.

    diff = changed_groups(candidate_record, baseline_record)
    timing = changed_group_timing(candidate, baseline, groups=diff["changed"], target=..., ...)

A whole-model measurement on the board costs a build, a queue slot and ~12 minutes of a shared FPGA; a
whole model on the emulator costs ~2 hours. Neither is a per-edit signal. What an edit changes is
usually a handful of groups, and a group's cycles are a property of its own program -- so each changed
group is built as a SMALL bare-metal program of its own and run on the emulator, for the candidate and
for the baseline, in parallel:

* **Which groups changed** is read off the two build records, never guessed: a group changed when who
  answered it, why, which object implements it (by content digest), or whether it gathers an operand
  differs (:func:`changed_groups`).
* **The per-group program is the whole-model program with one step.** It is rendered by the target's
  own whole-model driver from the same extraction, with the same kernel object (rebuilt from the
  package's own reply, compared by digest with the one the record names) or the same library call; the
  group's inputs are the values the reference statement computes for them (the model with no package),
  embedded as read-only data, so the group runs on the inputs the model would hand it. Each embedded
  value is laid out in the element type the program's own ABI header declares for its buffer
  (:func:`ctype_dtypes`), never in an assumed width.
* **Correctness is the group's own**: the emulator dumps the group's output at exit and
  :func:`merlin.perf.whole_model_build.grade_memory` grades it LOCALLY against the reference recomputed
  from the embedded inputs (:class:`~merlin.perf.whole_model_build.LocalReference`).
* **Cycles are the emulator's own device.** The emulator elaborates a different design from the board
  (a full-width accumulator readout), so its cycle counts are labelled as its own and are NEVER
  compared with a board number. What they are for is DIRECTION: did this edit make these groups faster
  or slower. :func:`direction_agreement` measures how often that direction matches the board's on
  pairs measured on both, per kind, and a kind whose agreement is poor is flagged.

Concurrency is capped by the host's load (the emulator is single-threaded and the host is shared): at
most ``max_parallel`` at once, and fewer while the one-minute load average is above the core count.
Nothing here names a target: the driver, the ISA and the emulator are the target's, reached by name.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

__all__ = [
    "DEVICE_LABEL",
    "SCHEMA",
    "GroupTimingError",
    "build_group_programs",
    "build_reference_group_programs",
    "changed_group_timing",
    "ctype_dtypes",
    "changed_groups",
    "direction_agreement",
    "time_group_programs",
]

SCHEMA = "whole_model_group_timing_v1"
DEVICE_LABEL = (
    "elaborated-RTL emulator (its own device: full-width readout); cycles are compared only with the "
    "same emulator, never with the board -- they predict the DIRECTION of a change"
)

#: The fields of a build record's per-group attribution that decide what a group's program IS.
_IDENTITY = ("on", "cause", "object_sha256", "gather", "arguments", "declined_as")


class GroupTimingError(RuntimeError):
    """A group program cannot be built or run, and the message says which part stopped it."""


# --------------------------------------------------------------------------------- what changed


def _attribution(record: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    rows = (record.get("attribution") or {}).get("per_group")
    if rows is None:
        raise GroupTimingError("the build record carries no per-group attribution; nothing can be compared")
    return {int(r["group"]): r for r in rows}


def changed_groups(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    """Which groups' programs differ between two whole-model build records, and in what.

    A group changed when any of :data:`_IDENTITY` differs: who answered it, the named cause, the
    object's content digest, whether it gathers an operand, or which buffers it is called with. A group
    present in only one record is changed. A group answered by the library in both, for the same
    reason, is unchanged -- the library call is the target's and does not move with the package.
    """
    ours, theirs = _attribution(candidate), _attribution(baseline)
    changed, unchanged, why = [], [], {}
    for group in sorted(set(ours) | set(theirs)):
        a, b = ours.get(group), theirs.get(group)
        if a is None or b is None:
            changed.append(group)
            why[group] = "present in only one build"
            continue
        diffs = [key for key in _IDENTITY if a.get(key) != b.get(key)]
        if diffs:
            changed.append(group)
            why[group] = {key: {"candidate": a.get(key), "baseline": b.get(key)} for key in diffs}
        else:
            unchanged.append(group)
    return {
        "candidate_elf": candidate.get("elf_sha256"),
        "baseline_elf": baseline.get("elf_sha256"),
        "changed": changed,
        "unchanged": unchanged,
        "why": why,
        "kinds": {g: (ours.get(g) or theirs.get(g) or {}).get("op") for g in changed},
    }


# ------------------------------------------------------------------------------- group programs


def _prepare(
    package_dir: str | Path | None,
    *,
    model_capsule: str | Path,
    target: str,
    work: Path,
    groups: Sequence[int] | None,
    decline: Sequence[Any],
    timeout: int,
    jobs: int,
    binder: Any = None,
    ask_only: bool = False,
    allow_regions: bool = False,
) -> dict[str, Any]:
    """The statement, the package's rows (objects built for ``groups`` only), the driver's model.

    ``allow_regions`` offers a package that opts in (``whole_model_regions``) fused regions of adjacent
    groups, exactly as the whole-model build does; with ``ask_only`` only the asked groups can form one
    (every other group is declined, and a region never spans a declined group).

    ``binder`` is the corpus binding a package build states every group under (the whole-model build's
    own, :func:`merlin.perf.whole_model_build.corpus_binder`).  ``ask_only`` puts only ``groups`` to the
    package: every other group is stated by the reference, as a caller-declined group is, so a one-group
    program costs one question rather than one per group of the model.  The asked group's own statement
    is unchanged -- its interface, its binding and its operands do not depend on who answers the others."""
    from merlin.runtime.backends import base as backends

    from . import whole_model_build as W

    capsule = W.load_model_capsule(model_capsule)
    unasked: list[Any] = []
    if ask_only and groups is not None and package_dir is not None:
        stated = W.state(capsule, target=target)
        keep = {int(g) for g in groups}
        unasked = [int(r["group"]) for r in stated["whole_program"]["per_group"] if int(r["group"]) not in keep]
    buffer = W.state(
        capsule,
        target=target,
        package_dir=package_dir,
        work=work / "lower",
        timeout=timeout,
        jobs=jobs,
        decline=[*unasked, *decline] if unasked else decline,
        binder=binder,
        allow_regions=allow_regions,
    )
    rows = W.decline_ops(W.bind_groups(buffer), decline)
    wanted = {int(r["group"]) for r in rows} if groups is None else {int(g) for g in groups}
    W._kernel_objects([r for r in rows if int(r["group"]) in wanted], target=target, out=work / "objects", jobs=jobs)
    # A region's internal members stand or fall with its boundary's kernel (an object that failed to build).
    W.settle_regions(rows)
    driver = backends.whole_model_driver(target)
    (value,) = capsule.inputs.values()
    (golden,) = capsule.outputs.values()
    model = driver.program.extract(
        None,
        target,
        sources={
            "linalg": capsule.interface,
            "weights_manifest": capsule.weights_manifest,
            "weights": capsule.weights,
            "input": value,
            "golden": golden,
        },
    )
    return {"capsule": capsule, "buffer": buffer, "rows": rows, "model": model, "driver": driver}


_REFERENCE: dict[tuple[str, str, str], dict[str, Any]] = {}


def reference_context(model_capsule: str | Path, *, target: str) -> dict[str, Any]:
    """The model with NO package, computed once per capsule (by its bytes) per process.

    ``values`` (every produced tensor's reference value), ``local`` (the
    :class:`~merlin.perf.whole_model_build.LocalReference` a group is graded with) and ``oracle`` (the
    whole model's chained digests). None of it depends on a package, and each is a full reference run
    of the model, so a caller timing many groups pays for it once.
    """
    import numpy as np

    from merlin.runtime import reference as REF
    from merlin.runtime.backends import base as backends

    from . import whole_model_build as W
    from . import whole_model_oracle as O

    capsule = W.load_model_capsule(model_capsule)
    key = (target, _sha256(capsule.interface), _sha256(capsule.weights))
    if key not in _REFERENCE:
        kept = _kept_reference(key)
        if kept is not None:
            _REFERENCE[key] = kept
    if key not in _REFERENCE:
        reference, leaves, _entry = O._reference_leaves(capsule, target=target)
        whole = {k: v for k, v in reference.items() if k != "outputs"}
        held = REF.reference_outputs(whole, {n: np.asarray(v).tolist() for n, v in leaves.items()})
        oracle, _entry = O._oracle(
            capsule, target=target, digest=backends.whole_model_driver(target).program.group_digest
        )
        _REFERENCE[key] = {
            "values": {str(k): np.asarray(v) for k, v in held.items()},
            "local": W.LocalReference(reference, leaves),
            "oracle": oracle,
        }
        _keep_reference(key, _REFERENCE[key])
    return _REFERENCE[key]


def _reference_code_digest() -> str:
    """The digest of the code a reference context is computed BY: a kept one is reused only while it
    is the same code, so a fix to the reference arithmetic is never masked by an older result."""
    from merlin.runtime import reference as REF

    from . import whole_model_build as W
    from . import whole_model_oracle as O

    digest = hashlib.sha256()
    for module in (W, O, REF):
        digest.update(Path(str(module.__file__)).read_bytes())
    return digest.hexdigest()


def _kept_path(key: tuple[str, str, str]) -> Path:
    from merlin.common.artifacts import cache_dir

    name = hashlib.sha256("\0".join((*key, _reference_code_digest())).encode()).hexdigest()
    return Path(cache_dir("group-reference")) / f"{name}.pickle"


def _kept_reference(key: tuple[str, str, str]) -> dict[str, Any] | None:
    """A reference context computed earlier for the same model bytes by the same code, or None.

    The whole-model reference run costs minutes, and every one-group program pays for it before it
    can embed its inputs; it is a pure function of the capsule's bytes and this code, so it is kept.
    Regenerable and purgeable (a cache), and read only from this host's own cache directory."""
    import pickle

    try:
        path = _kept_path(key)
        return pickle.loads(path.read_bytes()) if path.is_file() else None
    except Exception:  # noqa: BLE001 -- an unreadable cache entry is recomputed, never trusted
        return None


def _keep_reference(key: tuple[str, str, str], context: Mapping[str, Any]) -> None:
    import pickle

    try:
        path = _kept_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.with_suffix(f".{os.getpid()}.tmp")
        staging.write_bytes(pickle.dumps(dict(context), protocol=pickle.HIGHEST_PROTOCOL))
        os.replace(staging, path)
    except Exception:  # noqa: BLE001 -- a cache that cannot be written only costs the next caller time
        return


def ctype_dtypes(header: str | Path) -> dict[str, str]:
    """``{C type name: numpy dtype}`` for every scalar typedef the ABI ``header`` declares, resolved
    through chains of typedefs to a fixed-width integer (:data:`merlin.runtime.sdk_facts.TYPE_SIZES`).

    The widths a program's buffers have are the header's, and a machine's header can differ from
    another's for the same name; so they are read from the header the program is built with, by
    tokens, never assumed."""
    from merlin.runtime.sdk_facts import TYPE_SIZES, strip_comments

    text = strip_comments(Path(header).read_text(encoding="utf-8", errors="replace"))
    aliases: dict[str, str] = {}
    for chunk in text.split(";"):
        # Preprocessor lines are not part of the statement: a macro's parentheses before a typedef must
        # not make the typedef look like a function type.
        statement = " ".join(line for line in chunk.splitlines() if not line.strip().startswith("#"))
        tokens = statement.split()
        if "typedef" not in tokens or "{" in statement or "(" in statement:
            continue
        rest = tokens[tokens.index("typedef") + 1 :]
        if len(rest) >= 2:
            aliases[rest[-1]] = " ".join(rest[:-1])
    found: dict[str, str] = {}
    for name in aliases:
        base, seen = aliases[name], {name}
        while base in aliases and base not in seen:
            seen.add(base)
            base = aliases[base]
        words = [w for w in base.split() if w not in ("const", "volatile", "signed")]
        unsigned = "unsigned" in words or (len(words) == 1 and words[0].startswith("uint"))
        core = " ".join(w for w in words if w != "unsigned") or "unsigned"
        if core == "char" and "signed" not in base.split() and not unsigned:
            continue  # a plain char's signedness is the ABI's choice: unknown here, so not laid out
        if core in TYPE_SIZES:
            found[name] = f"<{'u' if unsigned else 'i'}{TYPE_SIZES[core]}"
    return found


def _names(value: Any) -> list[str]:
    """Every string a step names, its members' steps included (a fused region's step nests them)."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [name for item in value.values() for name in _names(item)]
    if isinstance(value, (list, tuple)):
        return [name for item in value for name in _names(item)]
    return []


def linked_region(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """``{"members", "boundary", "id"}`` when ``row`` is a member of a fused region the package answers
    and whose kernel is linked (its boundary on the package), else None."""
    from . import whole_model_build as W

    region = (row or {}).get("region")
    if not isinstance(region, Mapping) or (row or {}).get("on") != W.ON_PACKAGE:
        return None
    members = [int(g) for g in region.get("member_groups") or ()]
    if len(members) < 2:
        return None
    return {"members": members, "boundary": members[-1], "id": region.get("id")}


def _one_group_model(
    model: Mapping[str, Any], group: int, values: Mapping[str, Any], dtypes: Mapping[str, str]
) -> dict[str, Any]:
    """The driver's model with only ``group``'s step, its produced inputs embedded as read-only data.

    What the step reads and which arrays it needs are read off the model, not named here: a step field
    naming another buffer is an input (embedded from the reference ``values``, typed by ``dtypes``),
    and an array is kept unless only OTHER steps name it (another group's weights)."""
    import numpy as np

    steps = [s for s in model["steps"] if int(s["group"]) == int(group)]
    if len(steps) != 1:
        raise GroupTimingError(f"the model has {len(steps)} step(s) for group {group}")
    step = copy.deepcopy(steps[0])
    buffers = {str(b["name"]): b for b in model["buffers"]}
    out = str(step["out"])
    named = _names(step)
    # A fused region's step carries its members' steps: what they produce inside it is the region's own,
    # never an input; what they read from outside it is.
    inside = {str(m.get("out")) for m in step.get("members") or () if isinstance(m, Mapping)}
    reads = list(dict.fromkeys(n for n in named if n in buffers and n != out and n not in inside))
    elsewhere = {v for s in model["steps"] if s is not steps[0] for v in _names(s)}
    arrays = {k: v for k, v in model["arrays"].items() if k in named or k not in elsewhere}
    for name in reads:
        if name not in values:
            raise GroupTimingError(f"group {group} reads {name!r}, which the reference statement does not produce")
        ctype = str(buffers[name].get("ctype"))
        if ctype not in dtypes:
            raise GroupTimingError(
                f"the program's header declares no fixed-width type {ctype!r} for {name!r}; "
                "its bytes cannot be laid out"
            )
        flat = np.asarray(values[name]).reshape(-1)
        if flat.size != int(buffers[name]["elements"]):
            raise GroupTimingError(
                f"the reference holds {flat.size} element(s) of {name!r}, the program {buffers[name]['elements']}"
            )
        arrays[name] = np.ascontiguousarray(flat.astype(np.dtype(dtypes[ctype])))
    # The driver closes its program on the model's final dequantize and argmax. A one-step program
    # has no classifier, so it closes on one element of its own output at unit scale -- outside the
    # measured bracket, and never read as a classification.
    if step.get("dequantize") is None:
        step["dequantize"] = 1.0
    return {
        **{k: v for k, v in model.items() if k not in ("steps", "buffers", "arrays")},
        "steps": [step],
        "buffers": [buffers[out]],
        "arrays": arrays,
        "classes": 1,
        "device_groups": 1,
        "embedded_inputs": reads,
    }


def _graded_as(buffer: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    """How each group is graded, from the statement's own entry (as the whole-model memory map does)."""
    rows: dict[int, dict[str, Any]] = {}
    for row in (buffer.get("whole_program") or {}).get("per_group") or ():
        entry = row.get("entry") or {}
        if entry.get("bound_lsb") is not None:
            rows[int(row["group"])] = {
                "compare": "bounded",
                "bound_lsb": int(entry["bound_lsb"]),
                "lhs_scale": float(entry["lhs_scale"]),
                "rhs_scale": float(entry["rhs_scale"]),
                "relu": "relu" in (entry.get("epilogue") or ()),
            }
        else:
            rows[int(row["group"])] = {"compare": "exact"}
    return rows


def _addresses(elf: Path) -> dict[str, int]:
    """Every defined symbol's address, sized or not (the embedded arrays are unsized labels)."""
    from merlin.llvmlower import toolchain

    listed = subprocess.run(
        [str(toolchain.nm()), "--defined-only", "--radix=d", str(elf)], check=True, capture_output=True, text=True
    ).stdout
    found: dict[str, int] = {}
    for line in listed.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0].isdigit():
            found[parts[2]] = int(parts[0])
    return found


#: ``exactness(group, entry)``: a group's exactness contract (a dict), or None for the comparison its op implies.
ContractOf = Callable[[int, Mapping[str, Any]], Mapping[str, Any] | None]


def _entries(buffer: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    """Each group's statement entry (what its form, and so its exactness contract, is read from)."""
    return {
        int(r["group"]): dict(r.get("entry") or {}) for r in (buffer.get("whole_program") or {}).get("per_group") or ()
    }


def _contract(exactness: ContractOf | None, group: int, entries: Mapping[int, Mapping[str, Any]]):
    if exactness is None:
        return None
    found = exactness(int(group), entries.get(int(group)) or {})
    return dict(found) if found else None


def _group_map(
    elf: Path,
    one: Mapping[str, Any],
    full_row: Mapping[str, Any],
    producers: Mapping[str, int],
    *,
    contract: Mapping[str, Any] | None = None,
    stated: Mapping[str, Any] | None = None,
) -> dict:
    """A memory map (the build's own schema) for the one group: its output, and its embedded inputs --
    and, given one, the exactness ``contract`` the grade holds it to.  A fused region's program is mapped
    against the ``stated`` buffer, whose per-group entries say how its boundary is graded."""
    from . import whole_model_build as W

    layout = W.memory_map(elf, one, stated if stated is not None else {"whole_program": {"per_group": []}})
    (row,) = layout["groups"]
    addresses = _addresses(elf)

    def place(name: str) -> dict[str, Any]:
        array = one["arrays"][name]
        if name not in addresses:
            raise GroupTimingError(f"the linked program defines no symbol {name!r}")
        return {
            "symbol": name,
            "address": addresses[name],
            "bytes": int(array.nbytes),
            "elements": int(array.size),
            "element_bytes": int(array.itemsize),
        }

    row["inputs"] = [
        {**place(name), "produced_by": producers[name]} for name in one["embedded_inputs"] if name in producers
    ]
    if row.get("compare") != "region":  # a region is graded at its boundary from its members (memory_map)
        for key in ("compare", "bound_lsb", "lhs_scale", "rhs_scale", "relu"):
            if key in full_row:
                row[key] = full_row[key]
    if row.get("compare") == "bounded":
        step = one["steps"][0]
        row["lhs"], row["rhs"] = place(str(step["lhs"])), place(str(step["rhs"]))
    layout["dump"] = {"symbols": [row["symbol"]], "bytes": row["bytes"]}
    layout["note"] = "one-group program: inputs are embedded read-only data served from the ELF"
    if contract:
        row["exactness"] = dict(contract)
    return layout


def build_group_programs(
    package_dir: str | Path | None,
    groups: Sequence[int] | None,
    *,
    model_capsule: str | Path,
    target: str,
    machine: str,
    header: str | Path,
    header_sha256: str | None = None,
    out: str | Path,
    decline: Sequence[Any] = (),
    expect_objects: Mapping[int, str] | None = None,
    timeout: int = 600,
    jobs: int = 8,
    verify: str = "host_dump",
    prohibited_roles: Sequence[str] = (),
    harness_overrides: Sequence[str | Path] = (),
    ask_only: bool = False,
    phase0_recipe: str | Path | None = None,
    descriptor: str | Path | None = None,
    keep_statement: bool = False,
    exactness: ContractOf | None = None,
    allow_regions: bool = False,
    debug_companion: bool = False,
) -> dict[int, dict[str, Any]]:
    """One small program per group in ``groups``, built for ``machine``: ``{group: record}``.

    ``allow_regions`` lets the package answer adjacent asked groups as ONE fused region (see
    :func:`_prepare`).  A linked region is one program -- the driver's own region step
    (``program.region_steps``), graded at its boundary -- recorded on the boundary's record; each
    internal member's record names the boundary it is timed with (``timed_with``) and builds nothing, so a
    region's cycles are counted once and every member stays the package's.  A region the driver will not
    step is refused by name for each member, never silently split.

    ``exactness(group, entry)`` gives a group's exactness contract (:meth:`merlin.perf.exactness.
    Exactness.to_dict`) from its statement entry; it rides in the group's memory map, so the grade holds
    the group to exactly that contract and says so (:func:`merlin.perf.whole_model_memory.grade_memory`).

    Built the way the whole-model program it stands for is: under the corpus binding ``phase0_recipe``
    declares (required with a package, as for the whole-model build), with ``harness_overrides``
    replacing their namesakes in the harness tree, and refused unless the compiler READ the asserted
    header and each override.  The interface capsule the statement put to the package for the group is
    kept beside its program (``interface``); ``ask_only`` asks the package for ``groups`` alone.

    ``verify`` is the driver's verification mode (``host_dump`` for the emulator's memory dump; a
    functional-model check builds ``local_map``, which grades on the core and says where it is wrong).
    ``prohibited_roles`` builds the program under the same instruction rule as the run's own builds.
    ``keep_statement`` keeps the statement's work tree (``lower/``: each asked group's interface, the
    package's command buffer and target artifact), which is otherwise removed; an inspection reads it.
    ``debug_companion`` also links each program a second time from the same model, kernels and recipe
    with debug information added (:func:`_debug_companion`), so a PC count of the program can be read
    against source lines; the companion is recorded beside the program and never replaces it.

    Each record names the ELF, its memory map, the oracle (the whole model's, for the chained digest),
    who answered the group and the object's digest. ``expect_objects`` (``{group: object_sha256}``, from
    the build record the groups were diffed on) makes a rebuilt object that differs from the one the
    record named a REFUSAL for that group: the timing would otherwise be of a different kernel.
    """
    from merlin.runtime.backends import base as backends

    from . import whole_model_build as W

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    abi = W.machine_header(machine, header, header_sha256)
    binding = W.corpus_binder(target, phase0_recipe=phase0_recipe, descriptor=descriptor) if package_dir else None
    ctx = _prepare(
        package_dir,
        model_capsule=model_capsule,
        target=target,
        work=out,
        groups=groups,
        decline=decline,
        timeout=timeout,
        jobs=jobs,
        binder=binding.binder if binding is not None else None,
        ask_only=ask_only,
        allow_regions=allow_regions,
    )
    overrides = {str(Path(o).resolve()): W._sha256(o) for o in harness_overrides}
    reference = reference_context(model_capsule, target=target)
    values = reference["values"]
    (out / "oracle.json").write_text(json.dumps(reference["oracle"], indent=1) + "\n", encoding="utf-8")
    full_rows = _graded_as(ctx["buffer"])
    entries = _entries(ctx["buffer"])
    dtypes = ctype_dtypes(header)
    producers = {str(s["out"]): int(s["group"]) for s in ctx["model"]["steps"]}
    if groups is None:  # every DEVICE group of the model: the steps of its program
        groups = [int(s["group"]) for s in ctx["model"]["steps"]]
    rows = {int(r["group"]): r for r in ctx["rows"]}
    domain = ctx["buffer"]["whole_program"]["input_domain"]
    records: dict[int, dict[str, Any]] = {}
    for group in groups:
        here = out / f"g{int(group)}"
        row = rows.get(int(group)) or {}
        record: dict[str, Any] = {
            "group": int(group),
            "on": row.get("on"),
            "cause": row.get("cause"),
            # The package's own words when it did not answer (its decline reason, an exception it raised):
            # the agent reads its own error, never a paraphrase of it.
            "why": row.get("why"),
        }
        region = linked_region(row)
        if region is not None:
            record["region"] = region
            if int(group) != region["boundary"]:
                # AN INTERNAL MEMBER: answered by its region's one kernel, timed in the boundary's program.
                record.update(linked="submission", timed_with=region["boundary"])
                records[int(group)] = record
                continue
        try:
            want = (expect_objects or {}).get(int(group))
            if want and row.get("object_sha256") != want:
                raise GroupTimingError(
                    f"the rebuilt object is {str(row.get('object_sha256'))[:12]}, the build record named "
                    f"{str(want)[:12]}; timing it would time a different kernel"
                )
            stepped, member_rows = ctx["model"], ([row] if row else [])
            if region is not None:
                stepped, refused = ctx["driver"].program.region_steps(
                    ctx["model"], [{"members": region["members"], "boundary": region["boundary"]}]
                )
                if refused:
                    raise GroupTimingError(
                        f"the target's driver will not step the region {region['members']}: {refused}"
                    )
                member_rows = [rows[m] for m in region["members"] if m in rows]
            one = _one_group_model(stepped, int(group), values, dtypes)
            kernels = ctx["driver"].kernels.render_kernels(
                one,
                member_rows,
                entry=domain["tensor"],
                row_padding=W.pointee_row_padding(target)["multiple"],
            )
            recipe = W._with_header(
                backends.harness_build_recipe(target), Path(header), here / "harness", [Path(o) for o in overrides]
            )
            rule = (
                {"library_loops": False, "prohibited_selectors": sorted(W._prohibited(target, prohibited_roles))}
                if prohibited_roles
                else {}
            )

            def link(directory: Path, chosen: Any, one=one, kernels=kernels, rule=rule) -> Mapping[str, Any]:
                return ctx["driver"].program.build(
                    one,
                    None,
                    None,
                    directory,
                    sched_kernels=kernels,
                    extra_objects=[Path(o) for o in kernels["objects"]],
                    recipe=chosen,
                    verify=verify,
                    **rule,
                )

            receipt = link(here / "program", recipe)
            _require_headers_read(receipt, abi, overrides)
            layout = _group_map(
                Path(receipt["elf"]),
                one,
                full_rows[int(group)],
                producers,
                contract=_contract(exactness, group, entries),
                stated=ctx["buffer"] if region is not None else None,
            )
            (here / "memory_map.json").write_text(json.dumps(layout, indent=1) + "\n", encoding="utf-8")
            record.update(_kept_interface(out / "lower", int(group), here))
            census = {int(c["group"]): c for c in kernels["census"]}.get(int(group)) or {}
            record.update(
                {
                    "elf": receipt["elf"],
                    "elf_sha256": receipt["elf_sha256"],
                    "memory_map": str(here / "memory_map.json"),
                    "oracle": str(out / "oracle.json"),
                    "linked": census.get("on"),
                    "object_sha256": row.get("object_sha256"),
                    "gather": bool(census.get("gather")),
                    "abi_header": abi,
                    "harness_overrides": overrides,
                    "host_routed": receipt.get("host_routed") or [],
                    "library_paths": receipt.get("library_paths"),
                    "program_source": receipt.get("program_source"),
                    "program_object": receipt.get("program_object"),
                    "variant": _board_variant(receipt),
                }
            )
            if debug_companion:
                record["debug_companion"] = _debug_companion(link, recipe, here / "program.debug")
        except (Exception, SystemExit) as error:  # noqa: BLE001 -- a group that cannot be built is a named refusal
            # The driver reports a failed compile or link as SystemExit; a group whose program does not
            # build is a refusal for that group, never the end of every other group's build.
            record["refusal"] = f"{type(error).__name__}: {str(error)[-600:]}"
        records[int(group)] = record
    (out / "group_programs.json").write_text(json.dumps(records, indent=1, default=str) + "\n", encoding="utf-8")
    # The statement's work tree (every group's interface and the package's reply to it, hundreds of MB
    # for a ResNet) is regenerable -- the replies are cached by content -- and nothing reads it again.
    import shutil

    if not keep_statement:
        shutil.rmtree(out / "lower", ignore_errors=True)
    return records


#: The C compiler option that adds debug information (POSIX ``c99 -g``). It asks the compiler to describe
#: the code it emits, not to emit different code; :func:`merlin.perf.debug_companion.verify_debug_companion`
#: refuses a companion whose allocated bytes differ anyway, so a compiler that did change them is caught.
DEBUG_INFO_OPTION = "-g"


def _debug_companion(link: Callable[[Path, Any], Mapping[str, Any]], recipe: Any, directory: Path) -> dict[str, Any]:
    """The program linked again by ``link`` from the same inputs, under ``recipe`` with debug information.

    Every compile and link flag the program was built with is kept, in order, and the debug option is
    appended to them, so the companion is the recorded recipe plus debug information and nothing else.
    A companion that cannot be built is a refusal recorded here; the program itself stands either way."""
    import dataclasses

    try:
        chosen = dataclasses.replace(recipe, cflags=(*recipe.cflags, DEBUG_INFO_OPTION))
        receipt = link(directory, chosen)
    except (Exception, SystemExit) as error:  # noqa: BLE001 -- a companion that does not build is named, not fatal
        return {"refusal": f"{type(error).__name__}: {str(error)[-400:]}", "option": DEBUG_INFO_OPTION}
    return {
        "elf": receipt.get("elf"),
        "elf_sha256": receipt.get("elf_sha256"),
        "program_object": receipt.get("program_object"),
        "compiler": str(chosen.compiler),
        "option": DEBUG_INFO_OPTION,
        "cflags": list(chosen.cflags),
    }


def _program_files(receipt: Mapping[str, Any]) -> tuple[Path, Path]:
    """The rendered program and its relocatable, as the target's driver NAMES them in its receipt."""
    source, obj = receipt.get("program_source"), receipt.get("program_object")
    if not source or not obj:
        raise GroupTimingError("the target driver's receipt names no program_source/program_object")
    return Path(str(source)), Path(str(obj))


def _require_headers_read(receipt: Mapping[str, Any], abi: Mapping[str, Any], overrides: Mapping[str, str]) -> None:
    """Refuse a program the compiler did not build from the asserted header and every override."""
    from . import whole_model_build as W

    source, _obj = _program_files(receipt)
    read = W._headers_read(receipt, source)
    if abi.get("sha256") not in read.values():
        raise GroupTimingError(
            f"the program did not read the asserted parameter header {str(abi.get('sha256'))[:12]}; "
            f"it read {sorted(set(read.values()))[:4]}"
        )
    unread = [path for path, digest in overrides.items() if digest not in read.values()]
    if unread:
        raise GroupTimingError(f"the program did not read the harness override(s) {unread}")


def _kept_interface(lower: Path, group: int, here: Path) -> dict[str, Any]:
    """Copy the interface capsule the statement put to the package for ``group`` beside its program.

    The statement writes one ``g<group>.iface.mlir`` per group it asked; a group nothing was asked for
    (no package, or declined before the package) has none, and says so rather than carrying a path to
    a file that is about to be removed."""
    import shutil

    source = Path(lower) / f"g{int(group)}.iface.mlir"
    if not source.is_file():
        return {"interface": None, "interface_why": "the statement asked no package for this group"}
    here.mkdir(parents=True, exist_ok=True)
    kept = here / "interface.mlir"
    shutil.copyfile(source, kept)
    reply = Path(lower) / f"g{int(group)}.generated" / "command_buffer.json"
    found: dict[str, Any] = {"interface": str(kept), "interface_sha256": _sha256(kept)}
    if reply.is_file():
        shutil.copyfile(reply, here / "command_buffer.json")
        found["command_buffer"] = str(here / "command_buffer.json")
    return found


def _board_variant(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """What a batch needs to link this program beside others (the whole-model service's
    ``board_request.json`` ``variant`` shape), from the driver's receipt."""
    _source, obj = _program_files(receipt)
    return {
        "program_object": str(obj),
        "objects": [str(o.get("path")) for o in receipt.get("linked_objects") or ()],
        "supports": [str(s) for s in receipt.get("support_objects") or ()],
        "program": {
            "source_sha256": receipt.get("program_sha256"),
            "compiler": receipt.get("compiler"),
            "flags": receipt.get("flags"),
            "link_flags": receipt.get("link_flags"),
            "link_script": receipt.get("link_script"),
        },
    }


def build_reference_group_programs(
    groups: Sequence[int] | None,
    *,
    model_capsule: str | Path,
    target: str,
    machine: str,
    header: str | Path,
    header_sha256: str | None = None,
    out: str | Path,
    verify: str = "host_dump",
    harness_overrides: Sequence[str | Path] = (),
    exactness: ContractOf | None = None,
) -> dict[int, dict[str, Any]]:
    """The REFERENCE arm's one-group programs: the target's library answers the group (graded under the
    same ``exactness`` contract as the package arm, see :func:`build_group_programs`).

    :func:`merlin.perf.whole_model_builder.build_reference` restricted to one step -- the same driver,
    header assertion, harness overrides and no instruction rule (the reference arm is the bar, and the
    bar is the library as the vendor ships it) -- over the same one-group model and embedded inputs
    :func:`build_group_programs` builds a package's program from, so the two arms of one group differ
    only in who answered it."""
    from merlin.runtime.backends import base as backends

    from . import whole_model_build as W

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    abi = W.machine_header(machine, header, header_sha256)
    capsule = W.load_model_capsule(model_capsule)
    driver = backends.whole_model_driver(target)
    (value,) = capsule.inputs.values()
    (golden,) = capsule.outputs.values()
    model = driver.program.extract(
        None,
        target,
        sources={
            "linalg": capsule.interface,
            "weights_manifest": capsule.weights_manifest,
            "weights": capsule.weights,
            "input": value,
            "golden": golden,
        },
    )
    reference = reference_context(model_capsule, target=target)
    (out / "oracle.json").write_text(json.dumps(reference["oracle"], indent=1) + "\n", encoding="utf-8")
    stated = W.state(capsule, target=target)
    full_rows = _graded_as(stated)
    entries = _entries(stated)
    dtypes = ctype_dtypes(header)
    producers = {str(s["out"]): int(s["group"]) for s in model["steps"]}
    if groups is None:
        groups = [int(s["group"]) for s in model["steps"]]
    overrides = {str(Path(o).resolve()): W._sha256(o) for o in harness_overrides}
    records: dict[int, dict[str, Any]] = {}
    for group in groups:
        here = out / f"g{int(group)}"
        record: dict[str, Any] = {"group": int(group), "on": "vendor", "arm": "reference"}
        try:
            one = _one_group_model(model, int(group), reference["values"], dtypes)
            recipe = W._with_header(
                backends.harness_build_recipe(target), Path(header), here / "harness", [Path(o) for o in overrides]
            )
            receipt = driver.program.build(one, None, None, here / "program", recipe=recipe, verify=verify)
            _require_headers_read(receipt, abi, overrides)
            layout = _group_map(
                Path(receipt["elf"]),
                one,
                full_rows[int(group)],
                producers,
                contract=_contract(exactness, group, entries),
            )
            (here / "memory_map.json").write_text(json.dumps(layout, indent=1) + "\n", encoding="utf-8")
            record.update(
                {
                    "elf": receipt["elf"],
                    "elf_sha256": receipt["elf_sha256"],
                    "memory_map": str(here / "memory_map.json"),
                    "oracle": str(out / "oracle.json"),
                    "linked": "vendor",
                    "call": driver.program._call(one["steps"][0]).split("(", 1)[0].strip(),
                    "abi_header": abi,
                    "harness_overrides": overrides,
                    "host_routed": receipt.get("host_routed") or [],
                    "library_paths": receipt.get("library_paths"),
                    "program_source": receipt.get("program_source"),
                    "variant": _board_variant(receipt),
                }
            )
        except (Exception, SystemExit) as error:  # noqa: BLE001 -- a group that cannot be built is a named refusal
            record["refusal"] = f"{type(error).__name__}: {str(error)[-600:]}"
        records[int(group)] = record
    (out / "group_programs.json").write_text(json.dumps(records, indent=1, default=str) + "\n", encoding="utf-8")
    return records


# ---------------------------------------------------------------------------------------- timing


def _slots_free(max_parallel: int, running: int, *, min_parallel: int = 2) -> bool:
    """Room for one more emulator: under the cap, and within what the OTHER load leaves free.

    The emulator is single-threaded. The one-minute load average counts this caller's own runs too, so
    the room is the core count minus the load everyone else puts on the host; ``min_parallel`` runs are
    always allowed, so a busy host slows the tier down rather than stopping it.
    """
    if running >= max_parallel:
        return False
    if running < min_parallel:
        return True
    try:
        others = os.getloadavg()[0] - running
    except OSError:
        return True
    return running + 1 <= (os.cpu_count() or 1) - others


def time_group_programs(
    programs: Mapping[int, Mapping[str, Any]],
    *,
    target: str,
    model_capsule: str | Path,
    out: str | Path,
    max_parallel: int = 8,
    max_cycles: int = 60_000_000,
    timeout_s: float = 6 * 3600,
    emulator: str | Path | None = None,
    cache: str | Path | None = "default",
) -> dict[int, dict[str, Any]]:
    """Run each program on the dump-capable emulator; ``{group: {cycles, correct, grade, wall_s, ...}}``.

    Correctness is the group's LOCAL grade over its dumped output. A refused run (no line, no dump, a
    timeout) is reported as refused with its reason and carries no cycles.

    A GRADED run is kept in ``cache`` keyed by the program's bytes and the emulator's bytes, so the
    baseline side of the next comparison (usually the same best) is not paid for again; a carried
    row says so (``carried: true``). ``cache=None`` turns it off.
    """
    from . import whole_model_gsim as G

    out = Path(out)
    local = _local_for(model_capsule, target)
    engine = None
    if cache == "default":
        from merlin.common.artifacts import cache_dir

        cache = cache_dir("group-timing")
    try:
        engine = G._engine(target, emulator)["sha256"]
    except Exception:  # noqa: BLE001 -- no engine: every run refuses below, nothing is carried
        cache = None
    todo = [(g, p) for g, p in sorted(programs.items()) if not p.get("refusal")]
    results: dict[int, dict[str, Any]] = {
        g: {"group": g, "status": "refused", "refusal": p["refusal"]} for g, p in programs.items() if p.get("refusal")
    }

    def carried(program: Mapping[str, Any]) -> Path | None:
        if cache is None or not program.get("elf_sha256") or not engine:
            return None
        # A grade is a function of the program AND of the contract it was held to: a contract (in the
        # memory map, not the ELF) keys its own entry, so a changed contract is never served an old grade.
        try:
            rows = json.loads(Path(str(program.get("memory_map"))).read_text(encoding="utf-8")).get("groups") or []
        except (OSError, ValueError):
            rows = []
        contract = next((r.get("exactness") for r in rows if isinstance(r, Mapping) and r.get("exactness")), None)
        suffix = (
            "." + hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:16] if contract else ""
        )
        return Path(cache) / engine[:16] / f"{program['elf_sha256']}{suffix}.json"

    def one(group: int, program: Mapping[str, Any]) -> dict[str, Any]:
        kept = carried(program)
        if kept is not None and kept.is_file():
            row = json.loads(kept.read_text(encoding="utf-8"))
            return {**row, "group": group, "carried": True}
        started = time.monotonic()
        verdict = G.run_gsim_whole_model(
            program["elf"],
            program["memory_map"],
            program["oracle"],
            target=target,
            out=out / f"g{group}",
            emulator=emulator,
            max_cycles=max_cycles,
            timeout_s=timeout_s,
            local=local,
        )
        row = {
            "group": group,
            "status": verdict.get("status"),
            "refusal": verdict.get("refusal"),
            "wall_s": round(time.monotonic() - started, 1),
            "elf_sha256": program.get("elf_sha256"),
            "emulator_sha256": (verdict.get("emulator") or {}).get("sha256"),
            "machine": (verdict.get("machine") or {}).get("registry_entry"),
        }
        if verdict.get("status") == "graded":
            (entry,) = verdict["per_group"]
            grade = verdict.get("grade") or {}
            row.update(
                {
                    "kind": entry["kind"],
                    "cycles": entry["cycles"],
                    "correct": bool(entry["correct"]),
                    "failure": (grade.get("disagree") or grade.get("unverified") or [None])[0],
                    "chained_digest_agrees": str(group) in ((grade.get("chained") or {}).get("agree") or ()),
                    # Which exactness contract the grade held the group to, and its own numbers.
                    "exactness": (grade.get("contracts") or {}).get(str(group)),
                    "evidence": (grade.get("evidence") or {}).get(str(group)),
                }
            )
            if kept is not None:
                kept.parent.mkdir(parents=True, exist_ok=True)
                kept.write_text(json.dumps(row, indent=1) + "\n", encoding="utf-8")
        return row

    with ThreadPoolExecutor(max_workers=max(1, max_parallel)) as pool:
        pending, running = list(todo), {}
        while pending or running:
            while pending and _slots_free(max_parallel, len(running)):
                group, program = pending.pop(0)
                running[group] = pool.submit(one, group, program)
            finished = [g for g, f in running.items() if f.done()]
            for group in finished:
                try:
                    results[group] = running.pop(group).result()
                except Exception as error:  # noqa: BLE001 -- recorded per group, never dropped
                    results[group] = {
                        "group": group,
                        "status": "refused",
                        "refusal": f"{type(error).__name__}: {error}",
                    }
            if not finished:
                time.sleep(5)
    return results


def _local_for(model_capsule: str | Path, target: str) -> Any:
    try:
        return reference_context(model_capsule, target=target)["local"]
    except Exception:  # noqa: BLE001 -- no local reference: an exact group is reported unverified
        return None


def changed_group_timing(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    target: str,
    model_capsule: str | Path,
    machine: str,
    header: str | Path,
    header_sha256: str | None = None,
    out: str | Path,
    groups: Sequence[int] | None = None,
    max_parallel: int = 8,
    jobs: int = 8,
    phase0_recipe: str | Path | None = None,
    descriptor: str | Path | None = None,
    harness_overrides: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Tier B end to end: the changed groups of ``candidate`` against ``baseline``, both timed.

    ``candidate`` / ``baseline`` are ``{"package": dir or None, "record": build record, "decline": [...]}``
    (the record is the whole-model build record the groups are diffed on). ``groups`` overrides the
    diff. Returns per-group rows (candidate and baseline cycles on the emulator, the ratio, the
    direction, each side's local correctness) with the device labelled as its own.
    """
    out = Path(out)
    diff = changed_groups(candidate["record"], baseline["record"])
    chosen = [int(g) for g in (groups if groups is not None else diff["changed"])]
    started = time.monotonic()
    sides: dict[str, dict[int, dict[str, Any]]] = {}
    built: dict[str, dict[int, dict[str, Any]]] = {}
    for name, side in (("candidate", candidate), ("baseline", baseline)):
        objects = {g: r.get("object_sha256") for g, r in _attribution(side["record"]).items() if r.get("object_sha256")}
        built[name] = build_group_programs(
            side.get("package"),
            chosen,
            model_capsule=model_capsule,
            target=target,
            machine=machine,
            header=header,
            header_sha256=header_sha256,
            out=out / name,
            decline=side.get("decline") or (),
            expect_objects=None if side.get("allow_object_drift") else objects,
            jobs=jobs,
            phase0_recipe=phase0_recipe,
            descriptor=descriptor,
            harness_overrides=harness_overrides,
        )
    build_s = round(time.monotonic() - started, 1)
    together = {("candidate", g): p for g, p in built["candidate"].items()}
    together.update({("baseline", g): p for g, p in built["baseline"].items()})
    keyed = {index: program for index, program in enumerate(together.values())}
    names = dict(enumerate(together))
    timed = time_group_programs(
        keyed, target=target, model_capsule=model_capsule, out=out / "runs", max_parallel=max_parallel
    )
    for index, row in timed.items():
        side, group = names[index]
        sides.setdefault(side, {})[group] = {**row, "group": group}
    rows = []
    for group in chosen:
        a, b = sides.get("candidate", {}).get(group, {}), sides.get("baseline", {}).get(group, {})
        row = {
            "group": group,
            "kind": a.get("kind") or b.get("kind") or diff["kinds"].get(group),
            "candidate_cycles": a.get("cycles"),
            "baseline_cycles": b.get("cycles"),
            "candidate_correct": a.get("correct"),
            "baseline_correct": b.get("correct"),
            "candidate_refusal": a.get("refusal"),
            "baseline_refusal": b.get("refusal"),
            "wall_s": {"candidate": a.get("wall_s"), "baseline": b.get("wall_s")},
            "why_changed": diff["why"].get(group),
        }
        if row["candidate_cycles"] and row["baseline_cycles"]:
            row["ratio"] = round(row["candidate_cycles"] / row["baseline_cycles"], 4)
            row["direction"] = "faster" if row["ratio"] < 1 else "slower" if row["ratio"] > 1 else "unchanged"
        rows.append(row)
    document = {
        "schema": SCHEMA,
        "device": DEVICE_LABEL,
        "feeds_objective": False,
        "target": target,
        "machine": machine,
        "diff": diff,
        "groups": rows,
        "build_wall_s": build_s,
        "total_wall_s": round(time.monotonic() - started, 1),
        "max_parallel": max_parallel,
    }
    (out / "group_timing.json").write_text(json.dumps(document, indent=1, default=str) + "\n", encoding="utf-8")
    return document


# ------------------------------------------------------------------------------------ validation


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 3:
        return None

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2.0
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return round(cov / (vx * vy), 4) if vx and vy else None


def direction_agreement(
    rows: Sequence[Mapping[str, Any]],
    board: Mapping[int, tuple[int, int]],
    *,
    threshold: float = 0.05,
    poor: float = 0.7,
) -> dict[str, Any]:
    """How often the emulator's per-group direction matches the board's, overall and per kind.

    ``rows`` are :func:`changed_group_timing` rows; ``board`` is ``{group: (candidate, baseline)}`` board
    cycles for the same two programs. Only groups whose BOARD cycles moved by more than ``threshold``
    count (a change inside the noise has no direction). A kind agreeing on fewer than ``poor`` of its
    groups is flagged ``direction_unreliable``. Also the rank correlation of the two log-ratios.
    """
    import math

    counted, agree, by_kind = [], 0, {}
    for row in rows:
        group = int(row["group"])
        if group not in board or not row.get("ratio"):
            continue
        cand, base = board[group]
        if not cand or not base:
            continue
        board_ratio = cand / base
        if abs(board_ratio - 1) <= threshold:
            continue
        same = (board_ratio < 1) == (float(row["ratio"]) < 1)
        agree += same
        kind = str(row.get("kind"))
        slot = by_kind.setdefault(kind, {"n": 0, "agree": 0})
        slot["n"] += 1
        slot["agree"] += int(same)
        counted.append((math.log(float(row["ratio"])), math.log(board_ratio)))
    for slot in by_kind.values():
        slot["agreement"] = round(slot["agree"] / slot["n"], 3)
        slot["direction_unreliable"] = slot["agreement"] < poor
    return {
        "groups_counted": len(counted),
        "threshold": threshold,
        "agreement": round(agree / len(counted), 3) if counted else None,
        "spearman_log_ratio": _spearman([c[0] for c in counted], [c[1] for c in counted]),
        "by_kind": by_kind,
    }


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
