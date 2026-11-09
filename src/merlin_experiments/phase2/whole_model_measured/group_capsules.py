"""Per-group perf capsules: named groups of a model timed ALONE, both arms, the same way -- and the
cell machine's real measurer (:class:`GroupProgramMeasurer`).

A per-group perf capsule is the whole-model program with one step: the group the statement put to the
package, at model shapes, on the values the model hands it, graded against the reference recomputed
from those inputs (:func:`merlin.perf.whole_model_group_timing.build_group_programs`).  The REFERENCE
arm is the same one-step program with the target's library answering the group
(:func:`~merlin.perf.whole_model_group_timing.build_reference_group_programs`), so the two arms of one
group differ only in who answered it.

Each arm's recipe (machine, header, harness overrides, instruction rule, corpus binding) is the
whole-model build options of that arm (:func:`arm_from_options`), so a per-group number is measured
under the recipe the whole-model number it is compared with was.  Every package program's WHOLE ELF is
held to the arm's instruction rule (:func:`isa_scan`); programs are timed on the elaborated-RTL emulator
(:func:`time_on_gsim`), graded exactly from a memory dump.  :func:`validation_table` sets a standalone
count beside the group's in-model count on the same device, with the signed offset, so a systematic
offset is visible as one.

How a program is generated is the target's (its whole-model driver, in its support provider); nothing
here names a target, a machine or a group.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "merlin_group_perf_capsules_v1"
ARM_PACKAGE = "package"
ARM_REFERENCE = "reference"
DEVICE = "elaborated_rtl"
_PROHIBITED = "prohibited_roles"


class GroupCapsuleError(RuntimeError):
    """A per-group perf capsule cannot be built, run or admitted, and the message says which part."""


@dataclass(frozen=True)
class Arm:
    """How one arm's programs are built, as the whole-model build of that arm built its own."""

    name: str
    machine: str
    header: str
    header_sha256: str | None = None
    harness_overrides: tuple[str, ...] = ()
    prohibited_roles: tuple[str, ...] = ()
    phase0_recipe: str | None = None
    descriptor: str | None = None
    source: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)
    #: The sealed Phase 0 instruction policy the arm's ``prohibited_roles`` are held to (the job's own).
    instruction_policy: Mapping[str, Any] | None = None
    #: Whether the package arm may answer adjacent groups as one fused region (the whole-model build
    #: option of the same name; the reference arm is the library and never claims one).
    allow_regions: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "machine": self.machine,
            "header": self.header,
            "header_sha256": self.header_sha256,
            "harness_overrides": list(self.harness_overrides),
            "prohibited_roles": list(self.prohibited_roles),
            "phase0_recipe": self.phase0_recipe,
            "source": self.source,
            "allow_regions": self.allow_regions,
        }


def arm_from_options(
    options: Mapping[str, Any],
    *,
    name: str,
    source: str | None = None,
    instruction_policy: Mapping[str, Any] | None = None,
) -> Arm:
    """An arm from whole-model ``build_options``.

    The reference arm is the bar and is built under no instruction rule, exactly as the service exempts
    the reference from its instruction gate; reference options that declare prohibited roles are refused
    rather than silently measured under a different rule than the bar they stand for."""
    if not isinstance(options, Mapping):
        raise GroupCapsuleError(f"the {name} arm has no build options")
    missing = [key for key in ("machine", "header") if not options.get(key)]
    if missing:
        raise GroupCapsuleError(f"the {name} arm's build options name no {missing}")
    roles = tuple(str(r) for r in options.get(_PROHIBITED) or ())
    if name == ARM_REFERENCE and roles:
        raise GroupCapsuleError("reference options declaring prohibited roles are not the bar; refused")
    return Arm(
        name=name,
        machine=str(options["machine"]),
        header=str(options["header"]),
        header_sha256=options.get("header_sha256"),
        harness_overrides=tuple(str(o) for o in options.get("harness_overrides") or ()),
        prohibited_roles=roles,
        phase0_recipe=str(options["phase0_recipe"]) if options.get("phase0_recipe") else None,
        descriptor=str(options["descriptor"]) if options.get("descriptor") else None,
        source=source,
        extra={"model_capsule": options.get("model_capsule"), "verify": options.get("verify")},
        instruction_policy=dict(instruction_policy) if instruction_policy else None,
        allow_regions=bool(options.get("allow_regions")) and name != ARM_REFERENCE,
    )


def arms_from_jobs(package_job: str | Path, reference_job: str | Path) -> dict[str, Arm]:
    """Both arms from two whole-model ``job.json`` files, refused unless they are for one machine."""
    loaded = {}
    for name, path in ((ARM_PACKAGE, package_job), (ARM_REFERENCE, reference_job)):
        job = json.loads(Path(path).read_text(encoding="utf-8"))
        loaded[name] = arm_from_options(
            job.get("build_options") or {},
            name=name,
            source=str(path),
            instruction_policy=job.get("instruction_policy") if name == ARM_PACKAGE else None,
        )
    if loaded[ARM_PACKAGE].machine != loaded[ARM_REFERENCE].machine:
        raise GroupCapsuleError(
            f"the arms name different machines ({loaded[ARM_PACKAGE].machine} vs {loaded[ARM_REFERENCE].machine}); "
            "a per-group bar is only comparable on the machine the candidate is measured on"
        )
    return loaded


# ------------------------------------------------------------------------------------------ build


def build_arm_programs(
    arm: Arm,
    groups: Sequence[int],
    *,
    package_dir: str | Path | None,
    model_capsule: str | Path,
    target: str,
    out: str | Path,
    verify: str = "host_dump",
    jobs: int = 8,
    timeout: int = 600,
    exactness: Any = None,
) -> dict[int, dict[str, Any]]:
    """Each group's one-step program for ``arm``: ``{group: record}`` (plus ``arm``).  The package arm
    needs ``package_dir`` and asks the package for ``groups`` alone; the reference arm ignores it.
    ``exactness`` (a :class:`merlin.perf.exactness.Contract`) is the contract BOTH arms' groups are graded
    under; without one each group is graded by the comparison its op implies."""
    from .forms import contract_of_entry

    of_entry = contract_of_entry(exactness) if exactness is not None else None
    from merlin.perf import whole_model_group_timing as T

    if arm.name == ARM_PACKAGE:
        if package_dir is None:
            raise GroupCapsuleError("the package arm needs a package directory")
        records = T.build_group_programs(
            package_dir,
            [int(g) for g in groups],
            model_capsule=model_capsule,
            target=target,
            machine=arm.machine,
            header=arm.header,
            header_sha256=arm.header_sha256,
            out=out,
            verify=verify,
            prohibited_roles=arm.prohibited_roles,
            harness_overrides=arm.harness_overrides,
            timeout=timeout,
            jobs=jobs,
            ask_only=True,
            phase0_recipe=arm.phase0_recipe,
            descriptor=arm.descriptor,
            exactness=of_entry,
            allow_regions=arm.allow_regions,
        )
    elif arm.name == ARM_REFERENCE:
        records = T.build_reference_group_programs(
            [int(g) for g in groups],
            model_capsule=model_capsule,
            target=target,
            machine=arm.machine,
            header=arm.header,
            header_sha256=arm.header_sha256,
            out=out,
            verify=verify,
            harness_overrides=arm.harness_overrides,
            exactness=of_entry,
        )
    else:
        raise GroupCapsuleError(f"no arm named {arm.name!r}")
    for record in records.values():
        record["arm"] = arm.name
        record["arm_recipe"] = arm.to_dict()
        record["verify"] = verify
    return records


def _sealed_scan(record: Mapping[str, Any], arm: Arm, *, target: str) -> dict[str, Any]:
    """:func:`isa_scan` under ``arm``'s rule, refused (``clean: False``) when the scan prohibits less
    than the arm's sealed Phase 0 policy."""
    from . import gates as G

    report = isa_scan(record, target=target, roles=arm.prohibited_roles)
    weaker = G.scan_weaker_than_sealed(report, arm.instruction_policy, arm.prohibited_roles)
    if weaker:
        report = {**report, "clean": False, "error": report.get("error") or weaker}
    return report


def isa_scan(record: Mapping[str, Any], *, target: str, roles: Sequence[str]) -> dict[str, Any]:
    """The instruction rule over the WHOLE one-group ELF (kernel, library and host code alike), with the
    per-group census the same pass produces."""
    from merlin.perf import isa_prohibition as ISA

    if record.get("refusal") or not record.get("elf"):
        return {"clean": False, "error": "no program to scan", "summary": {}}
    variant = record.get("variant") or {}
    objects = {
        str(record["group"]): Path(str(o))
        for o in variant.get("objects") or ()
        if Path(str(o)).name.split(".", 1)[0] == f"g{record['group']}"
    }
    compiler = (variant.get("program") or {}).get("compiler")
    if not compiler:
        raise GroupCapsuleError(f"the program {record.get('elf')} records no compiler to disassemble with")
    return dict(
        ISA.check_program(
            Path(str(record["elf"])),
            target=target,
            roles=list(roles),
            compiler=compiler,
            group_objects=objects,
            library_groups=[] if record.get("linked") == "submission" else [str(record["group"])],
        )
    )


def time_on_gsim(
    programs: Mapping[str, Mapping[str, Any]],
    *,
    target: str,
    model_capsule: str | Path,
    out: str | Path,
    max_parallel: int = 4,
    max_cycles: int = 60_000_000,
    emulator: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Every labelled program on the elaborated-RTL emulator: ``{label: row}`` (a ranking measurement)."""
    from merlin.perf import whole_model_group_timing as T

    labels = sorted(programs)
    keyed = {index: dict(programs[label]) for index, label in enumerate(labels)}
    timed = T.time_group_programs(
        keyed,
        target=target,
        model_capsule=model_capsule,
        out=out,
        max_parallel=max_parallel,
        max_cycles=max_cycles,
        emulator=emulator,
    )
    rows = {}
    for index, label in enumerate(labels):
        row = dict(timed.get(index) or {"status": "refused", "refusal": "not timed"})
        row["group"] = programs[label].get("group")
        row["arm"] = programs[label].get("arm")
        row["device"] = DEVICE
        rows[label] = row
    return rows


def label_of(arm: str, group: int, model: str | None = None) -> str:
    return f"{model + '_' if model else ''}{arm}_g{int(group)}"


def program_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """What a measurement row carries about the program it measured (never its build tree)."""
    keys = (
        "group",
        "arm",
        "on",
        "cause",
        "why",
        "linked",
        "interface",
        "interface_sha256",
        "elf_sha256",
        "object_sha256",
    )
    return {key: record.get(key) for key in keys}


def measure_on_gsim(
    arms: Mapping[str, Arm],
    groups: Sequence[int],
    *,
    package_dir: str | Path | None,
    model_capsule: str | Path,
    target: str,
    out: str | Path,
    max_parallel: int = 4,
    model: str | None = None,
    max_cycles: int = 60_000_000,
    require_package: bool = False,
    diagnostics: Mapping[str, Any] | None = None,
    exactness: Any = None,
) -> dict[str, Any]:
    """Both arms' one-group programs for ``groups`` on the emulator, each graded exactly and each package
    program's WHOLE ELF held to its arm's instruction rule: ``{"rows": [...]}`` (the validation path; a
    cell measures through :class:`GroupProgramMeasurer` and :func:`..cells.measure_member`).
    ``diagnostics`` (``{"functional_model": spec, "rooflines": {group: roofline}}``) adds each correct
    package program's ``efficiency`` row (:func:`efficiency_row`) -- diagnostic only, never a change to
    the measurement.
    ``require_package`` refuses, BEFORE any emulator time, a package-arm program the package does not
    answer, quoting the package's own reason.  ``exactness`` is the contract both arms are graded under
    (the default -- every form exact, an op's own declared bound kept -- when None), recorded either way."""
    from merlin.perf import exactness as EX

    from . import gates as G

    exactness = exactness if exactness is not None else EX.Contract.default(target=target)
    out = Path(out)
    for name, arm in arms.items():
        unsealed = G.sealed_policy_problems(arm.instruction_policy, arm.prohibited_roles)
        if unsealed:
            # Before any program is built or timed: an arm held to roles no sealed Phase 0 policy
            # resolved would be scanned against a rule that may forbid nothing.
            raise GroupCapsuleError(
                f"the {name} arm declares prohibited roles {list(arm.prohibited_roles)} but carries no "
                f"enforceable sealed instruction policy ({'; '.join(unsealed)})"
            )
    programs: dict[str, dict[str, Any]] = {}
    for name, arm in arms.items():
        built = build_arm_programs(
            arm,
            groups,
            package_dir=package_dir,
            model_capsule=model_capsule,
            target=target,
            out=out / name,
            exactness=exactness,
        )
        for group, record in built.items():
            programs[label_of(name, group, model)] = record
    scans = {
        label: _sealed_scan(record, arms[record["arm"]], target=target)
        for label, record in programs.items()
        if arms[record["arm"]].prohibited_roles and not record.get("refusal")
    }
    unanswered = {
        label: (
            f"coverage: the package does not answer this group ({record.get('cause') or record.get('linked')}"
            + (f": {str(record['why'])[:300]}" if record.get("why") else "")
            + "); not timed"
        )
        for label, record in programs.items()
        if require_package
        and record.get("arm") == ARM_PACKAGE
        and not record.get("refusal")
        and record.get("linked") != "submission"
    }
    runnable = {
        label: record
        for label, record in programs.items()
        if not record.get("refusal") and label not in unanswered and (scans.get(label) or {"clean": True}).get("clean")
    }
    timed = time_on_gsim(
        runnable,
        target=target,
        model_capsule=model_capsule,
        out=out / "runs",
        max_parallel=max_parallel,
        max_cycles=max_cycles,
    )
    rows = []
    for label, record in sorted(programs.items()):
        row = {"label": label, "model": model, "device": DEVICE, **program_row(record)}
        if record.get("refusal"):
            row.update(status="refused", refusal=record["refusal"])
        elif label in unanswered:
            row.update(status="refused", refusal=unanswered[label])
        elif label in scans and not scans[label].get("clean"):
            scan = scans[label]
            why = scan.get("error") or scan.get("detail") or "no clean verdict"
            row.update(
                status="refused",
                refusal="isa_prohibited: "
                + (", ".join(sorted(scan.get("summary") or {})) or f"the program could not be checked ({why})"),
            )
        else:
            got = timed.get(label) or {}
            row.update(
                {
                    k: got.get(k)
                    for k in (
                        "status",
                        "kind",
                        "cycles",
                        "refusal",
                        "carried",
                        "emulator_sha256",
                        "machine",
                        "exactness",
                        "evidence",
                    )
                },
                correct=bool(got.get("correct")),
            )
        if label in scans:
            row["isa_clean"] = bool(scans[label].get("clean"))
            row["census"] = (scans[label].get("census") or {}).get("per_group")
        if diagnostics and record.get("arm") == ARM_PACKAGE and row.get("status") == "graded" and row.get("correct"):
            row["efficiency"] = efficiency_row(record, row, diagnostics, target=target, out=out / "efficiency" / label)
        rows.append(row)
    document = {"schema": SCHEMA, "device": DEVICE, "arms": {k: a.to_dict() for k, a in arms.items()}, "rows": rows}
    document["exactness"] = exactness.record()
    out.mkdir(parents=True, exist_ok=True)
    (out / "gsim_rows.json").write_text(json.dumps(document, indent=1, default=str) + "\n", encoding="utf-8")
    return document


def efficiency_row(
    record: Mapping[str, Any], row: Mapping[str, Any], diagnostics: Mapping[str, Any], *, target: str, out: Path
) -> dict[str, Any]:
    """One correct package program's efficiency diagnostics (:mod:`merlin.perf.group_efficiency`): what
    it issued per execution against its derived roofline -- or why no census could be taken."""
    from merlin.perf import group_efficiency as E

    group = str(record["group"])
    roofline = (diagnostics.get("rooflines") or {}).get(group)
    spec = diagnostics.get("functional_model")
    compiler = ((record.get("variant") or {}).get("program") or {}).get("compiler")
    objects = {
        group: Path(str(o))
        for o in ((record.get("variant") or {}).get("objects") or ())
        if Path(str(o)).name.split(".", 1)[0] == f"g{group}"
    }
    if not spec or not compiler or not objects or not record.get("elf"):
        return {
            "refusal": "no functional model, compiler, kernel object or program to take a census of",
            **E.efficiency(None, roofline=roofline, cycles=row.get("cycles")),
        }
    try:
        census = E.dynamic_census(
            Path(str(record["elf"])), target=target, compiler=compiler, group_objects=objects,
            functional_model=spec, out=out,
        )  # fmt: skip
    except Exception as error:  # noqa: BLE001 -- diagnostic only: recorded, never fails the measurement
        census = {"refusal": f"{type(error).__name__}: {error}"}
    report = E.efficiency((census.get("per_group") or {}).get(group), roofline=roofline, cycles=row.get("cycles"))
    if census.get("refusal"):
        report["refusal"] = census["refusal"]
    return report


# ------------------------------------------------------------------------------------ cell measurer


class GroupProgramMeasurer:
    """The cell machine's measurer (:class:`..cells.CellMeasurer`) over one-group programs on the
    emulator.  The package arm's recipe is the job's own build options (``spec["build_options"]``, which
    :func:`..cells.measure_cell` passes); the reference arm's is ``spec["reference_build_options"]``,
    else the package arm's without its instruction rule (the bar is the library as the vendor ships it)."""

    def __init__(self, spec: Mapping[str, Any]):
        self.spec = dict(spec)
        self.target = str(spec.get("target") or "")
        timing = dict(spec.get("timing") or {})
        self.max_parallel = int(timing.get("max_parallel") or 2)
        self.emulator = timing.get("emulator")
        self._arms: dict[str, Arm] = {}

    def arm(self, name: str) -> Arm:
        if name not in self._arms:
            options = dict(self.spec.get("build_options") or {})
            if name == ARM_REFERENCE:
                options = dict(self.spec.get("reference_build_options") or options)
                options.pop(_PROHIBITED, None)
            self._arms[name] = arm_from_options(options, name=name)
        return self._arms[name]

    def contract(self):
        """The exactness contract the cell's programs are graded under (carried by value on the spec)."""
        from merlin.perf import exactness as EX

        return EX.Contract.from_value(self.spec.get("exactness"), target=self.target)

    def programs(self, arm, groups, *, package_dir, member, out):
        built = build_arm_programs(
            self.arm(arm),
            groups,
            package_dir=package_dir,
            model_capsule=str(member.get("model_capsule")),
            target=str(member.get("target") or self.target),
            out=Path(out) / arm,
            exactness=self.contract(),
        )
        return {label_of(arm, group, member.get("label")): record for group, record in built.items()}

    def scan(self, record, *, roles):
        report = isa_scan(record, target=self.target, roles=roles)
        return {
            "clean": report.get("clean") is True,
            "summary": report.get("summary") or {},
            "prohibited": dict(report.get("prohibited") or {}),
            **({"error": report["detail"]} if report.get("detail") and not report.get("error") else {}),
            "census": (report.get("census") or {}).get("per_group")
            if isinstance(report.get("census"), Mapping)
            else None,
            **({"error": report["error"]} if report.get("error") else {}),
        }

    def time(self, records, *, member, max_cycles, out):
        target = str(member.get("target") or self.target)
        timed = time_on_gsim(
            records,
            target=target,
            model_capsule=str(member.get("model_capsule")),
            out=Path(out) / "runs",
            max_parallel=self.max_parallel,
            max_cycles=int(max_cycles),
            emulator=self.emulator,
        )
        # THE OBJECTIVE'S OWN PROGRAMS, on the package arm, carry their efficiency diagnostics: what each
        # issued against its derived roofline (the cell's ``diagnostics``). Never a held-out or a
        # collateral member's, never the reference's, and never a change to the measurement.
        diagnostics = member.get("diagnostics") if member.get("label") is None else None
        if diagnostics:
            for label, row in timed.items():
                record = records.get(label) or {}
                if record.get("arm") == ARM_PACKAGE and row.get("status") == "graded" and row.get("correct"):
                    row["efficiency"] = efficiency_row(
                        record, row, diagnostics, target=target, out=Path(out) / "efficiency" / label
                    )
        return timed


def cell_measurer(spec: Mapping[str, Any]) -> GroupProgramMeasurer:
    """The ``measurer`` a cell machine names (``module:callable``)."""
    return GroupProgramMeasurer(spec)


# ------------------------------------------------------------------------------------- validation


def in_model_cycles(result: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """``{"groups": {group: cycles}, "device": ..., "objective_cycles": ...}`` from a whole-model result."""
    document = json.loads(Path(result).read_text(encoding="utf-8")) if not isinstance(result, Mapping) else result
    verdict = document.get("verdict") or {}
    rows = [r for r in verdict.get("groups") or () if isinstance(r, Mapping) and r.get("group") is not None]
    device = document.get("device") or {}
    return {
        "groups": {str(r["group"]): r.get("cycles") for r in rows},
        "correct": {str(r["group"]): r.get("correct") for r in rows},
        "device": device.get("artifact"),
        "abi_header_sha256": device.get("abi_header_sha256"),
        "objective_cycles": document.get("objective_cycles"),
        "source": None if isinstance(result, Mapping) else str(result),
    }


def validation_table(
    measured: Sequence[Mapping[str, Any]], in_model: Mapping[tuple[str, str], Mapping[str, Any]], *, tolerance: float
) -> dict[str, Any]:
    """Each standalone count beside the in-model count of the same group, arm and device.  A row is
    within tolerance when ``|standalone / in_model - 1| <= tolerance``; every row keeps its signed offset
    and the offsets are summarized per device, so a SYSTEMATIC offset is visible as one."""
    rows = []
    for row in measured:
        key = (str(row.get("arm")), str(row.get("device")))
        reference = in_model.get(key)
        group = str(row.get("group"))
        entry: dict[str, Any] = {
            "group": group,
            "arm": key[0],
            "device": key[1],
            "standalone_cycles": row.get("cycles"),
            "standalone_correct": row.get("correct", row.get("admitted")),
            "in_model_cycles": None if reference is None else reference["groups"].get(group),
            "in_model_source": None if reference is None else reference.get("source"),
        }
        a, b = entry["standalone_cycles"], entry["in_model_cycles"]
        if a and b:
            entry.update(
                ratio=round(a / b, 5), offset_cycles=int(a) - int(b), within_tolerance=abs(a / b - 1.0) <= tolerance
            )
        else:
            entry.update(
                within_tolerance=False,
                why="no standalone count" if not a else "no in-model count for this arm and device",
            )
        rows.append(entry)
    summary: dict[str, Any] = {}
    for entry in rows:
        if "ratio" in entry:
            slot = summary.setdefault(entry["device"], {"n": 0, "ratios": []})
            slot["n"] += 1
            slot["ratios"].append(entry["ratio"])
    for slot in summary.values():
        ratios = sorted(slot["ratios"])
        slot["min_ratio"], slot["max_ratio"] = ratios[0], ratios[-1]
        slot["all_same_sign"] = all(r >= 1 for r in ratios) or all(r <= 1 for r in ratios)
    return {
        "schema": SCHEMA + "/validation",
        "tolerance": tolerance,
        "rows": rows,
        "per_device": summary,
        "passed": bool(rows) and all(e["within_tolerance"] and e["standalone_correct"] for e in rows),
    }


__all__ = [
    "ARM_PACKAGE",
    "efficiency_row",
    "ARM_REFERENCE",
    "Arm",
    "GroupCapsuleError",
    "GroupProgramMeasurer",
    "SCHEMA",
    "arm_from_options",
    "arms_from_jobs",
    "build_arm_programs",
    "cell_measurer",
    "in_model_cycles",
    "isa_scan",
    "measure_on_gsim",
    "time_on_gsim",
    "validation_table",
]
