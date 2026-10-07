"""Whole-model gates on a package: does it compile each declared model, fully, correctly, loop-free?

A capsule grade says a package is right on the forms the corpus holds. It cannot say the package
compiles a WHOLE model: a group whose form no capsule holds, a group the package declines, a kernel
that is right on its capsule and wrong at the model's own extents, and a library path the model
falls back to are all invisible to it. This gate answers that directly, per model the target's
experiment declares (``phase1_gates.whole_model.models``):

1. **build** the model as one program with the package (``whole_model_builder.build``, ``verify=
   "local"``), for the machine the model will be measured on, with the declared instruction roles
   prohibited -- the program phase 2 would measure, so the gate and the measurement cannot disagree;
2. **no prohibited instruction** anywhere in the linked program (``isa_prohibition.check_build``);
3. **coverage**: the share of the model's DEVICE groups the package itself compiled, at or above the
   declared floor. A group the builder puts on the core by design (the model's host region, or a
   readout the machine cannot express) is not the package's to compile and is outside the
   denominator; a group the package declined -- to the library, or to the core because the library's
   path would need a prohibited instruction -- counts against it;
4. **correctness** on the functional simulator: every group's own local check (its reference
   recomputed on the core from the inputs it actually held), held to the bound the target's EXACTNESS
   CONTRACT gives the group's form (:mod:`.exactness`; exact unless the contract says otherwise, and the
   contract applied is recorded), and the model's end result against the oracle;
5. **float accuracy**, for an open model: its output against the FLOAT model the capture quantized,
   at the capture's own declared tolerance (:mod:`.float_accuracy`). The checks in 4 are self-consistency --
   a quantized variant agrees with itself however far it is from the network -- so this one is
   separate and required, and a capture that recorded no float reference fails it closed.

WHAT THE AGENT IS TOLD, AND WHAT IT IS NOT. The feedback lines name the model, the failing group, its
op and op-form, and what failed (not compiled, and why; wrong, and how many elements; a prohibited
instruction, and where). They never carry a tensor value, a golden, or the model's expected output:
the end result is reported as agreeing or disagreeing with the oracle, not as the class it should be.
The model's tensors stay in the operator's tree; nothing here copies them anywhere an agent reads.

Nothing here names a target: the builder, the ISA facts, the simulator and the machine are all reached
through the target's own descriptor and backend.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "merlin_whole_model_gate_v1"
PASS, FAIL = "pass", "fail"
#: Routes a group is answered on, as the build's attribution spells them.
ROUTE_PACKAGE, ROUTE_HOST = "package", "host"
#: The cause the builder records for a declined group it moved to the core because the library's own
#: path for it would need a prohibited instruction. It is the package's decline, so it counts against.
CAUSE_LOOP_FREE_LIBRARY = "library_loop_free_path"
RESULT_FILE = "whole_model_gate.json"


# ------------------------------------------------------------------------------------------ config


def models_of(gate: Mapping[str, Any], *, root: Path) -> list[dict[str, Any]]:
    """The declared models with their paths resolved against ``root`` (the checkout's root)."""
    out = []
    for model in gate.get("models") or ():
        row = dict(model)
        for key in ("capsule", "header"):
            path = Path(str(row[key]))
            row[key] = str(path if path.is_absolute() else root / path)
        reference = row.get("price_reference")
        if reference:
            from merlin.common.paths import out_dir

            path = Path(str(reference))
            row["price_reference"] = str(path if path.is_absolute() else out_dir() / path)
        row.setdefault("name", Path(row["capsule"]).name)
        out.append(row)
    return out


def contract_of(gate: Mapping[str, Any], model: Mapping[str, Any] | None, *, root: Path, target: str):
    """The exactness contract a model is graded under: the model's own ``exactness`` path, else the
    gate's, else the default (every form exact).  A declared contract that cannot be read is an error,
    never the default."""
    from . import exactness as EX

    declared = (model or {}).get("exactness") or gate.get("exactness")
    if not declared:
        return EX.Contract.default(target=target)
    path = Path(str(declared))
    return EX.load(path if path.is_absolute() else root / path)


def grade_exactness(
    screen: Mapping[str, Any],
    expectations: Mapping[str, Any],
    contract: Any,
    *,
    routes: Sequence[Mapping[str, Any]],
    forms: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The screen's groups held to their exactness contract: ``(not_correct, record)``.

    A group whose contract is the check it already ran (exact on an exact check, an op's own declared
    bound on its bound check) keeps the check's verdict; any other is re-graded from the check's own
    numbers (:func:`.exactness.judge`), failing closed when they cannot show the bound held."""
    from . import exactness as EX

    groups = expectations.get("groups") or {}
    resolve = EX.resolver(
        contract,
        forms={str(k): v for k, v in (forms or {}).items() if isinstance(v, Mapping)},
        routes=routes,
        op_bounds={str(g): (e or {}).get("bound_lsb") for g, e in groups.items()},
    )
    per_group: dict[str, Any] = {}
    summary: dict[str, int] = {}
    wrong = []
    for row in screen.get("groups") or ():
        key = str(row["group"])
        compare = str((groups.get(key) or {}).get("compare") or "exact")
        try:
            exactness = resolve(key, row)
        except EX.ExactnessError as exc:
            wrong.append({"group": key, "kind": row.get("kind"), "local": "wrong", "failure": {"exactness": str(exc)}})
            continue
        label = exactness.label()
        summary[label] = summary.get(label, 0) + 1
        builtin = (exactness.mode == EX.EXACT and compare == "exact") or (
            exactness.declared_by == EX.DECLARED_OP and compare != "exact"
        )
        if builtin or row.get("local") == "absent":
            ok = row.get("local") == "correct"
            per_group[key] = {"contract": label, "regraded": False}
        else:
            check = row.get("check") or {}
            grade = EX.judge(
                exactness, max_abs=check.get("max_abs"), mismatches=check.get("mismatches"), elements=check.get("of")
            )
            ok = bool(grade["passed"])
            per_group[key] = {**grade, "regraded": True}
        if not ok:
            wrong.append(
                {
                    "group": key,
                    "kind": row.get("kind"),
                    "local": row.get("local") if not per_group[key].get("regraded") else "wrong",
                    "failure": row.get("failure") or per_group[key].get("why"),
                    "exactness": label,
                }
            )
    record = {"contract": contract.record(), "summary": summary, "label": EX.label_summary({"summary": summary})}
    record["per_group"] = per_group
    return wrong, record


def _headers_of(gate: Mapping[str, Any], *, root: Path) -> list[str]:
    """Every ABI header the gate may build against: the models' own and the declared ``headers``."""
    paths = [str(h) for h in gate.get("headers") or ()] + [str(m["header"]) for m in gate.get("models") or ()]
    return [str(Path(p) if Path(p).is_absolute() else root / p) for p in paths]


#: Where Phase 1 hands a grader the experiment's declared prohibited roles (the experiment adapter sets it).
ROLES_ENV = "MERLIN_PROHIBITED_INSTRUCTION_ROLES"


def gate_for(target: str, *, descriptor: str | Path | None = None) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    """``(whole-model gate, prohibited roles)``: the gate the target's descriptor declares
    (``phase1_gates.whole_model``), and the roles the experiment declares (handed to graders in
    :data:`ROLES_ENV`).  No gate declared is ``None``; no roles declared is an empty tuple."""
    import os

    import yaml

    if descriptor is None:
        from merlin.targetgen.corpora import descriptor_path

        descriptor = descriptor_path(target)
    document = yaml.safe_load(Path(descriptor).read_text(encoding="utf-8")) or {}
    gate = (document.get("phase1_gates") or {}).get("whole_model") if isinstance(document, Mapping) else None
    roles = tuple(r for r in (os.environ.get(ROLES_ENV) or "").split(",") if r.strip())
    return (dict(gate) if isinstance(gate, Mapping) else None), roles


# ---------------------------------------------------------------------------------------- coverage


def _price_table(model: Mapping[str, Any]) -> tuple[dict[str, int], str]:
    """``({group: price}, how priced)``: the declared reference run's per-group cycles when it exists,
    else one unit per group (and the pricing says so)."""
    reference = model.get("price_reference")
    if reference and Path(str(reference)).is_file():
        from . import whole_model_verdict as V

        document = json.loads(Path(str(reference)).read_text(encoding="utf-8"))
        table = {str(r["group"]): int(r.get("cycles") or 0) for r in V.group_table(document.get("verdict") or {})}
        if table and sum(table.values()) > 0:
            return table, f"per-group cycles of the reference run {Path(str(reference)).parent.name}"
    return {}, "one unit per group (no reference run declared or present)"


def coverage(per_group: Sequence[Mapping[str, Any]], model: Mapping[str, Any]) -> dict[str, Any]:
    """The package's share of the model's DEVICE groups, by count and priced, against the floor.

    Outside the denominator: groups the builder put on the core with no decline behind it (the model's
    host region, a readout the machine lacks). Inside it and not the package's: every declined group,
    whether it fell back to the library or -- its library path needing a prohibited instruction -- to
    the core."""
    price, pricing = _price_table(model)
    device, answered, declined, outside = [], [], [], []
    for row in per_group:
        group, on, cause = str(row.get("group")), row.get("on"), row.get("cause")
        if on == ROUTE_HOST and cause != CAUSE_LOOP_FREE_LIBRARY and not row.get("declined_as"):
            outside.append({"group": group, "op": row.get("op"), "cause": cause})
            continue
        device.append(group)
        (answered if on == ROUTE_PACKAGE else declined).append(row)
    unit = (lambda g: price.get(g, 0)) if price else (lambda g: 1)
    total = sum(unit(g) for g in device)
    priced = sum(unit(str(r.get("group"))) for r in answered)
    share = round(priced / total, 4) if total else 0.0
    floor = float(model["coverage_floor"])
    return {
        "device_groups": len(device),
        "package_groups": len(answered),
        "share": share,
        "floor": floor,
        "pricing": pricing,
        "passed": bool(device) and share + 1e-9 >= floor,
        # On the core by design (the model's host region, or a readout this machine lacks): not the package's.
        "outside_denominator": outside,
        "declined": [
            {
                "group": str(r.get("group")),
                "op": r.get("op"),
                "on": r.get("on"),
                "cause": r.get("declined_as") or r.get("cause"),
                "why": r.get("declined_why") or r.get("why"),
            }
            for r in declined
        ],
    }


# ------------------------------------------------------------------------------------------ one model


def _forms(target: str, capsule: str, forms: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Each group's op-form, as the caller derived it from the one grouping (``forms``); empty when none
    was supplied -- a form is a courtesy in the feedback, never a verdict input."""
    return {str(k): v for k, v in (forms or {}).items()}


def _form_text(form: Any) -> str:
    if not isinstance(form, Mapping):
        return ""
    parts = [f"{k}={form[k]}" for k in sorted(form) if form[k] not in (None, "", [], {})]
    return "{" + ", ".join(parts) + "}"


def _capsules_of_forms(groups: Sequence[str], routes: Mapping[str, Any], forms: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from .whole_model_capsules import resolve_capsules

        _chosen, mapping = resolve_capsules(",".join(f"g{g}" for g in groups), routes, forms=forms)
        return mapping
    except Exception:  # noqa: BLE001 -- ditto
        return {}


def _short(text: Any, limit: int = 240) -> str:
    line = " ".join(str(text or "").split())
    return line if len(line) <= limit else line[: limit - 3] + "..."


def _prune(build_dir: Path) -> None:
    """Drop what a whole-model build keeps only to link; keep its records and the ELF."""
    from . import whole_model_builder as B

    for name in ("lower", "objects", "harness", "program"):
        target = build_dir / name
        if target.is_dir():
            for path in target.rglob("*"):
                if path.is_file() and path.suffix != ".elf":
                    path.unlink()
    for pattern in B.PRUNABLE:
        for path in build_dir.glob(pattern):
            if path.is_file() and path.suffix != ".elf":
                path.unlink()


def end_result(screen: Mapping[str, Any], expectations: Mapping[str, Any]) -> dict[str, Any]:
    """The model's end result against the oracle, recorded WITHOUT the expected value.

    A classifier is graded on its class (``GM_ARGMAX``): agree or not -- the oracle's class is never
    recorded, it is the model's expected output. A model whose output is a tensor (an action) is graded
    on how many of its elements lie within the capsule's numeric policy of the oracle's (``GM_OUTPUT``):
    all of them, or not. A build that states neither is refused as ungradable, never passed.
    """
    if expectations.get("argmax") is not None:
        argmax = screen.get("argmax")
        observed = int(argmax[0]) if argmax else None
        agrees = observed is not None and observed == int(expectations["argmax"])
        return {"passed": agrees, "basis": "class", "agrees_with_oracle": agrees}
    elements = (
        (expectations.get("output") or {}).get("elements") if isinstance(expectations.get("output"), Mapping) else None
    )
    if elements:
        output = screen.get("output")
        if not output:
            return {"passed": False, "basis": "output tensor", "note": "the program printed no end-result line"}
        within, of = int(output[0]), int(output[1])
        agrees = of == int(elements) and within == of
        return {"passed": agrees, "basis": "output tensor", "within": within, "of": of, "agrees_with_oracle": agrees}
    return {"passed": False, "note": "the build states no end result to grade (no class and no output tensor)"}


# ------------------------------------------------------------------------------ which machine


#: The cause the builder records for a group whose result is the raw accumulator on a machine that
#: cannot read one out at full width: the group is computed on the core, and the package's kernel for
#: it is never run.
CAUSE_READOUT_UNAVAILABLE = "accumulator_readout_unavailable"
#: ``machine_limited_groups`` values: a group the declared machine cannot read out at full width either
#: MUST RUN (the gate moves to a full-width machine; the default) or runs ON THE CORE, as it does on the
#: declared device (the model is graded on the machine it is measured on).
MUST_RUN, ON_CORE = "must_run", "on_core"
#: The cause token a builder leads its refusal with when a model's groups commit an accumulator the
#: machine cannot read out at all (``merlin.perf.whole_model_open.MACHINE_CANNOT_READ_OUT``, restated
#: here so the gate does not import a builder to name it; a test holds the two equal).
MACHINE_CANNOT_READ_OUT = "machine_cannot_read_out_the_models_accumulators"
#: ``(capsule, preferred machine) -> chosen machine``: what a model needs of the readout is a property
#: of the model and the machine, never of the package, so it is decided once per process.
_READOUT_CHOICE: dict[tuple[str, str], dict[str, Any]] = {}


def _oracle_findings(expectations: Mapping[str, Any]) -> dict[str, Any] | None:
    """How the oracle and the capsule's golden stand against each other under the capsule's own policy --
    recorded beside every tensor end result so nobody re-derives why the oracle is not the reference."""
    source = expectations.get("source")
    if not source or not Path(source).is_file():
        return None
    import numpy as np

    stated = json.loads(Path(source).read_text(encoding="utf-8"))
    golden = np.asarray(stated.get("golden") or (), np.float64)
    oracle = np.asarray(stated.get("oracle_output") or (), np.float64)
    policy = stated.get("numeric_policy") or {}
    if golden.size == 0 or golden.size != oracle.size:
        return None
    atol, rtol = float(policy.get("atol", 0.0)), float(policy.get("rtol", 0.0))
    within = int((np.abs(oracle - golden) <= atol + rtol * np.abs(golden)).sum())
    return {"oracle_within_policy_of_golden": within, "of": int(golden.size), "policy": {"atol": atol, "rtol": rtol}}


def _against_reference(
    screen: Mapping[str, Any],
    build_record: Mapping[str, Any],
    expectations: Mapping[str, Any],
    model: Mapping[str, Any],
    *,
    target: str,
    out: Path,
    templates: Mapping[str, str] | None,
    timeout: int,
    jobs: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """A tensor model's end result, and every dispatch's result digest, against its REFERENCE ARM (the
    model's own host code with exact devices; :mod:`.whole_model_reference`). The criterion is the
    model's declaration (``end_result``), bit-identical by default; the oracle's policy check stays in
    the result as a diagnostic, beside how the oracle itself stands against the golden."""
    from . import whole_model_reference as R

    console_path = out / "screen" / "console.txt"
    console = console_path.read_text(encoding="utf-8", errors="replace") if console_path.is_file() else ""
    output = expectations.get("output") if isinstance(expectations.get("output"), Mapping) else {}
    verdict = R.end_result(
        console,
        build_record,
        target=target,
        simulator=str(screen.get("simulator") or model.get("simulator") or "spike"),
        declaration=model.get("end_result") if isinstance(model.get("end_result"), Mapping) else None,
        templates=templates,
        timeout=timeout,
        jobs=jobs,
        elements=output.get("elements"),
    )
    verdict["diagnostics"] = {
        "within_policy_of_oracle": end_result(screen, expectations),
        "oracle_vs_golden": _oracle_findings(expectations),
    }
    words = dict(verdict.get("words") or {})
    words["passed"] = bool(words) and not words.get("differ") and words.get("dispatches", 0) > 0
    if "note" in verdict and not words.get("dispatches"):
        words.update(passed=False, note=verdict["note"])
    return verdict, words


def _float_accuracy(out: Path, model: Mapping[str, Any]) -> dict[str, Any]:
    """The screened program's output against the capture's float reference (:mod:`.float_accuracy`)."""
    from . import float_accuracy as FA

    console_path = out / "screen" / "console.txt"
    console = console_path.read_text(encoding="utf-8", errors="replace") if console_path.is_file() else ""
    try:
        return FA.check(console, Path(str(model["capsule"])))
    except Exception as exc:  # noqa: BLE001 -- an unreadable reference is a failed check, never a pass
        return {"passed": False, "basis": "float reference", "note": _short(f"{type(exc).__name__}: {exc}")}


def _choice_key(model: Mapping[str, Any]) -> tuple[str, str]:
    return (str(model["capsule"]), str(model["machine"]))


def _full_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(Path(record["notes"]["build_record"]).read_text(encoding="utf-8"))


def _sha256(path: str | Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def full_width_machines(target: str, headers: Sequence[str | Path] = ()) -> list[dict[str, Any]]:
    """``[{machine, header_sha256}]`` the target's readout-width fact, DERIVED here from the target's own
    probes, elaboration and headers (:func:`merlin.targetgen.rtl.semantic_facts.readout_machines`), says
    read the accumulator out at full width, in the fact's own order.  ``headers`` are the declared header
    files a machine's header is found among by its registry ABI digest.  Empty when the fact is absent or
    not derived -- :func:`choose_machine` then refuses by name, never an error mid-gate."""
    import importlib

    try:
        SF = importlib.import_module("merlin.targetgen.rtl.semantic_facts")
        fact = SF.readout_machines(target, headers=[str(h) for h in headers])
    except Exception:  # noqa: BLE001 -- no derivation of the fact is no derived machine
        return []
    if fact.get("status") != "derived":
        return []
    out = []
    for machine, row in ((fact.get("value") or {}).get("machines") or {}).items():
        if isinstance(row, Mapping) and row.get("full_width_readout") is True and row.get("status") == "derived":
            header = row.get("header") or {}
            registry = header.get("registry_abi_header") or {}
            out.append(
                {
                    "machine": str(machine),
                    "header_sha256": str(header.get("sha256") or ""),
                    "registry_declared": bool(registry.get("declared_by")) and registry.get("agrees") is True,
                }
            )
    # A machine whose ABI header the hardware registry itself declares (and the fact agrees with) first:
    # its build asserts the header by the registry, not by this derivation.
    return sorted(out, key=lambda row: not row["registry_declared"])


def choose_machine(
    model: Mapping[str, Any], record: Mapping[str, Any], *, target: str, headers: Sequence[str]
) -> dict[str, Any]:
    """The machine a model's gate is built for, DERIVED from the readout-width fact.

    The declared machine (the one phase 2 measures on) when the model's build there leaves no group
    unexercised for want of a full-width accumulator readout. Otherwise the first machine the target's
    readout-width fact derives as reading out at full width, with its ABI header found among the
    declared ``headers`` BY CONTENT (the fact records each machine's header digest). Refused -- the gate
    fails closed -- when the model needs a full-width readout and no such machine and header are declared:
    a group whose kernel is never run cannot be graded, and a gate that skipped it would read as a pass.
    """
    rows = (record.get("attribution") or {}).get("per_group") or ()
    needs = list(record.get("_needs_full_width") or ()) or sorted(
        (str(r.get("group")) for r in rows if isinstance(r, Mapping) and r.get("cause") == CAUSE_READOUT_UNAVAILABLE),
        key=lambda g: int(g) if g.isdigit() else 10**9,
    )
    if not needs:
        return {"machine": str(model["machine"]), "header": str(model["header"]), "why": "the declared machine"}
    if model.get("machine_limited_groups") == ON_CORE:
        # DECLARED, not inferred: the model is graded on the device it is measured on, where these groups
        # run on the core by the machine's own fact. They are then outside the package's denominator.
        return {
            "machine": str(model["machine"]),
            "header": str(model["header"]),
            "why": (
                f"the declared machine; group(s) {', '.join('g' + g for g in needs[:12])} commit at full "
                f"width, which it cannot read out, and the model declares machine-limited groups run on the core"
            ),
            "machine_limited_groups": needs,
        }
    by_digest = {}
    for header in headers:
        if Path(header).is_file():
            by_digest.setdefault(_sha256(header), str(header))
    derived = full_width_machines(target, list(by_digest.values()))
    for row in derived:
        header = by_digest.get(row["header_sha256"])
        if header:
            return {
                "machine": row["machine"],
                "header": header,
                # Asserted by the fact only where the registry declares no header for the machine.
                "header_sha256": None if row["registry_declared"] else row["header_sha256"],
                "why": (
                    f"{model['machine']} cannot read out at full width the result of group(s) "
                    f"{', '.join('g' + g for g in needs[:12])}{' ...' if len(needs) > 12 else ''} "
                    f"({len(needs)} in all), so their kernels would never run; the readout-width fact derives "
                    f"{row['machine']} as reading them out"
                ),
            }
    raise ValueError(
        f"{len(needs)} group(s) of {model.get('name') or model['capsule']} commit at full accumulator width, which "
        f"{model['machine']} cannot read out, and no machine the readout-width fact derives as full-width has a "
        f"declared header ({[r['machine'] for r in derived] or 'none derived'}); the gate "
        "cannot grade those groups, so it refuses rather than skip them"
    )


def run_model(
    package: str | Path,
    model: Mapping[str, Any],
    *,
    target: str,
    roles: Sequence[str],
    out: str | Path,
    timeout: int = 7200,
    jobs: int | None = None,
    keep_build: bool = False,
    headers: Sequence[str] = (),
    chunk_ops: int | str | None = None,
    phase0_recipe: str | Path | None = None,
    descriptor: str | Path | None = None,
    exactness: Any = None,
) -> dict[str, Any]:
    """Every check of one declared model on ``package``; see the module docstring.

    ``exactness`` is the :class:`.exactness.Contract` the model's groups are held to (the default --
    every form exact -- when None); the correctness check records the contract it applied.

    ``phase0_recipe`` / ``descriptor`` name the corpus binding the builder states every group under; a
    package build refuses to guess it, so a gate that builds a package passes the experiment's own.

    ``chunk_ops`` is passed straight through to the builder (an open model only; ``None`` is the
    prior, unchunked build), so a gate run can carry its per-stage timing and dedup counts in
    ``checks["build"]`` without a separate path.
    """
    from . import isa_prohibition as ISA
    from . import whole_model_builder as B
    from . import whole_model_partial as PARTIAL
    from . import whole_model_screen as S

    out = Path(out)
    started = time.monotonic()
    name = str(model["name"])
    result: dict[str, Any] = {
        "model": name,
        "machine": model["machine"],
        "required": model.get("required", True) is not False,
        "checks": {},
        "status": FAIL,
    }
    checks = result["checks"]

    def _build(machine: str, header: str, header_sha256: str | None) -> dict[str, Any]:
        record = B.build(
            Path(package),
            target=target,
            out_dir=out / "build",
            model_capsule=str(model["capsule"]),
            machine=machine,
            header=header,
            header_sha256=header_sha256,
            verify="local",
            jobs=jobs,
            timeout=min(timeout, 3600),
            prohibited_roles=list(roles),
            chunk_ops=chunk_ops,
            phase0_recipe=None if phase0_recipe is None else str(phase0_recipe),
            descriptor=None if descriptor is None else str(descriptor),
        )
        PARTIAL.refuse(record, reader="the whole-model gate")  # a partial program is not the model
        return record

    try:
        chosen = _READOUT_CHOICE.get(_choice_key(model))
        if chosen is None:
            try:
                record = _build(str(model["machine"]), str(model["header"]), model.get("header_sha256"))
                built = _full_record(record)
            except Exception as refusal:  # noqa: BLE001 -- only the builder's own readout refusal is caught
                if not str(refusal).startswith(MACHINE_CANNOT_READ_OUT):
                    raise
                # A builder that refuses the machine outright (every device group commits the accumulator,
                # so none would run there) is the same need, stated before building anything.
                built = {"_needs_full_width": ["every device group"], "attribution": {}}
            chosen = choose_machine(model, built, target=target, headers=headers)
            _READOUT_CHOICE[_choice_key(model)] = chosen
            if chosen["machine"] != model["machine"]:
                record = _build(chosen["machine"], chosen["header"], chosen.get("header_sha256"))
        else:
            record = _build(chosen["machine"], chosen["header"], chosen.get("header_sha256"))
    except Exception as exc:  # noqa: BLE001 -- a model that cannot be built is the gate's first failure
        checks["build"] = {"passed": False, "error": _short(f"{type(exc).__name__}: {exc}", 600)}
        result["wall_s"] = round(time.monotonic() - started, 1)
        # A build that fails part-way has already written most of its per-group IR (11.4 GB for one
        # SmolVLA build that failed after 54 min); the error above is what a reader acts on.
        if not keep_build:
            _prune(out / "build")
        return result
    result["machine"] = chosen["machine"]
    result["machine_choice"] = {k: v for k, v in chosen.items() if k != "header"}
    full = _full_record(record)
    per_group = (full.get("attribution") or {}).get("per_group") or []
    checks["build"] = {
        "passed": True,
        "elf_sha256": record["elf_sha256"],
        "counts": full["attribution"]["counts"],
        # Present only for an open-model build (chunk_forward's home); absent (not present) for a
        # closed one, which has no per-stage record of its own yet.
        **({"stage_times": record["stage_times"]} if "stage_times" in record else {}),
        **({"object_dedup": record["object_dedup"]} if "object_dedup" in record else {}),
        **({"chunks": record["chunks"]} if "chunks" in record else {}),
    }
    routes = {str(r.get("group")): r for r in per_group}

    if roles:
        try:
            isa = ISA.check_build(record, target=target, roles=roles)
            checks["no_prohibited_instruction"] = {
                "passed": bool(isa["clean"]),
                "roles": list(roles),
                "summary": isa.get("summary") or {},
            }
        except Exception as exc:  # noqa: BLE001 -- an unread program is not a clean one
            checks["no_prohibited_instruction"] = {"passed": False, "error": _short(f"{type(exc).__name__}: {exc}")}

    checks["coverage"] = coverage(per_group, model)

    try:
        screen = S.structure_screen(
            record["elf"],
            groups={g: e["compare"] for g, e in record["expectations"]["groups"].items()},
            target=target,
            out=out / "screen",
            elf_sha256=record["elf_sha256"],
            templates=record.get("protocol"),
            simulator=str(model.get("simulator") or "spike"),
            timeout=timeout,
            routes=S.routes_of(full),
        )
    except Exception as exc:  # noqa: BLE001 -- a program that did not run is not a correct one
        screen = {"status": "error", "refusal": _short(f"{type(exc).__name__}: {exc}", 600)}
    if screen.get("status") != "screened":
        checks["correctness"] = {"passed": False, "refused": screen.get("refusal"), "wall_s": screen.get("wall_s")}
        checks["end_result"] = {"passed": False, "note": "the program's run could not be read"}
    else:
        from . import exactness as EX

        wrong, applied = grade_exactness(
            screen,
            record.get("expectations") or {},
            exactness if exactness is not None else EX.Contract.default(target=target),
            routes=per_group,
            forms=model.get("forms"),
        )
        checks["correctness"] = {
            "passed": not wrong,
            "groups": len(screen.get("groups") or ()),
            "not_correct": wrong,
            "wall_s": screen.get("wall_s"),
            # WHICH CONTRACT THE GROUPS WERE HELD TO -- "exact" only when every group was.
            "exactness": applied,
        }
        result["exactness"] = applied["label"]
        expectations = record.get("expectations") or {}
        checks["end_result"] = end_result(screen, expectations)
        if expectations.get("argmax") is None and (full.get("reference_identity") or {}):
            checks["end_result"], checks["words_vs_reference"] = _against_reference(
                screen,
                full,
                expectations,
                model,
                target=target,
                out=out,
                templates=record.get("protocol"),
                timeout=timeout,
                jobs=jobs,
            )
            # Self-consistency above cannot see a variant that is wrong about the model it quantized;
            # this REQUIRED check holds the output to the float model at the capture's own tolerance.
            checks["float_accuracy"] = _float_accuracy(out, model)

    failing_groups = sorted(
        {d["group"] for d in checks["coverage"]["declined"]}
        | {w["group"] for w in (checks["correctness"].get("not_correct") or [])},
        key=lambda g: int(g) if g.isdigit() else 10**9,
    )
    if failing_groups:
        forms = _forms(target, str(model["capsule"]), model.get("forms"))
        result["forms"] = {g: forms.get(g) for g in failing_groups if forms.get(g) is not None}
        result["capsules_of_form"] = _capsules_of_forms(failing_groups, routes, forms)
    result["status"] = PASS if all(c.get("passed") for c in checks.values()) else FAIL
    result["wall_s"] = round(time.monotonic() - started, 1)
    if not keep_build:
        _prune(out / "build")
    return result


# ------------------------------------------------------------------------------------------ feedback


def feedback_lines(result: Mapping[str, Any]) -> list[str]:
    """What the agent reads: one line per failing check, per group -- names, forms and counts only."""
    lines: list[str] = []
    for model in result.get("models") or ():
        name, checks = model["model"], model.get("checks") or {}
        if model.get("status") == PASS:
            lines.append(f"whole_model {name}: PASS")
            continue
        if model.get("required") is False:
            name += " (reported, not required)"
        build = checks.get("build") or {}
        if not build.get("passed"):
            lines.append(
                f"whole_model {name}: FAIL build -- the model could not be built with this package: "
                f"{build.get('error')}"
            )
            continue
        forms = model.get("forms") or {}
        mapping = model.get("capsules_of_form") or {}

        def _where(group: str) -> str:
            text = _form_text(forms.get(group))
            capsules = (
                (mapping.get(f"g{group}") or {}).get("capsules")
                if isinstance(mapping.get(f"g{group}"), Mapping)
                else None
            )
            return (f" form {text}" if text else "") + (
                f" [capsules of this form: {', '.join(capsules)}]" if capsules else ""
            )

        cov = checks.get("coverage") or {}
        if not cov.get("passed"):
            lines.append(
                f"whole_model {name}: FAIL coverage -- the package compiled {cov.get('package_groups')} of "
                f"{cov.get('device_groups')} device groups ({100 * float(cov.get('share') or 0):.1f}% priced by "
                f"{cov.get('pricing')}); the floor is {100 * float(cov.get('floor') or 0):.1f}%"
            )
            for row in cov.get("declined") or ():
                via = (
                    "its library path needs a prohibited instruction, so it would run on the core"
                    if row.get("on") == ROUTE_HOST
                    else "it falls back to the target's library"
                )
                lines.append(
                    f"  g{row['group']} {row.get('op')}{_where(row['group'])}: NOT COMPILED by the package "
                    f"({row.get('cause')}: {_short(row.get('why'), 200)}); {via}"
                )
        isa = checks.get("no_prohibited_instruction")
        if isa is not None and not isa.get("passed"):
            where = ", ".join(sorted(isa.get("summary") or {})) or isa.get("error") or "unread"
            lines.append(f"whole_model {name}: FAIL no-{'/'.join(isa.get('roles') or ['?'])} -- {where}")
        corr = checks.get("correctness") or {}
        if not corr.get("passed"):
            if corr.get("refused"):
                lines.append(
                    f"whole_model {name}: FAIL correctness -- the program's run was not readable: "
                    f"{_short(corr.get('refused'), 300)}"
                )
            for row in corr.get("not_correct") or ():
                failure = row.get("failure") or {}
                how = (
                    f"{failure.get('mismatches')} of {failure.get('of')} elements differ from a recomputation "
                    "on its own inputs"
                    if "mismatches" in failure
                    else f"{failure.get('over')} elements outside the declared bound"
                    if "over" in failure
                    else "its local check did not print"
                )
                lines.append(f"  g{row['group']} {row.get('kind')}{_where(row['group'])}: WRONG ({how})")
        end = checks.get("end_result") or {}
        if corr.get("passed") and not end.get("passed"):
            detail = (
                f"{end['within']} of {end['of']} output elements are within the model's numeric policy"
                if end.get("basis") == "output tensor" and end.get("of")
                else end.get("note") or "the model's final result disagrees with the oracle"
            )
            lines.append(f"whole_model {name}: FAIL end result -- every group is locally correct, but {detail}")
        accuracy = checks.get("float_accuracy")
        if accuracy is not None and not accuracy.get("passed"):
            detail = (
                f"{accuracy['within']} of {accuracy['of']} output elements are within the capture's tolerance "
                f"of the float model (max abs {accuracy.get('max_abs'):.4g})"
                if accuracy.get("of")
                else accuracy.get("note") or "no float reference to judge against"
            )
            lines.append(f"whole_model {name}: FAIL float accuracy -- {detail}")
    return lines


# ------------------------------------------------------------------------------------------ the gate


def run(
    package: str | Path,
    gate: Mapping[str, Any],
    *,
    target: str,
    roles: Sequence[str],
    out: str | Path,
    root: Path | None = None,
    only: Sequence[str] = (),
    keep_build: bool = False,
    phase0_recipe: str | Path | None = None,
    descriptor: str | Path | None = None,
) -> dict[str, Any]:
    """Every declared model's gate on ``package``; writes ``<out>/whole_model_gate.json``."""
    from merlin.common import provenance as PROV
    from merlin.common.paths import repo_root

    from .package_identity import program_digest

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    models = [m for m in models_of(gate, root=Path(root or repo_root())) if not only or m["name"] in only]
    results = [
        run_model(
            package,
            model,
            target=target,
            roles=roles,
            out=out / model["name"],
            timeout=int(gate.get("timeout_s") or 7200),
            keep_build=keep_build,
            headers=_headers_of(gate, root=Path(root or repo_root())),
            # A model may declare the chunk size its host code is built with; its reference arm uses
            # the same one (see whole_model_reference.produce).
            chunk_ops=model.get("chunk_ops"),
            phase0_recipe=phase0_recipe,
            descriptor=descriptor,
            exactness=contract_of(gate, model, root=Path(root or repo_root()), target=target),
        )
        for model in models
    ]
    try:
        digest = program_digest(Path(package))
    except Exception:  # noqa: BLE001 -- the digest labels the result; it is never a verdict input
        digest = None
    document = {
        "schema": SCHEMA,
        "target": target,
        "package_program_digest": digest,
        "roles": list(roles),
        # A model declared `required: false` is built, run and reported, and does not hold the gate.
        "passed": any(r["required"] for r in results) and all(r["status"] == PASS for r in results if r["required"]),
        "models": results,
        "wall_s": round(time.monotonic() - started, 1),
    }
    document["feedback"] = feedback_lines(document)
    # The gate states a verdict, so it names the hardware revision it came from, as its builds do.
    document["provenance"] = PROV.record(sources=[Path(package)] if package else ())
    (out / RESULT_FILE).write_text(json.dumps(document, indent=1, default=str) + "\n", encoding="utf-8")
    return document


def prepare_references(
    gate: Mapping[str, Any],
    *,
    target: str,
    root: Path | None = None,
    only: Sequence[str] = (),
    jobs: int | None = None,
) -> list[dict[str, Any]]:
    """Produce each declared open model's REFERENCE ARM ahead of any package's gate, on the machine the
    gate would choose for it, so a gate finds it cached rather than building and running it after the
    candidate (hours on a functional simulator for a transformer). A closed model is judged on its class
    and has no reference arm; it is listed and skipped."""
    from . import whole_model_open as WO
    from . import whole_model_reference as R

    base = Path(root or _repo_root())
    headers = _headers_of(gate, root=base)
    timeout = int(gate.get("timeout_s") or 7200)
    done: list[dict[str, Any]] = []
    for model in models_of(gate, root=base):
        if only and model["name"] not in only:
            continue
        if not WO.is_open_model(model["capsule"], target):
            done.append({"model": model["name"], "skipped": "a closed model is judged on its class"})
            continue
        chosen = _READOUT_CHOICE.get(_choice_key(model)) or {
            "machine": str(model["machine"]),
            "header": str(model["header"]),
            "header_sha256": model.get("header_sha256"),
        }

        def _prepare(choice: Mapping[str, Any]) -> dict[str, Any]:
            return R.prepare(
                model["capsule"],
                target=target,
                machine=str(choice["machine"]),
                header=str(choice["header"]),
                header_sha256=choice.get("header_sha256"),
                simulator=str(model.get("simulator") or "spike"),
                timeout=timeout,
                jobs=jobs,
                chunk_ops=model.get("chunk_ops"),
            )

        try:
            entry = _prepare(chosen)
        except Exception as refusal:  # noqa: BLE001 -- only the builder's own readout refusal is caught
            if not str(refusal).startswith(MACHINE_CANNOT_READ_OUT):
                raise
            chosen = choose_machine(
                model, {"_needs_full_width": ["every device group"], "attribution": {}}, target=target, headers=headers
            )
            _READOUT_CHOICE[_choice_key(model)] = chosen
            entry = _prepare(chosen)
        done.append(
            {
                "model": model["name"],
                "machine": chosen["machine"],
                "key": entry.get("key"),
                "output_digest": entry.get("output_digest"),
                "dispatches": entry.get("dispatches"),
                "wall_s": entry.get("wall_s"),
            }
        )
    return done


def _repo_root() -> Path:
    from merlin.common.paths import repo_root

    return repo_root()


def agent_view(document: Mapping[str, Any] | None, *, fresh: bool) -> dict[str, Any] | None:
    """The part of a gate result an agent's verdict carries: pass/fail per model, and the lines."""
    if not document:
        return None
    return {
        "passed": bool(document.get("passed")),
        "models": {m["model"]: m["status"] for m in document.get("models") or ()},
        "for_this_submission": fresh,
        "feedback": list(document.get("feedback") or []),
        "note": (
            "WHOLE-MODEL GATES: each declared model is built with your package for the machine it runs "
            "on, with the prohibited instruction roles trapped, and run on the functional simulator. "
            "The run cannot converge while one fails. Lines name the failing group, its op-form and what "
            "failed; no model tensor or expected value is ever given."
            + ("" if fresh else " These lines are for an EARLIER snapshot of your package.")
        ),
    }


# ------------------------------------------------------------------------------ within a graded run
#
# A run keeps every gate result by the PROGRAM digest of the package it judged (the bytes that can
# change the program: `package_identity.program_digest`), so a result is looked up by what it is
# about, never by when it ran, and an unchanged package is never rebuilt. The directory is the run's
# own (operator-side); the agent's workspace receives only `agent_view`.

GATE_DIR = "whole_model_gate"
_LOCK = threading.Lock()
_IN_FLIGHT: dict[str, Any] = {}


def program_digest_of(submission: str | Path) -> str | None:
    from .package_identity import program_digest

    try:
        return program_digest(Path(submission))
    except Exception:  # noqa: BLE001 -- a package with no program has nothing to look up
        return None


def lookup(run_dir: str | Path, digest: str | None) -> dict[str, Any] | None:
    """The gate result recorded in ``run_dir`` for the program ``digest``, or None."""
    if not digest:
        return None
    path = Path(run_dir) / GATE_DIR / "by_program" / f"{digest}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def latest(run_dir: str | Path) -> dict[str, Any] | None:
    path = Path(run_dir) / GATE_DIR / "latest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _record(run_dir: Path, document: Mapping[str, Any]) -> None:
    base = run_dir / GATE_DIR
    (base / "by_program").mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, indent=1, default=str) + "\n"
    (base / "by_program" / f"{document['package_program_digest']}.json").write_text(text, encoding="utf-8")
    (base / "latest.json").write_text(text, encoding="utf-8")


def evaluate(
    submission: str | Path,
    run_dir: str | Path,
    *,
    target: str,
    gate: Mapping[str, Any],
    roles: Sequence[str],
    key: str,
    root: Path | None = None,
) -> dict[str, Any] | None:
    """The gate result for ``submission``'s program: recorded, in flight (waited for), or run now on a
    private snapshot of it. None when the submission has no program to judge."""
    run_dir = Path(run_dir)
    digest = program_digest_of(submission)
    if digest is None:
        return None
    while True:
        found = lookup(run_dir, digest)
        if found is not None:
            return found
        with _LOCK:
            waiting = _IN_FLIGHT.get(digest)
            if waiting is None:
                event = threading.Event()
                _IN_FLIGHT[digest] = event
                break
        waiting.wait()
    try:
        work = run_dir / "_qa_work" / f"wmgate_{key}"
        if work.exists():
            shutil.rmtree(work)
        snapshot = work / "submission"
        shutil.copytree(Path(submission), snapshot, ignore=shutil.ignore_patterns("build", "__pycache__", ".git"))
        if program_digest_of(snapshot) != digest:
            return None  # edited while it was copied; the next grade judges the settled bytes
        document = run(snapshot, gate, target=target, roles=roles, out=work / "out", root=root)
        document["package_program_digest"] = digest
        document["key"] = key
        _record(run_dir, document)
        shutil.rmtree(snapshot, ignore_errors=True)
        return document
    finally:
        with _LOCK:
            _IN_FLIGHT.pop(digest, None)
        event.set()


def start(submission: str | Path, run_dir: str | Path, **kwargs: Any) -> bool:
    """:func:`evaluate` in a background thread, single-flight: False (and nothing started) while a gate
    is already running or this program already has a result."""
    with _LOCK:
        if _IN_FLIGHT:
            return False
    if lookup(run_dir, program_digest_of(submission)) is not None:
        return False

    def _body() -> None:
        try:
            evaluate(submission, run_dir, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- a background gate must never kill the grader
            print(f"[whole-model gate] {type(exc).__name__}: {exc}", flush=True)

    threading.Thread(target=_body, name="whole-model-gate", daemon=True).start()
    return True


def attach(
    verdict: dict[str, Any],
    submission: str | Path,
    run_dir: str | Path,
    *,
    target: str,
    gate: Mapping[str, Any],
    roles: Sequence[str],
    key: str,
    run_now: bool,
) -> dict[str, Any] | None:
    """Fold the gate into a capsule verdict: its agent view, and -- when the gate is required for the
    freeze -- ``all_pass`` only when the gate PASSED ON THESE BYTES. ``run_now`` runs it (synchronously)
    when a verdict would otherwise converge and no result for this program exists yet."""
    digest = program_digest_of(submission)
    document = lookup(run_dir, digest)
    if document is None and run_now and verdict.get("all_pass"):
        document = evaluate(submission, run_dir, target=target, gate=gate, roles=roles, key=key)
    fresh = document is not None
    shown = document if fresh else latest(run_dir)
    verdict["whole_model_gates"] = agent_view(shown, fresh=fresh) or {
        "passed": False,
        "for_this_submission": False,
        "feedback": [],
        "note": "WHOLE-MODEL GATES have not run on any snapshot of your package yet; they run on a "
        "schedule and before the run may converge.",
    }
    if gate.get("required_for_freeze", True) and verdict.get("all_pass") and not (fresh and document["passed"]):
        verdict["all_pass"] = False
        verdict["not_converged_reason"] = (
            "every capsule passes, but the whole-model gates "
            + ("have not passed on this package" if fresh else "have not yet run on this package")
            + " (see whole_model_gates)"
        )
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--descriptor", type=Path, help="the target descriptor declaring the gate (default: the target's)"
    )
    parser.add_argument("--package", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model", action="append", default=[], help="only this declared model (repeatable)")
    parser.add_argument("--keep-build", action="store_true", help="keep the build's intermediates")
    parser.add_argument(
        "--prepare-reference",
        action="store_true",
        help="produce each declared open model's reference arm (no package) so a later gate finds it cached",
    )
    args = parser.parse_args(argv)
    gate, roles = gate_for(args.target, descriptor=args.descriptor)
    if not gate:
        print(f"{args.target}: the descriptor declares no whole-model gate", file=sys.stderr)
        return 2
    if args.prepare_reference:
        for row in prepare_references(gate, target=args.target, only=args.model):
            print(json.dumps(row, default=str))
        return 0
    if args.package is None or args.out is None:
        parser.error("--package and --out are required unless --prepare-reference")
    document = run(
        args.package, gate, target=args.target, roles=roles, out=args.out, only=args.model, keep_build=args.keep_build
    )
    print("\n".join(document["feedback"]))
    print(f"passed={document['passed']} wall_s={document['wall_s']} -> {args.out / RESULT_FILE}")
    return 0 if document["passed"] else 1


__all__ = [
    "GATE_DIR",
    "SCHEMA",
    "agent_view",
    "attach",
    "choose_machine",
    "coverage",
    "end_result",
    "evaluate",
    "feedback_lines",
    "full_width_machines",
    "gate_for",
    "latest",
    "lookup",
    "main",
    "models_of",
    "prepare_references",
    "program_digest_of",
    "run",
    "run_model",
    "start",
]

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
