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

# ---- whole_model_gate --------------------------------------------------------------------------


def feedback_lines(result: Mapping[str, Any]) -> list[str]:
    """What the agent reads: one line per failing check, per group -- names, forms and counts only."""
    from merlin.analysis.whole_model.grading.gate import PASS, ROUTE_HOST, _form_text, _short

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
    from merlin.analysis.perf.measurement.authority import program_digest
    from merlin.analysis.whole_model.grading.gate import (
        PASS,
        RESULT_FILE,
        SCHEMA,
        _headers_of,
        contract_of,
        models_of,
        run_model,
    )
    from merlin.common import provenance as PROV
    from merlin.common.paths import repo_root

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
    from merlin.analysis.whole_model.build import open_program as WO
    from merlin.analysis.whole_model.build import oracle as R
    from merlin.analysis.whole_model.grading.gate import (
        _READOUT_CHOICE,
        MACHINE_CANNOT_READ_OUT,
        _choice_key,
        _headers_of,
        choose_machine,
        models_of,
    )

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


GATE_DIR = "whole_model_gate"


_LOCK = threading.Lock()


_IN_FLIGHT: dict[str, Any] = {}


def program_digest_of(submission: str | Path) -> str | None:
    from merlin.analysis.perf.measurement.authority import program_digest

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
    from merlin.analysis.whole_model.grading.gate import RESULT_FILE, gate_for

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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


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
