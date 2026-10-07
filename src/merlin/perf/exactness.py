"""The exactness contract: per form, how far a program's output may differ from its reference.

Exactness is a decision, not a default someone forgets to change.  A device requantization that rounds
half-to-even where the source rounds half-up is within one output LSB of the reference on every input
and never bit-exact; demanding bit-exactness of it makes the optimization unreachable, and accepting
"close" without saying how close makes every gate soft.  So the contract is REVIEWED DATA, declared per
form by the target's example (``examples/<target>/...``) against ``merlin/schemas/exactness_contract.
schema.yaml``, and every gate that grades a group enforces exactly the bound the contract gives that
group's form and records which contract it applied::

    schema: merlin.exactness_contract.v1
    target: <target>
    default: {mode: exact}                  # the only default there is
    forms:
      - name: <name>
        match: {op: <op>, ...}              # a subset of the group's form (its statement entry / form key)
        exactness:
          mode: bounded                     # exact | bounded
          bound: {max_abs_lsb: 1, max_fraction: 0.01}   # |delta| <= 1 LSB, on at most 1% of elements
          reason: <why this form cannot be bit-exact, reviewed>

THE RULES, EACH A REFUSAL WHEN BROKEN.

* The default is ``exact``; a contract that declares any other default is refused.  Nothing is ever
  loosened implicitly: a form that no entry matches is exact, and so is a form whose match names a
  field the grader cannot see for that group (an unseeable match never loosens).
* A ``bounded`` form states its bound in the source's own units -- the largest absolute difference in
  OUTPUT LSB (``max_abs_lsb``, an integer >= 1) and optionally the largest fraction of the elements that
  may differ at all (``max_fraction``) -- and a reason.  An ``exact`` form states no bound.
* An op whose statement itself declares a tolerance (``bound_lsb`` on its entry: a residual add's two
  rescales) is graded at that bound when no form entry speaks for it, and the record says the op
  declared it (``declared_by: op``).  A form entry overrides it either way -- including back to exact.
* Two matching entries that disagree are a refusal (ambiguous), never a choice.

WHAT A VERDICT SAYS.  ``exact`` or ``bounded(<=N LSB)`` / ``bounded(<=N LSB on <=f of elements)`` -- the
label of the contract that was applied, never "exact" for a bounded grade.  A grade needs evidence of
the size of a difference to pass a bounded contract: a mismatch count or a differing digest says that
something differs, not by how much, so it can satisfy ``exact`` (when it is zero) and fails closed under
``bounded`` otherwise (``verifiable: false``, with why).

Nothing here names a target, an op or a value; the contract is the target's data.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "merlin.exactness_contract.v1"
SCHEMA_NAME = "exactness_contract"
EXACT, BOUNDED = "exact", "bounded"
DECLARED_DEFAULT, DECLARED_OP = "default", "op"
_FORM_KEYS = {"name", "match", "exactness"}
_EXACTNESS_KEYS = {"mode", "bound", "reason"}
_BOUND_KEYS = {"max_abs_lsb", "max_fraction"}
_TOP_KEYS = {"schema", "target", "default", "forms", "notes"}


class ExactnessError(ValueError):
    """The contract is malformed, ambiguous for a group, or cannot be applied as declared."""


@dataclass(frozen=True)
class Exactness:
    """One group's contract: ``exact``, or ``bounded`` by ``max_abs_lsb`` (and optionally ``max_fraction``)."""

    mode: str = EXACT
    max_abs_lsb: int = 0
    max_fraction: float | None = None
    reason: str = ""
    declared_by: str = DECLARED_DEFAULT

    def label(self) -> str:
        if self.mode == EXACT:
            return "exact"
        if self.max_fraction is None:
            return f"bounded(<={self.max_abs_lsb} LSB)"
        return f"bounded(<={self.max_abs_lsb} LSB on <={self.max_fraction:g} of elements)"

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"mode": self.mode, "label": self.label(), "declared_by": self.declared_by}
        if self.mode == BOUNDED:
            out["bound"] = {"max_abs_lsb": self.max_abs_lsb, "max_fraction": self.max_fraction}
            out["reason"] = self.reason
        return out

    @classmethod
    def from_dict(cls, document: Mapping[str, Any] | None) -> Exactness:
        """The inverse of :meth:`to_dict` (an absent document is exact)."""
        if not document:
            return cls()
        bound = document.get("bound") or {}
        return cls(
            mode=str(document.get("mode") or EXACT),
            max_abs_lsb=int(bound.get("max_abs_lsb") or 0),
            max_fraction=None if bound.get("max_fraction") is None else float(bound["max_fraction"]),
            reason=str(document.get("reason") or ""),
            declared_by=str(document.get("declared_by") or DECLARED_DEFAULT),
        )


EXACTLY = Exactness()


def _exactness(body: Any, *, where: str) -> Exactness:
    if not isinstance(body, Mapping):
        raise ExactnessError(f"{where}: exactness must be a mapping")
    unknown = set(body) - _EXACTNESS_KEYS
    if unknown:
        raise ExactnessError(f"{where}: unknown exactness field(s) {sorted(unknown)}")
    mode = body.get("mode")
    if mode == EXACT:
        if body.get("bound") is not None:
            raise ExactnessError(f"{where}: an exact form states no bound")
        return Exactness(reason=str(body.get("reason") or ""), declared_by=where)
    if mode != BOUNDED:
        raise ExactnessError(f"{where}: mode is {EXACT!r} or {BOUNDED!r}, not {mode!r}")
    bound = body.get("bound")
    if not isinstance(bound, Mapping) or set(bound) - _BOUND_KEYS:
        raise ExactnessError(f"{where}: a bounded form states bound: {{max_abs_lsb[, max_fraction]}}")
    lsb = bound.get("max_abs_lsb")
    if isinstance(lsb, bool) or not isinstance(lsb, int) or lsb < 1:
        raise ExactnessError(f"{where}: max_abs_lsb is an integer number of output LSB, at least 1")
    fraction = bound.get("max_fraction")
    if fraction is not None and (
        isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < float(fraction) <= 1
    ):
        raise ExactnessError(f"{where}: max_fraction is a fraction of the elements in (0, 1]")
    reason = str(body.get("reason") or "").strip()
    if not reason:
        raise ExactnessError(f"{where}: a bounded form states the reviewed reason it cannot be exact")
    return Exactness(
        mode=BOUNDED,
        max_abs_lsb=int(lsb),
        max_fraction=None if fraction is None else float(fraction),
        reason=reason,
        declared_by=where,
    )


def _matches(match: Mapping[str, Any], form: Mapping[str, Any]) -> bool:
    """Every field of ``match`` is SEEN on ``form`` and equal to it (a list on a scalar field is one-of).
    A field the form does not carry never matches: an unseeable match never loosens a group."""
    for key, want in match.items():
        if key not in form:
            return False
        have = form[key]
        if isinstance(want, list) and not isinstance(have, list):
            if have not in want:
                return False
        elif have != want:
            return False
    return True


class Contract:
    """A validated exactness contract (see the module doc)."""

    def __init__(self, document: Mapping[str, Any], *, path: str | Path | None = None) -> None:
        if not isinstance(document, Mapping):
            raise ExactnessError("an exactness contract is a mapping")
        if document.get("schema") != SCHEMA:
            raise ExactnessError(f"an exactness contract declares schema {SCHEMA}")
        unknown = set(document) - _TOP_KEYS
        if unknown:
            raise ExactnessError(f"unknown top-level field(s) {sorted(unknown)}")
        from merlin.common.schemas import validate

        problems = validate(dict(document), SCHEMA_NAME)
        if problems:
            raise ExactnessError("; ".join(problems))
        default = document.get("default") or {"mode": EXACT}
        if not isinstance(default, Mapping) or default.get("mode") != EXACT or set(default) - {"mode"}:
            raise ExactnessError("the default is exact: a contract may loosen named forms, never every form")
        forms = document.get("forms")
        if not isinstance(forms, list):
            raise ExactnessError("forms is a list (empty when every form is exact)")
        seen: set[str] = set()
        self.forms: list[tuple[str, dict[str, Any], Exactness]] = []
        for index, entry in enumerate(forms):
            where = f"forms[{index}]"
            if not isinstance(entry, Mapping) or set(entry) - _FORM_KEYS:
                raise ExactnessError(f"{where}: a form entry has name, match and exactness")
            name = str(entry.get("name") or "").strip()
            if not name or name in seen:
                raise ExactnessError(f"{where}: every form entry has its own non-empty name")
            seen.add(name)
            match = entry.get("match")
            if not isinstance(match, Mapping) or not match:
                raise ExactnessError(f"{where} ({name}): match names at least one form field")
            self.forms.append((name, dict(match), _exactness(entry.get("exactness"), where=f"form:{name}")))
        self.document = copy.deepcopy(dict(document))
        self.path = str(path) if path is not None else None
        self.sha256 = hashlib.sha256(json.dumps(self.document, sort_keys=True).encode("utf-8")).hexdigest()
        #: What GRADES: each form's match and bound, without names, reasons or notes -- two contracts with
        #: the same semantics grade every group the same way, and a result graded under one stands under
        #: the other.
        semantics = sorted(
            json.dumps({"match": m, "mode": e.mode, "lsb": e.max_abs_lsb, "fraction": e.max_fraction}, sort_keys=True)
            for _n, m, e in self.forms
        )
        self.semantics_sha256 = hashlib.sha256(json.dumps(semantics).encode("utf-8")).hexdigest()

    def resolve(self, form: Mapping[str, Any] | None, *, op_bound_lsb: int | None = None) -> Exactness:
        """The contract for one group, given what is known of its form (statement entry fields and/or its
        form key) and the tolerance its op itself declares, if any (see the module doc)."""
        found = [(name, exactness) for name, match, exactness in self.forms if _matches(match, form or {})]
        if len({(e.mode, e.max_abs_lsb, e.max_fraction) for _name, e in found}) > 1:
            raise ExactnessError(
                f"the contract is ambiguous for {dict(form or {})}: entries {[n for n, _e in found]} disagree"
            )
        if found:
            return found[0][1]
        if op_bound_lsb is not None and int(op_bound_lsb) > 0:
            return Exactness(
                mode=BOUNDED,
                max_abs_lsb=int(op_bound_lsb),
                reason="the op's own statement declares this tolerance (bound_lsb)",
                declared_by=DECLARED_OP,
            )
        return EXACTLY

    def record(self) -> dict[str, Any]:
        """What a gate records about the contract it applied: its identity and every non-exact form."""
        return {
            "schema": SCHEMA,
            "sha256": self.sha256,
            "semantics_sha256": self.semantics_sha256,
            "path": self.path,
            "target": self.document.get("target"),
            "default": "exact",
            "forms": [{"name": n, "match": m, **e.to_dict()} for n, m, e in self.forms],
        }

    def to_document(self) -> dict[str, Any]:
        """The contract by value (what a run stamps into its config), with where it was read from."""
        return {"document": copy.deepcopy(self.document), "path": self.path, "sha256": self.sha256}

    @classmethod
    def default(cls, *, target: str | None = None) -> Contract:
        """Every form exact (op-declared tolerances still apply) -- what a target with no contract gets."""
        return cls({"schema": SCHEMA, "target": target or "", "default": {"mode": EXACT}, "forms": []})

    @classmethod
    def from_value(cls, value: Mapping[str, Any] | None, *, target: str | None = None) -> Contract:
        """A contract a run carried by value (:meth:`to_document`), or the default when it carried none."""
        if not value:
            return cls.default(target=target)
        contract = cls(value["document"], path=value.get("path"))
        if value.get("sha256") and value["sha256"] != contract.sha256:
            raise ExactnessError("the carried exactness contract does not hash to the digest it was sealed with")
        return contract


def load(path: str | Path) -> Contract:
    """The contract at ``path`` (YAML or JSON)."""
    import yaml

    path = Path(path)
    if not path.is_file():
        raise ExactnessError(f"no exactness contract at {path}")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ExactnessError(f"{path} is not YAML: {exc}") from exc
    return Contract(document, path=path)


# ------------------------------------------------------------------------------------------ grading


def judge(
    exactness: Exactness,
    *,
    max_abs: int | None = None,
    mismatches: int | None = None,
    elements: int | None = None,
    equal: bool | None = None,
) -> dict[str, Any]:
    """Grade one group's comparison evidence against its contract.

    Evidence, strongest first: ``max_abs`` (the largest absolute difference, in output LSB) with
    ``mismatches`` (how many elements differ at all) and ``elements``; else ``mismatches`` alone; else
    ``equal`` (a digest comparison).  ``passed`` is True only when the evidence shows the bound held;
    evidence that cannot show it under a bounded contract is ``verifiable: false`` and never passes."""
    out: dict[str, Any] = {"contract": exactness.label(), "mode": exactness.mode, "declared_by": exactness.declared_by}
    evidence = {
        k: v for k, v in (("max_abs", max_abs), ("mismatches", mismatches), ("elements", elements)) if v is not None
    }
    if equal is not None:
        evidence["digest_equal"] = bool(equal)
    out["evidence"] = evidence

    def done(passed: bool, why: str, *, verifiable: bool = True) -> dict[str, Any]:
        out.update(passed=bool(passed), verifiable=verifiable, why=why)
        return out

    nothing_differs = (max_abs is not None and int(max_abs) == 0) or (
        max_abs is None and mismatches is not None and int(mismatches) == 0 and (elements is None or int(elements) > 0)
    )
    if max_abs is None and mismatches is None and equal is None:
        return done(False, "no comparison evidence was recorded for this group", verifiable=False)
    if exactness.mode == EXACT:
        if max_abs is not None or mismatches is not None:
            return done(nothing_differs, "every element equals the reference" if nothing_differs else "elements differ")
        return done(bool(equal), "the digest equals the reference's" if equal else "the digest differs")
    # BOUNDED
    if nothing_differs or (max_abs is None and mismatches is None and equal):
        return done(True, "every element equals the reference (within any bound)")
    if max_abs is None:
        return done(
            False,
            f"{exactness.label()}: the check reports that elements differ, not by how much; the bound cannot be "
            "verified from it",
            verifiable=False,
        )
    if int(max_abs) > exactness.max_abs_lsb:
        return done(False, f"max |delta| {max_abs} LSB exceeds {exactness.label()}")
    if exactness.max_fraction is not None:
        if mismatches is None or not elements:
            return done(
                False,
                f"{exactness.label()}: the check states the largest difference but not how many elements "
                "differ, so the fraction cannot be verified",
                verifiable=False,
            )
        share = int(mismatches) / int(elements)
        if share > exactness.max_fraction:
            return done(False, f"{mismatches} of {elements} elements differ ({share:.4g}), over {exactness.label()}")
    return done(True, f"max |delta| {max_abs} LSB within {exactness.label()}")


def judge_arrays(got: Any, want: Any, exactness: Exactness) -> dict[str, Any]:
    """:func:`judge` over two integer arrays of one group's output (device and reference)."""
    import numpy as np

    got = np.asarray(got).astype(np.int64).reshape(-1)
    want = np.asarray(want).astype(np.int64).reshape(-1)
    if got.shape != want.shape:
        out = judge(exactness, mismatches=max(got.size, want.size, 1), elements=max(got.size, want.size, 1))
        out.update(
            passed=False, verifiable=True, why=f"the device holds {got.size} elements, the reference {want.size}"
        )
        return out
    delta = np.abs(got - want)
    return judge(
        exactness,
        max_abs=int(delta.max()) if delta.size else 0,
        mismatches=int((delta != 0).sum()),
        elements=int(delta.size),
    )


# ------------------------------------------------------------------- a whole-model verdict, re-judged

#: Group states a re-judgement may move (correct <-> failed); every other state (unverifiable, excused
#: because the reference fails it identically, a declared machine limit) is left as the rules set it.
_GRADED = ("correct", "failed")


def row_evidence(row: Mapping[str, Any]) -> dict[str, Any]:
    """The comparison evidence a verdict row carries (see :func:`merlin.perf.whole_model_verdict.judge`)."""
    evidence: dict[str, Any] = {}
    if row.get("max_abs") is not None:
        evidence["max_abs"] = int(row["max_abs"])
    if row.get("mismatches") is not None:
        evidence["mismatches"] = int(row["mismatches"])
    if row.get("elements") is not None:
        evidence["elements"] = int(row["elements"])
    if not evidence and row.get("basis") in ("oracle", "reference"):
        evidence["equal"] = row.get("state") == "correct"
    return evidence


def apply_to_verdict(
    verdict: Mapping[str, Any],
    resolve: Callable[[str, Mapping[str, Any]], Exactness],
    *,
    contract: Contract,
) -> dict[str, Any]:
    """``verdict`` with every graded group held to its own contract, and the contract recorded.

    ``resolve(group, row)`` gives a group's :class:`Exactness`.  A group whose board bytes differ from its
    functional model's is never re-judged (a race is not a tolerance); a group with no evidence of how far
    it differs fails a bounded contract closed.  Correctness, the timing status and the objective cycles
    are recomputed from the re-judged rows exactly as the verdict's own rules do."""
    from . import whole_model_verdict as V

    document = copy.deepcopy(dict(verdict))
    record: dict[str, Any] = {"contract": contract.record(), "per_group": {}, "summary": {}}
    if document.get("timing_status") not in (V.TIMING_MEASURED, V.TIMING_MEASURED_INVALID):
        document["exactness"] = record
        return document
    rows = {str(row["group"]): row for row in document.get("groups") or []}
    changed = []
    for key in sorted(rows, key=V._order):
        row = rows[key]
        try:
            exactness = resolve(key, row)
        except ExactnessError as exc:
            row.update(state=V.GROUP_FAILED, exactness={"contract": "ambiguous", "why": str(exc), "passed": False})
            changed.append(key)
            continue
        label = exactness.label()
        record["per_group"][key] = label
        record["summary"][label] = record["summary"].get(label, 0) + 1
        # THE VERDICT'S OWN RULE ALREADY IS THIS CONTRACT when an exact group is held exact, or an op's own
        # declared tolerance is the bound: then nothing is re-judged, only labelled.
        builtin = (exactness.mode == EXACT and row.get("compare") == V.COMPARE_EXACT) or (
            exactness.declared_by == DECLARED_OP and row.get("compare") == V.COMPARE_BOUNDED
        )
        if builtin or row.get("state") not in _GRADED or row.get("board_bytes_equal_functional_model") is False:
            row["exactness"] = {"contract": label, "declared_by": exactness.declared_by, "regraded": False}
            continue
        grade = judge(exactness, **row_evidence(row))
        row["exactness"] = grade
        state = V.GROUP_CORRECT if grade["passed"] else V.GROUP_FAILED
        if state != row["state"]:
            row["state"] = state
            row["detail"] = f"{row.get('detail') or ''}; exactness {label}: {grade['why']}".lstrip("; ")
            changed.append(key)
        row["correct"] = row["state"] == V.GROUP_CORRECT
    if changed:
        failures = sorted((k for k, r in rows.items() if r.get("state") == V.GROUP_FAILED), key=V._order)
        correctness = document.get("correctness") or {}
        argmax = correctness.get("argmax") or {}
        argmax_ok = bool(
            argmax.get("agrees_with_oracle")
            or correctness.get("argmax_vendor_also_fails")
            or correctness.get("argmax_machine_cannot_express")
        )
        correct = not failures and argmax_ok and not correctness.get("groups_absent")
        correctness.update(status="pass" if correct else "fail", groups_failed=failures)
        document["correctness"] = correctness
        document["timing_status"] = V.TIMING_MEASURED if correct else V.TIMING_MEASURED_INVALID
        document["objective_cycles"] = document.get("whole_window_cycles") if correct else None
        if correct:
            document.pop("invalid_reason", None)
        else:
            document["invalid_reason"] = (
                f"{len(failures)} group(s) failed their exactness contract or check ({failures[:12]})"
            )
    record["regraded_groups"] = changed
    document["exactness"] = record
    return document


#: The semantics digest of the default contract (every form exact): what a verdict graded before contracts
#: were recorded was graded under -- the verdict's own rules are exactly the default contract.
DEFAULT_SEMANTICS_SHA256 = hashlib.sha256(json.dumps([]).encode("utf-8")).hexdigest()


def graded_semantics(verdict: Mapping[str, Any] | None) -> str:
    """The semantics digest of the contract ``verdict`` was graded under (the default's when it recorded none)."""
    recorded = (((verdict or {}).get("exactness") or {}).get("contract") or {}).get("semantics_sha256")
    return str(recorded or DEFAULT_SEMANTICS_SHA256)


def label_summary(record: Mapping[str, Any] | None) -> str:
    """``exact`` when every group was held exact, else each contract with its group count."""
    summary = (record or {}).get("summary") or {}
    if not summary:
        return "unrecorded"
    if set(summary) == {"exact"}:
        return "exact"
    return ", ".join(f"{label} x{count}" for label, count in sorted(summary.items()))


def resolver(
    contract: Contract,
    *,
    forms: Mapping[str, Mapping[str, Any]] | None = None,
    routes: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]] | None = None,
    op_bounds: Mapping[str, int | None] | None = None,
) -> Callable[[str, Mapping[str, Any]], Exactness]:
    """``resolve(group, row)`` from what a caller knows of each group: its form (``forms``: a statement
    entry and/or form key per group), its route's op (``routes``: rows with ``group`` and ``op``) and the
    tolerance its op declares (``op_bounds``)."""
    if routes is not None and not isinstance(routes, Mapping):
        routes = {str(r.get("group")): r for r in routes if isinstance(r, Mapping)}
    forms = {str(k): dict(v) for k, v in (forms or {}).items() if isinstance(v, Mapping)}
    bounds = {str(k): v for k, v in (op_bounds or {}).items()}

    def resolve(group: str, row: Mapping[str, Any]) -> Exactness:
        form = dict(forms.get(str(group)) or {})
        op = ((routes or {}).get(str(group)) or {}).get("op")
        if op is not None:
            form.setdefault("op", op)
        return contract.resolve(form, op_bound_lsb=bounds.get(str(group)))

    return resolve


__all__ = [
    "BOUNDED",
    "Contract",
    "DECLARED_OP",
    "EXACT",
    "EXACTLY",
    "Exactness",
    "ExactnessError",
    "SCHEMA",
    "DEFAULT_SEMANTICS_SHA256",
    "apply_to_verdict",
    "graded_semantics",
    "judge",
    "judge_arrays",
    "label_summary",
    "load",
    "resolver",
    "row_evidence",
]
