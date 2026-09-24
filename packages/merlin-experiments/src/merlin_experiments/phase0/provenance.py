"""Host-owned Phase 0 provenance implementation."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

from merlin_experiments.corpus.phase_selection import generate_phase_selections


def _document_digest(document) -> str:
    """Stable digest for a parsed declaration/fact document."""
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# ------------------------------------------------------------------------------------------------
# deterministic local-path scrub (tracked-file hygiene — no /tmp, /scratch, /home paths ever ship).
# The m2m capture externalizes weights to a NON-deterministic temp dir and stamps its ABSOLUTE path into
# the captured linalg's ``prov.weights_file`` module attribute; the whole-model writer relativizes it but
# the fused/mapped op writer (in capsule_source, outside this file's edit boundary) does not — so we scrub
# the written capsule dir HERE, at the one place this generator owns. Op capsules ship NO weights file
# (their operands come from golden.yaml provenance), so the attribute is non-load-bearing and is STRIPPED;
# an already-relative value (the whole-model ``capsule.weights.safetensors``) is left untouched. Captured
# whole-model loader SOURCES (capsule.pytorch.py) may also carry upstream local-path env-defaults / docstring
# examples — those local-path tokens are redacted. Deterministic (temp-dir name never survives) + idempotent;
# structural string ops only (no regex).
# ------------------------------------------------------------------------------------------------
_LOCAL_ROOTS = ("/tmp", "/scratch", "/home")
_PATH_TERMINATORS = set("\"'") | set(" \t\r\n),]}>")


def _strip_weights_attr(text: str) -> str:
    """Strip an ABSOLUTE ``prov.weights_file = "<abs>"`` module attribute (with its separating comma); leave
    an already-relative value alone. There is exactly one per captured-linalg module."""
    key = 'prov.weights_file = "'
    i = text.find(key)
    if i == -1:
        return text
    j = text.find('"', i + len(key))
    if j == -1:
        return text
    if "/" not in text[i + len(key) : j]:  # already relative (e.g. capsule.weights.safetensors)
        return text
    start, end = i, j + 1
    if text[end : end + 2] == ", ":  # attribute is first: drop trailing ", "
        end += 2
    elif text[start - 2 : start] == ", ":  # attribute is later: drop leading ", "
        start -= 2
    return text[:start] + text[end:]


def _redact_local_paths(text: str) -> str:
    """Replace any absolute local-filesystem path token (``/tmp*`` / ``/scratch*`` / ``/home*``, consumed up
    to the next quote / whitespace / bracket) with the stable placeholder ``<path>``. Idempotent (the
    placeholder contains no local root)."""
    for root in _LOCAL_ROOTS:
        while True:
            i = text.find(root)
            if i == -1:
                break
            j = i + len(root)
            while j < len(text) and text[j] not in _PATH_TERMINATORS:
                j += 1
            text = text[:i] + "<path>" + text[j:]
    return text


def _scrub_capsule_dir(d) -> None:
    """Scrub every tracked-shippable text file in a written capsule dir of non-deterministic / local paths:
    strip the absolute ``prov.weights_file`` from ``*.mlir`` and redact local-path tokens from ``*.mlir`` +
    ``*.py``. Only rewrites a file when its content actually changes (keeps regenerations byte-stable)."""
    if d is None:
        return
    for p in sorted(Path(d).iterdir()):
        if p.suffix not in (".mlir", ".py") or not p.is_file():
            continue
        text = p.read_text(encoding="utf-8")
        scrubbed = text
        if p.suffix == ".mlir":
            scrubbed = _strip_weights_attr(scrubbed)
        scrubbed = _redact_local_paths(scrubbed)
        if scrubbed != text:
            p.write_text(scrubbed, encoding="utf-8")


def _capture_failure_reason(exc: Exception) -> str:
    """The one line that says WHY a capture failed, safe to write into a tracked file.

    Two things go wrong with the obvious ``str(exc)[:300]``. A capture failure carries the worker's
    stderr, and a traceback puts its cause LAST -- so the leading 300 characters are the frames that
    name nothing, and the recorded reason ends mid-path with no error in it, which is the same as
    recording nothing. And the frames are absolute local paths, which must not reach MANIFEST.yaml:
    this repo is published.

    So: the last UNINDENTED line, with local paths redacted by the same helper the capsule scrubber
    uses. Indentation is the structural signal, not a guess about wording -- a traceback indents its
    frame and source lines and leaves the exception flush left, so the last flush-left line is the
    thing that was raised. Taking merely the last non-empty line got this wrong on a real case: the
    export refusal ends with a trailing frame, and the recorded reason came out as ``next(self.gen)``.
    """
    lines = [ln for ln in str(exc).splitlines() if ln.strip()]
    flush = [ln for ln in lines if ln[:1] not in (" ", "\t")]
    tail = (flush or lines or [f"{type(exc).__name__} with no message"])[-1].strip()
    return _redact_local_paths(f"{type(exc).__name__}: {tail}")[:400]


def update_provenance_manifest(
    written,
    cap_root=None,
    *,
    target: str | None = None,
    performance_record: dict | None = None,
    unbuilt_roster: list | None = None,
    claim_model_evaluation: dict | None = None,
    unprovable_forbids: list | None = None,
    superseded: list | None = None,
) -> Path:
    """Rewrite ``MANIFEST.yaml``'s generated/hand_authored split from what this run actually emitted.

    The file's own header has always claimed the generator rewrites it, but no writer existed, so it was
    hand-maintained and silently drifted the moment the corpus grew -- 19 capsules appeared on disk that
    it never listed, and the only thing that noticed was a test telling you to "re-run generate_corpus.py",
    which did not do it.

    MERGE, never replace. A path this run emitted is ``generated``; everything else on disk keeps whatever
    classification it already had, defaulting to ``hand_authored`` for a capsule with no generator. That
    ordering matters: rebuilding the split from scratch would reclassify the frozen hand-authored
    source-of-record (A1, B3/B4, the held-out hidden set) as generated the first time a run happened to
    emit something at the same path.

    Scoped to the SHARED corpus (``<category>/<capsule>``, rel-depth 2). A target with its own nested
    corpus (``atlas/<category>/<capsule>``) carries its own provenance and is deliberately untouched.

    HOLDOUTS ARE COUNTED, NEVER NAMED. This file is tracked and sits inside the ``merlin/contract/``
    tree every arm is granted read-only, so listing a ``hidden/<capsule>`` path told the agent under
    test the op family of a held-out capsule. The generated/hand_authored split is provenance about
    the PUBLIC corpus; the holdouts contribute only a count, which reveals nothing.
    """
    if cap_root is None:
        from merlin.common.paths import checkout_root

        checkout = checkout_root()
        if checkout is None:
            raise ValueError("installed Phase 0 provenance requires an explicit corpus root")
        root = checkout / "merlin/contract/capsules"
    else:
        root = Path(cap_root)
    man_path = root / "MANIFEST.yaml"
    man = yaml.safe_load(man_path.read_text(encoding="utf-8")) if man_path.is_file() else {}
    gen, hand = set(man.get("generated") or []), set(man.get("hand_authored") or [])

    def _rel(d):
        try:
            r = Path(d).resolve().relative_to(root)
        except ValueError:
            return None
        return str(r) if len(r.parts) == 2 else None

    def _held(rel: str) -> bool:
        return rel.split("/", 1)[0] == "hidden"

    for d in written or []:
        rel = _rel(d)
        if rel:
            gen.add(rel)
            hand.discard(rel)
    on_disk = {str(rel) for c in root.rglob("capsule.yaml") if len((rel := c.parent.relative_to(root)).parts) == 2}
    hand |= on_disk - gen - hand  # never seen by a generator -> frozen source-of-record
    gen &= on_disk
    hand &= on_disk  # drop entries whose capsule is gone

    # Split the holdouts back out: they are counted here, never named (see the docstring).
    held_gen, held_hand = {r for r in gen if _held(r)}, {r for r in hand if _held(r)}
    gen -= held_gen
    hand -= held_hand

    man["generated_by"] = "merlin/contract/capsules/generate_corpus.py"
    man["generated"] = sorted(gen)
    man["hand_authored"] = sorted(hand)
    man["held_out"] = {"n_generated": len(held_gen), "n_hand_authored": len(held_hand)}
    if target is not None:
        per_target = dict(man.get("performance_generation") or {})
        per_target[target] = copy.deepcopy(performance_record or {})
        man["performance_generation"] = per_target
        performance_phase = (performance_record or {}).get("phase") or {}
        category = performance_phase.get("category") or "_perf"
        selected = [relative for directory in (written or []) if (relative := _rel(directory)) and not _held(relative)]
        phase_corpora = dict(man.get("phase_corpora") or {})
        phase_corpora[target] = generate_phase_selections(selected, performance_category=category)
        man["phase_corpora"] = phase_corpora
        # WHICH DECLARED ROSTER MODELS THIS TARGET HAS NO WHOLE-MODEL CAPSULE FOR, and why. Recorded
        # per target and rewritten on every full run, so a model that starts capturing stops being
        # listed rather than lingering as stale debt. An EMPTY list is written -- "no roster model is
        # unbuilt" is a result -- while ``None`` leaves the record untouched, because a caller that did
        # not walk the roster (a test rebuilding the split, say) has not learned that nothing is
        # missing. The two absences must not be spelled the same way.
        if unbuilt_roster is not None:
            per_roster = dict(man.get("roster_generation") or {})
            per_roster[target] = {"not_built": copy.deepcopy(unbuilt_roster)}
            man["roster_generation"] = per_roster
        if claim_model_evaluation is not None:
            per_claim = dict(man.get("claim_model_evaluation") or {})
            per_claim[target] = copy.deepcopy(claim_model_evaluation)
            man["claim_model_evaluation"] = per_claim
        # WHICH SYNTHESIZED NEGATIVE-LANE CAPSULES THIS TARGET HAS NONE OF, and why. Same two-absence
        # rule as the roster above: an empty list is the result "every forbid this axis derived is
        # provable", while `None` means nobody looked. Without this the family simply vanishes between
        # the requirement and the corpus, which is the one failure mode the whole axis exists to avoid.
        if unprovable_forbids is not None:
            per_lane = dict(man.get("lane_generation") or {})
            per_lane[target] = {"forbid_not_provable": copy.deepcopy(unprovable_forbids)}
            man["lane_generation"] = per_lane
        # WHICH SYNTHESIZED CAPSULES THIS RUN REMOVED because the requirement stopped asking for their
        # cell. Recorded for the same reason as the two above: a capsule that quietly disappears between
        # one regeneration and the next is indistinguishable from one that was never derived, and the
        # cover would go on citing it from disk until somebody noticed the count.
        if superseded is not None:
            per_sup = dict(man.get("superseded_generation") or {})
            per_sup[target] = {"removed": list(superseded)}
            man["superseded_generation"] = per_sup
    head = man_path.read_text(encoding="utf-8").split("generated_by:")[0] if man_path.is_file() else ""
    man_path.write_text(head + yaml.safe_dump(man, sort_keys=False), encoding="utf-8")
    return man_path
