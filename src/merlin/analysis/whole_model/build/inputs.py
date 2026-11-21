"""Load a whole-model capsule and its independent numerical reference."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

# ---- whole_model_capsule -----------------------------------------------------------------------


class WholeModelBuildError(RuntimeError):
    """The whole model cannot be built as one program, and the message says which part stopped it."""


# --------------------------------------------------------------------------------------------- capsule


@dataclasses.dataclass(frozen=True)
class ModelCapsule:
    """A model capsule's files and the numbers its golden was computed on."""

    directory: Path
    name: str
    interface: Path
    weights: Path
    weights_manifest: Path
    inputs: Mapping[str, Any]
    outputs: Mapping[str, Any]


def load_model_capsule(path: str | Path) -> ModelCapsule:
    """Read a model capsule the way the capsule runner does, refusing one that is not whole.

    The weights and the golden are gitignored beside the tracked capsule, so a checkout can hold the
    interface without them; that is named here rather than discovered as a missing file mid-build.
    """
    import numpy as np

    from merlin.development.targetgen.corpus import capsule_io as CC
    from merlin.development.targetgen.grading import golden_inputs as CI

    capsule = CC.load_capsule(path)
    directory = Path(capsule["__dir__"])
    attrs = (capsule.get("operation") or {}).get("attributes") or {}
    interface = directory / str(capsule.get("interface_mlir") or "capsule.interface.mlir")
    if not attrs.get("weights") or not attrs.get("weights_manifest"):
        raise WholeModelBuildError(f"capsule {directory.name!r} declares no weights, so it is not a model capsule")
    weights, manifest = directory / str(attrs["weights"]), directory / str(attrs["weights_manifest"])
    missing = [str(p) for p in (interface, weights, manifest) if not p.is_file()]
    if missing:
        raise WholeModelBuildError(f"capsule {directory.name!r} is missing {missing}")
    declared = CI.canonical_input_values(capsule, directory)
    if not declared:
        raise WholeModelBuildError(f"capsule {directory.name!r} records no decoded input in its golden")
    inputs = {
        name: np.asarray(spec["values"], dtype=np.float32).reshape([int(e) for e in spec["shape"]])
        for name, spec in declared.items()
    }
    outputs = {name: np.asarray(value, dtype=np.float32) for name, value in _model_golden(capsule, directory).items()}
    return ModelCapsule(directory, directory.name, interface, weights, manifest, inputs, outputs)


def _model_golden(capsule: Mapping[str, Any], directory: Path) -> dict[str, Any]:
    """A MODEL capsule's captured outputs: its independent golden, read as declared.

    A whole model's expectation is never recomputed here -- the integer Tensor engine cannot evaluate
    an ``op: model`` graph -- so a capsule whose golden declares the recompute source, or declares no
    outputs, is refused by name rather than graded against something invented.
    """
    from merlin.development.targetgen.grading import golden_inputs as GP
    from merlin.development.targetgen.grading.golden_store import load_golden

    if GP.golden_source(dict(capsule), directory) == "merlin_tensor_int":
        raise WholeModelBuildError(f"capsule {directory.name!r} declares no independent golden for its outputs")

    outputs = (load_golden(directory) or {}).get("outputs")
    if not outputs:
        raise WholeModelBuildError(f"capsule {directory.name!r} golden records no outputs")
    return dict(outputs)


# ---- whole_model_capsules ----------------------------------------------------------------------
# Which capsules answer for a whole-model group: its exact op-form first, its op otherwise.
#
# Shared by the whole-model gate and the measured mode's capsule check, so a failing group is mapped to
# the same capsules wherever it is reported.  The catalog is an explicit directory; the group's form
# comes from the caller (derived once from the one grouping), never recomputed here.

#: How many capsules a group name expands to, at most, so a check stays minutes long.
GROUP_CAPSULE_LIMIT = 12


def _capsule_rows(catalog: Path | None) -> list[dict[str, Any]]:
    import yaml

    if catalog is None or not catalog.is_dir():
        return []  # the launch names its capsule catalog; there is no implicit checkout corpus
    root = catalog
    rows = []
    for path in sorted(root.rglob("capsule.yaml")):
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        if document.get("label") not in (None, "public"):
            continue
        form = (
            ((document.get("model_form") or {}).get("form") or {})
            if isinstance(document.get("model_form"), Mapping)
            else {}
        )
        rows.append(
            {
                "name": str(document.get("name") or path.parent.name),
                "op": str(((document.get("operation") or {}).get("op")) or form.get("op") or ""),
                "model_form": bool(form),
                "form": dict(form) if form else None,
                "extends": document.get("extends"),
            }
        )
    return rows


def _form_text(form: Mapping[str, Any] | None) -> str | None:
    return json.dumps(form, sort_keys=True, default=str) if form else None


def resolve_capsules(
    names: str,
    routes: Mapping[str, Mapping[str, Any]],
    catalog: Path | None = None,
    forms: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """``names`` is a comma list of capsule names and/or groups (``g6``).

    A group becomes the capsules of ITS OWN op-form when ``forms`` names it (``{group: form}``, see
    the launch's declared ``group_forms``): the capsules whose declared model form is that form exactly, the quick one
    before its certificate sibling. When the corpus holds no capsule of that form the mapping says so
    (``exact_form: false``) and the group falls back to the capsules of its op -- model-form capsules
    first (they carry the model's own extents). Certificate-spill siblings are always left out (they
    are sized past a quick check); at most GROUP_CAPSULE_LIMIT per group."""
    rows = _capsule_rows(catalog)
    known = {row["name"] for row in rows}
    chosen: list[str] = []
    mapping: dict[str, Any] = {}
    for token in (part.strip() for part in str(names or "").split(",")):
        if not token:
            continue
        index = token[1:] if token[:1] in ("g", "G") and token[1:].isdigit() else (token if token.isdigit() else None)
        if index is None:
            mapping[token] = token if token in known else "unknown capsule (the check will say so)"
            if token not in chosen:
                chosen.append(token)
            continue
        op = str((routes.get(index) or {}).get("op") or "")
        if not op:
            mapping[token] = "no build of this model has named this group's op yet"
            continue
        wanted = _form_text((forms or {}).get(index))
        exact = [r for r in rows if wanted and _form_text(r.get("form")) == wanted and not r["name"].endswith("_spill")]
        if exact:
            candidates = sorted(exact, key=lambda r: (r["name"].endswith("_cert"), r["name"]))
        else:
            candidates = [r for r in rows if r["op"] == op and not r["name"].endswith("_spill")]
            candidates.sort(key=lambda r: (not r["model_form"], r["name"].endswith("_cert"), r["name"]))
        picked = [r["name"] for r in candidates[:GROUP_CAPSULE_LIMIT]]
        mapping[token] = {
            "op": op,
            "capsules": picked,
            "of_matching": len(candidates),
            "exact_form": bool(exact),
            **({"form": (forms or {}).get(index)} if wanted else {}),
            **(
                {}
                if exact or not wanted
                else {"note": "the corpus holds no capsule of this group's exact op-form; these share only its op"}
            ),
        }
        chosen.extend(name for name in picked if name not in chosen)
    return chosen, mapping


# ---- whole_model_passes ------------------------------------------------------------------------
# A package's own MLIR passes over the whole-model interface, run before the builder splits it.
#
# The whole-model statement (:mod:`merlin.compiler.lower.device.whole_program`) turns a captured model into compute
# GROUPS and puts each -- or, since :mod:`merlin.compiler.lower.device.region`, a legal run of them -- to
# the package. Neither lets a package change what the MODEL ITSELF says before that split happens: a
# package that would rather restate its own program (canonicalize a chain of casts, hoist a constant,
# whatever transform it can prove preserves the function) has no way to ask for it. This module is that
# way, and nothing more: it runs the package's OWN declared passes, over the package's OWN interpreter,
# through the exact entrypoint mechanism every other package call already uses
# (:func:`merlin.development.targetgen.packaging.loading.run_entrypoint`) -- never by importing or executing package code
# in this process.
#
# **Declaration is data, not code.** A package's ``manifest.yaml`` may carry a top-level
# ``whole_model_passes: [name, ...]`` list; each name must also be one of its own declared ``commands``
# (the same manifest section every entrypoint already comes from), run in the ORDER the list gives, each
# one's own declared ``argv`` invoked exactly like any other entrypoint (``{tool}``/``{input_mlir}``/
# ``{output_json}``) with the previous stage's output as the next stage's input. Nothing here executes a
# name the manifest omits, and nothing here decides what a pass may compute: verifying that a pass did
# not change the model's own function is a SEPARATE, independent check
# (:func:`merlin.analysis.whole_model.build.stages.verify_passes_preserve_semantics`), because deciding whether to
# trust a transform is not this module's job -- running the ones asked for, in order, and recording
# exactly what happened, is.
#
# **Fails closed, one pass at a time.** A pass that cannot even be invoked, or whose own reply is not
# parseable MLIR text, stops the chain right there: the caller gets back the LAST module every prior
# pass in the chain actually produced (the original, if the very first pass failed), together with a
# per-pass record naming which ones ran and which one stopped it. A caller that skipped this and used
# whatever text happened to be on disk would silently grade one package's chosen edits and call it every
# other package's default behavior.

#: The manifest's own list of pass command names, in the order they run. A top-level key (the
#: manifest schema's root already allows arbitrary additional properties), never inferred from a
#: naming convention on `commands` -- a sorted-by-name guess is an assumption about intent this
#: module does not make.
PASS_LIST_KEY = "whole_model_passes"


class PassDeclarationError(RuntimeError):
    """The manifest's own pass list cannot be trusted, named by what is wrong with it."""


def declared_passes(package: Any) -> tuple[str, ...]:
    """The package's own ordered pass list, or ``()`` when it declares none.

    Raises :class:`PassDeclarationError` when the list names something that is not one of the
    package's own declared commands: a manifest that cannot even name its passes correctly is not
    trusted to run any of them, rather than running the ones that happen to resolve and silently
    skipping the rest.
    """
    manifest = getattr(package, "manifest", None) or {}
    names = manifest.get(PASS_LIST_KEY)
    if not names:
        return ()
    if not isinstance(names, list) or not all(isinstance(n, str) and n for n in names):
        raise PassDeclarationError(f"{PASS_LIST_KEY!r} must be a list of non-empty command names, got {names!r}")
    commands = manifest.get("commands") or {}
    missing = [n for n in names if n not in commands]
    if missing:
        raise PassDeclarationError(
            f"{PASS_LIST_KEY!r} names {missing}, which the package declares no command for "
            f"(it declares {sorted(commands)})"
        )
    return tuple(names)


def apply_passes(
    interface_path: str | Path,
    package: Any,
    *,
    work: str | Path,
    timeout: int = 600,
    run=None,
) -> tuple[Path, list[dict[str, Any]], str]:
    """Run every pass ``package`` declares, in order, chaining each stage's output into the next.

    Returns ``(final_path, records, stopped_why)``: ``final_path`` is the last module any pass
    actually produced (``interface_path`` itself when no pass ran or the first one failed);
    ``records`` is one entry per ATTEMPTED pass (``name``, ``input``, ``output`` paths and digests,
    ``ok``); ``stopped_why`` is ``""`` when every declared pass ran, else the reason the chain
    stopped, naming which pass.
    """
    from merlin.common.provenance import source_digest
    from merlin.development.targetgen.packaging import loading as OR

    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    names = declared_passes(package)
    current = Path(interface_path)
    records: list[dict[str, Any]] = []
    for position, name in enumerate(names):
        destination = work / f"pass_{position:02d}_{name}.mlir"
        try:
            done = (run or OR.run_entrypoint)(package, name, current, destination, timeout=timeout)
        except Exception as error:  # noqa: BLE001 -- named and recorded, never raised past this point
            records.append({"pass": name, "ok": False, "why": f"{type(error).__name__}: {error}"})
            return current, records, f"pass {name!r} could not be invoked: {type(error).__name__}: {error}"
        if (
            getattr(done, "returncode", 1) != 0
            or not destination.is_file()
            or not destination.read_text(encoding="utf-8").strip()
        ):
            reason = (getattr(done, "stderr", "") or "the pass wrote no output")[:300]
            records.append({"pass": name, "ok": False, "why": reason})
            return current, records, f"pass {name!r} produced no usable module: {reason}"
        records.append(
            {
                "pass": name,
                "ok": True,
                "input": str(current),
                "input_sha256": source_digest([current]),
                "output": str(destination),
                "output_sha256": source_digest([destination]),
            }
        )
        current = destination
    return current, records, ""


# ---- whole_model_partial -----------------------------------------------------------------------
# A whole-model build of only some groups: how it is made, and why it can never be read as a whole model.
#
# ``only_groups=["g12"]`` (``--only-group g12``) builds the program with the package asked for g12 alone.
# Every other group keeps the target's library call exactly as a caller-declined group does
# (``decline=``): it is stated by the reference, never put to the package, so a one-group debug build
# costs one question instead of one per group. Which groups the model HAS is read off the statement
# with no package (the same reference statement the per-group timing programs use), so a misspelled
# group is refused rather than silently building nothing.
#
# THAT PROGRAM RUNS AND PRINTS EVERY GROUP LINE. Nothing in its log says the other groups are the
# library's, so the build marks itself: ``partial_build`` on the build record, on its oracle, on the
# expectations a measurement reads, and a ``PARTIAL_BUILD.json`` beside them. :func:`refuse` is what a
# verdict, a grade, a gate and a measurement call on what they are handed; a partial build is refused
# by each, never scored.

#: The key a partial build carries on every document a whole-model reader is handed.
MARKER = "partial_build"
#: The file a partial build writes beside its record, so a directory listing says so too.
MARKER_FILE = "PARTIAL_BUILD.json"


class PartialBuildRefused(ValueError):
    """A partial (``only_groups``) build was handed to something that reads a whole model."""


def parse_groups(items: Iterable[Any]) -> list[int]:
    """``["g12,g14", 3]`` -> ``[3, 12, 14]``: group indices as the build names them (``g12`` or ``12``)."""
    groups: set[int] = set()
    for item in items:
        if isinstance(item, bool):
            raise ValueError(f"{item!r} names no group")
        for token in str(item).split(",") if not isinstance(item, int) else [str(item)]:
            token = token.strip()
            if not token:
                continue
            digits = token[1:] if token[:1] == "g" else token
            if not digits.isdigit():
                raise ValueError(f"{token!r} names no group; spell a group as g<index> (e.g. g12)")
            groups.add(int(digits))
    return sorted(groups)


def unasked(capsule: Any, *, target: str, only: Sequence[Any]) -> list[int]:
    """Every group of ``capsule``'s model that is NOT in ``only`` -- the groups a partial build declines.

    Read off the reference statement (no package asked anything), so the indices are the model's own;
    a group ``only`` names that the model does not have is refused."""
    from merlin.analysis.whole_model.build import stages as W

    keep = parse_groups(only)
    if not keep:
        raise W.WholeModelBuildError("only_groups names no group")
    stated = W.state(capsule, target=target)
    present = [int(row["group"]) for row in stated["whole_program"]["per_group"]]
    missing = sorted(set(keep) - set(present))
    if missing:
        raise W.WholeModelBuildError(
            f"only_groups names {[f'g{g}' for g in missing]}, which this model does not have "
            f"(its groups are g{min(present)}..g{max(present)})"
        )
    return [group for group in present if group not in keep]


def marker(only: Sequence[Any], attribution: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """What a partial build was: the groups asked for, the ones the package answered, and why it is not a
    whole model."""
    keep = parse_groups(only)
    answered = sorted(int(r["group"]) for r in attribution if r.get("on") == "package")
    return {
        "only_groups": [f"g{g}" for g in keep],
        "package_answered": [f"g{g}" for g in answered],
        "why": (
            "built with only_groups: every other group is the target's library call, so this program is "
            "not the candidate's whole model and is never measured, graded or gated as one"
        ),
    }


def mark(record: dict[str, Any], out: str | Path, only: Sequence[Any]) -> dict[str, Any]:
    """Put the marker on ``record``, on the oracle the build wrote, and beside them as :data:`MARKER_FILE`."""
    note = marker(only, (record.get("attribution") or {}).get("per_group") or ())
    record[MARKER] = note
    out = Path(out)
    (out / MARKER_FILE).write_text(json.dumps(note, indent=1) + "\n", encoding="utf-8")
    oracle = out / "oracle.json"
    if oracle.is_file():
        document = json.loads(oracle.read_text(encoding="utf-8"))
        document[MARKER] = note
        oracle.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    return note


def refuse(document: Mapping[str, Any] | None, *, reader: str) -> None:
    """Raise :class:`PartialBuildRefused` when ``document`` (a build record, an oracle, expectations)
    carries the partial-build marker; ``reader`` names who refused it."""
    note = (document or {}).get(MARKER) if isinstance(document, Mapping) else None
    if note:
        groups = ", ".join((note or {}).get("only_groups") or []) if isinstance(note, Mapping) else ""
        raise PartialBuildRefused(
            f"{reader} refuses a partial build (only {groups or 'some groups'}): the other groups are the "
            "target's library call, so it is not the whole model"
        )


__all__ = [
    "GROUP_CAPSULE_LIMIT",
    "PASS_LIST_KEY",
    "apply_passes",
    "declared_passes",
    "resolve_capsules",
]
