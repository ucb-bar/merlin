"""Build the operator's reference-compiler baselines file from measurement studies, mapped to corpus cases.

The author sees, per large test case, the cycles of reference compilers the operator measured
(:mod:`merlin_experiments.phase1.feedback.reference_baselines`). This builder reads the operator's
study JSON files, keeps only the implementations it is told are references (``--source
<study key>=<label>``), drops every implementation whose key or label carries an ``--exclude`` token
(by default ``exo``: private kernels the operator withholds from author inputs), maps each measured
row to a PUBLIC corpus case -- by case id, else by the exact shape+semantics key of its tensors -- and
writes a ``merlin.performance_baselines.v1`` file of cycle numbers and labels only.

A study row is any JSON object holding, under an implementation key, an object with positive
``gsim_cycles`` and/or ``firesim_cycles`` (one baseline per engine) and a ``case``; or a flat object
with ``implementation``, ``case`` and those cycle fields. Rows that map to no public corpus case are
listed under ``unmapped`` with the reason, never guessed.

Stage the output as the ``baselines.json`` file of a reviewed reference origin named
``performance_baselines`` (staged at ``performance_baselines/baselines.json``; the file stanza and the
author-guidance paragraph are printed); :func:`verify_corpus` re-checks every row against the released
corpus when the release is prepared. Historical v1 staging at ``performance-baselines/`` is still read.

Usage::

    python -m merlin_experiments.corpus.performance_baselines \\
        --corpus <source-release>/payload/corpus \\
        --study-dir out/artifacts/perf-studies/fresh1-firesim \\
        --source fsm=fsm-library --source merlin=merlin-reference \\
        --out <origin-dir>/baselines.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from ..phase1.feedback import reference_baselines as RB
from ..spec import SpecError

#: Display name and kind of the reference labels this repository's operators use; others pass ``--display``.
KNOWN_SOURCES = {
    "fsm-library": {"display": "the FSM software library", "kind": "vendor software library"},
    "merlin-reference": {"display": "Merlin's reference compiler", "kind": "human-built compiler"},
}
DEFAULT_EXCLUDE = ("exo",)
#: Corpus subtrees that are never public author inputs.
PRIVATE_DIRS = {"hidden", "private"}
_MAX_STUDY_BYTES = 64 << 20
_WINDOW_KEYS = ("window", "harness")


def _positive(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tokens(text) -> set[str]:
    """Lower-case alphanumeric words of ``text`` (structural split, no regex)."""
    words, current = set(), []
    for char in str(text or "").lower():
        if char.isalnum():
            current.append(char)
        elif current:
            words.add("".join(current))
            current = []
    if current:
        words.add("".join(current))
    return words


def _epilogue(value) -> str:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value or "")


def shape_key(op, tensors, *, epilogue, output_dtype) -> str | None:
    """``op|role:shape:dtype,...|epilogue=...|out=...`` over the non-output tensors; None when incomplete.

    ``tensors`` is ``[(role, shape, dtype)]``; names are not part of the key."""
    parts = []
    for role, shape, dtype in tensors:
        if role == "output":
            continue
        if not isinstance(shape, list) or not all(type(d) is int for d in shape) or not dtype:
            return None
        parts.append(f"{role}:{'x'.join(str(d) for d in shape)}:{dtype}")
    if not op or not parts or not output_dtype:
        return None
    return f"{op}|{','.join(sorted(parts))}|epilogue={_epilogue(epilogue)}|out={output_dtype}"


def _capsule_key(capsule: dict) -> str | None:
    operation = capsule.get("operation") if isinstance(capsule.get("operation"), dict) else {}
    attributes = operation.get("attributes") if isinstance(operation.get("attributes"), dict) else {}
    tensors = [
        (row.get("role"), row.get("shape"), row.get("dtype"))
        for row in capsule.get("inputs") or ()
        if isinstance(row, dict)
    ]
    return shape_key(
        operation.get("op"), tensors, epilogue=attributes.get("epilogue"), output_dtype=attributes.get("output_dtype")
    )


def _row_key(row: dict) -> str | None:
    """The key of a study row that carries ``tensors: {name: [shape, dtype, role]}`` (the shapes.json form)."""
    tensors = row.get("tensors")
    if not isinstance(tensors, dict):
        return None
    listed = []
    for value in tensors.values():
        if not isinstance(value, list) or len(value) != 3:
            return None
        shape, dtype, role = value
        listed.append((role, shape, dtype))
    return shape_key(row.get("op"), listed, epilogue=row.get("epilogue"), output_dtype=row.get("output_dtype"))


def corpus_cases(corpus: Path) -> dict:
    """``{case: {interface_sha256, shape_key}}`` of every public case under ``corpus`` (private trees skipped)."""
    import yaml

    corpus = Path(corpus)
    found: dict = {}
    for directory, subdirs, files in os.walk(corpus, followlinks=False):
        subdirs[:] = sorted(d for d in subdirs if d not in PRIVATE_DIRS and not d.startswith("."))
        if "capsule.yaml" not in files:
            continue
        here = Path(directory)
        try:
            capsule = yaml.safe_load((here / "capsule.yaml").read_text(encoding="utf-8"))
        except (OSError, ValueError, yaml.YAMLError):
            continue
        if not isinstance(capsule, dict):
            continue
        name = str(capsule.get("name") or here.name)
        interface = here / str(capsule.get("interface_mlir") or "capsule.interface.mlir")
        found[name] = {
            "interface_sha256": _sha256(interface) if interface.is_file() and not interface.is_symlink() else None,
            "shape_key": _capsule_key(capsule),
        }
    return found


def _study_rows(document):
    """``(row id, study key, implementation object, parent row)`` for every implementation measurement."""
    stack = [document]
    while stack:
        node = stack.pop()
        if isinstance(node, list):
            stack.extend(reversed(node))
            continue
        if not isinstance(node, dict):
            continue
        flat = node.get("implementation")
        if isinstance(flat, str) and any(_positive(node.get(f"{e}_cycles")) for e in RB.ENGINES):
            yield str(node.get("case") or node.get("layer") or ""), flat, node, node
            continue
        for key, value in node.items():
            if isinstance(value, dict) and any(_positive(value.get(f"{e}_cycles")) for e in RB.ENGINES):
                row_id = str(node.get("layer") or node.get("id") or value.get("case") or node.get("case") or "")
                yield row_id, key, value, node
            elif isinstance(value, (dict, list)):
                stack.append(value)


def _instrument(document: dict, engine: str, study: Path) -> tuple[str, dict]:
    instrument = document.get("instrument") if isinstance(document.get("instrument"), dict) else {}
    suffix = "emulator_sha256"
    # ``<engine>_emulator_sha256`` names that engine's binary; an unprefixed one names the study's only engine.
    emulator = next(
        (
            v
            for k, v in instrument.items()
            if isinstance(v, str) and str(k).endswith(suffix) and str(k)[: -len(suffix)].rstrip("_") in ("", engine)
        ),
        None,
    )
    window = next((instrument[k] for k in _WINDOW_KEYS if isinstance(instrument.get(k), str)), None)
    record = {"engine": engine}
    if emulator:
        record["emulator_sha256"] = emulator
    if window:
        record["window"] = " ".join(window.split())[: RB._MAX_TEXT]
    memory = instrument.get("memory_model")
    if isinstance(memory, str):
        record["memory_model"] = " ".join(memory.split())[: RB._MAX_TEXT]
    name = f"{engine}:{emulator[:12]}" if emulator else f"{engine}:{study.stem}"
    return name, record


def build(studies, corpus: Path, *, sources: dict, exclude=DEFAULT_EXCLUDE, display: dict | None = None) -> dict:
    """The baselines document from ``studies`` (later files override earlier rows of the same case/source/engine).

    ``sources`` maps a study implementation key to a source label."""
    exclude = {token.lower() for token in exclude}
    for key, label in sources.items():
        if (_tokens(key) | _tokens(label)) & exclude:
            raise SpecError(f"source {key}={label} carries an excluded token")
    cases = corpus_cases(corpus)
    by_shape: dict = {}
    for name, case in cases.items():
        if case["shape_key"]:
            by_shape.setdefault(case["shape_key"], []).append(name)
    instruments, chosen, unmapped = {}, {}, []
    for study in studies:
        study = Path(study)
        if study.is_symlink() or not study.is_file() or study.stat().st_size > _MAX_STUDY_BYTES:
            continue
        try:
            document = json.loads(study.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        digest = _sha256(study)
        for row_id, key, measured, parent in _study_rows(document):
            label_text = measured.get("label")
            # An excluded implementation is never read, whatever source mapping names it.
            if (_tokens(key) | _tokens(label_text)) & exclude or key not in sources:
                continue
            source = sources[key]
            case = measured.get("case") or parent.get("case")
            case = case if case in cases else None
            if case is None:
                key_of_row = _row_key(parent) or _row_key(measured)
                matches = by_shape.get(key_of_row) if key_of_row else None
                if matches and len(matches) == 1:
                    case = matches[0]
                else:
                    reason = (
                        "shape matches several corpus cases"
                        if matches
                        else "not a public corpus case and no exact shape+semantics match"
                    )
                    unmapped.append({"row": row_id[:120] or "unnamed", "source": source, "reason": reason})
                    continue
            for engine in RB.ENGINES:
                cycles = measured.get(f"{engine}_cycles")
                if not _positive(cycles):
                    continue
                name, record = _instrument(document, engine, study)
                instruments[name] = record
                provenance = {"study": study.name, "study_sha256": digest, "row": row_id[:120] or case}
                for field in ("elf_sha256", "kernel_o_sha256", "checksum"):
                    if isinstance(measured.get(field), str):
                        provenance[field] = measured[field][: RB._MAX_TEXT]
                row = {
                    "case": case,
                    "interface_sha256": cases[case]["interface_sha256"],
                    "shape_key": cases[case]["shape_key"],
                    "source": source,
                    "engine": engine,
                    "cycles": cycles,
                    "instrument": name,
                    "revision": str(label_text)[:120] if isinstance(label_text, str) else None,
                    "provenance": provenance,
                }
                if record.get("window"):
                    row["window"] = record["window"]
                macs = parent.get("useful_macs")
                if _positive(macs):
                    row["useful_macs"] = macs
                chosen[(case, source, engine)] = {k: v for k, v in row.items() if v is not None}
    declared = {}
    for label in sorted(set(sources.values())):
        declared[label] = {**KNOWN_SOURCES.get(label, {"display": label}), **((display or {}).get(label) or {})}
    manifest = Path(corpus) / "MANIFEST.yaml"
    document = {
        "schema": RB.SCHEMA,
        "advisory_only": True,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "note": RB.NOTE,
        "corpus": {"manifest_sha256": _sha256(manifest)} if manifest.is_file() else {},
        "sources": declared,
        "excluded_sources": [f"{token}: withheld from author inputs" for token in sorted(exclude)],
        "instruments": instruments,
        "baselines": [chosen[k] for k in sorted(chosen)],
        "unmapped": sorted(unmapped, key=lambda r: (r["row"], r["source"])),
    }
    return RB.validate(document)


def verify_corpus(reference_root: Path, corpus: Path) -> dict | None:
    """Re-check a staged baselines file against the released corpus; None when none is staged."""
    path = RB.staged_file(Path(reference_root))
    if path is None:
        return None
    try:
        document = RB.validate(json.loads(path.read_text(encoding="utf-8")))
    except ValueError as exc:
        raise SpecError(f"performance baselines are malformed: {exc}") from exc
    cases = corpus_cases(corpus)
    for row in document["baselines"]:
        case = cases.get(row["case"])
        if case is None:
            raise SpecError(f"performance baseline names {row['case']!r}, which is not a public corpus case")
        if row.get("interface_sha256") and row["interface_sha256"] != case["interface_sha256"]:
            raise SpecError(f"performance baseline of {row['case']!r} was measured on another interface")
    return {"path": str(path), "sha256": _sha256(path), "baselines": len(document["baselines"])}


def _pairs(values, what: str) -> dict:
    out = {}
    for value in values or ():
        key, sep, label = value.partition("=")
        if not sep or not key or not label:
            raise SystemExit(f"--{what} expects KEY=VALUE, got {value!r}")
        out[key] = label
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus", required=True, type=Path, help="the release payload's corpus directory")
    parser.add_argument("--study", action="append", type=Path, default=[], help="a study JSON (repeatable)")
    parser.add_argument("--study-dir", action="append", type=Path, default=[], help="every *.json in a directory")
    parser.add_argument("--source", action="append", default=[], help="STUDY_KEY=LABEL of a reference to keep")
    parser.add_argument("--display", action="append", default=[], help="LABEL=display name override")
    parser.add_argument("--exclude", action="append", default=None, help="token never read (default: exo)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--guidance-out", type=Path, help="write the author-guidance paragraph here")
    args = parser.parse_args(argv)
    studies = [*sorted(p for d in args.study_dir for p in Path(d).glob("*.json")), *args.study]
    sources = _pairs(args.source, "source") or {"fsm": "fsm-library", "merlin": "merlin-reference"}
    display = {label: {"display": name} for label, name in _pairs(args.display, "display").items()}
    document = build(
        studies,
        args.corpus,
        sources=sources,
        exclude=tuple(args.exclude) if args.exclude else DEFAULT_EXCLUDE,
        display=display,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(document, indent=1, sort_keys=True) + "\n").encode("utf-8")
    args.out.write_bytes(payload)
    stanza = {
        "path": RB.STAGED,
        # A positively reviewed (v2) reference manifest names it as this origin's file.
        "origin": RB.STAGED_ORIGIN,
        "destination": RB.STAGED_DESTINATION,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "source": "operator-measured reference compiler cycles; numbers and labels only",
    }
    text = RB.guidance(document)
    if args.guidance_out:
        args.guidance_out.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {len(document['baselines'])} baseline(s), {len(document['unmapped'])} unmapped row(s)")
    for row in document["baselines"]:
        print(f"  {row['case']}  {row['source']}  {row['engine']}  {row['cycles']}")
    print("reference-manifest member:", json.dumps(stanza))
    print("author guidance:\n" + text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
