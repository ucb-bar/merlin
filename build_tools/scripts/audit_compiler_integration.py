#!/usr/bin/env python3
"""Inventory retained compiler sources and emitted pass diagnostics without execution.

This is an operator audit, not a grader. Imports are static observations; pass
records are candidate-authored claims. Neither establishes semantic preservation,
hardware execution, performance or qualification. Missing records stay missing.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_inventory(submission, baseline):
    sources, overrides = [], []
    for path in sorted(submission.rglob("*.py")):
        if path.is_symlink():
            raise ValueError("audit requires regular retained source files")
        relative = path.relative_to(submission)
        tree = ast.parse(path.read_text())
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module or "")
        sources.append({"path": relative.as_posix(), "sha256": digest(path),
                        "imports": sorted(imports)})
        if relative.parts[0] == "compiler_library":
            original = baseline.joinpath(*relative.parts[1:])
            baseline_hash = digest(original) if original.is_file() else None
            overrides.append({"path": relative.as_posix(), "sha256": digest(path),
                              "baseline_sha256": baseline_hash,
                              "comparison": "missing_baseline" if baseline_hash is None else
                              "unchanged" if baseline_hash == digest(path) else "changed"})
    return sources, overrides


def json_string_attribute(text, name):
    """Read the retained generic-MLIR string attribute's JSON payload.

    This bounded reader handles MLIR byte escapes, not JSON string escapes.
    It only inventories an emitted diagnostic attribute; it does not parse or
    admit the surrounding program. Unsupported encodings refuse explicitly.
    """
    marker = name + ' = "'
    start = text.find(marker)
    if start < 0:
        return None
    position = start + len(marker)
    output = bytearray()
    while position < len(text):
        char = text[position]
        position += 1
        if char == '"':
            return json.loads(output.decode("utf-8"))
        if char == "\\":
            if position >= len(text):
                break
            char = text[position]
            position += 1
            if char in ('"', "\\"):
                output.extend(char.encode())
                continue
            pair = char + text[position:position + 1]
            if len(pair) != 2 or not all(c in "0123456789abcdefABCDEF" for c in pair):
                raise ValueError("unsupported MLIR string escape")
            output.append(int(pair, 16))
            position += 1
        else:
            output.extend(char.encode("utf-8"))
    raise ValueError("unterminated MLIR string attribute")


def pass_records(value):
    if isinstance(value, dict):
        if isinstance(value.get("family"), str) and isinstance(value.get("result"), str):
            yield value
        for child in value.values():
            yield from pass_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from pass_records(child)


def receipt_inventory(results, attribute, sources):
    rows, errors = [], []
    counts = Counter()
    for path in sorted(results.glob("*/capsule_result.json")):
        result = json.loads(path.read_text())
        artifact = path.parent / "generated/lowered.target.mlir"
        source = result.get("compiler_source", {}).get("candidate_sha256")
        row = {"capsule": path.parent.name, "reported_source": source,
               "numeric": result.get("numeric", {}).get("status"),
               "final_status": result.get("status"), "artifact_sha256": None,
               "source_copy_matches": None, "families": [], "diagnostic": "missing_artifact"}
        if artifact.is_file():
            row["artifact_sha256"] = digest(artifact)
            # Never follow a candidate-provided arbitrary source path.
            if isinstance(source, str) and len(source) == 64 and all(c in "0123456789abcdef" for c in source):
                snapshot = path.parent / "generated/compiler-source" / source
                row["source_copy_matches"] = bool(sources) and all(
                    (snapshot / member["path"]).is_file()
                    and not (snapshot / member["path"]).is_symlink()
                    and digest(snapshot / member["path"]) == member["sha256"] for member in sources)
            try:
                schedule = json_string_attribute(artifact.read_text(), attribute)
                families = sorted({(item["family"], item["result"]) for item in pass_records(schedule)})
                row["families"] = [{"family": a, "result": b} for a, b in families]
                row["diagnostic"] = "present" if schedule is not None else "missing_attribute"
                # Unmatched sources remain visible but cannot credit this submission.
                if row["source_copy_matches"]:
                    counts.update(families)
            except (ValueError, UnicodeError) as error:
                row["diagnostic"] = "unreadable_attribute"
                errors.append({"capsule": path.parent.name, "error": type(error).__name__})
        rows.append(row)
    return {"cases": rows, "parse_errors": errors,
            "matched_sources": sum(row["source_copy_matches"] is True for row in rows),
            "per_family_matched_cases": [
                {"family": a, "result": b, "cases": count}
                for (a, b), count in counts.most_common()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="selected immutable Python library root containing merlin/")
    parser.add_argument("--results", type=Path, required=True,
                        help="directory containing per-capsule result directories")
    parser.add_argument("--schedule-attribute", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for directory in (args.submission, args.baseline, args.results):
        if not directory.is_dir():
            parser.error("every input directory must exist")
    sources, overrides = source_inventory(args.submission, args.baseline)
    document = {"schema": "merlin.compiler_integration_audit.v1",
                "scope": "static imports and candidate-authored emitted diagnostics; no qualification",
                "sources": sources, "overrides": overrides,
                "receipts": receipt_inventory(args.results, args.schedule_attribute, sources)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps({"source_files": len(sources), "overrides": len(overrides),
                      "matched_sources": document["receipts"]["matched_sources"],
                      "parse_errors": len(document["receipts"]["parse_errors"])}))
    return int(bool(document["receipts"]["parse_errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
