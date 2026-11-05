#!/usr/bin/env python3
"""Check pinned operator-coverage records without executing a compiler or test case."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath

SCHEMA = "merlin.core_aten_saved_evidence.v1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(root, name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or str(path) != name or ".." in path.parts or "\\" in name:
        raise ValueError("unsafe evidence path")
    return root.joinpath(*path.parts)


def indexed(rows, key, *, normalize=False):
    result = {}
    for row in rows:
        value = row[key].removeprefix("aten.") if normalize else row[key]
        if not value or value in result:
            raise ValueError(f"empty or duplicate {key}")
        result[value] = row
    return result


def count(row, key):
    value = int(row[key])
    if value < 0:
        raise ValueError(f"negative {key}")
    return value


def validate(root, manifest):
    root = root.absolute()
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported coverage evidence manifest")
    files = manifest["files"]
    if not files or len(files) > 5000 or sum(r["bytes"] for r in files.values()) > 64 * 1024**2:
        raise ValueError("coverage evidence exceeds source bounds")
    entries = list(root.rglob("*"))
    if root.is_symlink() or any(p.is_symlink() for p in entries):
        raise ValueError("indirect evidence path")
    if {p.relative_to(root).as_posix() for p in entries if p.is_file()} != set(files):
        raise ValueError("coverage evidence membership differs")
    for name, record in files.items():
        path = relative(root, name)
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError(f"coverage evidence checksum differs: {name}")

    def rows(key):
        with relative(root, manifest[key]).open(newline="") as stream:
            return list(csv.DictReader(stream))

    catalog = json.loads(relative(root, manifest["catalog"]).read_text())
    canonical = indexed(catalog, "overload", normalize=True)
    if len({r["n"] for r in catalog}) != len(catalog):
        raise ValueError("catalog repeats a case index")
    classes = Counter(r["klass"] for r in catalog)
    if dict(classes) != manifest["expected_class_counts"]:
        raise ValueError("operator class counts differ")
    table = indexed(rows("table"), "overload", normalize=True)
    numerical = indexed(rows("case_results"), "overload", normalize=True)
    credits = indexed(rows("credit_map"), "overload", normalize=True)
    corpus_rows = rows("corpus_results")
    corpus = indexed(corpus_rows, "capsule")
    if set(canonical) != set(table) or set(canonical) != set(numerical):
        raise ValueError("operator result rosters differ")
    device_names = {n for n, r in canonical.items() if r["klass"] == "DEVICE57"}
    if set(credits) != device_names:
        raise ValueError("device credit roster differs")
    details = []
    for name, entry in canonical.items():
        row, case = table[name], numerical[name]
        if row["klass"] != entry["klass"] or case["klass"] != entry["klass"]:
            raise ValueError("operator classification differs across records")
        if entry["klass"] != "DEVICE57":
            continue
        mapping = credits[name]
        if mapping["tier"] != "tier1":
            raise ValueError("unsupported operator credit tier")
        names = [n for n in mapping["tier1_capsules"].split(";") if n]
        if len(names) != len(set(names)):
            raise ValueError("operator repeats a credited test")
        graded = [n for n in names if n in corpus]
        ungraded = sorted(set(names) - set(graded))
        passed = [n for n in graded if corpus[n]["outcome"] == "pass"]
        observed = {"credited": len(names), "credited_graded": len(graded),
                    "credited_not_graded": len(ungraded), "credited_pass": len(passed),
                    "credited_fail": len(graded) - len(passed)}
        if any(count(row, field) != value for field, value in observed.items()):
            raise ValueError(f"credited test counts differ: {name}")
        full = (case["ok"] == "PASS" and case["result"] == "lowered-on-device"
                and count(case, "device_insns") > 0
                and all(case[k] == "pass" for k in ("numerical", "host_compute", "trace_check")))
        if row["meets_full"] not in {"YES", "NO"} or case["meets_full"] not in {"YES", "NO"}:
            raise ValueError("unknown FULL verdict")
        if full != (row["meets_full"] == "YES") or full != (case["meets_full"] == "YES"):
            raise ValueError(f"FULL verdict differs from its components: {name}")
        supported = full and observed["credited_graded"] > 0 and observed["credited_fail"] == 0
        if row["proven"] not in {"YES", "NO"} or supported != (row["proven"] == "YES"):
            raise ValueError(f"support verdict differs from graded cases: {name}")
        details.append({"overload": name, "full_definition": full,
                        "graded_corpus_clean": observed["credited_fail"] == 0,
                        "reported_supported": supported, **observed,
                        "ungraded_test_names": ungraded})
    groups = Counter((r["group"], r["outcome"]) for r in corpus_rows)
    return {
        "schema": "merlin.core_aten_saved_verification.v1",
        "source_commit": manifest["source_commit"],
        "verified_files": len(files), "operator_classes": dict(classes),
        "device_lanes": dict(Counter(r["form_lane"] for r in catalog if r["klass"] == "DEVICE57")),
        "device_full_definition": sum(r["full_definition"] for r in details),
        "device_supported_on_graded_cases": sum(r["reported_supported"] for r in details),
        "host_reported_supported": sum(r["proven"] == "YES" for r in table.values() if r["klass"] == "HOST136"),
        "ungraded_device_credit_slots": sum(r["credited_not_graded"] for r in details),
        "corpus": {group: {outcome: n for (g, outcome), n in groups.items() if g == group}
                   for group in sorted({g for g, outcome in groups})},
        "nonpassing_corpus": [r for r in corpus_rows if r["outcome"] != "pass"],
        "device_details": details,
        "scope": "Saved records and byte identity only. Selected forms, not all PyTorch semantics. No fresh numerical or hardware execution.",
        "archived_programs_executed": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = validate(args.root, json.loads(args.manifest.read_text()))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"coverage evidence verification failed: {error}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
