"""Phase 0 entry points for the performance-demand and redundancy stages over pinned inputs.

``derive_demand`` reads each selected program's frontend trace by an explicit sha256 pin, keeps only
its value-free projection (:func:`merlin_experiments.corpus.model_shape_roster.project`), derives the
plan of :mod:`.performance_demand` and writes the public source contracts plus the holdouts as an
owner-only Phase 0 hidden-profile sidecar. ``assemble_epoch`` applies :mod:`.case_redundancy` to a
previous release together with that plan and writes the next public scalability suite and public
supplement selections (the release inputs :func:`merlin_experiments.corpus.release.prepare` stages),
plus the derived performance-scope roster for the generated cohort, whose sealed members can only
be collapsed by the next Phase 0 generation. Nothing here executes a case or seals a release.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from . import case_redundancy as R
from . import performance_demand as D


def parse_program(spec: str) -> tuple[str, str, Path, str]:
    """``MODEL/PROGRAM=PATH@SHA256`` -> its parts; every program is pinned by its trace bytes."""
    label, sep, pinned = spec.partition("=")
    path, at, digest = pinned.rpartition("@")
    model, slash, program = label.partition("/")
    if not sep or not at or not slash or not model or not program or len(digest) != 64:
        raise ValueError(f"program selection {spec!r} must be MODEL/PROGRAM=PATH@SHA256")
    return model, program, Path(path), digest


def project_trace(model: str, program: str, path: Path, sha256: str) -> dict[str, Any]:
    from merlin_experiments.corpus.model_shape_roster import project

    raw = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError(f"{model}/{program}: selected trace bytes changed")
    record = project(json.loads(raw), model=model, program=program, source_sha256=sha256)
    del raw
    record["graphs"] = [graph for graph in record["graphs"] if graph["stage"] == "prepared"]
    for graph in record["graphs"]:
        graph["connected_region_proposals"] = []
    return record


def existing_raw_contracts(root: Path | None) -> dict[str, list[int]]:
    """Public raw int8-to-int32 matrix contracts of a staged suite, by name -> ``[M, K, N]``."""
    out: dict[str, list[int]] = {}
    for path in sorted(Path(root).glob("*/capsule.yaml")) if root is not None else ():
        cap = yaml.safe_load(path.read_text()) or {}
        attrs = (cap.get("operation") or {}).get("attributes") or {}
        if ((cap.get("operation") or {}).get("op") != "matmul" or attrs.get("epilogue") != []
                or attrs.get("output_dtype") != "i32" or cap.get("numeric_policy") != {"compare": "exact_int",
                                                                                       "dtype": "i32"}):
            continue
        shapes = {row.get("name"): row.get("shape") for row in cap.get("inputs") or ()}
        lhs, rhs = shapes.get(attrs.get("lhs")), shapes.get(attrs.get("weight"))
        if isinstance(lhs, list) and isinstance(rhs, list) and len(lhs) == len(rhs) == 2 and lhs[1] == rhs[0]:
            out[str(cap["name"])] = [lhs[0], lhs[1], rhs[1]]
    return out


def derive_demand(programs: list[str], *, policy: Path, facts: Path, template: Path, output: Path,
                  existing_suite: Path | None = None) -> dict[str, Any]:
    """Write ``plan.json``, ``sources/`` and the owner-only ``hidden-profile.yaml`` under ``output``."""
    if output.exists():
        raise ValueError("performance demand needs a fresh output directory")
    selected = [parse_program(spec) for spec in programs]
    projections = [{"model": m, "program": p, "projection": project_trace(m, p, path, sha)}
                   for m, p, path, sha in selected]
    plan = D.derive(projections, policy=yaml.safe_load(policy.read_text()),
                    facts=json.loads(facts.read_text()), existing=existing_raw_contracts(existing_suite))
    plan["inputs"] = {
        "policy": {"path": str(policy), "sha256": hashlib.sha256(policy.read_bytes()).hexdigest()},
        "rtl_facts": {"path": str(facts), "sha256": hashlib.sha256(facts.read_bytes()).hexdigest()},
        "template": str(template), "existing_suite": str(existing_suite) if existing_suite else None,
        "programs": [{"model": m, "program": p, "trace_sha256": sha} for m, p, _, sha in selected],
    }
    output.mkdir(parents=True)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    sources = D.write_public_sources(plan, template=template, destination=output / "sources")
    hidden = output / "hidden-profile.yaml"
    descriptor = os.open(hidden, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(yaml.safe_dump(D.hidden_profile(plan, template=template), sort_keys=False))
    return {"plan": str(output / "plan.json"), "plan_sha256": plan["sha256"], "public_sources": len(sources),
            "hidden": sum(m["split"] == "hidden" for m in plan["members"]),
            "satisfied_by_existing": sum(m["satisfied_by_existing"] for m in plan["members"]),
            "coverage": {k: {f: round(v[f], 4) for f in ("exact_public_share", "represented_public_share",
                                                          "hidden_share")} | {"status": v["status"]}
                         for k, v in plan["coverage"].items()}}


def _release_rows(release: Path, sources: Path | None,
                  extra_sources: tuple[Path, ...] = ()) -> tuple[list[dict], dict[str, str]]:
    preparation = json.loads((release / "private/preparation.json").read_text())
    supplement = set((preparation["assembly"].get("public_supplement") or {}).get("members") or ())
    corpus = release / "payload/corpus"
    rows, origin = [], {}
    for path in sorted(corpus.glob("*/*/capsule.yaml")):
        member = path.parent.relative_to(corpus).as_posix()
        if member.startswith(("_", "hidden/")):
            continue
        cohort = "supplement" if member in supplement else "generated"
        rows.append(_row(path, member, cohort))
        origin[rows[-1]["name"]] = cohort
    for cohort, root in (("scalability", corpus / "_public_scalability/cases"), ("derived", sources),
                         *(("derived", extra) for extra in extra_sources)):
        for path in sorted(Path(root).glob("*/capsule.yaml")) if root is not None else ():
            rows.append(_row(path, path.parent.name, "scalability"))
            origin[rows[-1]["name"]] = cohort
    return rows, origin


def _row(path: Path, member: str, cohort: str) -> dict[str, Any]:
    binding = path.parent / "source-shape-binding.json"
    loaded = json.loads(binding.read_text()) if binding.is_file() else None
    capsule = yaml.safe_load(path.read_text()) or {}
    return {"name": path.parent.name, "member": member, "cohort": cohort, "dir": path.parent, "capsule": capsule,
            "binding": loaded if isinstance(loaded, dict) else None,
            "program_sha256": R.program_sha256(path.parent, capsule)}


def _copy_scoped(source: Path, destination: Path, scope: str) -> str:
    from merlin_experiments.runner import fingerprint

    from ..corpus.preparation import copy_input

    copy_input(source, destination, private=True)
    path = destination / "capsule.yaml"
    cap = yaml.safe_load(path.read_text())
    if cap.get("performance_scope", R.SCORED) != scope:
        cap["performance_scope"] = scope
        path.chmod(0o600)
        path.write_text(yaml.safe_dump(cap, sort_keys=False))
    return fingerprint(destination)


def assemble_epoch(previous: Path, *, plan: Path, sources: Path, output: Path, evidence_root: Path,
                   target: str, chain_plan: Path | None = None, chain_sources: Path | None = None) -> dict[str, Any]:
    """Next scalability suite and supplement selections with redundancy applied; see the module doc.

    ``chain_plan``/``chain_sources`` add a chain-demand stage's performance members
    (:mod:`.chain_corpus`): their connected cases are admitted through the same intake as the derived
    contractions and, like them, are preferred as scorers of their own work.
    """
    from merlin.common.digest import sha256_file

    from ..corpus.public_scalability import read as read_suite
    from ..corpus.public_supplement import read as read_supplement
    from ..corpus.scalability_intake import prepare as intake
    from .evidence import load_exported_evidence

    if output.exists():
        raise ValueError("epoch assembly needs a fresh output directory")
    if (chain_plan is None) != (chain_sources is None):
        raise ValueError("select both a chain plan and its sources, or neither")
    demand = json.loads(plan.read_text())
    chains = json.loads(chain_plan.read_text()) if chain_plan is not None else None
    if chains is not None:
        from .chain_demand import PLAN_SCHEMA

        if chains.get("schema") != PLAN_SCHEMA:
            raise ValueError("chain plan has an unknown schema")
        written = {path.parent.name for path in Path(chain_sources).glob("*/capsule.yaml")}
        expected = {m["name"] for m in chains["members"] if m["role"] == "performance"}
        if written != expected:
            raise ValueError("chain sources differ from the plan's performance members")
        chained = [{"name": name, "split": "public", "represents": []} for name in sorted(expected)]
        demand = {**demand, "members": [*demand["members"], *chained]}
    rows, origin = _release_rows(previous, sources, (chain_sources,) if chain_sources is not None else ())
    # Operator-staged sweep members lost their ``performance`` block at intake; their sweep role still
    # marks a point of a fitted or paired family, so none of them is collapsed.
    protected = {row["name"] for row in rows if origin[row["name"]] in {"supplement", "scalability"}
                 and row["capsule"].get("source_role") == "derived_sweep"}
    record = R.derive(rows, demand=demand, protected=protected)
    collapsed = {row["removed"] for row in record["collapsed"]}
    scope = record["performance_scope"]
    output.mkdir(parents=True)
    (output / "case-redundancy.json").write_text(json.dumps(record, indent=2) + "\n")
    by_name = {row["name"]: row for row in rows}
    # Derived contracts get their software screen through the ordinary scalability intake.
    derived = [row["dir"] for row in rows if origin[row["name"]] == "derived" and row["name"] not in collapsed]
    intaken = {}
    if derived:
        suite = intake(derived, output / "derived-intake", target=target,
                       evidence=load_exported_evidence(evidence_root), evidence_root=evidence_root)
        intaken = {Path(r["member"]).name: (suite.parent / r["member"], r) for r in json.loads(suite.read_text())["members"]}
    staged = json.loads((previous / "payload/corpus/_public_scalability/selection.json").read_text())
    old_rows = {Path(r["member"]).name: r for r in json.loads(Path(staged["source"]["selection"]["path"]).read_text())["members"]}
    members, lineage = [], []
    suite_root = output / "scalability-suite"
    for name in sorted(n for n, o in origin.items() if o in {"scalability", "derived"} and n not in collapsed):
        source, base = intaken[name] if origin[name] == "derived" else (by_name[name]["dir"], old_rows[name])
        digest = _copy_scoped(source, suite_root / "sources" / name, scope[name])
        row = {key: base[key] for key in ("source_group", "split", "execution_role") if key in base}
        members.append({"member": f"sources/{name}", "sha256": digest, **row})
        lineage.append({"name": name, "origin": origin[name], "performance_scope": scope[name]})
    provenance = suite_root / "provenance.json"
    provenance.write_text(json.dumps({
        "schema": "merlin.phase0.performance_epoch.v1", "previous_release": str(previous),
        "previous_suite": staged["source"]["selection"], "demand_plan": {"path": str(plan),
                                                                      "sha256": sha256_file(plan)},
        **({"chain_plan": {"path": str(chain_plan), "sha256": sha256_file(chain_plan)}} if chain_plan else {}),
        "case_redundancy_sha256": record["sha256"], "members": lineage,
        "collapsed": [row for row in record["collapsed"] if origin.get(row["removed"]) != "generated"],
    }, indent=2) + "\n")
    suite_path = suite_root / "suite.json"
    suite_path.write_text(json.dumps({**{k: staged["source"][k] for k in ("schema", "target", "policy")},
                                      "origin": "operator_requested", "members": members,
                                      "provenance": {"path": str(provenance), "sha256": sha256_file(provenance)}},
                                     indent=2) + "\n")
    suite = read_suite(suite_path, target=target)
    supplement_root, supplement_members = output / "supplement", []
    for name in sorted(n for n, o in origin.items() if o == "supplement" and n not in collapsed):
        member = by_name[name]["member"]
        supplement_members.append({"member": member, "sha256": _copy_scoped(by_name[name]["dir"],
                                                                             supplement_root / member, scope[name])})
    supplement_provenance = supplement_root / "provenance.json"
    supplement_provenance.write_text(json.dumps({"schema": "merlin.phase0.performance_epoch.v1",
                                                 "previous_release": str(previous),
                                                 "case_redundancy_sha256": record["sha256"]}, indent=2) + "\n")
    selection = supplement_root / "selection.json"
    selection.write_text(json.dumps({"schema": "merlin.corpus.public_supplement.v1", "target": target,
                                     "origin": "operator_requested", "members": supplement_members,
                                     "provenance": {"path": str(supplement_provenance),
                                                    "sha256": sha256_file(supplement_provenance)}},
                                    indent=2) + "\n")
    read_supplement(selection, target=target)
    generated = sorted(n for n, o in origin.items() if o == "generated")
    roster = {"schema": "merlin.phase1.performance_scope_roster.v1",
              # Exclusions a generated case cannot carry itself: its scorer is in another cohort, or a
              # sealed duplicate stays graded until regeneration collapses it. Merge as the policy's
              # ``performance_scope.excluded`` list.
              "excluded": sorted(n for n in generated if (scope.get(n) == R.EXCLUDED or n in collapsed)
                                 and by_name[n]["capsule"].get("performance_scope") != R.EXCLUDED),
              "declared_in_cases": sorted(n for n in generated
                                          if by_name[n]["capsule"].get("performance_scope") == R.EXCLUDED),
              "collapse_pending_regeneration": sorted(n for n in generated if n in collapsed),
              "case_redundancy_sha256": record["sha256"]}
    (output / "performance-scope.json").write_text(json.dumps(roster, indent=2) + "\n")
    return {"scalability_suite": str(suite_path), "scalability_members": len(suite["members"]),
            "supplement_selection": str(selection), "supplement_members": len(supplement_members),
            "collapsed": len(record["collapsed"]), "excluded": sum(v == R.EXCLUDED for v in scope.values()),
            "generated_excluded_roster": len(roster["excluded"]),
            "generated_collapse_pending": len(roster["collapse_pending_regeneration"])}
