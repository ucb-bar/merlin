"""Materialize ONLY bound Lean source/query/root sidecars into an exclusive NEW tree.

Stdlib and reviewed closed-header source parser only. No provider, controller,
SDK, compiler, object, archive, installation or native action. A completed copy
is DATA preparation and grants no semantic, native, human or graph authority.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_core import PlanError, closure, decode, header, identifier, pin, relative, require, topo


def component(value):
    require(
        type(value) is str and value and value.isascii() and all(c.isalnum() or c in "_-" for c in value),
        "safe profile/case component required",
    )
    return value


def read_bound(packet, reference, suffix):
    require(
        type(reference) is dict and set(reference) == {"file", "sha256", "bytes"}, "closed pinned reference required"
    )
    require(
        type(reference["bytes"]) is int and 0 <= reference["bytes"] <= 32 * 1024 * 1024, "bounded byte pin required"
    )
    path = packet / str(relative(reference["file"]))
    require(path.suffix == suffix, "only the selected source/query/JSON role is allowed")
    require(not path.is_symlink() and path.resolve().is_relative_to(packet), "source reference escaped packet")
    require(all(not p.is_symlink() for p in path.parents if p.is_relative_to(packet)), "linked input parent refused")
    require(stat.S_ISREG(path.stat().st_mode), "regular source/data file required")
    raw = path.read_bytes()
    require(
        pin(raw) == {k: reference[k] for k in ("sha256", "bytes")}, "missing/changed pinned input: " + reference["file"]
    )
    return raw


def file_names(root):
    found = set()
    for directory, dirs, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        require(all(not (parent / d).is_symlink() for d in dirs), "linked directory refused")
        for name in files:
            path = parent / name
            require(not path.is_symlink(), "linked file refused")
            found.add(path.relative_to(root).as_posix())
    return found


def validate_spec(packet, spec):
    require(set(spec) == {"schema", "primary_baseline", "profiles"}, "closed staging specification required")
    require(spec["schema"] == "merlin.graph-staging-spec.v1", "unknown staging schema")
    require(type(spec["profiles"]) is list and bool(spec["profiles"]), "nonempty profile list required")
    profiles, expected_files, profile_ids = [], {"STAGING_SPEC.json"}, set()
    for profile in spec["profiles"]:
        require(
            set(profile) == {"id", "project_prefixes", "sources", "dependency_order", "cases", "provenance"},
            "closed profile record required",
        )
        name = component(profile["id"])
        require(name not in profile_ids, "duplicate profile id")
        profile_ids.add(name)
        prefixes = tuple(profile["project_prefixes"])
        require(
            prefixes and all(type(p) is str and p.endswith(".") for p in prefixes), "explicit project prefixes required"
        )
        sources, graph, externals = {}, {}, {}
        for source in profile["sources"]:
            require(set(source) == {"module", "ref"}, "closed source record required")
            module = identifier(source["module"])
            require(module.startswith(prefixes), "nonproject source namespace refused")
            require(module not in sources, "duplicate/conflicting module ID: " + module)
            raw = read_bound(packet, source["ref"], ".lean")
            parsed = header(raw)
            graph[module] = sorted(n for n in parsed["imports"] if n.startswith(prefixes))
            externals[module] = sorted(
                set(n for n in parsed["imports"] if not n.startswith(prefixes))
                | ({"Init"} if parsed["implicit_Init"] else set())
            )
            sources[module] = source["ref"]
            expected_files.add(source["ref"]["file"])
        order = topo(graph)
        require(order == profile["dependency_order"], "dependency order differs from deterministic source DAG")
        cases, case_ids = [], set()
        for case in profile["cases"]:
            require(set(case) == {"id", "module", "query", "roots"}, "closed query record required")
            case_id = component(case["id"])
            require(case_id not in case_ids, "duplicate case ID")
            case_ids.add(case_id)
            module = identifier(case["module"])
            query = read_bound(packet, case["query"], ".lean")
            query_header = header(query)
            require(
                not any(line.lstrip().startswith("#") for line in query.decode("utf-8").splitlines()),
                "query contains source-line diagnostic/evaluation directive",
            )
            needed = closure([n for n in query_header["imports"] if n.startswith(prefixes)], graph)
            roots_raw = read_bound(packet, case["roots"], ".json")
            roots = decode(roots_raw)
            require(roots["schema"] == "merlin.graph-staging-roots.v1", "root sidecar schema differs")
            require(
                (roots["profile"], roots["case"], roots["module"]) == (name, case_id, module),
                "query/root identity mismatch",
            )
            require(roots["query_pin"] == {k: case["query"][k] for k in ("sha256", "bytes")}, "query/root pin mismatch")
            final, support = identifier(roots["final_root"]), roots["support_roots"]
            require(type(support) is list and all(identifier(r) for r in support), "exact support list required")
            require(len(set(support)) == len(support) and final not in support, "duplicate final/support roots")
            require(
                roots["exact_final_declaration"] in query.decode("utf-8"), "final declaration/query association differs"
            )
            require(
                roots["native_context_request"] == "UNSELECTED"
                and roots["compile"] == roots["audit"] == roots["qualification"] == "UNEXECUTED",
                "staging cannot select context or executed status",
            )
            require(
                roots["native_authorization"] is False
                and roots["graph_approval"] is False
                and roots["human_review"] is False,
                "staging cannot carry approval authority",
            )
            expected_files.update((case["query"]["file"], case["roots"]["file"]))
            cases.append(
                {
                    "id": case_id,
                    "module": module,
                    "query": case["query"],
                    "roots": case["roots"],
                    "project_order": [n for n in order if n in needed],
                    "sidecar": roots,
                }
            )
        profiles.append(
            {
                "id": name,
                "sources": sources,
                "DAG": graph,
                "order": order,
                "external_names_only": externals,
                "cases": cases,
                "provenance": profile["provenance"],
            }
        )
    require(
        file_names(packet) == expected_files, "extra or missing capsule payload; no object/SDK/other files admitted"
    )
    return profiles


def write_new(destination, relative_name, raw, inventory):
    path = destination / str(relative(relative_name))
    require(path.suffix in (".lean", ".json"), "only Lean source or pure JSON sidecars may be materialized")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
    require(path.read_bytes() == raw, "copied bytes differ")
    inventory[relative_name] = pin(raw)


def verify_materialized(destination):
    destination = Path(destination).absolute()
    require(not destination.is_symlink() and destination.resolve() == destination, "canonical staging root required")
    manifest = decode((destination / "MATERIALIZATION.json").read_bytes())
    require(manifest["schema"] == "merlin.graph-source-materialization.v1", "materialization schema differs")
    require(
        manifest["data_only"] is True
        and manifest["native_context_request"] == "UNSELECTED"
        and manifest["native_authorization"] is False
        and manifest["graph_approval"] is False
        and manifest["human_review"] is False
        and manifest["compile_audit_qualification"] == "UNEXECUTED",
        "no authority from staging",
    )
    expected = set(manifest["files"]) | {"MATERIALIZATION.json"}
    require(file_names(destination) == expected, "extra/missing staged payload")
    for name, expected_pin in manifest["files"].items():
        path = destination / str(relative(name))
        require(path.suffix in (".lean", ".json"), "staged object/SDK/non-source file refused")
        require(pin(path.read_bytes()) == expected_pin, "staged source/query/sidecar changed: " + name)
    return {
        "status": "MATERIALIZED_DATA_INTEGRITY_MATCHED",
        "files": len(expected),
        "data_only": True,
        "native_authorization": False,
        "graph_approval": False,
        "compile_audit_qualification": "UNEXECUTED",
    }


def materialize(packet, destination):
    packet, destination = Path(packet).absolute(), Path(destination).absolute()
    require(not packet.is_symlink() and packet.resolve() == packet, "canonical packet root required")
    require(
        destination.parent.is_dir() and destination.parent.resolve() == destination.parent,
        "existing canonical parent required",
    )
    require(
        not destination.is_relative_to(packet) and not packet.is_relative_to(destination),
        "input/output tree overlap refused",
    )
    # Never touch an existing original/main/private context, even on an error.
    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    inventory = {}
    try:
        spec_path = packet / "STAGING_SPEC.json"
        require(not spec_path.is_symlink() and stat.S_ISREG(spec_path.stat().st_mode), "regular staging spec required")
        raw_spec = spec_path.read_bytes()
        write_new(
            destination,
            "INTENT.json",
            (
                json.dumps(
                    {
                        "schema": "merlin.graph-staging-intent.v1",
                        "specification": pin(raw_spec),
                        "data_only": True,
                        "native_context_request": "UNSELECTED",
                        "compile_audit_qualification": "UNEXECUTED",
                    },
                    indent=2,
                )
                + "\n"
            ).encode(),
            inventory,
        )
        profiles = validate_spec(packet, decode(raw_spec))
        for profile in profiles:
            prefix = "profiles/" + profile["id"]
            for module in profile["order"]:
                # Re-read/re-hash immediately before each copy, not a cached preflight result.
                raw = read_bound(packet, profile["sources"][module], ".lean")
                write_new(destination, prefix + "/project/" + module.replace(".", "/") + ".lean", raw, inventory)
            order = {
                "schema": "merlin.graph-staged-dependency-order.v1",
                "profile": profile["id"],
                "dependency_order": profile["order"],
                "DAG": profile["DAG"],
                "external_names_only": profile["external_names_only"],
                "build": "UNEXECUTED",
            }
            write_new(
                destination, prefix + "/BUILD_ORDER.json", (json.dumps(order, indent=2) + "\n").encode(), inventory
            )
            sidecar = {
                "schema": "merlin.graph-staged-profile.v1",
                "profile": profile["id"],
                "provenance": profile["provenance"],
                "native_context_request": "UNSELECTED",
                "compile_audit_qualification": "UNEXECUTED",
                "source_union_is_semantic_theorem": False,
            }
            write_new(destination, prefix + "/PROFILE.json", (json.dumps(sidecar, indent=2) + "\n").encode(), inventory)
            for case in profile["cases"]:
                sub = prefix + "/queries/" + case["id"]
                write_new(
                    destination,
                    sub + "/" + case["module"].replace(".", "/") + ".lean",
                    read_bound(packet, case["query"], ".lean"),
                    inventory,
                )
                write_new(destination, sub + "/roots.json", read_bound(packet, case["roots"], ".json"), inventory)
        manifest = {
            "schema": "merlin.graph-source-materialization.v1",
            "specification": pin(raw_spec),
            "files": inventory,
            "profiles": [p["id"] for p in profiles],
            "data_only": True,
            "native_context_request": "UNSELECTED",
            "compile_audit_qualification": "UNEXECUTED",
            "native_authorization": False,
            "graph_approval": False,
            "human_review": False,
        }
        with (destination / "MATERIALIZATION.json").open("x") as stream:
            stream.write(json.dumps(manifest, indent=2) + "\n")
        return verify_materialized(destination)
    except BaseException as error:
        # Preserve exactly the partial first failure; no rollback, cleanup or retry
        # into this now-existing tree. An existing destination never reaches here.
        with (destination / "FIRST_FAILURE.json").open("x") as stream:
            stream.write(
                json.dumps(
                    {
                        "status": "SOURCE_STAGING_REFUSED_OR_UNAVAILABLE",
                        "error_type": type(error).__name__,
                        "reason": str(error),
                        "completed_files": inventory,
                        "partial_tree_preserved": True,
                        "native_context_request": "UNSELECTED",
                        "native_authorization": False,
                        "graph_approval": False,
                    },
                    indent=2,
                )
                + "\n"
            )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--verify-staged", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.verify_staged is not None:
            require(args.packet is None and args.destination is None, "verify mode is read-only")
            result = verify_materialized(args.verify_staged)
        else:
            require(args.packet is not None and args.destination is not None, "packet and NEW destination required")
            result = materialize(args.packet, args.destination)
    except (PlanError, OSError, KeyError, TypeError, UnicodeError) as error:
        result = {
            "status": "STAGING_UNAVAILABLE" if isinstance(error, OSError) else "STAGING_REFUSED",
            "error": str(error),
            "data_only": True,
            "native_authorization": False,
            "graph_approval": False,
        }
        print(json.dumps(result, indent=2))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
