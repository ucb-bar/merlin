"""Stdlib-only SOURCE/DATA DAG planning. Never run Lean, providers or toolchains.

Paths in original metadata are provenance only. Read only caller-selected packet
relative refs. A source/header plan proves neither elaboration nor root meaning.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import stat
from pathlib import Path, PurePosixPath


class PlanError(ValueError):
    pass


def require(value, message):
    if not value:
        raise PlanError(message)


def pin(raw):
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _pairs(rows):
    result = {}
    for key, value in rows:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def decode(raw):
    try:
        return json.loads(
            raw, object_pairs_hook=_pairs, parse_constant=lambda _: (_ for _ in ()).throw(PlanError("nonfinite JSON"))
        )
    except (ValueError, UnicodeError) as error:
        raise PlanError(str(error)) from error


def relative(value):
    require(type(value) is str and value and "\\" not in value, "relative path text required")
    path = PurePosixPath(value)
    require(
        not path.is_absolute() and str(path) == value and all(x not in (".", "..") for x in path.parts),
        "noncanonical relative path",
    )
    return path


def identifier(value):
    require(type(value) is str and bool(value), "module/root identifier required")
    for part in value.split("."):
        require(
            part
            and part.isascii()
            and (part[0].isalpha() or part[0] == "_")
            and all(c.isalnum() or c in "_'" for c in part),
            "unsupported module/root name: " + value,
        )
    return value


class Reader:
    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.cache = {}

    def read(self, reference):
        require(type(reference) is dict, "pinned ref required")
        path = self.root / str(relative(reference.get("file")))
        require(
            path.suffix in (".lean", ".py", ".json", ".mlir", ".md", ".txt", ".xml", ".patch"),
            "only source/query/metadata refs may be read",
        )
        require(
            type(reference.get("bytes")) is int and 0 <= reference["bytes"] <= 32 * 1024 * 1024,
            "bounded exact source/data size required",
        )
        require(type(reference.get("sha256")) is str and len(reference["sha256"]) == 64, "source/data SHA256 required")
        key = reference["file"]
        if key not in self.cache:
            require(not path.is_symlink() and path.resolve().is_relative_to(self.root), "packet ref escaped root")
            require(stat.S_ISREG(path.stat().st_mode), "source/data must be a regular file")
            raw = path.read_bytes()
            self.cache[key] = raw
        raw = self.cache[key]
        require(pin(raw) == {k: reference[k] for k in ("sha256", "bytes")}, "source/data mutation: " + key)
        return raw

    def document(self, reference):
        return decode(self.read(reference))


# Independent closed header decoder. Nested/line comments are recognized; source
# imports are ASCII literal dotted identifiers, one per line. No Lean code runs.
_BODY = {
    "namespace",
    "section",
    "end",
    "open",
    "set_option",
    "def",
    "theorem",
    "lemma",
    "inductive",
    "structure",
    "class",
    "instance",
    "abbrev",
    "variable",
    "variables",
    "universe",
    "universes",
    "noncomputable",
    "private",
    "protected",
    "attribute",
    "example",
    "export",
    "opaque",
    "axiom",
    "syntax",
    "macro",
    "notation",
    "infix",
    "infixl",
    "infixr",
    "prefix",
    "postfix",
}


def header(raw):
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise PlanError("invalid source UTF8") from error
    require("\x00" not in text and not text.startswith("\ufeff"), "unsupported source prefix/control")
    position, imports, prelude = 0, [], False

    def skip():
        nonlocal position
        newline = False
        while position < len(text):
            if text[position].isspace():
                newline |= text[position] in "\r\n"
                position += 1
            elif text.startswith("--", position):
                end = text.find("\n", position)
                position = len(text) if end < 0 else end
            elif text.startswith("/-", position):
                position += 2
                depth = 1
                while depth and position < len(text):
                    if text.startswith("/-", position):
                        depth += 1
                        position += 2
                    elif text.startswith("-/", position):
                        depth -= 1
                        position += 2
                    else:
                        newline |= text[position] in "\r\n"
                        position += 1
                require(depth == 0, "unterminated header comment")
            else:
                break
        return newline

    def token():
        nonlocal position
        start = position
        while position < len(text) and (text[position].isalnum() or text[position] in "_.'"):
            position += 1
        return text[start:position]

    skip()
    while position < len(text):
        word = token()
        if word == "prelude":
            require(not prelude and not imports, "prelude must be first and unique")
            prelude = True
            require(skip() or position == len(text), "prelude must end its header line")
        elif word == "import":
            # A newline between import and its name is deliberately unsupported.
            require(not skip(), "import must have a literal module on the same line")
            name = identifier(token())
            require(name not in imports, "duplicate literal import")
            imports.append(name)
            require(skip() or position == len(text), "one literal import per line required")
        else:
            require(
                word in _BODY or (not word and text.startswith("@[", position)),
                "unknown/unsupported header command: " + repr(word),
            )
            return {
                "imports": imports,
                "prelude": prelude,
                "implicit_Init": not prelude,
                "body_start": position - len(word),
                "decoder": "closed-literal-source-header-v1",
            }
    require(bool(imports) or prelude, "source lacks a recognized header/body")
    return {
        "imports": imports,
        "prelude": prelude,
        "implicit_Init": not prelude,
        "body_start": position,
        "decoder": "closed-literal-source-header-v1",
    }


def module_from_path(path):
    require(type(path) is str, "logical original source path required")
    marker = "/lean/"
    if marker not in "/" + path or not path.endswith(".lean"):
        return None
    tail = ("/" + path).split(marker, 1)[1]
    if not tail.startswith(("Merlin/", "MerlinVerify/")):
        return None
    return identifier(tail[:-5].replace("/", "."))


def project_records(rows):
    require(type(rows) is list, "bound source rows must be a list")
    result = {}
    for row in rows:
        if type(row) is list:
            require(len(row) == 4, "project source tuple must have4 fields")
            name, path, digest, size = row
        else:
            require(type(row) is dict, "source record required")
            path = row.get("path")
            name = row.get("module") or module_from_path(path)
            if name is None:
                continue
            digest, size = row.get("sha256"), row.get("bytes", row.get("size"))
        identifier(name)
        value = {"sha256": digest, "bytes": size}
        require(name not in result or result[name] == value, "conflicting bound project source: " + name)
        result[name] = value
    return result


def bound(document, descriptor):
    for key in descriptor["path"]:
        document = document[key]
    if descriptor.get("last"):
        return document[-1]
    if descriptor.get("all_but_last"):
        return document[:-1]
    return document


def topo(graph):
    degree = {name: len(dependencies) for name, dependencies in graph.items()}
    dependents = {name: [] for name in graph}
    for name, dependencies in graph.items():
        for dependency in dependencies:
            require(dependency in graph, "missing project source: " + dependency)
            dependents[dependency].append(name)
    ready = sorted(name for name, count in degree.items() if not count)
    heapq.heapify(ready)
    result = []
    while ready:
        name = heapq.heappop(ready)
        result.append(name)
        for dependent in sorted(dependents[name]):
            degree[dependent] -= 1
            if degree[dependent] == 0:
                heapq.heappush(ready, dependent)
    require(len(result) == len(graph), "cyclic project sources: " + ",".join(sorted(n for n, d in degree.items() if d)))
    return result


def closure(roots, graph):
    found, todo = set(), list(roots)
    while todo:
        name = todo.pop()
        require(name in graph, "missing project source: " + name)
        if name not in found:
            found.add(name)
            todo.extend(graph[name])
    return found


def profile_plan(profile, reader, future_root):
    name = profile["id"]
    selection = hashlib.sha256(canonical(profile)).hexdigest()
    output_prefix = future_root + "/" + name + "/" + selection
    require(len({c["id"] for c in profile["cases"]}) == len(profile["cases"]), "duplicate case id")
    relative(name)
    prefixes = tuple(profile["project_prefixes"])
    require(all(type(p) is str and p.endswith(".") for p in prefixes), "explicit project prefixes required")
    sources, headers, graph, external = {}, {}, {}, {}
    for row in profile["catalog"]:
        module = identifier(row["module"])
        require(module.startswith(prefixes), "project source outside selected namespace")
        ref = row["source"]
        reader.read(ref)
        if module in sources:
            require(sources[module]["source"]["sha256"] == ref["sha256"], "conflicting duplicate module: " + module)
            sources[module]["provenance"].append(row["provenance"])
        else:
            sources[module] = {"source": ref, "provenance": [row["provenance"]]}
    if profile.get("project_inventory"):
        declared = project_records(reader.document(profile["project_inventory"]))
        require(
            declared == {n: {k: r["source"][k] for k in ("sha256", "bytes")} for n, r in sources.items()},
            "project inventory differs from catalog pins",
        )
    for module, row in sorted(sources.items()):
        parsed = header(reader.read(row["source"]))
        headers[module] = parsed
        graph[module] = sorted(n for n in parsed["imports"] if n.startswith(prefixes))
        external[module] = sorted(
            set(n for n in parsed["imports"] if not n.startswith(prefixes))
            | ({"Init"} if parsed["implicit_Init"] else set())
        )
    order = topo(graph)
    cases = []
    for case in profile["cases"]:
        if case.get("status") == "UNAVAILABLE":
            cases.append(case)
            continue
        try:
            query = reader.read(case["query"])
            request = reader.document(case["request_metadata"])
            prepared = reader.document(case["prepared_metadata"])
            prepared = prepared.get("prepared", prepared)
            for original_key, case_key in (("domain", "domain"), ("relation", "relation"), ("premises", "assumptions")):
                if original_key in prepared:
                    require(
                        prepared[original_key] == case[case_key], "semantic metadata association differs: " + case_key
                    )
            if "query" in prepared:
                require(prepared["query"] == query.decode("utf-8"), "prepared/query bytes differ")
            if "query_sha256" in prepared:
                require(prepared["query_sha256"] == pin(query)["sha256"], "prepared query pin differs")
            if "roots" in prepared:
                require(
                    prepared["roots"] == case["support_roots"] + [case["final_root"]],
                    "prepared/request root roster differs",
                )
            for role, descriptor in case["bindings"].items():
                observed = bound(request, descriptor)
                expected = query.decode("utf-8") if role == "query" else case[role]
                require(observed == expected, "query/root binding mismatch: " + case["id"] + ":" + role)
            declared = case.get("project_sources_binding")
            if declared:
                expected = project_records(bound(request, declared))
                actual = {n: {k: r["source"][k] for k in ("sha256", "bytes")} for n, r in sources.items()}
                require(expected == actual, "request project source selection differs from profile")
            final = identifier(case["final_root"])
            support = case["support_roots"]
            require(type(support) is list and all(identifier(r) for r in support), "root list required")
            require(len(set(support)) == len(support) and final not in support, "duplicate/final support root")
            parsed = header(query)
            roots = [n for n in parsed["imports"] if n.startswith(prefixes)]
            needed = closure(roots, graph)
            for row in case["buffers"]:
                reader.read(row["ref"])
            if case.get("exact_final_declaration"):
                require(case["exact_final_declaration"] in query.decode("utf-8"), "final declaration/query mismatch")
            prefix = output_prefix + "/" + case["id"]
            relative(prefix)
            cases.append(
                {
                    **case,
                    "status": "SOURCE_QUERY_DAG_PREPARED_NOT_CHECKED",
                    "query_header": parsed,
                    "query_project_imports": roots,
                    "query_external_names_only": sorted(
                        set(n for n in parsed["imports"] if not n.startswith(prefixes))
                        | ({"Init"} if parsed["implicit_Init"] else set())
                    ),
                    "main_project_compile_order": [n for n in order if n in needed],
                    "source_build_roles_in_dependency_order": [
                        {"module": n, "role": "direct_query_import" if n in roots else "transitive_support"}
                        for n in order
                        if n in needed
                    ],
                    "output_directory_request": {
                        "source": prefix + "/source",
                        "objects": prefix + "/objects",
                        "query": prefix + "/query",
                        "audits": prefix + "/audits",
                    },
                    "future_steps": {
                        k: "UNEXECUTED"
                        for k in (
                            "source_build",
                            "query_kernel_compilation",
                            "exact_root_axiom_type_audit",
                            "qualification",
                            "fresh_replay",
                        )
                    },
                    "root_resolution": (
                        "exact metadata/query pin association only; "
                        "kernel name/type/axiom audit UNEXECUTED"
                    ),
                }
            )
        except (PlanError, OSError, KeyError, UnicodeError) as error:
            cases.append({"id": case["id"], "status": "UNAVAILABLE", "reason": str(error), "query": case.get("query")})
    identity = hashlib.sha256(canonical({"profile": name, "sources": sources, "cases": cases})).hexdigest()
    return {
        "id": name,
        "successor_of": profile.get("successor_of"),
        "source_plan_identity": identity,
        "status": "SOURCE_DAG_PREPARED_NOT_COMPILED",
        "sources": sources,
        "headers": headers,
        "project_DAG": graph,
        "external_module_names_only": external,
        "support_library_build_order": order,
        "support_library_output_request": output_prefix + "/library",
        "cases": cases,
        "runtime_context": "MISSING_NEW_INDEPENDENT_F_OFFICIAL_SDK_INTAKE",
        "compiling_source_union_proves_combined_semantics": False,
    }


def build(inputs, reader):
    require(inputs.get("schema") == "merlin.fresh-graph-plan-inputs.v1", "unknown input schema")
    for ref in inputs.get("packet_data_refs", []):
        reader.read(ref)
    require(len({p["id"] for p in inputs["profiles"]}) == len(inputs["profiles"]), "duplicate profile id")
    profiles = []
    for profile in inputs["profiles"]:
        try:
            profiles.append(profile_plan(profile, reader, str(relative(inputs["future_root"]))))
        except (PlanError, OSError, KeyError, UnicodeError) as error:
            profiles.append(
                {
                    "id": profile["id"],
                    "status": "UNAVAILABLE",
                    "reason": str(error),
                    "successor_of": profile.get("successor_of"),
                    "compile": "UNEXECUTED",
                }
            )
    shared = {}
    for p in profiles:
        for name, row in p.get("sources", {}).items():
            shared.setdefault(name, {}).setdefault(row["source"]["sha256"], []).append(p["id"])
    return {
        "schema": "merlin.fresh-graph-source-build-plan.v1",
        "input_identity": hashlib.sha256(canonical(inputs)).hexdigest(),
        "primary_baseline": inputs["primary_baseline"],
        "profiles": profiles,
        "shared_source_conflicts": [
            {"module": n, "variants": rows} for n, rows in sorted(shared.items()) if len(rows) > 1
        ],
        "runtime_or_native_qualification": False,
        "historical_authority_transferred": False,
        "implicit_Init_and_external_names": "names only; no Std/SDK/source/object files opened",
        "all_compile_audit_qualification_replay_steps": "UNEXECUTED",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--inputs", default="INPUTS.json")
    parser.add_argument("--output", default="PLAN.json")
    args = parser.parse_args(argv)
    reader = Reader(args.packet)
    inputs_path = reader.root / str(relative(args.inputs))
    output = reader.root / str(relative(args.output))
    inputs = decode(inputs_path.read_bytes())
    require(not output.is_symlink() and output.resolve().is_relative_to(reader.root), "output escaped packet")
    protected = [inputs_path, *(reader.root / str(relative(r["file"])) for r in inputs.get("packet_data_refs", []))]
    require(
        all(
            output.resolve() != path.resolve() and not (output.exists() and path.exists() and output.samefile(path))
            for path in protected
        ),
        "output may not overwrite source/query/metadata inputs",
    )
    result = build(inputs, reader)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "SOURCE_PLAN_WRITTEN",
                "profiles": len(result["profiles"]),
                "unavailable_profiles": [p["id"] for p in result["profiles"] if p["status"] == "UNAVAILABLE"],
                "integrity_only": True,
                "proof_or_native_qualification": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
