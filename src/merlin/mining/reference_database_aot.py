"""Optional ahead-of-time compilation of selected C originals into derived, pinned machine products.

Every executable, flag and include byte is an explicit spec input verified by hash. The
compiled ELF is a derived product beside the originals, never a replacement for them. Its
source/machine correspondence comes only from the produced objects' symbol tables.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .reference_database_io import copy_checked, digest, safe_member, write_json

UNAVAILABLE = "aot_compile_unavailable"
FAILED = "aot_compile_failed"
DERIVED_ROOT = Path("derived") / "aot"


def _expand(template, values):
    """Expand whole-token placeholders; list values splice, others substitute in place."""
    argv = []
    for token in template:
        if token in values and isinstance(values[token], list):
            argv.extend(values[token])
            continue
        for key, value in values.items():
            if not isinstance(value, list):
                token = token.replace(key, value)
        argv.append(token)
    return argv


def dependency_paths(text):
    """Parse a make-format dependency file structurally: targets end with ':', '\\' continues."""
    paths, escaped = [], ""
    for token in text.split():
        if token == "\\":
            continue
        if token.endswith("\\"):
            escaped += token[:-1] + " "
            continue
        token, escaped = escaped + token, ""
        if token.endswith(":"):
            continue
        paths.append(token)
    return list(dict.fromkeys(paths))


def _pinned_file(path, root):
    path = Path(path)
    resolved = path if path.is_absolute() else Path(root) / path
    if not resolved.is_file():
        return dict(path=str(path), status="unresolved")
    try:
        label = resolved.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        label = str(resolved.resolve())
    return dict(path=label, sha256=digest(resolved), size_bytes=resolved.stat().st_size)


def _run(argv, cwd, log_stem):
    """Run one pinned tool; output is retained verbatim and failure is reported, never hidden."""
    completed = subprocess.run(argv, cwd=cwd, capture_output=True, check=False)
    Path(str(log_stem) + ".stdout").write_bytes(completed.stdout)
    Path(str(log_stem) + ".stderr").write_bytes(completed.stderr)
    return completed


def _toolchain(recipe, root):
    """Verify the pinned driver and its declared components; absent tools are a diagnostic."""
    driver = recipe["driver"]
    path = Path(driver["path"])
    if not path.is_absolute() or not path.is_file():
        return None, dict(kind=UNAVAILABLE, reason="toolchain_driver_absent", path=str(path))
    if digest(path) != driver["sha256"]:
        raise ValueError("selected toolchain driver differs from its pinned hash")
    record = dict(driver=dict(path=str(path), sha256=driver["sha256"]), components={})
    for name, component in sorted(recipe.get("components", {}).items()):
        query = None
        if "path" in component:
            located = Path(component["path"])
        else:
            query = _expand(component["resolve_argv"], {"{driver}": str(path)})
            completed = subprocess.run(query, capture_output=True, check=False)
            located = Path(completed.stdout.decode().strip()) if not completed.returncode else None
        if located is None or not located.is_absolute() or not located.is_file():
            return None, dict(kind=UNAVAILABLE, reason="toolchain_component_absent", component=name)
        if digest(located) != component["sha256"]:
            raise ValueError("selected toolchain component differs from its pinned hash: " + name)
        record["components"][name] = dict(path=str(located), sha256=component["sha256"], resolve_argv=query)
        record.setdefault("placeholders", {})["{" + name + "}"] = str(located)
    identity = recipe.get("identity_argv")
    if identity:
        argv = _expand(identity, {"{driver}": str(path)})
        completed = subprocess.run(argv, capture_output=True, check=False)
        record["identity"] = dict(argv=argv, returncode=completed.returncode, stdout=completed.stdout.decode())
    return record, None


def _include_trees(recipe, root, artifacts):
    """Materialize declared include layouts from staged originals and extra pinned headers."""
    trees = {}
    for tree in recipe.get("include_trees", []):
        base = DERIVED_ROOT / "include" / safe_member(tree["id"])
        files = {}
        for relative, source in sorted(tree["files"].items()):
            destination = Path(root) / base / safe_member(relative)
            if "artifact" in source:
                original = artifacts[source["artifact"]]
                copy_checked(Path(root) / original["path"], destination, original["sha256"])
                files[relative] = dict(artifact=source["artifact"], sha256=original["sha256"])
            else:
                if not Path(source["path"]).is_file():
                    return None, dict(kind=UNAVAILABLE, reason="include_input_absent", tree=tree["id"], file=relative)
                copy_checked(source["path"], destination, source["sha256"])
                files[relative] = dict(
                    path=source["path"], sha256=source["sha256"], origin=source.get("origin", "unrecorded")
                )
        trees[tree["id"]] = dict(path=base.as_posix(), files=files)
    return trees, None


def decoded_lines(text):
    """Parse a decoded DWARF line-table listing: CU headers, then file/line/address rows."""
    rows, unit = [], None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("CU:"):
            unit = stripped[3:].strip().rstrip(":")
            continue
        tokens = stripped.split()
        if len(tokens) < 3 or not tokens[1].isdigit() or not tokens[2].startswith("0x"):
            continue
        try:
            address = int(tokens[2], 16)
        except ValueError:
            continue
        rows.append(dict(address=address, file=tokens[0], line=int(tokens[1]), unit=unit))
    return rows


def subprogram_ranges(text):
    """Parse a DWARF debug-info listing into subprogram/inlined ranges using only its tags/attributes."""
    entries, current = [], None
    for line in text.splitlines():
        tokens = line.split()
        tags = [token.strip("()") for token in tokens if token.strip("()").startswith("DW_TAG_")]
        if tags:
            current = dict(tag=tags[0], attributes={})
            entries.append(current)
            continue
        attributes = [token for token in tokens if token.startswith("DW_AT_")]
        if current is not None and attributes:
            # Values follow the last ": "; leading "(form)" tokens name the DWARF encoding only.
            value = line.rpartition(": ")[2].split()
            while value and value[0].startswith("(") and value[0].endswith(")"):
                value = value[1:]
            current["attributes"][attributes[0]] = " ".join(value)
    ranges = []
    for entry in entries:
        if entry["tag"] not in ("DW_TAG_subprogram", "DW_TAG_inlined_subroutine"):
            continue
        attributes = entry["attributes"]
        try:
            low = int(attributes["DW_AT_low_pc"], 16)
            high = int(attributes["DW_AT_high_pc"], 16)
        except (KeyError, ValueError):
            continue
        # DWARF 4+ encodes high_pc as a length when it is a constant smaller than low_pc.
        stop = low + high if high < low else high
        ranges.append(
            dict(
                tag=entry["tag"],
                name=attributes.get("DW_AT_name"),
                origin=attributes.get("DW_AT_abstract_origin"),
                address_range=[low, stop],
                decl_line=attributes.get("DW_AT_decl_line"),
            )
        )
    return ranges


def debug_correspondence(root, directory, dumps, values):
    """Run pinned debug-info dumpers over the derived ELF and retain their parsed tables."""
    result = dict(basis="DWARF line table and subprogram ranges emitted by the pinned compiler", dumps={})
    parsers = dict(line_table=decoded_lines, subprograms=subprogram_ranges)
    for name, template in sorted(dumps.items()):
        if name not in parsers:
            raise ValueError("unsupported debug dump: " + name)
        argv = _expand(template, values)
        completed = _run(argv, root, root / directory / ("dwarf-" + name))
        if completed.returncode:
            result["dumps"][name] = dict(argv=argv, status="dump_failed", returncode=completed.returncode)
            continue
        parsed = parsers[name](completed.stdout.decode(errors="replace"))
        relative = directory / ("dwarf-" + name + ".json")
        write_json(Path(root) / relative, dict(argv=argv, rows=parsed))
        result["dumps"][name] = dict(argv=argv, status="parsed", rows=len(parsed), path=relative.as_posix())
    return result


def attach_debug_lines(records, occurrences, root):
    """Summarize per-span file:line rows from the tool-produced line table, by address only."""
    import bisect
    import json

    by_build = {record["id"]: record for record in records if record["status"] == "compiled"}
    for row in occurrences:
        record = by_build.get(row["artifact_id"])
        dumps = (record or {}).get("debug_correspondence", {}).get("dumps", {})
        table = dumps.get("line_table", {})
        if table.get("status") != "parsed" or row.get("address") is None:
            continue
        lines = json.loads((Path(root) / table["path"]).read_bytes())["rows"]
        lines.sort(key=lambda item: item["address"])
        keys = [item["address"] for item in lines]
        start, stop = row["address"], row["address"] + row["size_bytes"]
        selected = lines[bisect.bisect_left(keys, start) : bisect.bisect_left(keys, stop)]
        files = sorted({item["file"] for item in selected})
        row["debug_lines"] = dict(
            source=table["path"],
            rows=len(selected),
            files=files,
            line_span={
                name: [
                    min(i["line"] for i in selected if i["file"] == name),
                    max(i["line"] for i in selected if i["file"] == name),
                ]
                for name in files
            },
            basis="DWARF line rows whose address lies in this span",
        )


def compile_declared(root, recipe, artifacts):
    """Return (derived artifacts, per-build records, diagnostics); originals are only read."""
    from merlin.compile.analysis.artifact_reader import PinnedELF, open_ordinary

    from .reference_database_analysis import elf_function_symbols

    if not recipe:
        return [], [], []
    root = Path(root)
    by_id = {row["id"]: row for row in artifacts}
    builds = recipe.get("builds", [])
    toolchain, problem = _toolchain(recipe["toolchain"], root)
    trees = None
    if problem is None:
        trees, problem = _include_trees(recipe, root, by_id)
    if problem is not None:
        diagnostics = [dict(problem, build=build["id"], sources=build["sources"]) for build in builds]
        write_json(root / DERIVED_ROOT / "diagnostics.json", dict(diagnostics=diagnostics))
        return [], [], diagnostics
    derived, records, diagnostics = [], [], []
    for build in builds:
        directory = DERIVED_ROOT / safe_member(build["id"])
        (root / directory).mkdir(parents=True, exist_ok=True)
        include_dirs = []
        for entry in build.get("include_dirs", []):
            if "tree" in entry:
                include_dirs.append(trees[entry["tree"]]["path"])
            else:
                include_dirs.append(Path(by_id[entry["artifact_dir"]]["path"]).parent.as_posix())
        include_args = [build["include_flag"] + path for path in include_dirs]
        units, failed = [], None
        for source_id in build["sources"]:
            source = by_id[source_id]
            if source["format"] != "c_source" or digest(root / source["path"]) != source["sha256"]:
                raise ValueError("AOT source is not a staged, pinned C original: " + source_id)
            stem = directory / safe_member(source_id)
            values = {
                **toolchain.get("placeholders", {}),
                "{driver}": toolchain["driver"]["path"],
                "{root}": str(root),
                "{includes}": include_args,
                "{source}": source["path"],
                "{object}": stem.as_posix() + ".o",
                "{depfile}": stem.as_posix() + ".d",
            }
            argv = _expand(build["compile_argv"], values)
            completed = _run(argv, root, root / stem)
            unit = dict(
                source=source_id,
                source_sha256=source["sha256"],
                argv=argv,
                returncode=completed.returncode,
                stderr=stem.as_posix() + ".stderr",
            )
            units.append(unit)
            if completed.returncode or not (root / values["{object}"]).is_file():
                failed = dict(kind=FAILED, build=build["id"], stage="compile", source=source_id)
                break
            depfile = root / values["{depfile}"]
            unit["dependencies"] = (
                [_pinned_file(path, root) for path in dependency_paths(depfile.read_text())]
                if depfile.is_file()
                else "not_recorded"
            )
            unit["object"] = _pinned_file(values["{object}"], root)
            obj = PinnedELF(open_ordinary(root / values["{object}"]), values["{object}"])
            try:
                unit["defined_functions"] = sorted({row["name"] for row in elf_function_symbols(obj)})
            finally:
                obj.close()
        output = directory / (safe_member(build["id"]).name + ".elf")
        link = None
        if failed is None:
            values = {
                **toolchain.get("placeholders", {}),
                "{driver}": toolchain["driver"]["path"],
                "{objects}": [unit["object"]["path"] for unit in units],
                "{output}": output.as_posix(),
            }
            argv = _expand(build["link_argv"], values)
            completed = _run(argv, root, root / directory / "link")
            trace = [line.strip() for line in completed.stdout.decode().splitlines() if line.strip()]
            link = dict(
                argv=argv,
                returncode=completed.returncode,
                # A traced input may be named as "archive(member)"; pin the archive file itself.
                inputs=[_pinned_file(line.partition("(")[0], root) for line in trace],
            )
            if completed.returncode or not (root / output).is_file():
                failed = dict(kind=FAILED, build=build["id"], stage="link")
        record = dict(
            id=build["id"],
            sources=build["sources"],
            toolchain=toolchain,
            include_trees={
                key: trees[key] for key in sorted({e["tree"] for e in build.get("include_dirs", []) if "tree" in e})
            },
            include_dirs=include_dirs,
            units=units,
            link=link,
            status="failed" if failed else "compiled",
            relation="derived product compiled from the listed originals; not an original artifact",
            byte_equality_with_published_binaries="not_claimed",
        )
        if failed is None and build.get("debug_dumps"):
            record["debug_correspondence"] = debug_correspondence(
                root,
                directory,
                build["debug_dumps"],
                dict(toolchain.get("placeholders", {}), **{"{output}": output.as_posix()}),
            )
        if failed:
            diagnostics.append(failed)
        else:
            record["output"] = _pinned_file(output, root)
            derived.append(
                dict(
                    id=build["id"],
                    path=output.as_posix(),
                    sha256=record["output"]["sha256"],
                    format="elf",
                    derived=True,
                    sources=build["sources"],
                    origin=dict(
                        kind="aot_compiled_derivative",
                        build_record=(directory / "build.json").as_posix(),
                        sources={sid: by_id[sid]["sha256"] for sid in build["sources"]},
                    ),
                )
            )
        write_json(root / directory / "build.json", record)
        records.append(record)
    write_json(root / DERIVED_ROOT / "diagnostics.json", dict(diagnostics=diagnostics))
    return derived, records, diagnostics


def _subprograms(root, record):
    dump = record.get("debug_correspondence", {}).get("dumps", {}).get("subprograms", {})
    if root is None or dump.get("status") != "parsed":
        return None
    import json

    rows = json.loads((Path(root) / dump["path"]).read_bytes())["rows"]
    return [row for row in rows if row["tag"] == "DW_TAG_subprogram"]


def symbol_correspondence(records, occurrences, root=None):
    """Join derived machine occurrences to parsed C definitions by tool-produced symbol names only."""
    by_artifact = {}
    for row in occurrences:
        by_artifact.setdefault(row["artifact_id"], []).append(row)
    matches = []
    for record in records:
        if record["status"] != "compiled":
            continue
        definers = {}
        for unit in record["units"]:
            for name in unit.get("defined_functions", []):
                definers.setdefault(name, []).append(unit["source"])
        subprograms = _subprograms(root, record)
        for machine in by_artifact.get(record["id"], []):
            symbol = machine.get("symbol")
            units = definers.get(symbol, [])
            if symbol is None or len(units) != 1:
                machine["source_correspondence"] = "unknown"
                continue
            parsed = [
                row["id"]
                for row in by_artifact.get(units[0], [])
                if row.get("symbol") == symbol and row["boundary_status"] == "parsed_source_definition"
            ]
            if len(parsed) != 1:
                machine["source_correspondence"] = "unknown"
                continue
            machine["source_correspondence"] = "aot_symbol_definition"
            span = [machine.get("address"), (machine.get("address") or 0) + (machine.get("size_bytes") or 0)]
            named = [row["address_range"] for row in subprograms or () if row["name"] == symbol]
            dwarf = (
                "not_recorded"
                if subprograms is None
                else "absent"
                if not named
                else "exact"
                if span in named
                else "overlapping"
                if any(lo < span[1] and span[0] < hi for lo, hi in named)
                else "disjoint"
            )
            matches.append(
                dict(
                    machine_occurrence=machine["id"],
                    source_occurrence=parsed[0],
                    translation_unit=units[0],
                    symbol=symbol,
                    symbol_status=machine.get("symbol_status"),
                    dwarf_subprogram_range=dwarf,
                    basis="STT_FUNC defined by this unit's compiled object == parsed C definition name in that unit",
                    build_record=(DERIVED_ROOT / record["id"] / "build.json").as_posix(),
                    scope="compiled from that unit with the recorded flags; inlined callees are not separated",
                    semantic_equivalence_to_published_binaries="unknown",
                )
            )
    return matches
