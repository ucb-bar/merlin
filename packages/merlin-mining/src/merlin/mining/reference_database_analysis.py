"""Bottom-up CCA extraction with occurrence-local provenance, without optimization selection."""

from __future__ import annotations

import ast
import bisect
import hashlib
import json
import struct
import time
from pathlib import Path

from .reference_database_io import store_body, write_json

ADVICE_KEYS = frozenset(
    (
        "next_action",
        "library_routes",
        "transfer_candidates",
        "transfer_questions",
        "experiment_tasks",
        "optimization_tasks",
        "knob_proposals",
    )
)


def observations_only(value):
    if isinstance(value, dict):
        return {key: observations_only(item) for key, item in value.items() if key not in ADVICE_KEYS}
    if isinstance(value, (tuple, list)):
        return [observations_only(item) for item in value]
    return value


def elf_function_symbols(elf):
    """Read original ELF function metadata; absence never creates a guessed source name."""
    functions = []
    for header in elf.headers:
        _, kind, _, _, offset, size, link, _, _, stride = header
        if kind not in (2, 11):
            continue
        if stride != 24 or size % stride or link >= len(elf.headers):
            raise ValueError("invalid ELF symbol table")
        strings = elf.headers[link]
        if strings[1] != 3:
            raise ValueError("invalid ELF symbol strings")
        names, symbols = elf.read_at(strings[4], strings[5]), elf.read_at(offset, size)
        for at in range(0, size, stride):
            name, info, _, owner, address, length = struct.unpack_from("<IBBHQQ", symbols, at)
            if info & 15 != 2 or not length or not 0 < owner < len(elf.headers):
                continue
            stop = names.find(b"\0", name, name + 1025)
            if not 0 <= name < len(names) or stop < 0:
                raise ValueError("invalid ELF symbol name")
            functions.append(dict(name=names[name:stop].decode(), address=address, size_bytes=length))
    return functions


def machine_regions(root, artifact, *, facts, decoder):
    from merlin_experiments.phase1.feedback import reference_analysis as analysis

    from merlin.compile.analysis.artifact_reader import PinnedELF, open_ordinary
    from merlin.compile.analysis.riscv_machine import MASK, decode_cpu, region_index
    from merlin.perf.pc_instruction_listing import normalized_text

    path = Path(root) / artifact["path"]
    elf = PinnedELF(open_ordinary(path), artifact["path"])
    try:
        if elf.pin["sha256"] != artifact["sha256"]:
            raise ValueError("model changed after staging")
        rows = elf.instructions()
        symbols = elf_function_symbols(elf)
        table, _ = analysis._table(facts)
        opcode = table.get("custom_opcode")
        if opcode is None:
            raise ValueError("selected ISA has no derived accelerator opcode")
        indexed = region_index(rows, artifact_sha256=artifact["sha256"], custom_opcode=opcode)
        # This existing owner lifts every accelerator span, without a first-page sample.
        typed_ccas = []
        lifted = analysis._lift(
            elf,
            "elf",
            elf.pin,
            facts=facts,
            offset=0,
            limit=len(indexed),
            candidate=False,
            decoder=decoder,
            ccas=typed_ccas,
        )
        if lifted["next_offset"] is not None:
            raise ValueError("incomplete CCA region census")
        reports = {tuple(row.get("address_range", ())): row for row in lifted["regions"]}
        cca_objects = {
            tuple(row.get("address_range", ())): cca for row, cca in zip(lifted["regions"], typed_ccas, strict=True)
        }
        outputs, edges, bodies = [], [], []
        for scope, body in indexed:
            start, stop = scope["address_range"]
            (section,) = [entry for entry in elf.sections if entry[0] == scope["section"]]
            payload = elf.read_at(section[1] + start - section[3], stop - start)
            stored = store_body(root, "machine", payload, ".bin")
            bodies.append(stored)
            occurrence_id = artifact["id"] + ":" + scope["region_id"]
            directory = Path("mappings") / artifact["id"] / scope["region_id"]
            (Path(root) / directory).mkdir(parents=True, exist_ok=True)
            listing = directory / "instructions.dis"
            (Path(root) / listing).write_text(
                "Disassembly of section " + scope["section"] + ":\n" + normalized_text(body)
            )
            report = reports.get(tuple(scope["address_range"]))
            if report is None:
                report = dict(
                    kind="static_host_region",
                    listed_instructions=len(body),
                    cca=None,
                    semantic_region_status="unknown",
                    reason="no_selected_accelerator_sites; host computation semantics not reconstructed",
                )
            report_path = directory / "analysis.json"
            write_json(Path(root) / report_path, observations_only(report))
            if report.get("cca"):
                write_json(Path(root) / directory / "cca.json", report["cca"])
                from merlin.kernels.cca_mlir import to_mlir

                (Path(root) / directory / "cca.mlir").write_text(to_mlir(cca_objects[tuple(scope["address_range"])]))
            outputs.append(
                dict(
                    id=occurrence_id,
                    artifact_id=artifact["id"],
                    kernel_id=stored["id"],
                    symbol=None,
                    section=scope["section"],
                    address=start,
                    size_bytes=stop - start,
                    boundary_status=scope["boundary_status"],
                    accelerator_sites=scope["accelerator_sites"],
                    listing=listing.as_posix(),
                    analysis=report_path.as_posix(),
                    source_correspondence="unknown",
                    model_layer="unknown",
                    numerical_equivalence="unknown",
                    performance="unknown",
                    containing_symbols=[
                        symbol
                        for symbol in symbols
                        if symbol["address"] <= start < stop <= symbol["address"] + symbol["size_bytes"]
                    ],
                )
            )
            exact = [
                symbol["name"]
                for symbol in symbols
                if symbol["address"] == start and symbol["size_bytes"] == stop - start
            ]
            if len(set(exact)) == 1:
                outputs[-1]["symbol"] = exact[0]
            for index, row in enumerate(body):
                decoded = decode_cpu(row)
                if decoded.kind not in ("call", "call_indirect"):
                    continue
                target, basis = decoded.target, "decoded_direct_call"
                if decoded.kind == "call_indirect":
                    target, basis = None, "unresolved_indirect_call"
                    if index:
                        previous, high = body[index - 1], decode_cpu(body[index - 1])
                        if (
                            high.kind == "auipc"
                            and high.rd
                            and high.rd == decoded.rs1
                            and previous.addr + high.size == row.addr
                        ):
                            target = (previous.addr + high.imm + decoded.imm) & MASK & ~1
                            basis = "adjacent_decoded_auipc_call"
                edges.append(
                    dict(
                        caller_id=occurrence_id,
                        call_pc=row.addr,
                        target=target,
                        basis=basis,
                        callee_id=None,
                        dynamic_multiplicity="unknown",
                    )
                )
        # Maps identify code spans, not logical layers, invocation order, shapes or semantic equality.
        starts = sorted((row["address"], row["address"] + row["size_bytes"], row["id"]) for row in outputs)
        keys = [row[0] for row in starts]
        for edge in edges:
            if edge["target"] is not None:
                index = bisect.bisect_right(keys, edge["target"]) - 1
                if index >= 0 and starts[index][0] <= edge["target"] < starts[index][1]:
                    edge["callee_id"] = starts[index][2]
                    edge["target_offset"] = edge["target"] - starts[index][0]
        # Every selected custom issue must occur in exactly one materialized span.
        expected = sum(row["accelerator_sites"] for row in lifted["regions"])
        observed = sum(row["accelerator_sites"] for row in outputs)
        if observed != expected or sum(len(body) for _, body in indexed) != len(rows):
            raise ValueError("accelerator decomposition coverage differs from original CCA inventory")
        return (
            bodies,
            outputs,
            edges,
            dict(
                regions=len(outputs),
                accelerator_regions=sum(row["accelerator_sites"] > 0 for row in outputs),
                accelerator_sites=observed,
                all_instructions=len(rows),
                decomposed_instructions=sum(len(body) for _, body in indexed),
                boundary_status="inferred_static_span; original ELF has no asserted source boundaries",
                complete_accelerator_site_coverage=True,
            ),
        )
    finally:
        elf.close()


def source_regions(root, artifact):
    from merlin.kernels.reference_inventory import inspect_source

    raw = (Path(root) / artifact["path"]).read_bytes()
    started = time.monotonic()
    inventory = inspect_source(raw, artifact["format"])
    if inventory["parser"] == "unavailable":
        raise ValueError("source parser unavailable: " + json.dumps(inventory["diagnostics"]))
    text = raw.decode()
    lines = text.splitlines(keepends=True)
    module = ast.parse(text) if artifact["format"] == "python_schedule" else None
    definitions = {node.lineno: node for node in module.body if isinstance(node, ast.FunctionDef)} if module else {}
    bodies, outputs, edges = [], [], []
    for index, function in enumerate(inventory["functions"]):
        start = function["line"]
        node = definitions.get(start)
        if node is not None:
            start = min([start, *(decorator.lineno for decorator in node.decorator_list)])
        payload = "".join(lines[start - 1 : function["end_line"]]).encode()
        stored = store_body(
            root, artifact["format"], payload, ".py" if artifact["format"] == "python_schedule" else ".c"
        )
        bodies.append(stored)
        identity = artifact["id"] + ":source-" + str(index)
        relative = Path("mappings") / artifact["id"] / ("source-" + str(index) + ".json")
        report = dict(
            kind="source_syntax",
            parser=inventory["parser"],
            function=function,
            cca=None,
            semantic_region_status="unknown",
            machine_correspondence="unknown",
            reason="existing source parser supplies syntax; no verified source-to-command adapter for this input",
            diagnostics=inventory["diagnostics"],
            advice_generated=False,
        )
        write_json(Path(root) / relative, observations_only(report))
        outputs.append(
            dict(
                id=identity,
                artifact_id=artifact["id"],
                kernel_id=stored["id"],
                symbol=function["symbol"],
                start_line=start,
                end_line=function["end_line"],
                boundary_status="parsed_source_definition",
                accelerator_sites=None,
                analysis=relative.as_posix(),
                source_correspondence="original_exact_source_span",
                model_layer="unknown",
                numerical_equivalence="unknown",
                performance="unknown",
            )
        )
        for callee, count in function["calls"].items():
            edges.append(
                dict(
                    caller_id=identity,
                    callee_symbol=callee,
                    static_sites=count,
                    callee_id=None,
                    basis="parsed_source_call",
                    dynamic_multiplicity="unknown",
                )
            )
    exported_bindings = 0
    if module is not None:
        # Record the actual upstream factories/exports as syntax without importing Exo or choosing a schedule.
        exports = [
            dict(line=node.lineno, end_line=node.end_lineno, statement=ast.get_source_segment(text, node))
            for node in module.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
        ]
        write_json(
            Path(root) / "mappings" / artifact["id"] / "module-exports.json",
            dict(kind="parsed_module_assignments", exports=exports, executed=False),
        )
        by_symbol = {row["symbol"]: row["id"] for row in outputs}

        def bindings(target, selector=()):
            if isinstance(target, ast.Name):
                yield target.id, selector
            elif isinstance(target, (ast.Tuple, ast.List)):
                for index, child in enumerate(target.elts):
                    yield from bindings(child, (*selector, index))

        for node in module.body:
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            for target in node.targets:
                for name, selector in bindings(target):
                    statement = ast.get_source_segment(text, node)
                    recipe = dict(statement=statement, result_selector=list(selector), executed=False)
                    payload = (json.dumps(recipe, sort_keys=True, indent=2) + "\n").encode()
                    stored = store_body(root, "source_export", payload, ".json")
                    bodies.append(stored)
                    identity = artifact["id"] + ":export-" + name
                    relative = Path("mappings") / artifact["id"] / ("export-" + name + ".json")
                    write_json(
                        Path(root) / relative,
                        dict(
                            kind="parsed_export_recipe",
                            recipe=recipe,
                            cca=None,
                            semantic_region_status="unknown",
                            compiled=False,
                            factory=ast.unparse(node.value.func),
                            reason="Original optimized-procedure recipe; Exo was not imported or executed.",
                        ),
                    )
                    outputs.append(
                        dict(
                            id=identity,
                            artifact_id=artifact["id"],
                            kernel_id=stored["id"],
                            symbol=name,
                            start_line=node.lineno,
                            end_line=node.end_lineno,
                            boundary_status="parsed_source_export",
                            accelerator_sites=None,
                            analysis=relative.as_posix(),
                            source_correspondence="original_exact_export_recipe",
                            model_layer="unknown",
                            numerical_equivalence="unknown",
                            performance="unknown",
                        )
                    )
                    edges.append(
                        dict(
                            caller_id=identity,
                            callee_id=by_symbol.get(ast.unparse(node.value.func)),
                            callee_symbol=ast.unparse(node.value.func),
                            basis="parsed_factory_name_reference",
                            result_selector=list(selector),
                            dynamic_multiplicity="unknown",
                        )
                    )
                    exported_bindings += 1
    write_json(Path(root) / "mappings" / artifact["id"] / "source-inventory.json", observations_only(inventory))
    return (
        bodies,
        outputs,
        edges,
        dict(
            functions=len(inventory["functions"]),
            exported_bindings=exported_bindings,
            parser=inventory["parser"],
            diagnostics=observations_only(inventory["diagnostics"]),
            seconds=time.monotonic() - started,
            source_sha256=hashlib.sha256(raw).hexdigest(),
            semantic_lift="unknown",
        ),
    )
