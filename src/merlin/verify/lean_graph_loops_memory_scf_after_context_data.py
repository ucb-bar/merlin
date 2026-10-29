"""Independent full-module decoding and finite one-call context hints.

New wrappers are source-level test extensions, never original producer captures.
Hints do not establish semantic equality: emitted Lean checks the full source,
normalized module, expansion and independently decoded final body.
"""

from copy import deepcopy
import hashlib

from .lean_graph_input_admission import parse_admitted
from .lean_graph_loops_memory_scf_after_entries_data import callgraph, headers, literal_opcodes, reachable
from .lean_graph_loops_shape_after_projection import decode, normalized_code
from .lean_graph_loops_shape_capture import export


def extend_hint(adapter, raw_source, *, parser, index_bits, max_nodes):
    exported = export(parse_admitted(raw_source, parser, calls="shape-direct"))
    actual = decode(exported, index_bits=index_bits, max_nodes=max_nodes)
    hint = deepcopy(adapter)
    original_names = {row["name"] for row in hint["shape"]["module"]["functions"]}
    additions = [row for row in actual["module"]["functions"] if row["name"] not in original_names]
    hint["shape"]["module"] = actual["module"]
    hint["shape"]["changed_sites"] = actual["shape_sites"]
    hint["shape"]["structural_nodes"] = actual["structural_nodes"]
    hint["shape"]["source_binding"] = hashlib.sha256(raw_source.encode()).hexdigest()
    hint["shape"]["target_binding"] = "UNTRUSTED-NORMALIZATION-HINT-FOR-SOURCE-LEVEL-EXTENSION"
    for label in ("normalized_candidate", "candidate_plain"):
        hint[label]["functions"].extend(dict(row, code=normalized_code(row["code"], {})) for row in additions)
        hint[label]["calls"] = deepcopy(actual["module"]["calls"])
    return hint, actual, exported


def _suffix_refusals(code, name, calls, opcodes, literals, indices):
    refused = []
    call_targets = dict(calls)
    opcode_map = dict(opcodes)
    literal_keys, index_keys = {row[0] for row in literals}, {row[0] for row in indices}
    while code["tag"] != "returned":
        tag = code["tag"]
        if tag == "region":
            refused.append(dict(entry=name, code="continuation_region_not_supported"))
        elif tag == "action":
            key = code["key"]
            selected = opcode_map.get(key, {}).get("tag")
            if key in call_targets:
                refused.append(dict(entry=name, code="post_call_callee_not_supported", target=call_targets[key],
                                    obligation="related initial private-clock/heap stability and contextual call congruence"))
            elif key not in literal_keys and key not in index_keys and selected not in ("binary", "load", "store"):
                refused.append(dict(entry=name, code="state_inspecting_or_unknown_suffix_action", key=key))
            if any(row["kind"]["tag"] == "view" for row in code["results"]):
                refused.append(dict(entry=name, code="view_result_not_supported"))
        elif tag == "cast" and code["source"] != code["target"]:
            refused.append(dict(entry=name, code="nonidentical_alias_not_supported"))
        code = code["rest"]
    if any(row["kind"]["tag"] == "view" for row in code["values"]):
        refused.append(dict(entry=name, code="descriptor_return_not_supported"))
    return refused


def contexts(whole, adapter):
    source = whole["independent_source"]["module"]
    final = whole["actual_final"]["model"]["module"]
    source_headers = headers(whole["source_export"])
    final_headers = headers(whole["actual_final"]["actual_export"])
    first_graph, last_graph = callgraph(source), callgraph(final)
    kernel = adapter["step_binding"]["function"]
    literals = literal_opcodes(source, final)
    wrappers, rows, refusals = [], [], []
    if source_headers != final_headers:
        refusals.append(dict(code="complete_roster_or_interface_changed"))
    if any(row["declaration"] for row in source_headers + final_headers):
        refusals.append(dict(code="external_declaration_not_supported"))
    for function in adapter["normalized_candidate"]["functions"]:
        name = function["name"]
        closure = reachable(name, first_graph, last_graph)
        if name == kernel:
            rows.append(dict(name=name, route="kernel"))
        elif kernel not in closure:
            rows.append(dict(name=name, route="unchanged", reachable=closure))
        else:
            first = function["code"]
            targets = dict(source["calls"])
            if first["tag"] != "action" or targets.get(first["key"]) != kernel:
                refusals.append(dict(entry=name, code="prefix_or_transitive_kernel_call_not_supported",
                                     obligation="initial-clock and prefix/frame stability must be proved"))
                rows.append(dict(name=name, route="unchanged", reachable=closure))
                continue
            wrapper = dict(function=function, key=first["key"], callee=kernel, arguments=first["operands"],
                           results=first["results"], suffix=first["rest"])
            wrappers.append(wrapper)
            rows.append(dict(name=name, route="caller", wrapper=len(wrappers)-1))
            refusals.extend(_suffix_refusals(first["rest"], name, source["calls"], adapter["step_binding"]["opcodes"],
                                             literals, whole["actual_final"]["model"]["index_opcodes"]))
            if any(binding["kind"]["tag"] == "view" for binding in first["results"]):
                refusals.append(dict(entry=name, code="private_callee_result_may_escape"))
    return dict(source_headers=source_headers, final_headers=final_headers, rows=rows, wrappers=wrappers,
                source_callgraph=first_graph, final_callgraph=last_graph, literals=literals,
                kernel_reachable=reachable(kernel, first_graph, last_graph), refusals=refusals,
                provenance="source-level wrapper extension of retained raw modules; not original producer capture",
                kernel_outcomes="UNOBSERVED", lean_status="UNCOMPILED")
