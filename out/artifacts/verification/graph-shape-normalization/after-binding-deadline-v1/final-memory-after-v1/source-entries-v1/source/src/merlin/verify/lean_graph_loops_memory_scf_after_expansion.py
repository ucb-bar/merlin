"""Expanded typed target and independently decoded materialization obligation.

Expansion is an untrusted candidate, not a verified lowering. The generated
Lean query checks its complete normalization against the actual final module.
Source-plan-to-expansion simulation remains a separate, explicitly false flag.
No old whole-tree final matcher or native context is invoked by this module.
"""

from __future__ import annotations

import copy
import hashlib
import json

from .lean_graph_loops_memory_scf_after import prepare_final
from .lean_graph_loops_shape_after_projection import definition_roster, permutation_hint
from .lean_graph_loops_shape_epoch_model import module_literal
from .lean_graph_loops_unroll import _list, _nat, _string


def _sequence(atoms, returned):
    result = dict(tag="returned", values=copy.deepcopy(returned))
    for atom in reversed(atoms):
        result = dict(copy.deepcopy(atom), rest=result)
    return result


def _tiled_atoms(uniform, tile, fresh, index_keys):
    tiled = dict(uniform=copy.deepcopy(uniform))
    for key in ("tile", "tileIV", "sum", "bound"):
        tiled[key] = dict(name=next(fresh), kind=dict(tag="index"), known=tile if key == "tile" else None)
    tiled.update(addKey=index_keys["add"], minKey=index_keys["minimum"])
    store = dict(
        tag="action", key=uniform["storeKey"], results=[],
        operands=[uniform[key] for key in ("seed", "buffer", "outer", "inner")],
    )
    column = dict(
        tag="region", key=uniform["loopKey"],
        operands=[uniform[key] for key in ("zero", "columns", "one")],
        parameters=[uniform["inner"]], results=[], body=_sequence([store], []),
    )
    point = dict(
        tag="region", key=uniform["loopKey"],
        operands=[tiled["tileIV"], tiled["bound"], uniform["one"]],
        parameters=[uniform["outer"]], results=[], body=_sequence([column], []),
    )
    add = dict(tag="action", key=tiled["addKey"], operands=[tiled["tileIV"], tiled["tile"]], results=[tiled["sum"]])
    minimum = dict(
        tag="action", key=tiled["minKey"], operands=[tiled["sum"], uniform["rows"]], results=[tiled["bound"]],
    )
    outer = dict(
        tag="region", key=uniform["loopKey"], operands=[uniform["zero"], uniform["rows"], tiled["tile"]],
        parameters=[tiled["tileIV"]], results=[], body=_sequence([add, minimum, point], []),
    )
    return [dict(tag="constant", result=tiled["tile"], value=tile), outer], tiled


def expand(adapter, index_opcodes):
    """Use canonical source commands, never copy the actual final body.

    The retained adapter is a hint. Its old proofs do not automatically establish
    this new expansion. Every generated/removed site is returned for the pending
    per-fragment proof and signed/scoped checks.
    """
    module = copy.deepcopy(adapter["normalized_candidate"])
    plan = adapter["step_binding"]
    epochs = adapter["memory"]["reuse"]["epochs"]
    views = {item["object"]: item for item in plan["privateViews"]}
    keys = {}
    for key, opcode in index_opcodes:
        if opcode in keys and keys[opcode] != key:
            raise ValueError("current expansion requires one exact key per generated index opcode")
        keys[opcode] = key
    if set(keys) != {"add", "minimum"}:
        raise ValueError("actual generated index meanings must contain add and minimum")
    maximum = max(definition_roster(module), default=-1)
    fresh = iter(range(maximum + 1, maximum + 1 + 4 * len(adapter["initializer_links"])))
    atoms, sites, owners = [], [], {}
    physical_clock = 0
    for step in plan["steps"]:
        instruction, atom = step["instruction"], step["atom"]
        kind = instruction["kind"]
        start = len(atoms)
        site = dict(source_position=step["position"], instruction=copy.deepcopy(instruction), physical_clock=physical_clock)
        if kind == "allocate":
            obj = instruction["object"]
            item, slot = views[obj], epochs[obj]["slot"]
            if instruction["retained"]:
                if slot != physical_clock or slot in owners:
                    raise ValueError("physical slot does not equal this retained allocation ordinal")
                owners[slot] = item
                atoms.append(atom)
                physical_clock += 1
                site.update(kind="retained-allocation", object=obj, slot=slot)
            else:
                owner = owners.get(slot)
                if owner is None or (owner["shape"], owner["element"]) != (item["shape"], item["element"]):
                    raise ValueError("removed allocation lacks an earlier same-type retained slot owner")
                view_type = item["binding"]["kind"]["type"]
                atoms.append(dict(tag="cast", origin=owner["binding"], result=item["binding"], source=view_type, target=view_type))
                site.update(kind="alias", owner=owner["object"], object=obj, slot=slot)
        elif kind == "release" and not instruction["retained"]:
            site["kind"] = "empty-release"
        elif kind == "release":
            slot = epochs[instruction["object"]]["slot"]
            if slot not in owners:
                raise ValueError("retained release has no live physical slot owner")
            del owners[slot]
            atoms.append(atom)
            site.update(kind="retained-release", object=instruction["object"], slot=slot)
        elif kind == "initialize":
            link = adapter["initializer_links"][step["initializer"]]
            generated, tiled = _tiled_atoms(link["canonical"], link["config"]["tile"], fresh, keys)
            atoms.extend(generated)
            site.update(kind="tiled-initializer", initializer=tiled, link=step["initializer"])
        else:
            atoms.append(atom)
            site["kind"] = "retained"
        sites.append(dict(site, fragment_start=start, fragment_stop=len(atoms), next_physical_clock=physical_clock))
    matches = [function for function in module["functions"] if function["name"] == plan["function"]]
    if len(matches) != 1:
        raise ValueError("one complete source kernel must bind the expansion")
    matches[0]["code"] = _sequence(atoms, plan["returned"])
    return dict(module=module, sites=sites, final_physical_clock=physical_clock)


def materialization_obligation(expanded, actual, frame, bits):
    swaps = permutation_hint(expanded, actual["module"])
    pairs = _list(swaps, lambda pair: "(" + _nat(pair[0]) + ", " + _nat(pair[1]) + ")")
    text = f"""import Merlin.GraphLoopsShapeAfter
set_option autoImplicit false
set_option maxRecDepth 16384
set_option maxHeartbeats 8000000
open Merlin Merlin.GraphLoopsShape
namespace GraphMemoryAfterMaterialization
def expandedTarget : Module := {module_literal(expanded)}
def actualFinal : Module := {module_literal(actual["module"])}
def actual : After.Projection :=
  ⟨{_nat(bits)}, expandedTarget, actualFinal, {pairs},
    {_list(frame, _string)}, {_list(actual["frame"], _string)}⟩
theorem accepted : After.check actual = true := by decide
theorem actual_final_syntax :
    After.permuteModule actual.swaps actualFinal = Epoch.normalizedModule expandedTarget :=
  (After.check_sound accepted).2.2.2
theorem expanded_to_actual {{Value World : Type}} (meaning : Interpretation Value World)
    (name : String) (inputs : List (Runtime Value)) (before : World)
    (outputs : List (Runtime Value)) (after : World) :
    complete meaning expandedTarget false name inputs before outputs after ↔
      complete meaning actualFinal false name inputs before outputs after :=
  After.checked_complete accepted meaning name inputs before outputs after
end GraphMemoryAfterMaterialization
"""
    return text, swaps


def prepare_expansion(adapter, source_frame, raw_final, *, parser, index_bits, max_nodes):
    actual = prepare_final(raw_final, parser=parser, index_bits=index_bits, max_nodes=max_nodes)
    expanded = expand(adapter, actual["model"]["index_opcodes"])
    query, swaps = materialization_obligation(expanded["module"], actual["model"], source_frame, index_bits)
    return dict(
        expansion=expanded, actual_final=actual, query=query, swaps=swaps,
        adapter_hint_sha256=hashlib.sha256(json.dumps(adapter, sort_keys=True).encode()).hexdigest(),
        source_frame=source_frame,
        final_root="GraphMemoryAfterMaterialization.expanded_to_actual",
        support_roots=["GraphMemoryAfterMaterialization.accepted", "GraphMemoryAfterMaterialization.actual_final_syntax"],
        source_plan_semantics_proved=False, kernel_checked=False, qualified=False,
        pipeline_receipt_issued=False, complete_memory_bridge_proved=False,
    )
