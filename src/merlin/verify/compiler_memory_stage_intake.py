"""Independently decode actual memory-DCE endpoints and emit closed facts.

The Python pruning and SSA permutation are hints. The kernel must check the
actual target against the pruned actual source, including the complete frame.
No acceptance, live authority or execution result is issued by this module.
"""

from __future__ import annotations

import copy

from .buffer_reuse_bridge import decode_pair
from .lean_graph_input_admission import ModelRefusal
from .lean_graph_loops_shape_after_projection import SingleDecoder, definition_roster
from .lean_graph_loops_shape_epoch_model import module_literal
from .lean_graph_loops_unroll import _list, _nat, _string


class DceDecoder(SingleDecoder):
    def check_loop_controls(self, operands):
        # The rule retains the original region, operands and body and works for
        # arbitrary partial region meanings. It needs no trip-count, success or
        # totality premise, including when an index control is dynamic.
        if any(row["kind"] != dict(tag="index") for row in operands[:3]):
            raise ModelRefusal("memory DCE SCF controls must be index bindings")


def decode(exported, *, index_bits, max_nodes):
    decoder = DceDecoder(exported, index_bits=index_bits, max_nodes=max_nodes)
    module = decoder.ordinary_module()
    return dict(module=module, frame=decoder.frame)


def free_reads(code):
    tag = code["tag"]
    if tag == "returned":
        return [row["name"] for row in code["values"]]
    rest = free_reads(code["rest"])
    if tag == "constant":
        return [name for name in rest if name != code["result"]["name"]]
    if tag == "cast":
        return [code["origin"]["name"], *[name for name in rest if name != code["result"]["name"]]]
    if tag == "dimension":
        return [
            code["view"]["name"],
            code["axis"]["name"],
            *[name for name in rest if name != code["result"]["name"]],
        ]
    results = [row["name"] for row in code["results"]]
    reads = [row["name"] for row in code["operands"]]
    if tag == "region":
        parameters = [row["name"] for row in code["parameters"]]
        reads += [name for name in free_reads(code["body"]) if name not in parameters]
    elif tag != "action":
        raise ValueError("unknown typed Code node")
    return [*reads, *[name for name in rest if name not in results]]


def entry_plan(code):
    if code["tag"] == "returned":
        return copy.deepcopy(code), []
    rest, names = entry_plan(code["rest"])
    if code["tag"] == "constant" and code["result"]["name"] not in free_reads(rest):
        return rest, [*names, code["result"]["name"]]
    return {**copy.deepcopy(code), "rest": rest}, names


def prune_constants(code, removals, removed):
    if code["tag"] == "returned":
        return copy.deepcopy(code)
    rest = prune_constants(code["rest"], removals, removed)
    if (
        code["tag"] == "constant"
        and code["result"]["name"] in removals
        and code["result"]["name"] not in free_reads(rest)
    ):
        removed.append(dict(binding=copy.deepcopy(code["result"]), value=code["value"]))
        return rest
    # Keep every action and region boundary. The kernel separately proves
    # equality of the recursively pruned callback given to each partial handler.
    node = {**copy.deepcopy(code), "rest": rest}
    if code["tag"] == "region":
        node["body"] = prune_constants(code["body"], removals, removed)
    return node


def removal_plan(module):
    # The original compiler pass traverses function entry blocks. The theorem
    # supports arbitrary recursive plans; this source-derived hint preserves
    # nested constants that the actual current producer leaves in place.
    return [name for function in module["functions"] for name in entry_plan(function["code"])[1]]


def pruned_module(module, removals=None):
    removals = removal_plan(module) if removals is None else removals
    removed, functions = [], []
    for function in module["functions"]:
        sites = []
        functions.append({**copy.deepcopy(function), "code": prune_constants(function["code"], removals, sites)})
        removed.extend({"function": function["name"], **row} for row in sites)
    return {**copy.deepcopy(module), "functions": functions}, removed


def permutation_hint(expected, actual):
    # Unlike the shape-normalization hint, this same-mode pass keeps casts and
    # dimensions. Align the complete ordinary rosters without normalizing them.
    swaps = []
    for source, target in zip(definition_roster(actual), definition_roster(expected)):
        current = source
        for left, right in reversed(swaps):
            current = right if current == left else left if current == right else current
        if current != target:
            swaps.insert(0, [current, target])
    return swaps


def query(projection):
    source, final = map(module_literal, (projection["before"], projection["after"]))
    before_frame, after_frame = (_list(projection[side + "Frame"], _string) for side in ("before", "after"))
    swaps = _list(projection["swaps"], lambda pair: "(" + _nat(pair[0]) + ", " + _nat(pair[1]) + ")")
    removals = _list(projection["removals"], _nat)
    return f"""import Merlin.GraphLoopsShapeDeadConstants
set_option autoImplicit false
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
namespace ActualMemoryStage
open Merlin.GraphLoopsShape
def actualBefore : Module := {source}
def actualAfter : Module := {final}
def beforeFrame : List String := {before_frame}
def afterFrame : List String := {after_frame}
def swaps : List (Nat × Nat) := {swaps}
def removals : List Nat := {removals}
theorem before_valid : actualBefore.Valid 64 := by decide
theorem after_valid : actualAfter.Valid 64 := by decide
theorem frame_exact : beforeFrame = afterFrame := by decide
theorem actual_pruned : After.permuteModule swaps actualAfter =
    DeadConstants.pruneModule removals actualBefore := by decide
theorem complete {{Value World : Type}} (meaning : Interpretation Value World)
    (name : String) (inputs : List (Runtime Value)) (before : World)
    (outputs : List (Runtime Value)) (after : World) :
    Merlin.GraphLoopsShape.complete meaning actualBefore false name inputs before outputs after ↔
      Merlin.GraphLoopsShape.complete meaning actualAfter false name inputs before outputs after :=
  DeadConstants.checked_complete actualBefore actualAfter removals swaps actual_pruned
    meaning name inputs before outputs after
theorem contextual {{Value World Observation : Type}} (meaning : Interpretation Value World)
    (context : (List String × (String → RegionMeaning Value World)) → Observation) :
    context (beforeFrame, Merlin.GraphLoopsShape.complete meaning actualBefore false) =
      context (afterFrame, Merlin.GraphLoopsShape.complete meaning actualAfter false) := by
  have same : Merlin.GraphLoopsShape.complete meaning actualBefore false =
      Merlin.GraphLoopsShape.complete meaning actualAfter false := by
    funext name inputs before outputs after
    exact propext (complete meaning name inputs before outputs after)
  rw [frame_exact, same]
end ActualMemoryStage
"""


def prepare(source, final, *, index_bits=64, max_nodes=5000):
    if type(index_bits) is not int or index_bits != 64:
        raise ValueError("explicit memory DCE requires index64")
    if any(type(text) is not str or len(text.encode()) > 256 * 1024 for text in (source, final)):
        raise ValueError("bounded complete source/final raw text required")
    actual = decode_pair(source, final)
    before = decode(actual["source_export"], index_bits=index_bits, max_nodes=max_nodes)
    after = decode(actual["final_export"], index_bits=index_bits, max_nodes=max_nodes)
    expected, removed = pruned_module(before["module"])
    projection = dict(
        before=before["module"],
        after=after["module"],
        beforeFrame=before["frame"],
        afterFrame=after["frame"],
        swaps=permutation_hint(expected, after["module"]),
        removals=removal_plan(before["module"]),
    )
    return dict(projection=projection, query=query(projection), proposed_rewrite_sites=len(removed))
