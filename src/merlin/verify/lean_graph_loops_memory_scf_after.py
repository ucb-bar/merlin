"""Prepare an independent actual-final term and exact initializer obligations.

This development API does not claim the whole memory bridge: slot/register and
complete-function simulation remain to be derived. Native checks require a new
separately reviewed context. No old pair matcher or saved plan is called here.
"""

from __future__ import annotations

import hashlib
import json

from .lean_graph_input_admission import parse_admitted
from .lean_graph_loops_memory_scf_after_projection import decode_final
from .lean_graph_loops_shape_capture import export
from .lean_graph_loops_shape_epoch_model import module_literal, uniform_literal
from .lean_graph_loops_shape_model import binding
from .lean_graph_loops_unroll import _list, _nat, _string


def _single(values, label):
    if len(values) != 1:
        raise ValueError("one " + label + " required in the rectangular target subset")
    return values[0]


def initializer(region):
    """Extract a hint; emitted equality checks the complete original region."""
    addition = region["body"]
    if addition["tag"] != "action" or json.loads(addition["key"])["name"] != "arith.addi":
        raise ValueError("actual outer body must begin with index addition")
    minimum = addition["rest"]
    if minimum["tag"] != "action" or json.loads(minimum["key"])["name"] != "arith.minui":
        raise ValueError("actual outer body must retain the generated minimum")
    inner = minimum["rest"]
    if inner["tag"] != "region" or inner["body"]["tag"] != "region":
        raise ValueError("actual target requires inner point and column loops")
    row, store = inner["body"], inner["body"]["body"]
    if store["tag"] != "action" or json.loads(store["key"])["name"] != "memref.store":
        raise ValueError("actual rectangular target requires its store")
    if any(len(node["operands"]) != 3 for node in (region, inner, row)) or len(store["operands"]) != 4:
        raise ValueError("rectangular target control/store arity differs")
    uniform = dict(
        storeKey=store["key"],
        loopKey=region["key"],
        seed=store["operands"][0],
        buffer=store["operands"][1],
        outer=_single(inner["parameters"], "point induction binding"),
        inner=_single(row["parameters"], "column induction binding"),
        zero=region["operands"][0],
        rows=region["operands"][1],
        columns=row["operands"][1],
        one=inner["operands"][2],
    )
    kind = uniform["buffer"]["kind"]
    if kind["tag"] != "view" or len(kind["type"]["shape"]) != 2:
        raise ValueError("two-dimensional static final initializer required")
    if uniform["zero"]["known"] != 0 or uniform["one"]["known"] != 1:
        raise ValueError("current final initializer subset requires zero lower bound and unit point stride")
    values = [uniform[name]["known"] for name in ("rows", "columns")]
    if values != kind["type"]["shape"]:
        raise ValueError("current final initializer must cover its complete rectangular view")
    return dict(
        uniform=uniform,
        tile=region["operands"][2],
        tileIV=_single(region["parameters"], "tile induction binding"),
        sum=_single(addition["results"], "addition result"),
        bound=_single(minimum["results"], "minimum result"),
        addKey=addition["key"],
        minKey=minimum["key"],
        shape=kind["type"]["shape"],
        element=kind["type"]["element"],
    )


def initializers(module):
    result = []
    for function in module["functions"]:
        code, position = function["code"], 0
        while code["tag"] != "returned":
            if code["tag"] == "region":
                result.append(dict(initializer(code), function=function["name"], position=position))
                position += 1
            code = code["rest"]
    return result


def _initializer_execution(name, site, bits):
    """Instantiate the local semantic root; whole-function joining stays separate."""
    rows, columns = map(_nat, site["shape"])
    tile, element = _nat(site["tile"]["known"]), _string(site["element"])
    return f"""theorem {name}_execution {{Value Public : Type}}
    (base : Interpretation Value (MemoryWorld Value Public)) (classify : String → MemoryOpcode)
    (object : Nat) (value : Value)
    (store : classify {name}.uniform.storeKey = .store)
    (loop : classify {name}.uniform.loopKey = .loop)
    (add : (withMemory base classify).action {name}.addKey =
      GraphLoopsMemorySCF.After.IndexAction {_nat(bits)} .add)
    (minimum : (withMemory base classify).action {name}.minKey =
      GraphLoopsMemorySCF.After.IndexAction {_nat(bits)} .minimum)
    (aliases : Aliases) (env : Merlin.GraphLoopsShape.Env Value)
    (inputs : {name}.Inputs env object {rows} {columns} {tile} {element} value)
    (meaning : GraphLoopsMemorySCF.PureMeaning Value)
    (before after : MemoryWorld Value Public) (outputs : List (Runtime Value)) :
    Eval (withMemory base classify) false aliases {name}.code env before outputs after ↔
      outputs = [] ∧ after.observed = before.observed ∧
        GraphLoopsMemorySCF.HeapRun meaning
          (GraphLoopsMemorySCF.Witness.uniform {rows} {columns}
            {name}.uniform.outer.name {name}.uniform.inner.name {name}.uniform.seed.name)
          {tile} object true (Epoch.eraseEnv env) before.heap after.heap :=
  {name}.eval_epoch base classify {_nat(bits)} {rows} {columns} {tile} object {element} value
    {name}_bounds_valid {name}_uniform_facts {name}_control_facts
    store loop add minimum aliases env inputs meaning before after outputs
"""


def obligation(model, sites, bits):
    roots = ["GraphMemoryAfterCandidate.actual_final_typed"]
    text = """import Merlin.GraphLoopsMemorySCFAfterTiled
import Merlin.GraphLoopsShapeEpochModel
set_option autoImplicit false
set_option maxRecDepth 16384
set_option maxHeartbeats 8000000
open Merlin Merlin.GraphLoopsShape
namespace GraphMemoryAfterCandidate
"""
    text += "def actualFinal : Module := " + module_literal(model["module"]) + "\n"
    text += (
        "def actualIndexOpcodes : GraphLoopsMemorySCF.After.IndexOpcodes := "
        + _list(model["index_opcodes"], lambda pair: "(" + _string(pair[0]) + ", ." + pair[1] + ")")
        + "\n"
    )
    text += "theorem actual_final_typed : actualFinal.Valid " + _nat(bits) + " := by decide\n"
    for index, site in enumerate(sites):
        name = "initializer" + str(index)
        fields = [uniform_literal(site["uniform"])] + [binding(site[key]) for key in ("tile", "tileIV", "sum", "bound")]
        fields += [_string(site[key]) for key in ("addKey", "minKey")]
        text += "def " + name + " : GraphLoopsMemorySCF.After.TiledInitializer := ⟨" + ", ".join(fields) + "⟩\n"
        text += f"""theorem {name}_syntax :
    (functionAt actualFinal {_string(site["function"])}).map
      (fun function => (Epoch.topRegions function.code)[{_nat(site["position"])}]?) =
        some (some {name}.code) := by decide
def {name}_bounds : GraphLoopsSegment.Bounds :=
  ⟨{_nat(bits)}, 0, {_nat(site["shape"][0])}, 1, {_nat(site["tile"]["known"])}⟩
theorem {name}_bounds_valid : {name}_bounds.Valid := by decide
theorem {name}_uniform_facts :
    {name}.uniform.Facts {_nat(site["shape"][0])} {_nat(site["shape"][1])} {_string(site["element"])} := by decide
theorem {name}_control_facts : {name}.ControlFacts := by decide
"""
        text += _initializer_execution(name, site, bits)
        roots += [
            "GraphMemoryAfterCandidate." + name + suffix
            for suffix in ("_syntax", "_bounds_valid", "_uniform_facts", "_control_facts", "_execution")
        ]
    text += "end GraphMemoryAfterCandidate\n"
    return text, roots


def prepare_final(raw, *, parser, index_bits, max_nodes, tiled=True):
    parsed = parse_admitted(raw, parser, calls="shape-direct")
    actual = export(parsed)
    model = decode_final(actual, index_bits=index_bits, max_nodes=max_nodes)
    sites = initializers(model["module"]) if tiled else []
    query, roots = obligation(model, sites, index_bits)
    return dict(
        original_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        actual_export=actual,
        model=model,
        initializers=sites,
        query=query,
        roots=roots,
        query_format="pure-declarations",
        query_version="actual-final-local-tiled-v2",
        local_initializer_root="Merlin.GraphLoopsMemorySCF.After.TiledInitializer.eval_epoch",
        local_initializer_root_kernel_checked=False,
        complete_memory_bridge_proved=False,
        kernel_checked=False,
        qualified=False,
        pipeline_receipt_issued=False,
    )
