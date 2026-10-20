"""Select exact physical fusion from pinned software semantics and RTL bounds.

Reuse the existing scalar MLIR correspondence and physical fusion planner.
Target callbacks emit instructions; selected declarations bind the operation,
layout and resource domain. Neither source text nor an identity qualifies it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

from merlin.sched.ir import Call, evaluate
from merlin.targetgen.isa_spec import resolve_fact_integer
from merlin.verify.epilogue_applicability import ReadoutCapability

from .fusion import FusionCapability
from .fusion_semantics import ElementwiseSemantics, UnsupportedElementwise, scalar_graph
from .storage_contracts import binding_value, resolve_element_count


def _fields(value, fields, label):
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError(f"fusion {label}: expected exactly {sorted(fields)}")
    return value


def _strings(value, label):
    if type(value) is not list or not value or len(value) > 256 or any(type(x) is not str or not x for x in value):
        raise ValueError(f"fusion {label}: requires a bounded nonempty string list")
    return value


@dataclass(frozen=True)
class SelectedFusionContract:
    declarations_json: str

    def bind(self, adapter, *, emitters, qualification_identity, max_operations, max_ir_bytes):
        """Build qualified semantics/capabilities; preserve legal unsupported routes."""
        specification = json.loads(self.declarations_json)
        if emitters is None:
            emitters = {}
        if (type(emitters) is not dict or set(emitters) - set(specification["readouts"])
                or any(not callable(emit) for emit in emitters.values())):
            raise ValueError("fusion emitters must name declared readouts and be callable")
        if qualification_identity is not None and (
            type(qualification_identity) is not str or not qualification_identity.strip()
        ):
            raise ValueError("fusion qualification must be an independently selected nonempty identity")
        emitters = dict(emitters)
        qualified = qualification_identity is not None
        admitted, refusals = {}, {}
        for name, row in specification["elementwise"].items():
            if not qualified:
                refusals[name] = "missing_execution_qualification"
                continue
            if row["execution"] != "contiguous_read_then_write":
                refusals[name] = "unsupported_or_unknown_execution"
                continue
            try:
                graph = scalar_graph(row["scalar_ir"].encode(), max_operations, max_ir_bytes)
            except UnsupportedElementwise as exc:
                refusals[name] = str(exc)
                continue
            if (*graph.input_bits, graph.output_bits) != tuple(row["element_bits"]):
                raise ValueError("fusion scalar MLIR widths disagree with selected RTL element widths")
            if len(row["byte_orders"]) != len(row["element_bits"]):
                raise ValueError("fusion byte layouts must cover every scalar input and result")
            admitted[name] = graph

        def semantics(call):
            if call.instr not in admitted:
                return None
            row = specification["elementwise"][call.instr]
            values = {key: evaluate(value, {}) for key, value in call.args}
            elements = binding_value({"product": row["elements"], "scale": 1}, values)
            if row["readout"] is not None:
                physical = specification["readouts"].get(row["readout"])
                if physical is None or physical["max_elements"] is None or elements > physical["max_elements"]:
                    raise UnsupportedElementwise("actual readout instruction exceeds its selected RTL resource domain")
            return ElementwiseSemantics(row["scalar_ir"].encode(), elements, tuple(row["stages"]),
                                        tuple(row["byte_orders"]), row["readout"])

        def project(call):
            effect = adapter.project(call)
            if effect is None:
                return None
            selected = semantics(call)
            if selected is not None and effect.elementwise is not None:
                actual = effect.elementwise
                graph = scalar_graph(actual.function_ir, max_operations, max_ir_bytes)
                if (graph != admitted[call.instr] or replace(actual, function_ir=selected.function_ir) != selected):
                    raise ValueError("provider elementwise projection disagrees with selected semantics/layout")
            return replace(effect, elementwise=selected)

        capabilities, unavailable = [], {}
        for selector, readout in specification["readouts"].items():
            if not qualified or selector not in emitters or readout["max_elements"] is None:
                unavailable[selector] = ("missing_execution_qualification" if not qualified else
                                         "missing_physical_emitter" if selector not in emitters
                                         else "unknown_RTL_bound")
                continue
            routes = tuple((r["producer"], r["consumer"], r["result"]) for r in readout["routes"]
                           if {r["producer"], r["consumer"], r["result"]} <= set(admitted))
            if not routes:
                unavailable[selector] = "unsupported_computation_or_execution"
                continue

            def emit(producer, consumer, *, selected=selector, declared=readout, routes=routes):
                matching = [result for p, c, result in routes if (producer.instr, consumer.instr) == (p, c)]
                if not matching or semantics(producer).elements > declared["max_elements"]:
                    return None
                call = emitters[selected](producer, consumer)
                if call is not None and (type(call) is not Call or call.instr not in matching):
                    raise ValueError("physical fusion emitter returned an undeclared instruction composition")
                return call

            capabilities.append(FusionCapability(
                ReadoutCapability(selector, frozenset(readout["applies"]), readout["evidence"]), emit,
                adapter.configuration_identity, adapter.numerical_policy_identity, qualification_identity,
            ))
        return replace(adapter, project=project), tuple(capabilities), {
            "schema": "merlin.selected_fusion_binding.v1", "qualification_identity": qualification_identity,
            "admitted_instructions": sorted(admitted), "unavailable_instructions": refusals,
            "selected_readouts": [cap.readout.selector for cap in capabilities], "unavailable_readouts": unavailable,
        }


def resolve_fusion_contract(declaration, facts, isa, storage):
    """Resolve only explicit typed semantics, physical routes and structural bounds."""
    row = _fields(declaration, ("schema", "elementwise", "readouts"), "contract")
    if row["schema"] != "merlin.physical_fusion_contract.v1" or storage is None:
        raise ValueError("selected fusion requires its versioned contract and selected storage effects")
    effects = json.loads(storage.declarations_json)
    elementwise = row["elementwise"]
    if type(elementwise) is not dict or len(elementwise) > 4096:
        raise ValueError("fusion elementwise declarations must be a bounded instruction mapping")
    normalized = {}
    for name, item in elementwise.items():
        instruction = isa.instruction(name)
        _fields(item, ("scalar_ir", "elements", "element_bits", "stages", "byte_orders", "readout", "execution"),
                "elementwise declaration")
        if (name not in effects or effects[name]["exact_copy"]
                or effects[name]["nonmemory_effects"] and not effects[name]["configuration_reads"]
                or effects[name]["completion"] != "synchronous"):
            raise ValueError("fusion requires synchronous memory-only effects or checked read-only configuration")
        if type(item["scalar_ir"]) is not str or not item["scalar_ir"]:
            raise ValueError("fusion computation must be explicit scalar MLIR text")
        if type(item["element_bits"]) is not list or not item["element_bits"] or len(item["element_bits"]) > 256:
            raise ValueError("fusion requires a bounded list of RTL element widths")
        widths = [resolve_fact_integer(v, facts, label="fusion element bits", minimum=1) for v in item["element_bits"]]
        if len(widths) != len(effects[name]["accesses"]):
            raise ValueError("fusion RTL widths must correspond to every selected memory access")
        orders = _strings(item["byte_orders"], "byte orders")
        if any(order not in ("little", "big") for order in orders):
            raise ValueError("fusion byte order must be explicit little or big")
        stages = _strings(item["stages"], "ordered stages")
        if item["readout"] is not None and (type(item["readout"]) is not str or not item["readout"]):
            raise ValueError("fusion readout identity must be explicit or absent")
        if type(item["execution"]) is not str or not item["execution"]:
            raise ValueError("fusion element execution semantics must be explicit or unknown")
        normalized[name] = {**item, "element_bits": widths, "stages": stages, "byte_orders": orders,
                            "elements": resolve_element_count(item["elements"], instruction)}
    readouts = row["readouts"]
    if type(readouts) is not dict or len(readouts) > 256:
        raise ValueError("fusion readouts must be a bounded mapping")
    physical = {}
    for selector, readout in readouts.items():
        if type(selector) is not str or not selector:
            raise ValueError("fusion requires named physical readouts")
        _fields(readout, ("applies", "evidence", "max_elements", "routes"), "physical readout")
        _strings(readout["applies"], "readout stages")
        if type(readout["evidence"]) is not str or not readout["evidence"]:
            raise ValueError("fusion requires explicit readout declaration evidence")
        maximum = (None if readout["max_elements"] is None else
                   resolve_fact_integer(readout["max_elements"], facts, label="fusion maximum elements", minimum=1))
        if type(readout["routes"]) is not list or not readout["routes"] or len(readout["routes"]) > 256:
            raise ValueError("fusion requires bounded explicit instruction compositions")
        for route in readout["routes"]:
            _fields(route, ("producer", "consumer", "result"), "physical composition")
            if any(type(name) is not str or name not in normalized for name in route.values()):
                raise ValueError("fusion composition must name selected scalar instructions")
            producer, consumer, result = (normalized[route[k]] for k in ("producer", "consumer", "result"))
            if result["stages"] != producer["stages"] + consumer["stages"] or result["readout"] != selector:
                raise ValueError("fusion result must bind one readout and the complete ordered source stages")
        physical[selector] = {**readout, "max_elements": maximum}
    return SelectedFusionContract(json.dumps({"elementwise": normalized, "readouts": physical}, sort_keys=True))
