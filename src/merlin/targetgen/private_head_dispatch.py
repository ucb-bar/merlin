"""Independent head invocation and actual stock descriptor-shim custody checks."""

import json
from pathlib import Path


def verify_bridge(entry, component, *, built, target, collection=None):
    from merlin.common import invocation_record
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.compilation_recipe import verify_completed_recipe
    from merlin.llvmlower.device_shim import emit_logical_translation_unit, logical_buffers
    from merlin.llvmlower.head_source_route import validate_head_member_ranges

    from .native_model_execution import _digest
    from .owned_host_model_execution import _pinned

    whole = component.get("whole_batch", False)
    if type(whole) is not bool:
        raise ValueError("head bridge has no explicit original batch ownership")
    kind = "whole_logical_batch" if whole else "stock_logical_batch_slices"
    products = validate_head_member_ranges(component)
    custody = (entry.get("custody") or {}).get("head_source")
    if (not isinstance(custody, dict) or custody.get("source_ir_sha256") != component["source_ir_sha256"]
            or custody.get("interface_sha256") != component["interface_sha256"]
            or custody.get("matrix_invocations") != component["matrix_invocations"]
            or custody.get("bridge_kind") != kind
            or whole and (custody.get("kernel_invocations") != 1
                          or custody.get("mathematical_products") != products)):
        raise ValueError("head bridge lacks exact original source/interface/invocation custody")
    bridge = custody.get("descriptor_bridge")
    if not isinstance(bridge, dict) or set(bridge) != {"source", "object", "compilation"}:
        raise ValueError("head bridge lacks actual source/object/compiler custody")
    source, obj = _pinned(bridge, "source"), _pinned(bridge, "object")
    sidecar_pin = built["device_sidecar"]
    sidecar = json.loads(_pinned({"sidecar": {**sidecar_pin,
                         "size_bytes": Path(sidecar_pin["path"]).stat().st_size}}, "sidecar").read_text())
    kernels = built["device_artifacts"]["kernels"]
    symbols = {}
    for row in sidecar["routed"]:
        name = row["symbol"]
        chosen = sidecar["expected_interfaces"][name]
        from hashlib import sha256

        if sha256(chosen["mlir"].encode()).hexdigest() != chosen["sha256"]:
            raise ValueError("head matrix interface changed after source admission")
        buffers, reason = logical_buffers(chosen["mlir"])
        if buffers is None:
            raise ValueError("head matrix interface refused: " + reason)
        symbols[name] = buffers, sidecar["call_buffers"][name]
    unit = emit_logical_translation_unit(target, symbols,
                                         kernel_symbol_for=lambda name: kernels[name]["kernel_symbol"])
    expected = unit.text
    if collection is not None:
        from merlin.llvmlower.sibling_source_route import borrowed_bridge_source

        for sibling in collection.get("components", []):
            if sibling.get("schema") != "merlin.original_sibling_source_composition.v1":
                continue
            call = sidecar["call_buffers"][sibling["physical_symbol"]]
            outputs = tuple(i for i, row in enumerate(call) if row["role"].startswith("out_"))
            expected += "\n" + borrowed_bridge_source(
                sibling["physical_symbol"], sibling["borrowed_entry_symbol"], call, outputs)
    if set(unit.symbols) != set(symbols) or source.read_text() != expected or unit.skipped:
        raise ValueError("head descriptor bridge differs from exact stock batch-slice transport")
    call = sidecar["call_buffers"][component["physical_symbol"]]
    h = component["matrix_invocations"]
    m, k, n = component["geometry_m_k_n"]
    if ([r["shape"] for r in call] != [[h, m, k], [h, k, n], [h, m, n]]
            or [r["dtype"] for r in call] != ["i8", "i8", "i32"]):
        raise ValueError("head slice bridge no longer covers every original product/view")
    buffers, reason = logical_buffers(sidecar["expected_interfaces"][component["physical_symbol"]]["mlir"])
    expected_shapes = [[h, m, k], [h, k, n], [h, m, n]] if whole else [[m, k], [k, n], [m, n]]
    if buffers is None or [list(b.shape) for b in buffers] != expected_shapes:
        raise ValueError("head interface lost original whole/sliced shape ownership: " + (reason or "shape drift"))
    observed = invocation_record.verify(_pinned(bridge, "compilation"))

    def pin(path):
        return {key: _digest(path)[key] for key in ("path", "sha256")}

    if (observed.get("stage") != "device_shim_compilation" or observed.get("inputs") != [pin(source)]
            or observed.get("outputs") != [pin(obj)]
            or Path(observed["executable"]["path"]) != toolchain.clang().resolve()
            or observed["argv"][-4:] != ["-c", str(source), "-o", str(obj)]):
        raise ValueError("head bridge lost the original observed C compiler boundary")
    recipe = built["compilation_recipe"]
    compilation = verify_completed_recipe(Path(recipe["path"]), executable=Path(built["elf"]),
                                          expected_recipe_sha256=recipe["sha256"])
    linked = {(row["path"], row["sha256"]) for row in compilation["commands"][-1]["inputs"]}
    if (str(obj), _digest(obj)["sha256"]) not in linked:
        raise ValueError("head descriptor bridge object is not an actual final ELF input")
    result = {"source": _digest(source), "object": _digest(obj),
            "compilation": _digest(_pinned(bridge, "compilation")), "matrix_invocations": h,
            "scope": "exact stock positive-count loop over disjoint original head slices; "
                     "one synchronous matrix artifact invocation per original product"}
    if products != h:
        result["scope"] = ("exact stock positive-count loop over source-proven grouped row slices; "
                           "every original product retains its independently rederived row range")
    if whole:
        result.update(kernel_invocations=1, mathematical_products=products,
                      scope="exact ranked whole-batch interface; one kernel call covers every original product")
    return result


def mandatory_heads(function, commands, obligations, component, *, mandatory):
    """Keep every source obligation; whole-batch object budgets are checked separately."""
    from merlin.llvmlower.head_source_route import validate_head_member_ranges

    members = component["members"]
    products = validate_head_member_ranges(component)
    h = component["matrix_invocations"]
    if type(h) is not int or h < 2:
        raise ValueError("head dispatch requires every original positive-count slice")
    by_group = {r["source_group"]: r for r in members}
    if len(by_group) != products:
        raise ValueError("head dispatch repeats an original product")
    whole = component.get("whole_batch", False)
    if whole and ({r["source_group"] for r in obligations} != set(by_group)
                  or component.get("mathematical_products") != products or component.get("kernel_invocations") != 1):
        raise ValueError("whole head dispatch lacks its complete independently admitted product roster")
    result = []
    for obligation in obligations:
        member = by_group.get(obligation["source_group"])
        if (member is None or obligation["region_id"] != member["source_region_id"]
                or obligation["semantic_family"] != "contraction"):
            raise ValueError("raw head invocation cannot cover original supporting source work")
        checked = mandatory(function, commands, [obligation])
        ranges = ({key: member[key] for key in ("projected_head", "row_start", "row_count")}
                  if "projected_head" in member else {})
        result.extend({**row, **ranges, "head_slice": member["head"],
                       "invocation_basis": ("independently verified complete whole-batch interface and object budget"
                                            if whole else "independently verified stock batch-slice descriptor loop")}
                      for row in checked)
    return result
