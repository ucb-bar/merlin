"""Pure, bounded fusion enumeration from a frozen output path, spec and facts.

This is a proposal denominator, not fusion admission. ``enumerate_candidates``
never reads target sources or executes a backend. The caller freezes the result
of ``capability_manifests.derived_output_path`` in the contract first.

The minimal profile uses parameter classes (not shape samples): nonzero integer
residuals cover the derived interval, partitioned by sign/tile count.
Contradictory one-producer rows prune their graph extensions; those rows retain
an explicit ``pruning_scope``. Pending rows still participate in shared graphs.
Additional evidence can live in facts.fusion_enumeration or
readout_semantics.fusion_enumeration as {status: derived, value: ..., evidence:
[locator, ...]} records. Unknown controls/resources are never defaulted to true.
"""

from __future__ import annotations

import copy
import inspect
from itertools import combinations_with_replacement, product
from pathlib import Path

from merlin.common.digest import sha256_file
from merlin.common.jsonio import canonical_json, canonical_sha256, write_canonical_json
from merlin.xdsl_dialects.lowering.k_extension import lower_k_extension

PROFILE = {
    "schema": "merlin.fusion_enumeration_profile.minimal.v1",
    "max_producers": 2,
    "coefficient_tiles": "derived_from_k_and_prefix_bounds",
    "case_parameters": ["shapes", "tails", "edge_values", "output_scales"],
    "fractional_controls": ["-1/2", "1/2"],
    "max_rows": 1_000_000,
    "serial_edges": "out_of_profile",
    "incompatible_node_extensions": "pruned_with_witness",
    "dataflow_variants": "mesh_product_graphs_only",
}


def _unordered_inputs(value, key=""):
    """Normalize set-like inventories; ordered physical paths remain ordered."""
    if isinstance(value, dict):
        normalized = {k: _unordered_inputs(v, k) for k, v in value.items()}
        if key == "dataflows" and isinstance(normalized.get("value"), list):
            normalized["value"] = sorted(normalized["value"], key=canonical_json)
        return normalized
    if isinstance(value, list):
        members = [_unordered_inputs(v) for v in value]
        if key in {
            "operations",
            "schemes",
            "constraints",
            "arrays",
            "memories",
            "datapaths",
            "compute_datapaths",
            "storage_datapaths",
            "interfaces",
            "evidence",
            "families",
            "ranks",
            "layouts",
            "ops",
            "unknown",
            "scale_value",
            "scale_source",
            "weight_scale_granularity",
            "activation_scale_granularity",
            "readout_exit",
            "operand_sum_rounding",
            "residual_coefficient",
            "reductions",
            "accumulator_stages",
        }:
            return sorted(members, key=canonical_json)
        return members
    return value


def _spec_input(spec):
    # No composition declarations are input authority, even for input identity.
    return _unordered_inputs({k: v for k, v in spec.items() if k != "compositions"})


def _finding(state, reason, evidence):
    result = {"state": state, "reason": reason, "evidence": evidence}
    if state == "pending_facts":
        result["missing_fact"] = evidence
    return result


def _require(record, locator, reason, *, expected=None):
    if record.get("status") == "conflict":
        return [_finding("incompatible", "conflicting evidence: " + reason, locator)]
    from merlin.targetgen.quantization_spec import unresolved_values

    if record.get("status") != "derived" or not record.get("evidence") or unresolved_values(record.get("value")):
        details = [
            item["reason"] for item in record.get("evidence", []) if isinstance(item, dict) and item.get("reason")
        ]
        if details:
            reason += ": " + "; ".join(details)
        return [_finding("pending_facts", reason, locator)]
    if expected is not None and record["value"] != expected:
        return [_finding("incompatible", reason, locator)]
    return []


def _record(value, locator):
    return {"status": "derived" if value is not None else "unknown", "value": value, "evidence": [locator]}


def _explicit(body, contract):
    result = copy.deepcopy((contract.get("readout_semantics") or {}).get("fusion_enumeration") or {})
    # Conflicting premises are retained, never resolved by precedence.
    for key, record in (body.get("fusion_enumeration") or {}).items():
        if key in result and _unordered_inputs(result[key], key) != _unordered_inputs(record, key):
            result[key] = {
                "status": "conflict",
                "value": None,
                "evidence": [
                    f"facts#/fusion_enumeration/{key}",
                    f"contract#/readout_semantics/fusion_enumeration/{key}",
                ],
            }
        else:
            result[key] = copy.deepcopy(record)
    return result


def normalize_fusion_facts(*, target, facts, contract, spec=None):
    """Consume the shared composition derivation and frozen output controls."""
    from merlin.targetgen.capability_manifests import derived_output_path
    from merlin.targetgen.composition_facts import derive
    from merlin.targetgen.quant_methodology import output_path_of

    path = derived_output_path(target, facts=facts, contract=contract) if output_path_of(contract) else None
    composition = derive(facts=facts, contract=contract, software_spec=_spec_input(spec or {}), output_path=path)
    fields = _explicit(facts.get("facts", facts), contract)
    aliases = {"operand_signed_bits": "operand_bits"}
    for name, value in composition["facts"].items():
        _install_derived(
            fields,
            aliases.get(name, name),
            {
                "status": "derived",
                "value": value,
                "evidence": copy.deepcopy(composition["evidence"][name])
                if isinstance(composition["evidence"][name], list)
                else [composition["evidence"][name]],
            },
        )
    for name, reason in composition["unknown"].items():
        record = _record(None, f"composition_facts#/unknown/{name}")
        record["evidence"].append({"reason": reason})
        fields.setdefault(aliases.get(name, name), record)
    epilogue = contract.get("output_epilogue") or {}
    for key in ("bias_preload", "accumulate"):
        source = epilogue.get(key) or {}
        if source.get("status") == "derived" and source.get("evidence"):
            _install_derived(fields, key, _record(True, f"contract#/output_epilogue/{key}"))
    operand_sum = (contract.get("readout_semantics") or {}).get("operand_sum") or {}
    if operand_sum.get("schema") == "operand_sum_contract_v1" and operand_sum.get("operands") == 2:
        if all(key in operand_sum for key in ("operand_dtype", "scale_dtype", "operand_rounding", "operand_saturates")):
            _install_derived(fields, "operand_sum", _record(True, "contract#/readout_semantics/operand_sum"))
    return _unordered_inputs({"schema": "merlin.fusion_facts.v1", "output_path": path, "fields": fields})


def _install_derived(fields, name, record):
    authored = fields.get(name) or {}
    if authored.get("status") == "conflict" or (
        authored.get("value") is not None
        and canonical_json(_unordered_inputs({name: {"value": authored["value"]}}))
        != canonical_json(_unordered_inputs({name: {"value": record["value"]}}))
    ):
        raise ValueError(f"authored fusion_enumeration {name} conflicts with derived value {record['value']!r}")
    fields[name] = record


def _families(spec):
    from merlin.targetgen.semantic_families import from_op

    found = set()
    declarations = spec.get("operations") or []
    for row in declarations.values() if isinstance(declarations, dict) else declarations:
        if row.get("placement") == "host":
            continue
        found.update(row.get("families") or [])
        found.update(f for op in row.get("ops") or [] if (f := from_op(op)))
    if isinstance(declarations, dict):
        found.update(name for name, row in declarations.items() if row.get("placement") != "host")
    return found


# Whether each producer form executes a product on the contraction mesh, as
# fixed by its lowering: contractions and the ones-operand window mean are mesh
# products; scaled loads accumulate on load and a pool window is a load plus a
# pooled store. Only a mesh product has a dataflow to select.
_MESH_PRODUCT = {
    ("contraction", "matmul"): True,
    ("contraction", "conv_as_matmul"): True,
    ("window_movement", "mean"): True,
    ("operand_sum", "scaled_loads"): False,
    ("window_movement", "pool"): False,
}
DATAFLOW_NOT_APPLICABLE = "not_applicable"


def uses_mesh_product(node):
    """Return whether a semantic node executes a mesh product; unknown forms fail closed."""
    key = (node.get("form"), node.get("realization"))
    if key not in _MESH_PRODUCT:
        raise ValueError(f"UNKNOWN: producer form {key!r} has no declared mesh-product semantics")
    return _MESH_PRODUCT[key]


def dataflow_variants(nodes):
    """Every selectable dataflow for a graph with a mesh product, else exactly one."""
    # Classify every node first so one unknown form cannot hide behind a known product.
    if any([uses_mesh_product(node) for node in nodes]):
        return ("WS", "OS")
    return (DATAFLOW_NOT_APPLICABLE,)


def _producer_forms(spec):
    families = _families(spec)
    forms = []
    if "contraction" in families:
        forms.extend([("contraction", "matmul"), ("contraction", "conv_as_matmul")])
    if "operand_sum" in families or "elementwise_map" in families:
        forms.append(("operand_sum", "scaled_loads"))
    if families & {"movement", "reduction", "window_mean"}:
        forms.append(("window_movement", "pool"))
    if "window_mean" in families:
        forms.append(("window_movement", "mean"))
    return forms


def _routes(path, fields):
    if path is None:
        return [
            (
                {"exit": "UNKNOWN", "stage_indices": []},
                [_finding("pending_facts", "ordered output path is missing", "contract#/output_epilogue/output_path")],
            )
        ]
    routes = []
    for exit_name in ("accumulator", "requant"):
        availability = path["exits"].get(exit_name)
        if availability is False:
            routes.append(
                (
                    {"exit": exit_name, "stage_indices": []},
                    [_finding("incompatible", "exit is not available", f"output_path#/exits/{exit_name}")],
                )
            )
            continue
        masks = [()] if exit_name == "accumulator" else product((False, True), repeat=len(path["readout"]))
        for mask in masks:
            indices = [i for i, keep in enumerate(mask) if keep]
            issues = (
                []
                if availability is True
                else [_finding("pending_facts", "exit availability is unknown", f"output_path#/exits/{exit_name}")]
            )
            if exit_name == "requant":
                issues.extend(_route_requirements(path, fields, indices))
            route = {"exit": exit_name, "stage_indices": indices}
            pools = [i for i in indices if path["readout"][i]["kind"] == "pool"]
            if pools:
                route["pool_padding"] = (
                    "zero_or_padded" if all(_nonnegative_prefix(path, indices, i) for i in pools) else "unpadded_only"
                )
            if pools:
                padding_classes = ("unpadded", "padded") if route["pool_padding"] == "zero_or_padded" else ("unpadded",)
                routes.extend(({**route, "pool_padding_class": padding}, issues) for padding in padding_classes)
            else:
                routes.append((route, issues))
    # One witness per static contradiction covers its extensions. Numerically
    # different masks of the same impossible type path are not candidates.
    valid, contradictions = [], {}
    for route, issues in routes:
        incompatible = [issue for issue in issues if issue["state"] == "incompatible"]
        if not incompatible:
            valid.append((route, issues))
            continue
        rule = canonical_json(incompatible[0])
        if rule not in contradictions:
            contradictions[rule] = (route, issues)
    return valid + [contradictions[key] for key in sorted(contradictions)]


def _route_requirements(path, fields, indices):
    from merlin.targetgen.quantization_spec import unresolved_values

    issues = []
    kinds = [path["readout"][i]["kind"] for i in indices]
    for i in indices:
        for unknown in unresolved_values(path["readout"][i]):
            issues.append(
                _finding("pending_facts", "readout stage field is unresolved", f"output_path#/readout/{i}/{unknown}")
            )
    for i, stage in enumerate(path["readout"]):
        if i not in indices:
            issues.extend(
                _require(
                    fields.get(f"bypass.{i}", {}),
                    f"fusion_facts#/fields/bypass.{i}",
                    "omitted stage needs an identity or disable realization",
                    expected=True,
                )
            )
        elif stage["kind"] == "pool" and stage.get("padding", "unknown").lower() == "unknown":
            issues.append(
                _finding("pending_facts", "pool padding semantics are missing", f"output_path#/readout/{i}/padding")
            )
    if "scale" in kinds and "convert" not in kinds:
        issues.append(_finding("incompatible", "scale requires the preceding typed conversion", "output_path#/readout"))
    if "requant" not in kinds:
        issues.append(
            _finding("incompatible", "narrow exit requires the narrowing stage", "output_path#/exits/requant")
        )
    return issues


def _nonnegative_prefix(path, indices, stop):
    """Prove a lower bound of zero through the selected typed readout stages."""
    nonnegative = False
    for index in indices:
        if index >= stop:
            break
        stage = path["readout"][index]
        kind = stage["kind"]
        if kind == "activation":
            nonnegative = stage.get("function") == "relu"
        elif kind == "requant":
            clamp = stage.get("clamp")
            nonnegative = bool(clamp and clamp[1] >= 0 and (nonnegative or clamp[0] >= 0))
        elif kind not in {"convert", "scale", "pool"}:
            nonnegative = False
        # Scale values are case parameters; positive multipliers are an existing
        # composition obligation when activation precedes the hardware scale.
    return nonnegative


def _extension_bounds(fields):
    """Bound the aligned representative with the production all-prefix proof.

    Geometry is a case parameter. The shared panel width supplies a reproducible
    representative K=N; every E2 geometry must recheck these same guards.
    """
    required = ("operand_bits", "k_extension_within", "weight_panel_width")
    if any(_require(fields.get(k, {}), f"fusion_facts#/fields/{k}", k) for k in required):
        return {
            "status": "unknown",
            "missing": [k for k in required if _require(fields.get(k, {}), f"fusion_facts#/fields/{k}", k)],
        }
    bits = fields["operand_bits"]["value"]
    within = fields["k_extension_within"]["value"]
    width = fields["weight_panel_width"]["value"]
    maximum = (1 << (bits - 1)) - 1
    source_bound = width * (1 << (bits - 1)) ** 2
    arguments = dict(
        source_k=width,
        residual_width=width,
        digit_bits=bits,
        max_extended_k=within["max_k"],
        accumulator_bits=within["partial_sum_bits"],
        source_bound=source_bound,
        residual_range=[-(1 << (bits - 1)), maximum],
    )
    # Monotonic capacity and all-prefix predicates: binary search the lowering,
    # rather than implementing a second arithmetic legality rule.
    low, high = 0, max(0, (within["max_k"] - width) // width) * maximum
    while low < high:
        middle = (low + high + 1) // 2
        try:
            lower_k_extension(middle, **arguments)
        except ValueError:
            high = middle - 1
        else:
            low = middle
    tiles = (low + maximum - 1) // maximum
    partitions = [
        dict(sign=sign, tiles=t, abs_min=(t - 1) * maximum + 1, abs_max=min(t * maximum, low))
        for sign, t in product((-1, 1), range(1, tiles + 1))
    ]
    try:
        proofs = [lower_k_extension(c, **arguments) for c in sorted({-low, 0, low})]
    except ValueError:
        proofs = []  # Even the no-extension representative exceeds the supplied bound.
    return dict(
        status="derived",
        T_max=tiles,
        coefficient_bound=low,
        digit_max=maximum,
        representative=arguments,
        partitions=partitions,
        boundary_proofs=proofs,
    )


def _residual_classes(fields):
    bounds = _extension_bounds(fields)
    issues = [
        _finding("pending_facts", "coefficient domain needs " + name, f"fusion_facts#/fields/{name}")
        for name in bounds.get("missing", [])
    ]
    integer = {
        "kind": "integer_constant",
        "domain": "extension_bounds",
        "T_max": bounds.get("T_max", "UNKNOWN"),
        "coefficient_bound": bounds.get("coefficient_bound", "UNKNOWN"),
    }
    if bounds.get("T_max") == 0:
        issues.append(_finding("incompatible", "derived coefficient bound is 0", "extension_bounds#/coefficient_bound"))
    return [(None, []), (integer, issues), ({"kind": "fractional", "values": PROFILE["fractional_controls"]}, [])]


def _extension_requirements(residual):
    if residual is None:
        return []
    if residual["kind"] == "fractional":
        # Exercise the same exact-integer refusal used by production lowering.
        try:
            lower_k_extension(
                "1/2",
                source_k=1,
                residual_width=1,
                digit_bits=2,
                max_extended_k=3,
                accumulator_bits=3,
                source_bound=0,
                residual_range=[-1, 1],
            )
        except ValueError as error:
            return [_finding("incompatible", str(error), "source#lower_k_extension/integer_coefficient")]
        raise AssertionError("fractional coefficient unexpectedly lowered")
    return []  # The coefficient domain carries lower_k_extension's proof.


def _flow_requirements(flow, fields):
    available = fields["dataflows"]
    issues = _require(available, "fusion_facts#/fields/dataflows", "dataflow selection must be derived")
    labels = {"WS": {"WS", "weight_stationary"}, "OS": {"OS", "output_stationary"}}
    if not issues and not labels[flow].intersection(available["value"]):
        issues.append(_finding("incompatible", "dataflow is not selectable", "fusion_facts#/fields/dataflows"))
    if flow == "OS":
        issues.extend(
            _require(
                fields.get("mesh_signed_bits", {}),
                "fusion_facts#/fields/mesh_signed_bits",
                "constant-weight all-prefix OS guard needs the mesh signed width",
            )
        )
    return issues


def _node_requirements(node, normalized, scheme):
    fields = normalized["fields"]
    family = "movement" if node["form"] == "window_movement" else node["form"]
    placements = fields.get("producer_placement", {})
    issues = _require(placements, "fusion_facts#/fields/producer_placement", "producer placement must be derived")
    if not issues and "accelerator" not in placements["value"].get(family, []):
        issues.append(
            _finding(
                "pending_facts",
                "accelerator placement is missing for " + family,
                f"fusion_facts#/fields/producer_placement/{family}",
            )
        )
    path = normalized["output_path"]
    if path and not (path.get("accumulator") or {}).get("dtype"):
        issues.append(_finding("pending_facts", "accumulator type is missing", "output_path#/accumulator/dtype"))
    if scheme.get("status") != "resolved":
        issues.extend(
            _finding("pending_facts", "scheme field is unresolved: " + name, f"schemes#/{scheme['id']}/{name}")
            for name in scheme.get("unknown", [])
        )
    if node["seed"]:
        issues.extend(
            _require(
                fields.get("bias_preload", {}),
                "fusion_facts#/fields/bias_preload",
                "bias seed requires an exact accumulator preload",
                expected=True,
            )
        )
    if node["form"] == "operand_sum":
        issues.extend(
            _require(
                fields.get("accumulate", {}),
                "fusion_facts#/fields/accumulate",
                "two scaled loads require accumulate-on-load",
                expected=True,
            )
        )
        issues.extend(
            _require(
                fields.get("operand_sum", {}),
                "fusion_facts#/fields/operand_sum",
                "load scale, rounding and saturation contract is required",
                expected=True,
            )
        )
    if node["form"] == "window_movement":
        required_stage = "pool" if node["realization"] == "pool" else "scale"
        if path and not any(stage["kind"] == required_stage for stage in node["readout"]["stages"]):
            issues.append(
                _finding(
                    "incompatible", "window realization requires its readout " + required_stage, "output_path#/readout"
                )
            )
        key = "pool_window_within_register_fields" if node["realization"] == "pool" else "mean_realization"
        if node["realization"] == "mean" and scheme.get("readout_strategy") != "f32_product_rne":
            issues.append(
                _finding(
                    "incompatible",
                    "ones-matmul mean requires the single f32 boundary",
                    f"schemes#/{scheme['id']}/readout_strategy",
                )
            )
        issues.extend(
            _require(
                fields.get(key, {}),
                f"fusion_facts#/fields/{key}",
                "window realization and register bounds must be evidenced",
            )
        )
    issues.extend(_extension_requirements(node["residual"]))
    if node["residual"] and scheme.get("operand_sum_rounding") == "per_operand":
        issues.append(
            _finding(
                "incompatible",
                "K extension cannot erase per-operand rounding",
                f"schemes#/{scheme['id']}/operand_sum_rounding",
            )
        )
    return issues


def seed_source(value):
    """Classify the typed initializer owner, independently of SSA names/markers."""
    from xdsl.dialects import arith, builtin, tensor
    from xdsl.ir import BlockArgument

    if isinstance(value, BlockArgument):
        return "runtime_argument"
    owner = value.owner
    if isinstance(owner, arith.ConstantOp) and isinstance(
        owner.value, (builtin.DenseIntOrFPElementsAttr, builtin.DenseResourceAttr, builtin.IntegerAttr)
    ):
        return "constant"
    if isinstance(owner, tensor.SplatOp):
        return seed_source(owner.operands[0])
    return "UNKNOWN"


def _scale_source_variants(schemes, axes) -> list[dict]:
    """Keep base schemes and cross the declared runtime axis by numeric semantics."""
    variants = [copy.deepcopy(row) for row in schemes]
    if "runtime_input" in axes.get("scale_source", ()):
        for row in schemes:
            if (
                row.get("scale_source", "constant") == "constant"
                and row.get("scale_value") == "arbitrary_f32"
                and row.get("weight_scale_granularity") == "tensor"
                and row.get("activation_scale_granularity") == "tensor"
                and row.get("readout_strategy") == "f32_product_rne"
                and row.get("operand_sum_rounding") == "single_rounding"
            ):
                variants.append({**copy.deepcopy(row), "scale_source": "runtime_input"})
    return sorted(variants, key=canonical_json)


def _runtime_scale_requirements(node, normalized, spec):
    """Runtime narrow witnesses require the full arithmetic boundary and evidence.

    Raw exits carry no scale operand: a runtime label on a raw-only node is not
    a distinct semantic variant. Unsupported narrow crossings remain visible.
    """
    if node["regime"].get("scale_source") != "runtime_input":
        return []
    issues = _require(
        normalized["fields"].get("runtime_readout_scale", {}),
        "fusion_facts#/fields/runtime_readout_scale",
        "runtime scale transport requires role, layout and scalar ABI evidence",
    )
    policy = (
        ((spec.get("quantization") or {}).get("numerics", {}).get("policies") or {})
        .get("scale_source", {})
        .get("runtime_input", {})
    )
    if policy.get("value") != "runtime_input" or not policy.get("evidence") or not policy.get("basis"):
        issues.append(
            _finding(
                "pending_facts",
                "runtime scale policy is not evidenced",
                "spec#/quantization/numerics/policies/scale_source/runtime_input",
            )
        )
    stages = node["readout"]["stages"]
    arithmetic = [stage for stage in stages if stage["kind"] != "activation"]
    bits = (normalized["fields"].get("operand_bits") or {}).get("value")
    if type(bits) is not int or bits < 2:
        issues.append(
            _finding(
                "pending_facts",
                "runtime scale narrow format needs derived operand bits",
                "fusion_facts#/fields/operand_bits",
            )
        )
        return issues
    valid = (
        [s["kind"] for s in arithmetic] == ["convert", "scale", "requant"]
        and arithmetic[0].get("to") == "f32"
        and arithmetic[0].get("rounding") == "half_even"
        and arithmetic[1].get("dtype") == "f32"
        and arithmetic[1].get("rounding") == "half_even"
        and arithmetic[2].get("to") == f"i{bits}"
        and arithmetic[2].get("rounding") == "half_even"
        and arithmetic[2].get("clamp") == [-(1 << (bits - 1)), (1 << (bits - 1)) - 1]
    )
    activation = [s for s in stages if s["kind"] == "activation"]
    valid = valid and (
        not activation
        or (
            len(activation) == 1
            and stages[0] == activation[0]
            and activation[0].get("function") == "relu"
            and activation[0].get("domain") == "accumulator"
        )
    )
    if (
        not valid
        or node["readout"]["exit"] != "requant"
        or node["form"] != "contraction"
        or node["realization"] != "matmul"
        or node["seed"]
        or node["residual"]
    ):
        issues.append(
            _finding(
                "incompatible",
                "runtime v1 requires unseeded matmul and complete f32 narrow readout without residual or pool",
                "output_path#/readout",
            )
        )
    return issues


def _nodes(spec, normalized, schemes, constraints):
    for form, realization in _producer_forms(spec):
        residuals = _residual_classes(normalized["fields"]) if form == "contraction" else [(None, [])]
        seeds = (None, "runtime_argument", "constant") if form == "contraction" else (None,)
        for (route, route_issues), source, (residual, residual_issues), scheme in product(
            _routes(normalized["output_path"], normalized["fields"]), seeds, residuals, schemes
        ):
            route = {
                **route,
                "stages": [normalized["output_path"]["readout"][i] for i in route["stage_indices"]]
                if normalized["output_path"]
                else [],
            }
            # A raw exit consumes no scale, so do not manufacture a runtime operand.
            if scheme.get("scale_source") == "runtime_input" and route["exit"] == "accumulator":
                continue
            node = {
                "form": form,
                "realization": realization,
                "readout": route,
                "seed": source is not None,
                **({"seed_source": source} if source is not None else {}),
                "residual": residual,
                "scheme": scheme["id"],
                "regime": {k: v for k, v in scheme.items() if k not in {"id", "status", "unknown", "graded"}},
            }
            yield (
                node,
                (
                    route_issues
                    + residual_issues
                    + _node_requirements(node, normalized, scheme)
                    + _runtime_scale_requirements(node, normalized, spec)
                    + _scheme_axes(node, spec["quantization"]["schemes"]["axes"], scheme, constraints)
                ),
            )


def _scheme_axes(node, axes, scheme, constraints):
    exit_axis = {"accumulator": "accumulator_full", "requant": "narrow_saturating"}.get(node["readout"]["exit"])
    residual_axis = "integer_constant" if node["residual"] else "none"
    issues = []
    for axis, value in (("readout_exit", exit_axis), ("residual_coefficient", residual_axis)):
        if value is not None and (value not in axes[axis] or (axis in scheme and scheme[axis] != value)):
            issues.append(
                _finding(
                    "incompatible",
                    "regime is outside the declared scheme axis",
                    f"spec#/quantization/schemes/axes/{axis}",
                )
            )
    from merlin.targetgen.quantization_spec import unresolved_values

    regime = {**scheme, "readout_exit": exit_axis, "residual_coefficient": residual_axis}
    for index, constraint in enumerate(constraints):
        if all(regime.get(axis) == value for axis, value in constraint["when"].items()):
            for unknown in unresolved_values(constraint):
                issues.append(
                    _finding(
                        "pending_facts",
                        "crossed scheme constraint is unresolved: " + unknown,
                        f"scheme_resolution#/constraints/{index}/{unknown}",
                    )
                )
    return issues


def _generator_inventory():
    from merlin.common import jsonio, quant_formats
    from merlin.compile import composition_fusion
    from merlin.targetgen import (
        capability_manifests,
        composition_facts,
        output_path,
        quant_methodology,
        quantization_spec,
        required_regimes,
        semantic_families,
        software_spec,
        spec_fact_drift,
    )
    from merlin.xdsl_dialects.lowering import k_extension, output_fusion, runtime_readout

    modules = [
        composition_fusion,
        capability_manifests,
        composition_facts,
        output_path,
        quantization_spec,
        quant_methodology,
        semantic_families,
        software_spec,
        spec_fact_drift,
        jsonio,
        quant_formats,
        k_extension,
        output_fusion,
        runtime_readout,
        required_regimes,
    ]
    from merlin.targetgen.contract.schemas import contract_dir

    return {
        "enumerator": sha256_file(Path(__file__)),
        "fusion_candidates.schema.json": sha256_file(contract_dir() / "schemas" / "fusion_candidates.schema.json"),
        "quantization_schemes.schema.json": sha256_file(
            contract_dir() / "schemas" / "quantization_schemes.schema.json"
        ),
        **{module.__name__: sha256_file(inspect.getfile(module)) for module in modules},
    }


def _state(issues):
    states = {issue["state"] for issue in issues}
    return next(
        (state for state in ("out_of_profile", "incompatible", "pending_facts") if state in states), "candidate"
    )


def _shared_requirements(nodes, fields):
    issues = _require(
        fields.get("shared_input_resident", {}),
        "fusion_facts#/fields/shared_input_resident",
        "the complete aligned live allocation must fit the derived operand capacity",
    )
    if "shared_input" in fields:
        issues.extend(
            _require(
                fields["shared_input"],
                "fusion_facts#/fields/shared_input",
                "explicit sharing control must hold",
                expected=True,
            )
        )
    # Weight and readout scales may differ. Shared codes, zero point and layout
    # need their own equality proof; equal original floating tensors do not suffice.
    representations = fields.get("shared_input_representations", {})
    issues.extend(
        _require(
            representations,
            "fusion_facts#/fields/shared_input_representations",
            "shared input quantization representations must be explicit",
        )
    )
    if representations.get("status") != "derived" or not representations.get("evidence"):
        return issues
    values = representations.get("value") or {}
    inputs = [values.get(node["scheme"]) for node in nodes]
    if any(
        not isinstance(item, dict)
        or not all(key in item for key in ("dtype", "zero_point", "layout", "code_scale", "immutable"))
        for item in inputs
    ):
        issues.append(
            _finding(
                "pending_facts",
                "shared input representation is incomplete",
                "fusion_facts#/fields/shared_input_representations",
            )
        )
    elif inputs[0] != inputs[1] or inputs[0]["immutable"] is not True:
        issues.append(
            _finding(
                "incompatible",
                "shared input codes, zero point, layout or mutability differ",
                "fusion_facts#/fields/shared_input_representations",
            )
        )
    return issues


def _input_representations(fields, schemes):
    """Bind the shared typed leaf; output weights/scales do not alter its codes.

    Input scale domains must agree. The graph shares one immutable integer
    operand, including its symbolic zero point, layout and activation scale.
    """
    bits = fields["operand_bits"].get("value")
    if type(bits) is not int:
        return
    fields.setdefault(
        "shared_input_representations",
        _record(
            {
                scheme["id"]: dict(
                    dtype=f"i{bits}",
                    zero_point="shared_input_0.zero_point",
                    layout="shared_input_0.layout",
                    immutable=True,
                    code_scale={k: scheme.get(k, "UNKNOWN") for k in ("scale_value", "activation_scale_granularity")},
                )
                for scheme in schemes
            },
            "spec#/quantization/schemes",
        ),
    )


def _shared_classes(siblings, fields):
    """Unordered producer multisets with representative readout relations.

    Do not square every readout, seed, scheme and coefficient instance. Group
    output routes by same/different path and exit; output scale values are E2.
    """
    groups = {}
    for node, issues in siblings:
        core = {k: v for k, v in node.items() if k not in {"readout", "scheme", "regime"}}
        groups.setdefault(canonical_json(core), []).append((node, issues))
    for left, right in combinations_with_replacement(sorted(groups), 2):
        representatives = {}
        for a, b in product(groups[left], groups[right]):
            ra, rb = a[0]["readout"], b[0]["readout"]
            values = fields.get("shared_input_representations", {}).get("value") or {}
            if a[0]["scheme"] != b[0]["scheme"]:
                left_input, right_input = values.get(a[0]["scheme"]), values.get(b[0]["scheme"])
                if left_input is None or left_input != right_input:
                    continue
            relation = (
                tuple(sorted((a[0]["scheme"], b[0]["scheme"]))),
                # Effective regimes are separate even when their base scheme ID agrees.
                tuple(sorted(canonical_json(n["regime"]) for n in (a[0], b[0])))
                if any(n["regime"].get("scale_source") == "runtime_input" for n in (a[0], b[0]))
                else (),
                tuple(sorted(canonical_json(n["readout"]["stages"]) for n in (a[0], b[0])))
                if any(n["regime"].get("scale_source") == "runtime_input" for n in (a[0], b[0]))
                else (),
                ra["exit"] == rb["exit"],
                ra["stage_indices"] == rb["stage_indices"],
                ra["exit"] if ra["exit"] == rb["exit"] else "mixed",
                tuple(sorted((ra.get("pool_padding_class", "none"), rb.get("pool_padding_class", "none")))),
            )
            nodes = sorted([a[0], b[0]], key=canonical_json)
            key = (sum(issue["state"] == "pending_facts" for issue in a[1] + b[1]), canonical_json(nodes))
            if relation not in representatives or key < representatives[relation][0]:
                representatives[relation] = (key, nodes, a[1] + b[1])
        for _, nodes, issues in representatives.values():
            shared_issues = _shared_requirements(nodes, fields)
            yield nodes, issues + shared_issues


def enumerate_candidates(*, target, facts, contract, spec):
    """Return the complete minimal-profile denominator, including prune witnesses.

    Unordered independent siblings are canonicalized together with their schemes,
    seeds and readouts. Serial edges have one skeleton witness per ordered form
    pair; its scope covers all parameter/readout extensions outside the profile.
    """
    normalized = normalize_fusion_facts(target=target, facts=facts, contract=contract, spec=spec)
    context = {"output_path": normalized["output_path"]}
    for name, record in normalized["fields"].items():
        if record.get("status") == "derived" and record.get("evidence"):
            context[name] = record.get("value")
    bits = context.get("operand_bits")
    if type(bits) is int:
        context["dtype"] = f"i{bits}"
    context["max_k_extension"] = (context.get("k_extension_within") or {}).get("max_k", "UNKNOWN")
    from merlin.targetgen.quantization_spec import resolve_quantization_schemes

    resolution, _ = resolve_quantization_schemes(_spec_input(spec), context)
    schemes = sorted(resolution["schemes"], key=lambda row: row["id"])
    _input_representations(normalized["fields"], schemes)
    schemes = _scale_source_variants(schemes, resolution["axes"])
    rows, siblings = {}, []
    for node, issues in _nodes(spec, normalized, schemes, resolution["constraints"]):
        _add_rows(rows, "single", [node], issues, normalized)
        if node["form"] == "contraction" and _state(issues) != "incompatible":
            siblings.append((node, issues))
    for nodes, issues in _shared_classes(siblings, normalized["fields"]):
        _add_rows(rows, "shared_input", nodes, issues, normalized)
    for left, right in product(_producer_forms(spec), repeat=2):
        semantic = {"profile": PROFILE["schema"], "structure": "serial", "producer_forms": [list(left), list(right)]}
        _insert(
            rows,
            semantic,
            [
                _finding(
                    "out_of_profile", "serial producer edges are outside the minimal profile", "profile#/serial_edges"
                )
            ],
            "all_extensions",
        )
    return _manifest(target, facts, contract, spec, normalized, resolution, rows)


def _add_rows(rows, structure, nodes, issues, normalized):
    for flow in dataflow_variants(nodes):
        semantic = {
            "profile": PROFILE["schema"],
            "structure": structure,
            "nodes": nodes,
            "dataflow": flow,
            "shared_leaf": "immutable_input_0" if structure == "shared_input" else None,
            "guards": (["constant_weights", "all_prefix_sums_fit_mesh_signed_bits"] if flow == "OS" else [])
            + (["complete_aligned_live_allocation_fits_shared_input_resident"] if structure == "shared_input" else [])
            + (["per_case_mean_division_matches_product"] if any(n["realization"] == "mean" for n in nodes) else [])
            + (["per_case_k_and_all_prefix_extension_proof"] if any(n["residual"] for n in nodes) else []),
        }
        _insert(
            rows,
            semantic,
            issues + (_flow_requirements(flow, normalized["fields"]) if flow != DATAFLOW_NOT_APPLICABLE else []),
            "node_extensions" if structure == "single" else "row",
        )


def _insert(rows, semantic, issues, scope):
    identity = canonical_sha256(semantic)
    reasons = sorted({canonical_json(issue): issue for issue in issues}.values(), key=canonical_json)
    rows[identity] = {
        "candidate_id": identity,
        "semantic": semantic,
        "state": _state(reasons),
        "negative_control": any(
            (node["residual"] or {}).get("kind") == "fractional" for node in semantic.get("nodes", [])
        ),
        "reasons": reasons,
        "pruning_scope": scope if _state(reasons) in {"incompatible", "out_of_profile"} else "row",
    }
    if len(rows) > PROFILE["max_rows"]:
        raise ValueError("fusion candidate safety limit exceeded; enumeration was not truncated")


def _manifest(target, facts, contract, spec, normalized, resolution, rows):
    return {
        "schema": "merlin.fusion_candidates.v1",
        "target": target,
        "profile": copy.deepcopy(PROFILE),
        "identity": {
            "facts_sha256": canonical_sha256(_unordered_inputs(facts)),
            "spec_sha256": canonical_sha256(_spec_input(spec)),
            "contract_sha256": canonical_sha256(_unordered_inputs(contract)),
            "generator_sha256": canonical_sha256(_generator_inventory()),
            "scheme_resolution_sha256": canonical_sha256(_unordered_inputs(resolution)),
            "profile_sha256": canonical_sha256(PROFILE),
        },
        "normalized_facts": normalized,
        "scheme_resolution": _unordered_inputs(resolution),
        "extension_bounds": _extension_bounds(normalized["fields"]),
        "summary": _summary(rows.values()),
        "enumeration_complete": True,
        "candidate_count": len(rows),
        "rows": [rows[key] for key in sorted(rows)],
    }


def _summary(rows):
    from collections import Counter

    rows = list(rows)

    def counts(values):
        return dict(sorted(Counter(values).items()))

    return {
        "states": counts(r["state"] for r in rows),
        "forms": counts(
            r["semantic"]["structure"] if r["semantic"]["structure"] != "single" else r["semantic"]["nodes"][0]["form"]
            for r in rows
        ),
        "schemes": counts(
            " + ".join(sorted(n["scheme"] for n in r["semantic"].get("nodes", []))) or "none" for r in rows
        ),
        "pending_facts_by_missing_fact": counts(
            fact
            for r in rows
            if r["state"] == "pending_facts"
            for fact in {reason["missing_fact"] for reason in r["reasons"] if reason["state"] == "pending_facts"}
        ),
    }


def campaign_stratum(semantic):
    """Practice axes from typed graph semantics, without coefficient magnitudes."""
    from merlin.targetgen.composition_corpus import fusion_structure_class

    nodes = semantic.get("nodes", [])
    structure = fusion_structure_class(semantic)
    pools = sorted(
        {
            (
                tuple(sorted(stage.get("reductions", []))),
                stage.get("padding", "UNKNOWN"),
                node["readout"].get("pool_padding_class", "unpadded"),
            )
            for node in nodes
            for stage in node["readout"]["stages"]
            if stage["kind"] == "pool"
        }
    )
    if structure == "shared_input" and pools:
        structure = "shared_input_pool"
    return dict(
        structure=structure,
        schemes=sorted({node["scheme"] for node in nodes}),
        exits=sorted({node["readout"]["exit"] for node in nodes}),
        pool_kind_padding=[
            dict(kinds=list(kinds), padding=padding, padding_class=padding_class)
            for kinds, padding, padding_class in pools
        ],
        seed_sources=sorted(
            {node.get("seed_source", "runtime_argument" if node.get("seed") else "none") for node in nodes}
        ),
        dataflow=semantic.get("dataflow"),
        shared_input_fan_out=len(nodes) if semantic["structure"] == "shared_input" else 0,
    )


STRATIFICATION = "merlin.fusion_campaign.required_regime_strata.v1"


def campaign_policy(spec):
    """The declared per-stratum budget and the spec's required regimes, as recorded in a campaign."""
    from merlin.targetgen.required_regimes import required_regimes

    limit = (spec.get("compositions") or {}).get("campaign_per_stratum", 2)
    if type(limit) is not int or limit < 1:
        raise ValueError("compositions.campaign_per_stratum must be a positive integer")
    regimes = [
        dict(composition=copy.deepcopy(chain), scheme=scheme, axes=copy.deepcopy(axes))
        for chain, scheme, axes in required_regimes(spec)
    ]
    return limit, regimes


def _stratify(manifest, limit, regimes):
    """Strata per required regime, then the practice key for rows witnessing no regime.

    A candidate witnessing required regimes (merlin.targetgen.required_regimes) belongs to
    one stratum per matching regime, keyed by that regime's identity, the practice axes of
    campaign_stratum and the node realizations: inside one regime, matmul and conv_as_matmul
    siblings are distinct witnesses, so neither may starve the other. Every other candidate
    keeps the plain campaign_stratum key. Each stratum admits its first `limit` members in
    candidate_id order, and a candidate executes when any of its strata admits it. Thus every
    stratum, hence every regime with e eligible candidates, executes at least min(limit, e).
    """
    from merlin.targetgen.required_regimes import regime_identity, represents_required_regime

    result = copy.deepcopy(manifest)
    counts = {}
    coverage = [
        dict(regime=regime_identity(r["composition"], r["scheme"], r["axes"]), eligible=0, executed=0, strata=set())
        for r in regimes
    ]
    for row in sorted(result["rows"], key=lambda row: row["candidate_id"]):
        if row["state"] != "candidate":
            continue
        practice = campaign_stratum(row["semantic"])
        matches = [
            index
            for index, r in enumerate(regimes)
            if represents_required_regime(row["semantic"], r["composition"], r["scheme"], r["axes"])
        ]
        if matches:
            practice["realizations"] = sorted({node.get("realization", "none") for node in row["semantic"]["nodes"]})
            row["stratum_key"] = dict(required_regimes=[coverage[index]["regime"] for index in matches], **practice)
        else:
            row["stratum_key"] = practice
        strata = {index: canonical_json(dict(regime=coverage[index]["regime"], **practice)) for index in matches}
        admitted = False
        for key in list(strata.values()) or [canonical_json(practice)]:
            counts[key] = counts.get(key, 0) + 1
            admitted = admitted or counts[key] <= limit
        if not admitted:
            row["state"] = "deferred_by_stratification"
        for index, key in strata.items():
            coverage[index]["eligible"] += 1
            coverage[index]["executed"] += admitted
            coverage[index]["strata"].add(key)
    for entry in coverage:
        entry["strata"] = len(entry["strata"])
        if entry["executed"] < min(limit, entry["eligible"]):
            raise ValueError("required-regime stratification lost its per-regime guarantee")
    result["campaign"] = dict(
        per_stratum=limit,
        order="candidate_id",
        stratum_count=len(counts),
        stratification=STRATIFICATION,
        required_regimes=regimes,
        regime_coverage=coverage,
    )
    return result


def stratify_candidates(manifest, spec):
    """Select a bounded campaign while retaining every enumeration row and ID."""
    return _stratify(manifest, *campaign_policy(spec))


def verify_stratification(manifest, spec=None, *, required=False):
    """Reconstruct campaign states from the retained enumeration denominator.

    With a spec, the budget and required regimes are re-derived from it; without one,
    from the recorded policy. Either way the executed set must be reproduced exactly.
    """
    if "campaign" not in manifest:
        if required or spec is not None:
            raise ValueError("fusion campaign is required for the selected spec")
        return
    original = copy.deepcopy(manifest)
    policy = original.pop("campaign")
    for row in original["rows"]:
        row.pop("stratum_key", None)
        if row["state"] == "deferred_by_stratification":
            row["state"] = "candidate"
    if spec is not None:
        expected = stratify_candidates(original, spec)
    elif policy.get("stratification") != STRATIFICATION:
        raise ValueError("fusion campaign stratification definition differs from this reader")
    else:
        expected = _stratify(original, policy["per_stratum"], policy["required_regimes"])
    if expected != manifest:
        raise ValueError("fusion campaign differs from deterministic stratification")


def write_manifest(manifest, *, destination=None):
    """Write canonical bytes under the repository's artifact layout by default."""
    from merlin.common.paths import artifacts_dir

    path = (
        Path(destination)
        if destination
        else artifacts_dir() / "verification" / manifest["target"] / "fusion-candidates.json"
    )
    write_canonical_json(path, manifest)
    return path


def main(argv=None):
    """Freeze selected derivations, then write the pure candidate manifest."""
    import argparse
    import json

    import yaml

    from merlin.targetgen.capability_manifests import _derived_output_epilogue
    from merlin.targetgen.rtl.facts import load_facts
    from merlin.targetgen.target_registry import load_contract

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--facts", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    facts = load_facts(args.target, explicit=args.facts, regenerate=False)
    contract = yaml.safe_load(args.contract.read_text()) if args.contract else load_contract(args.target)
    contract = copy.deepcopy(contract)
    if not (contract.get("output_epilogue") or {}).get("output_path"):
        contract["output_epilogue"] = _derived_output_epilogue(args.target, contract, facts)
    from merlin.targetgen.software_spec import load_software_spec
    from merlin.targetgen.spec_fact_drift import _resolve_chain, fact_capabilities, resolve_spec

    declared_spec = load_software_spec(args.spec, target=args.target)
    spec = _spec_input(declared_spec)
    capabilities = fact_capabilities(target=args.target, contract=contract, raw_facts=facts, data_only=True)
    spec, _ = resolve_spec(spec, capabilities)
    producers = sorted({"movement" if form == "window_movement" else form for form, _ in _producer_forms(spec)})
    placement = _resolve_chain(
        {"producer": producers, "constraints": []},
        capabilities,
        {"output_path": contract["output_epilogue"].get("output_path")},
    )
    spec["composition_resolution"] = {"chains": [placement]}
    manifest = enumerate_candidates(target=args.target, facts=facts, contract=contract, spec=spec)
    manifest = stratify_candidates(manifest, declared_spec)
    path = write_manifest(manifest, destination=args.out)
    print(
        json.dumps(
            {
                "manifest": str(path),
                "candidate_count": manifest["candidate_count"],
                **manifest["summary"],
                "extension_bounds": {k: manifest["extension_bounds"].get(k) for k in ("T_max", "coefficient_bound")},
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
