"""Reference-provider compilation, exact Spike readback and retired fusion checks."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from .fusion_cases import canonical, chain_for_node, digest, file_digest


def source_members(source):
    from merlin.common import mlir_query as mq

    module = mq.parse(source)
    function = next(op for op in module.body.block.ops if op.name == "func.func")
    operations = [op for op in function.body.block.ops if op.name != "func.return"]
    all_indices = {op: index for index, op in enumerate(module.walk())}
    stages = {
        op.attributes["prov.trace_role"].data: [index]
        for index, op in enumerate(operations)
        if "prov.trace_role" in op.attributes
    }
    return module, operations, stages, all_indices


def prove_placement(source, proof):
    """Numerical admission of a producer alone does not admit its host epilogue."""
    _, operations, stages, indices = source_members(source)
    members = {index for values in stages.values() for index in values}
    if not members:
        return None, "pending_audit: source has no bound composition members"
    bindings = proof.get("bindings", [])
    owners = []
    for binding in bindings:
        covered = set(
            binding.get(
                "replaced_ordinals", binding.get("source_operation_ordinals", [binding.get("source_operation_ordinal")])
            )
        )
        matched = {index for index in members if indices[operations[index]] in covered}
        if matched:
            owners.append((binding, matched))
    if not owners or set.union(*(members for _, members in owners)) != members:
        return None, "host_fallback: admission does not cover every source fusion member"
    if len(owners) != 1 or owners[0][1] != members:
        return None, "split_kernels: fusion members do not occupy one admitted kernel"
    binding = owners[0][0]
    return dict(
        symbol=binding["symbol"],
        source_op_indices=sorted(members),
        stages=stages,
        program_sha256=__import__("hashlib").sha256(source.encode()).hexdigest(),
    ), None


def _selected_policy(spec, candidate):
    from merlin.targetgen.core_aten_spec_int8 import case_readout

    policy = copy.deepcopy(spec)
    nodes = candidate["semantic"]["nodes"]
    strategies = {node["regime"]["readout_strategy"] for node in nodes}
    if len(strategies) != 1:
        raise ValueError("backend_unsupported: mixed readout strategies have no provider policy")
    for node in nodes:
        scheme = {**node["regime"], "id": node["scheme"], "status": "resolved", "unknown": []}
        policy = case_readout(dict(spec=policy), scheme)["spec"]
    # A derived compatibility view selects semantic routes, never historical IDs.
    chains = [chain_for_node(node, candidate["semantic"]["dataflow"]) for node in nodes]
    if candidate["semantic"]["structure"] == "shared_input":
        chains[0]["multiplicity"] = dict(min=len(nodes))
        chains[0]["exits"] = sorted({node["readout"]["exit"] for node in nodes})
        chains[0]["constraints"].extend(["per_output_readout_independent", dict(shared_input_resident="from_facts")])
    policy["compositions"] = dict(
        schema="merlin.software_spec.compositions.v1",
        status="reviewed",
        grading="placement_and_instruction_audit",
        chains=chains,
    )
    return policy


def readout_options(selection, policy, facts, candidate):
    strategy = policy["numerical_semantics"]["readout"]["strategy"]
    return dict(
        path=selection["candidates"]["normalized_facts"]["output_path"],
        scheme={**candidate["semantic"]["nodes"][0]["regime"], "readout_strategy": strategy},
        facts=facts,
        compositions=policy["compositions"],
        numerical_semantics=policy["numerical_semantics"],
    )


def _write_bundle(payload, directory):
    from merlin.targetgen.core_aten_batch import _write_npz

    (directory / "model.mlir").write_text(payload["source"])
    arrays, manifest = [], {}
    for index, spec in enumerate(payload["inputs"]):
        # Each f32 JSON value is the exact double of a binary32, so this cast
        # restores its bits, including negative zero and subnormals.
        storage = {"i8": np.int8, "i32": np.int32, "f32": np.float32}[spec["dtype"]]
        arrays.append(np.asarray(payload["input_values"][spec["name"]], dtype=storage))
        manifest[str(index)] = dict(kind="input", name=f"arg{index}", shape=spec["shape"], dtype=spec["dtype"])
    _write_npz(directory / "inputs.npz", arrays)
    (directory / "weights.safetensors.manifest.json").write_bytes(canonical(manifest))
    (directory / "input_order.json").write_bytes(canonical({f"arg{i}": i for i in range(len(arrays))}))


# Lint runtime_oracle_obligations. Spike's narrow ReLU order is a property of
# the selected extension bytes, so parity is an explicit reviewed pin.
RELU_SCALE_ORDER_OBLIGATION = "relu_scale_order_parity_or_nonnegative_test_input"
RELU_ORDER_FILE = "spike-extension-relu-order.yaml"
RELU_ORDER_SCHEMA = "merlin.spike_extension_relu_order.v1"
_RELU_ORDERS = frozenset({"pre_scale", "post_scale"})


class OracleObligationRefused(ValueError):
    """An execution case whose oracle obligations are not discharged earns no admission."""


def relu_order_registry(target, path=None):
    """Read the target's reviewed extension ReLU-order identities, refusing a missing or malformed file.

    Returns {extension sha256: {artifact, relu_scale_order, evidence}}. Each
    identity must be the digest its hardware_pins artifact declares.
    """
    import yaml

    from merlin.common.provenance import load_artifacts
    from merlin.targetgen.target_registry import reference_targets

    def refuse(reason):
        return OracleObligationRefused(f"oracle_obligation_unchecked: relu-order registry {reason}")

    if path is None:
        base = reference_targets().get(target)
        if base is None:
            raise refuse(f"has no reviewed target directory for {target!r}")
        path = Path(base) / RELU_ORDER_FILE
    path = Path(path)
    if not path.is_file():
        raise refuse(f"{path} is missing")
    try:
        document = yaml.safe_load(path.read_bytes())
    except yaml.YAMLError as error:
        raise refuse(f"{path} is not YAML") from error
    if (
        not isinstance(document, dict)
        or set(document) != {"schema", "target", "extensions"}
        or document["schema"] != RELU_ORDER_SCHEMA
        or document["target"] != target
        or not isinstance(document["extensions"], dict)
    ):
        raise refuse(f"{path} is malformed")
    artifacts = load_artifacts()
    registry = {}
    for identity, entry in document["extensions"].items():
        if (
            not isinstance(identity, str)
            or len(identity) != 64
            or not set(identity) <= set("0123456789abcdef")
            or not isinstance(entry, dict)
            or set(entry) != {"artifact", "relu_scale_order", "evidence"}
            or entry["relu_scale_order"] not in _RELU_ORDERS
            or not isinstance(entry["evidence"], str)
            or not entry["evidence"].strip()
        ):
            raise refuse(f"{path} has a malformed extension identity")
        pinned = artifacts.get(entry["artifact"]) if isinstance(entry["artifact"], str) else None
        if pinned is None or pinned.role != "spike_extension" or pinned.target != target or pinned.digest != identity:
            raise refuse(f"{path} names {identity[:12]}, which is not a pinned {target} spike_extension digest")
        registry[identity] = dict(entry)
    return registry


def _proof_member(proof, symbol, member_index, members):
    """The admission record that binds one source output's readout, or {} when none does."""
    kernels = [k for k in proof.get("kernels", ()) if k.get("symbol") == symbol]
    if len(kernels) > 1:
        raise ValueError("pending_audit: admission has ambiguous kernel proofs")
    if not kernels:
        return {}
    shared = kernels[0].get("shared_input")
    if isinstance(shared, dict):
        rows = shared.get("members", [])
        return rows[member_index] if member_index < len(rows) and isinstance(rows[member_index], dict) else {}
    schedule = kernels[0].get("schedule")
    return schedule if members == 1 and isinstance(schedule, dict) else {}


def _source_scale_input(payload, member_index):
    """Payload input index of one output's runtime readout scale, decided by the SOURCE.

    None means the source readout scale is constant. The admission proof never
    selects this: a runtime source whose proof binds no scale operand refuses.
    """
    from .runtime_scale_reference import needs_runtime_scale

    cap = payload["nodes"][member_index]
    if not needs_runtime_scale(cap):
        return None
    name = cap["operation"]["attributes"]["case"].get("runtime_scale_input")
    matches = [i for i, spec in enumerate(payload["inputs"]) if spec["name"] == f"node{member_index}_{name}"]
    if not isinstance(name, str) or len(matches) != 1:
        raise ValueError("pending_audit: runtime readout source has no unique scale input")
    return matches[0]


def _proved_scale_operand(payload, proof, symbol, member_index):
    """Return (source scale input or None, proof member), refusing any source/proof disagreement."""
    source = _source_scale_input(payload, member_index)
    member = _proof_member(proof, symbol, member_index, len(payload["nodes"]))
    operand = member.get("scale_operand")
    if source is None:
        if operand is not None:
            raise ValueError("pending_audit: admission binds a runtime readout scale the source does not declare")
        return None, member
    bound = operand.get("source") if isinstance(operand, dict) else None
    if not isinstance(operand, dict) or operand.get("kind") != "runtime_input" or not isinstance(bound, dict):
        raise ValueError("pending_audit: runtime source readout has no proved scale_operand")
    if bound.get("func_arg") != source:
        raise ValueError("pending_audit: proved scale_operand differs from the source scale input")
    return source, member


def _bound_argument(binding, args, source):
    names = [name for name, row in binding.get("operands", {}).items() if row.get("func_arg") == source]
    indices = [i for i, arg in enumerate(args) if arg["name"] in names]
    return indices[0] if len(names) == 1 and len(indices) == 1 else None


def _relu_runtime_refs(payload, proof, symbol, args, binding):
    """List the kernel scale arguments of ReLU runtime readouts, derived from the source."""
    refs = set()
    try:
        for member_index, cap in enumerate(payload.get("nodes", ())):
            source, member = _proved_scale_operand(payload, proof, symbol, member_index)
            if source is None:
                continue
            relu = "activation" in cap["operation"]["attributes"]["case"]["readout_stages"]
            if member.get("relu", relu) is not relu:
                raise ValueError("runtime readout activation differs between source and admission")
            if relu:
                index = _bound_argument(binding, args, source)
                if index is None:
                    raise ValueError("ReLU runtime readout scale is not one bound kernel argument")
                refs.add(index)
    except ValueError as error:
        raise OracleObligationRefused(f"oracle_obligation_unchecked: {error}") from error
    return refs


def discharge_oracle_obligations(payload, proof, catalog, symbol, extension_sha256, *, relu_order):
    """Discharge every lint runtime_oracle_obligation against this execution.

    The obligation holds for an ACTIVE ReLU over a runtime scale. RTL applies
    ReLU before the scale; an oracle that applies it after agrees only when the
    scale is nonnegative. It is discharged either by the selected extension
    being a pre_scale identity in ``relu_order`` (relu_order_registry), or by
    the executed input: a finite f32 >= 0. Negative zero is accepted: every
    product with -0 is a signed zero, which rounds to integer 0 under either
    order (RNE to integer, then clamp), so the stored i8 bytes cannot differ.
    Anything else refuses.
    Required obligations are re-derived from the source readouts, which the
    admission proof must bind, so a catalog that drops one also refuses.
    """
    kernels = [k for k in catalog.get("kernels", ()) if k.get("symbol") == symbol]
    bindings = [b for b in catalog.get("bindings", ()) if b.get("symbol") == symbol]
    if len(kernels) != 1 or len(bindings) != 1:
        raise OracleObligationRefused("oracle_obligation_unchecked: compiled catalog has no unique kernel binding")
    kernel, binding = kernels[0], bindings[0]
    required = _relu_runtime_refs(payload, proof, symbol, kernel["args"], binding)
    obligations = kernel.get("runtime_oracle_obligations")
    if obligations is None:
        obligations = []
    if not isinstance(obligations, list):
        raise OracleObligationRefused("oracle_obligation_unchecked: runtime_oracle_obligations is malformed")
    args, rows, declared = kernel["args"], [], set()
    for obligation in obligations:
        if (
            not isinstance(obligation, dict)
            or set(obligation) != {"instruction_index", "scale_operand", "requires"}
            or type(obligation["instruction_index"]) is not int
            or not isinstance(obligation["scale_operand"], list)
            or len(obligation["scale_operand"]) != 2
            or type(obligation["scale_operand"][0]) is not int
            or obligation["scale_operand"][1] != 0
        ):
            raise OracleObligationRefused("oracle_obligation_unchecked: malformed runtime oracle obligation")
        if obligation["requires"] != RELU_SCALE_ORDER_OBLIGATION:
            raise OracleObligationRefused(
                f"oracle_obligation_unchecked: no consumer for requirement {obligation['requires']!r}"
            )
        index = obligation["scale_operand"][0]
        if not 0 <= index < len(args):
            raise OracleObligationRefused("oracle_obligation_unchecked: obligation names no kernel argument")
        arg = args[index]
        source = binding.get("operands", {}).get(arg["name"], {}).get("func_arg")
        if (
            arg["role"] != "input"
            or arg["dtype"] != "f32"
            or arg["shape"] != [1]
            or arg["name"] not in kernel.get("runtime_scale_arguments", ())
            or type(source) is not int
            or not 0 <= source < len(payload["inputs"])
        ):
            raise OracleObligationRefused("oracle_obligation_unchecked: obligation is not a bound runtime scale input")
        spec = payload["inputs"][source]
        value = np.asarray(payload["input_values"][spec["name"]], dtype=np.float32).reshape(-1)
        if spec["dtype"] != "f32" or spec["shape"] != [1] or value.shape != (1,):
            raise OracleObligationRefused("oracle_obligation_unchecked: executed scale input is not f32[1]")
        bits = int(value.view(np.uint32)[0])
        row = dict(
            obligation,
            argument=arg["name"],
            source_input=spec["name"],
            input_bits=f"{bits:#010x}",
            extension_sha256=extension_sha256,
        )
        identity = relu_order.get(extension_sha256, {})
        if identity.get("relu_scale_order") == "pre_scale":
            row["basis"] = "parity_proved_extension"
        elif np.isfinite(value[0]) and value[0] >= 0:
            row["basis"] = "nonnegative_test_input"
            row["input_class"] = "negative_zero" if bits == 0x80000000 else "finite_nonnegative"
        else:
            known = identity.get("evidence", "no parity proof is pinned")
            raise OracleObligationRefused(
                f"oracle_obligation_undischarged: instruction {obligation['instruction_index']} ReLU scale "
                f"{arg['name']}={float(value[0])!r} ({bits:#010x}) is negative and extension "
                f"{extension_sha256[:12]} is not relu-order parity-proved ({known})"
            )
        declared.add(index)
        rows.append(row)
    if required - declared:
        raise OracleObligationRefused(
            "oracle_obligation_missing: ReLU runtime readout has no lint obligation for scale argument(s) "
            + ", ".join(str(i) for i in sorted(required - declared))
        )
    return dict(
        schema="merlin.runtime_oracle_discharge.v1",
        status="discharged",
        symbol=symbol,
        extension_sha256=extension_sha256,
        obligations=rows,
    )


def _catalog(directory):
    return json.loads((directory / "spike-build" / "region_offload" / "core_aten_catalog.json").read_bytes())


def _oracle_discharge(payload, proof, placement, directory, extension_sha256, target, relu_order_path=None):
    """Return (record, refusal reason); the record is retained either way."""
    try:
        record = discharge_oracle_obligations(
            payload,
            proof,
            _catalog(directory),
            placement["symbol"],
            extension_sha256,
            relu_order=relu_order_registry(target, relu_order_path),
        )
        reason = None
    except OracleObligationRefused as error:
        record = dict(schema="merlin.runtime_oracle_discharge.v1", status="refused", reason=str(error))
        reason = str(error)
    (directory / "oracle-obligations.json").write_bytes(canonical(record))
    return record, reason


def _qualify(outcome, discharge):
    """A passing case needs a discharged oracle-obligation record; absence fails closed."""
    if not isinstance(discharge, dict) or discharge.get("status") != "discharged":
        outcome["reason"] = "oracle_obligation_unchecked: no discharged runtime oracle obligation record"
        return outcome
    outcome["outcome"] = "pass"
    return outcome


def _runtime_abi(directory, proof, symbol):
    """Bind the compiled pointer ABI, not guessed DMA addresses, to the trace."""
    import math
    import subprocess

    from merlin.common.quant_formats import storage_bits
    from merlin.runtime.host_provider import read_host_function_abi

    root = directory / "spike-build" / "region_offload"
    catalog = json.loads((root / "core_aten_catalog.json").read_bytes())
    if catalog["bindings"] != proof["bindings"]:
        raise ValueError("pending_audit: compiled runtime bindings differ from admission")
    kernel = next(k for k in catalog["kernels"] if k["symbol"] == symbol)
    args = kernel["args"]
    abi = read_host_function_abi(root / "core_aten.ll", symbol)
    header = subprocess.check_output(
        ["readelf", "--file-header", str(directory / "spike-build" / "model.elf")], text=True
    )
    if (
        "RISC-V" not in header
        or "ELF64" not in header
        or abi.result != "void"
        or abi.variadic
        or abi.arguments != ("ptr",) * len(args)
        or len(args) > 8
    ):
        raise ValueError("pending_audit: compiled kernel has no supported register pointer ABI")
    # RISC-V psABI pointer registers, as in the shared native trace reader.
    arguments = [
        dict(arg_index=i, register=register, byte_extent=math.prod(arg["shape"]) * storage_bits(arg["dtype"]) // 8)
        for i, (register, arg) in enumerate(zip(range(10, 18), args))
    ]
    return {symbol: arguments}, args


def _bind_result_slices(decoded, binding, args):
    """Project the packed output allocation into the original disjoint results."""
    import math

    from merlin.common.quant_formats import storage_bits

    if binding.get("result_buffers"):
        outputs = [a for a in args if a["role"] == "output"]
        buffers = binding["result_buffers"]
        if len(outputs) != len(buffers) or any(
            b["argument"] != a["name"]
            or b["shape"] != a["shape"]
            or b["byte_offset"] != 0
            or b["byte_length"] != math.prod(a["shape"]) * storage_bits(a["dtype"]) // 8
            for b, a in zip(buffers, outputs, strict=True)
        ):
            raise ValueError("pending_audit: typed result buffers differ from the compiled ABI")
        return
    slices = binding.get("result_slices")
    if not slices:
        return
    outputs = [(i, a) for i, a in enumerate(args) if a["role"] == "output"]
    if len(outputs) != 1:
        raise ValueError("pending_audit: packed result has no unique output allocation")
    index, arg = outputs[0]
    element = storage_bits(arg["dtype"]) // 8
    cursor = 0
    for part in slices:
        if part["offset"] != cursor:
            raise ValueError("pending_audit: result slices do not partition the output allocation")
        cursor += math.prod(part["shape"])
    if cursor != math.prod(arg["shape"]):
        raise ValueError("pending_audit: incomplete packed result partition")
    for row in decoded["instructions"]:
        buffers = row.get("runtime_buffer_bindings")
        if buffers is None:
            continue
        output = next(b for b in buffers if b["arg_index"] == index)
        row["runtime_buffer_bindings"] = [b for b in buffers if b["arg_index"] != index] + [
            dict(
                arg_index=len(args) + i,
                base=output["base"] + part["offset"] * element,
                byte_extent=math.prod(part["shape"]) * element,
            )
            for i, part in enumerate(slices)
        ]


def _operand_sum_proof(proof, symbol):
    kernel = next(k for k in proof["kernels"] if k["symbol"] == symbol)
    declared = kernel["operand_sum"]
    if kernel["status"] != "proven" or declared["status"] != "proven":
        raise ValueError("pending_audit: operand-sum register choices have no numeric proof")
    return declared


def _screen_per_operand_case(cap, declared, facet):
    """Recheck source parameters against provider register choices, independently."""
    from merlin.targetgen.operand_sum_numeric import audit_i8_per_operand_sum

    attrs = cap["operation"]["attributes"]
    case, scheme = attrs["case"], attrs["quantization_scheme"]
    if (
        case["exit"] != "requant"
        or set(case["readout_stages"]) - {"activation", "convert", "scale", "requant"}
        or case.get("accumulator_stages")
        or case.get("coefficient") is not None
        or declared.get("operand_sum_rounding") != "per_operand"
    ):
        raise ValueError("pending_audit: per-operand contract does not cover this source tail")
    loads, gains = declared["load_scale_bits"], declared["multiplier_bits"]
    if len(loads) != 2 or len(gains) != attrs["N"]:
        raise ValueError("pending_audit: per-operand register choices do not match source columns")

    def value(bits):
        if type(bits) is not int or not 0 <= bits < 2**32:
            raise ValueError("pending_audit: invalid f32 register bits")
        return float(np.array(bits, dtype=np.uint32).view(np.float32))

    width = scheme["columns_per_scale"] if scheme["weight_scale_granularity"] == "column_block" else attrs["N"]
    screens, seen = [], set()
    for column, gain in enumerate(gains):
        source_gain = case["multipliers"][column // width]
        if (source_gain, gain) in seen:
            continue
        seen.add((source_gain, gain))
        result = audit_i8_per_operand_sum(
            lhs_scale=case["operand_multipliers"][0],
            rhs_scale=case["operand_multipliers"][1],
            readout_scale=source_gain,
            relu="activation" in case["readout_stages"],
            facet=facet,
            device=dict(
                lhs_scale=value(loads[0]), rhs_scale=value(loads[1]), readout_scale=value(gain), relu=declared["relu"]
            ),
        )
        if result["status"] != "within_bound":
            raise ValueError(f"pending_audit: per-operand exhaustive screen failed: {result}")
        screens.append(dict(result, column=column))
    return screens


def _per_operand_numeric_screens(payload, proof, symbol, facts, target):
    from merlin.targetgen.readout_facet import for_target
    from merlin.targetgen.target_registry import load_contract

    members = [
        cap
        for cap in payload["nodes"]
        if cap["operation"]["attributes"]["case"]["producer"] == "operand_sum"
        and cap["operation"]["attributes"]["quantization_scheme"]["operand_sum_rounding"] == "per_operand"
    ]
    if not members:
        return []
    facets = [facet.to_dict() for facet in for_target(target, facts=facts["raw"], contract=load_contract(target))]
    selected = [facet for facet in facets if (facet.get("operand_sum") or {}).get("operand_dtype") in {"i8", "int8"}]
    if len(selected) != 1:
        raise ValueError("pending_audit: per-operand screen has no unique selected load/readout model")
    declared = _operand_sum_proof(proof, symbol)
    return [screen for cap in members for screen in _screen_per_operand_case(cap, declared, selected[0])]


def _runtime_scale_member(payload, proof, symbol, args, member_index):
    """Return (kernel argument, frozen input bits) for a runtime-scale output, else None.

    The SOURCE decides whether the output has a runtime scale; the proof must
    then bind that scale input (func_arg) through the compiled operand binding.
    The bits come from this execution's frozen input bytes, never case metadata.
    """
    source, member = _proved_scale_operand(payload, proof, symbol, member_index)
    if source is None:
        return None
    input_spec = payload["inputs"][source]
    value = np.asarray(payload["input_values"][input_spec["name"]], dtype=np.float32)
    if input_spec["dtype"] != "f32" or input_spec["shape"] != [1] or value.shape != (1,) or not np.isfinite(value[0]):
        raise ValueError("pending_audit: runtime readout has no finite f32[1] input")
    binding = next(b for b in proof["bindings"] if b["symbol"] == symbol)
    index = _bound_argument(binding, args, source)
    if (
        index is None
        or args[index]["role"] != "input"
        or args[index]["dtype"] != "f32"
        or args[index]["shape"] != [1]
        or member.get("scale_ref", [index, 0]) != [index, 0]
    ):
        raise ValueError("pending_audit: runtime scale source is not one bound f32[1] kernel argument")
    return index, int(value.view(np.uint32)[0])


def _runtime_scale_loads(payload, proof, symbol, args, elf, trace, runtime_abi, facts, outputs):
    """Audit the retired load-to-CONFIG_ST chain for every runtime-scale sibling."""
    from merlin.common.facts_view import interface
    from merlin.targetgen.runtime_scale_trace import audit_runtime_scale_loads

    inputs = {}
    for member_index in range(outputs):
        found = _runtime_scale_member(payload, proof, symbol, args, member_index)
        if found is not None:
            inputs[found[0]] = found[1]
    if not inputs:
        return None
    if not runtime_abi:
        raise ValueError("pending_audit: runtime readout has no compiled pointer ABI")
    raw = facts["raw"]
    table = interface(raw.get("facts", raw), "funct_decode_table") or {}
    slot = facts.get("runtime_readout_scale")
    if not isinstance(slot, dict):
        raise ValueError("pending_audit: runtime readout scale slot is underived")
    return audit_runtime_scale_loads(
        elf=elf,
        trace=trace,
        runtime_abi=runtime_abi,
        scale_inputs=[dict(kernel=symbol, arg_index=i, offset=0, dtype="f32", shape=[1]) for i in sorted(inputs)],
        input_bits=inputs,
        slot=slot,
        custom_opcode=table.get("custom_opcode"),
    )


def _readout_bindings(payload, proof, symbol, args, scale_loads=None):
    """Bind each source output's columns to its compiled logical allocation."""
    from merlin.targetgen.functional_truth_compositions import _audit_parameters

    binding = next(b for b in proof["bindings"] if b["symbol"] == symbol)
    outputs = [i for i, arg in enumerate(args) if arg["role"] == "output"]
    if binding.get("result_slices"):
        outputs = list(range(len(args), len(args) + len(binding["result_slices"])))
    if len(outputs) != len(payload["nodes"]) or len(outputs) != len(payload["references"]):
        raise ValueError("pending_audit: source outputs do not match compiled allocations")
    partitions = []
    for member_index, (cap, reference, output) in enumerate(
        zip(payload["nodes"], payload["references"], outputs, strict=True)
    ):
        attrs = cap["operation"]["attributes"]
        case, scheme = attrs["case"], attrs["quantization_scheme"]
        if case["exit"] == "accumulator":
            continue
        runtime = _runtime_scale_member(payload, proof, symbol, args, member_index)
        if runtime is not None:
            if case["producer"] == "operand_sum" or scale_loads is None:
                raise ValueError("pending_audit: runtime readout has no retired scale-load evidence")
            index, bits = runtime
            # A runtime partition carries scale_input instead of column_scale_bits, never both.
            partitions.append(
                dict(
                    output=dict(kind="argbase", arg_index=output, offset=0),
                    shape=reference["shape"],
                    scale_input=dict(kind="argbase", arg_index=index, offset=0, dtype="f32", shape=[1]),
                    expected_input_bits=bits,
                    scale_load_evidence_id=scale_loads["evidence"][f"{symbol}:{index}"],
                )
            )
            continue
        bits = _audit_parameters(scheme, case)["readout_multiplier_bits"]
        columns = reference["shape"][-1]
        width = scheme["columns_per_scale"] if scheme["weight_scale_granularity"] == "column_block" else columns
        gains = [bits[c // width] for c in range(columns)]
        if case["producer"] == "operand_sum":
            # The exhaustive source-domain proof may select equivalent register
            # factors. Reopening admission re-derives these choices independently.
            gains = _operand_sum_proof(proof, symbol)["multiplier_bits"]
            if len(gains) != columns:
                raise ValueError("pending_audit: proved readout columns differ from source")
        partitions.append(
            dict(
                output=dict(kind="argbase", arg_index=output, offset=0),
                shape=reference["shape"],
                column_scale_bits=gains,
            )
        )
    return dict(partitions=partitions)


def _operand_bindings(payload, proof, symbol, args):
    declared = _operand_sum_proof(proof, symbol)
    shape = payload["inputs"][0]["shape"]
    if [declared["rows"], declared["cols"]] != shape:
        raise ValueError("pending_audit: proved operand extent differs from source")
    inputs = [i for i, arg in enumerate(args) if arg["role"] == "input"]
    bits = declared["load_scale_bits"]
    if len(inputs) != 2 or len(bits) != 2:
        raise ValueError("pending_audit: operand sum needs two bound scaled loads")
    return dict(
        operands=[
            dict(
                input=dict(kind="argbase", arg_index=index, offset=0),
                shape=shape,
                scale_bits=scale,
                accumulate=position != 0,
            )
            for position, (index, scale) in enumerate(zip(inputs, bits, strict=True))
        ]
    )


def _linked_constant_input(elf, globals_by_name):
    from merlin.targetgen.elf_lanes import _section_headers
    from merlin.targetgen.native_instruction_audit import elf_object_buffers

    blob, matches = elf.read_bytes(), []
    sections = _section_headers(blob)
    for buffer in elf_object_buffers(elf):
        data = globals_by_name.get(buffer["symbol"])
        if data is None or len(data) != buffer["byte_extent"]:
            continue
        for _, kind, flags, offset, size, address in sections:
            start = buffer["base"] - address
            if kind == 1 and flags & 2 and not flags & 1 and 0 <= start and start + len(data) <= size:
                if blob[offset + start : offset + start + len(data)] == data:
                    matches.append(dict(kind="global", symbol=buffer["symbol"], offset=0))
    if len(matches) != 1:
        raise ValueError("pending_audit: source constant seed has no unique immutable linked object")
    return matches[0]


def _constant_seed_input(case, directory, symbol, elf):
    """Bind source seed bytes through typed LLVM globals to the same final ELF."""
    from xdsl.context import Context
    from xdsl.dialects import builtin, llvm
    from xdsl.parser import Parser

    from merlin.targetgen.elf_lanes import _endian

    values = np.asarray(case["constant_seed"], dtype=_endian(elf.read_bytes()) + "i4")
    encodings = {values.tobytes()}
    if np.all(values == values[0]):
        encodings.add(values[0].tobytes())
    context = Context()
    context.load_dialect(builtin.Builtin)
    context.load_dialect(llvm.LLVM)
    module = Parser(context, (directory / "spike-build/region_offload/core_aten.llvm.mlir").read_text()).parse_module()
    function = next(op for op in module.walk() if isinstance(op, llvm.FuncOp) and op.sym_name.data == symbol)
    references = {op.global_name.root_reference.data for op in function.walk() if isinstance(op, llvm.AddressOfOp)}
    globals_by_name = {
        op.sym_name.data: op.value.data.data
        for op in module.body.block.ops
        if isinstance(op, llvm.GlobalOp)
        and op.constant is not None
        and isinstance(op.value, builtin.DenseIntOrFPElementsAttr)
        and op.sym_name.data in references
        and op.value.data.data in encodings
    }
    return _linked_constant_input(elf, globals_by_name)


def _extension_parameters(case, facts):
    from merlin.xdsl_dialects.lowering.k_extension import lower_k_extension

    extension = lower_k_extension(
        case["coefficient"],
        source_k=case["source_k"],
        residual_width=case["residual_width"],
        digit_bits=facts["operand_signed_bits"],
        max_extended_k=facts["k_extension_within"]["max_k"],
        accumulator_bits=facts["k_extension_within"]["partial_sum_bits"],
        source_bound=case["source_partial_bound"],
        residual_range=case["residual_range"],
    )
    # A diagonal residual contributes one digit tile to each output tile,
    # independent of how many other output columns the source contains.
    return dict(source_k=case["source_k"], extension_k=len(extension["tile_digits"]) * facts["tile_dim"])


def _trace_contexts(payload, candidate, path):
    from merlin.compile.fusion_enumeration import dataflow_variants
    from merlin.targetgen.functional_truth_compositions import _audit_modes

    nodes = candidate["semantic"]["nodes"]
    # A mesh product must carry a selected dataflow; a graph without one has none to audit.
    if candidate["semantic"]["dataflow"] not in dataflow_variants(nodes):
        raise ValueError("pending_audit: candidate dataflow contradicts its mesh-product semantics")

    contexts = []
    for node, cap in zip(nodes, payload["nodes"], strict=True):
        case = cap["operation"]["attributes"]["case"]
        chain = chain_for_node(node, candidate["semantic"]["dataflow"])
        if candidate["semantic"]["structure"] == "shared_input":
            chain["multiplicity"] = dict(min=len(nodes))
        modes = dict.fromkeys(_audit_modes(chain, case, path), True)
        if case["exit"] == "requant":
            modes["readout_multipliers"] = True
        contexts.append((case, chain, modes))
    return contexts


def _trace_modes(payload, candidate, path):
    # Compatibility view for callers auditing a single member.
    contexts = _trace_contexts(payload, candidate, path)
    if any(modes != contexts[0][2] for _, _, modes in contexts[1:]):
        raise ValueError("pending_audit: siblings require output-scoped trace contexts")
    return contexts[0]


def _scoped_sibling_modes(contexts, placement, parameters, partitions, outputs):
    """Keep the full producer stream, but bind readout duties to each source output."""
    scoped, scoped_parameters = {}, {}
    if len(outputs) != len(contexts):
        raise ValueError("pending_audit: sibling outputs do not match trace contexts")
    for stage in placement["stages"]:
        owners = [i for i in range(len(contexts)) if stage.startswith(f"node{i}_")]
        if len(contexts) == 1:
            owners = [0]
        if len(owners) != 1:
            raise ValueError("pending_audit: source stage has no unique sibling owner")
        i = owners[0]
        case, _, modes = contexts[i]
        params = copy.deepcopy(parameters)
        output = outputs[i]
        if case["exit"] == "accumulator":
            params["i32_exit"] = dict(outputs=[output])
            if any("scale_input" in part for part in partitions):
                # Raw siblings of a runtime readout need explicit identity state.
                params["i32_exit"]["identity_config"] = True
            params.pop("readout_multipliers", None)
        else:
            selected = [part for part in partitions if part["output"] == output]
            if len(selected) != 1:
                raise ValueError("pending_audit: narrow sibling has no unique readout partition")
            # The auditor's partitions route checks a whole allocation set. Its
            # single-output route instead filters commits before checking coverage.
            params["readout_multipliers"] = dict(selected[0], outputs=[output])
        scoped[stage], scoped_parameters[stage] = modes, params
    return scoped, scoped_parameters


def _trace_audit(payload, candidate, placement, proof, directory, build, facts, target, *, retain=True):
    from merlin.llvmlower.device_build import load_region_offload
    from merlin.targetgen.composition_audit import check_composition
    from merlin.targetgen.frozen_placement import host_chain_consumers
    from merlin.targetgen.native_instruction_audit import (
        audit_case_instructions,
        decode_retirement_trace,
        elf_object_buffers,
    )

    sidecar = load_region_offload(directory / "spike-build")
    routed = sidecar.get("routed", [])
    if len(routed) != 1 or routed[0]["symbol"] != placement["symbol"]:
        raise ValueError("split_kernels: linked routing differs from admitted fused kernel")
    if routed != proof.get("bindings"):
        raise ValueError("pending_audit: linked source binding differs from independently admitted source")
    region = (
        routed[0].get("source_region_id")
        or routed[0].get("region_id")
        or digest(dict(source_sha256=placement["program_sha256"], source_op_indices=placement["source_op_indices"]))
    )
    routed = [
        dict(routed[0], symbol=placement["symbol"], region_id=region, source_op_indices=placement["source_op_indices"])
    ]
    frozen = dict(
        program_sha256=placement["program_sha256"],
        source_placements={str(i): "accelerator" for i in placement["source_op_indices"]},
    )
    nodes = candidate["semantic"]["nodes"]
    if candidate["semantic"]["structure"] == "shared_input":
        if (
            not all(node["form"] == "contraction" for node in nodes)
            or routed[0].get("numeric", {}).get("shared_input") is not True
            or routed[0].get("software_declaration", "contraction") != "contraction"
        ):
            raise ValueError("pending_audit: shared source is not bound to independent contractions")
        # The fused routing op represents the verified contraction producers.
        # Bind their declaration so the native audit also checks its obligations.
        routed[0]["software_declaration"] = "contraction"
    contexts = _trace_contexts(payload, candidate, facts["output_path"])
    case, chain, modes = contexts[0]
    if len(contexts) > 1 and any(
        c[0]["producer"] != case["producer"]
        or c[0].get("pool_window") != case.get("pool_window")
        or c[0].get("pool_stride") != case.get("pool_stride")
        or c[0].get("padding") != case.get("padding")
        or c[0].get("seed_source") != case.get("seed_source")
        or c[1].get("extension") != chain.get("extension")
        for c in contexts[1:]
    ):
        raise ValueError("pending_audit: heterogeneous producer/pool/seed parameters need per-member audit")
    if any(n["residual"] != nodes[0]["residual"] or n["seed"] != nodes[0]["seed"] for n in nodes[1:]):
        raise ValueError("pending_audit: heterogeneous sibling seed/extension needs per-member audit")
    parameters = {}
    if nodes[0]["residual"]:
        parameters["k_extension"] = _extension_parameters(case, facts)
    if "pool" in case["readout_stages"]:
        attrs = payload["nodes"][0]["operation"]["attributes"]
        h, w = case["pool_in_dims"]
        py, px = case["padding"]
        if len(set(case["pool_window"])) != 1 or len(set(case["pool_stride"])) != 1:
            raise ValueError("pending_audit: rectangular pool register realization is unavailable")
        parameters["pooled_store"] = dict(
            geometry=dict(
                image=[h + 2 * py, w + 2 * px, attrs["N"]], window=case["pool_window"][0], stride=case["pool_stride"][0]
            )
        )
        if py or px:
            size, stride = case["pool_window"][0], case["pool_stride"][0]
            oh, ow = (h + 2 * py - size) // stride + 1, (w + 2 * px - size) // stride + 1
            # Bind register padding to the unpadded source image. The readout
            # partition audit separately requires complete, disjoint output coverage.
            parameters["pooled_store"] = dict(
                fields=dict(
                    pool_stride=stride,
                    pool_size=size,
                    pool_out_dim=ow,
                    porows=oh,
                    pocols=ow,
                    orows=h,
                    ocols=w,
                    upad=py,
                    lpad=px,
                )
            )
        if case["producer"] == "movement":
            modes["scratchpad_source_pooling_fenced"] = True
    composition = dict(
        source_op_indices=placement["source_op_indices"],
        producer_source_op_indices=[
            i
            for name, values in placement["stages"].items()
            if "mac" in name or "operand_sum" in name or (case["producer"] == "movement" and "pool" in name)
            for i in values
        ],
        stages=placement["stages"],
        producer=case["producer"],
    )
    expected = dict(
        composition=composition,
        modes=modes,
        mode_parameters=parameters,
        stage_scoped_modes={name: modes for name in placement["stages"]},
    )
    trace, elf = directory / "retirement.log", Path(build["elf"])
    native = audit_case_instructions(elf=elf, trace=trace, routed=routed, facts=facts["raw"])
    runtime_abi, args = None, []
    if nodes[0]["seed"] or chain.get("multiplicity") or case["exit"] == "requant":
        runtime_abi, args = _runtime_abi(directory, proof, placement["symbol"])
    scale_loads = None
    if any(c[0]["exit"] == "requant" for c in contexts):
        scale_loads = _runtime_scale_loads(
            payload, proof, placement["symbol"], args, elf, trace, runtime_abi, facts, len(contexts)
        )
        parameters["readout_multipliers"] = _readout_bindings(payload, proof, placement["symbol"], args, scale_loads)
    if case["producer"] == "operand_sum":
        parameters["scaled_accumulate_load"] = _operand_bindings(payload, proof, placement["symbol"], args)
    if nodes[0]["seed"]:
        if case.get("seed_source") == "constant":
            seed = _constant_seed_input(case, directory, placement["symbol"], elf)
        else:
            bias = next(i for i, arg in enumerate(args) if arg["name"] == "bias")
            seed = dict(kind="argbase", arg_index=bias, offset=0)
        parameters["accumulator_preload"] = dict(input=seed)
    if chain.get("multiplicity"):
        shared = next(i for i, arg in enumerate(args) if arg["name"] == "a")
        parameters["shared_input"] = dict(input=dict(kind="argbase", arg_index=shared, offset=0))
    if len(contexts) > 1:
        binding = proof["bindings"][0]
        if binding.get("result_slices"):
            output_indices = list(range(len(args), len(args) + len(binding["result_slices"])))
        else:
            output_indices = [i for i, arg in enumerate(args) if arg["role"] == "output"]
        outputs = [dict(kind="argbase", arg_index=i, offset=0) for i in output_indices]
        scoped, scoped_parameters = _scoped_sibling_modes(
            contexts, placement, parameters, parameters.get("readout_multipliers", {}).get("partitions", []), outputs
        )
        expected["stage_scoped_modes"] = scoped
        expected["stage_scoped_mode_parameters"] = scoped_parameters
        expected["modes"] = {mode: True for _, _, duties in contexts for mode in duties}
    decoded = decode_retirement_trace(
        elf=elf,
        trace=trace,
        routed=routed,
        target=target,
        runtime_abi=runtime_abi,
        static_buffers=elf_object_buffers(elf) if runtime_abi else None,
    )
    _bind_result_slices(decoded, proof["bindings"][0], args)
    decoded["composition_evidence"] = dict(
        frozen_placement=frozen,
        source_sha256=placement["program_sha256"],
        routed_source_sha256=placement["program_sha256"],
        routed=routed,
        host_intermediate_consumers=host_chain_consumers(payload["source"], frozen, composition),
        native_instruction_audit=native,
    )
    if scale_loads is not None:
        decoded["composition_evidence"]["runtime_scale_loads"] = scale_loads
    result = check_composition(decoded, expected, target=target)
    if result["status"] == "pass" and case["producer"] == "operand_sum":
        screens = _per_operand_numeric_screens(payload, proof, placement["symbol"], facts, target)
        if screens:
            result["numeric_screens"] = screens
    if retain:
        retained = dict(result=result, native=native, expected=expected)
        if scale_loads is not None:
            retained["runtime_scale_loads"] = scale_loads
        (directory / "composition-audit.json").write_bytes(canonical(retained))
    return result


def execute_case(payload, candidate, selection, directory):
    from merlin.runtime.backends import spike_model
    from merlin.targetgen.core_aten_device import load_execution_provider, region_routing_for_bundle, selected_facts
    from merlin.targetgen.software_spec import load_software_spec

    target, stage = selection["target"], "admission"
    outcome = dict(outcome="not_admitted", numeric=False, placement=False, trace=False, reason="pending_execution")
    discharge = None
    try:
        package = Path(selection["package"])
        with selected_facts(target, Path(selection["facts"])) as raw_facts:
            provider = load_execution_provider(target, Path(selection["provider"]), candidate_writable_roots=())
            policy = _selected_policy(load_software_spec(Path(selection["spec"]), target=target), candidate)
            options = readout_options(selection, policy, raw_facts, candidate)
            proof = provider.admit_numeric_schedule(
                payload["source"], package=package, numeric_policy=policy, readout_options=options
            )
            (directory / "admission.json").write_bytes(canonical(proof))
            if proof.get("status") != "proven":
                outcome["reason"] = "backend_unsupported: " + proof.get("reason", "provider declined")
                return outcome
            placement, reason = prove_placement(payload["source"], proof)
            if reason:
                outcome["reason"] = reason
                return outcome
            outcome["placement"] = True
            stage = "compile"
            _write_bundle(payload, directory)

            def offload(prepared, workdir, **kwargs):
                return provider.core_aten_offload(
                    prepared, workdir, numeric_policy=policy, readout_options=options, **kwargs
                )

            routing = region_routing_for_bundle(target, package, offload)
            build = spike_model.build(
                directory,
                directory / "spike-build",
                inputs_npz=directory / "inputs.npz",
                arena_mb=16,
                backend="scalar",
                host_roundeven=True,
                device=routing,
                dump_all_outputs=True,
            )
            (directory / "build.json").write_bytes(
                canonical(dict(elf=str(build["elf"]), mem_bytes=build["mem_bytes"], vlen=build.get("vlen")))
            )
            stage = "execution"
            runner_options = provider.runner_options()
            extension = pinned_extension(target, Path(selection["extension"]), runner_options)
            if file_digest(extension) != selection["extension_sha256"]:
                raise ValueError("selected patched extension digest changed")
            runner_options.update(extlib=extension)
            stage = "oracle_obligations"
            discharge, refusal = _oracle_discharge(
                payload, proof, placement, directory, selection["extension_sha256"], target
            )
            if refusal:
                outcome["reason"] = refusal
                return outcome
            stage = "execution"
            run = spike_model.run(
                build["elf"],
                mem_bytes=build["mem_bytes"],
                timeout=selection["timeout"],
                vlen=build.get("vlen"),
                trace_path=directory / "retirement.log",
                log_commits=True,
                # Simulator diagnostics and guest HTIF share stdout. Flush host
                # lines so buffered diagnostics cannot splice into shape frames.
                command_prefix=["stdbuf", "-oL"],
                **runner_options,
            )
            outputs = run["output_bytes"]
            wanted = [
                np.asarray(r["values"], dtype={"i8": np.int8, "i32": np.int32}[r["dtype"]]).tobytes()
                for r in payload["references"]
            ]
            output_shapes = run.get("output_shapes")
            if not output_shapes or any(not shape for shape in output_shapes):
                raise ValueError("runner did not measure output shapes")
            outcome["numeric"] = outputs == wanted and output_shapes == [r["shape"] for r in payload["references"]]
            (directory / "output.json").write_bytes(
                canonical(dict(bytes=[value.hex() for value in outputs], shapes=output_shapes))
            )
            stage = "trace"
            from .fusion_cases import facts_for_cases

            facts = facts_for_cases(selection["candidates"], raw_facts)
            facts["raw"] = raw_facts
            audit = _trace_audit(payload, candidate, placement, proof, directory, build, facts, target)
            outcome["trace"] = audit["status"] == "pass"
            outcome["reason"] = (
                "pass"
                if outcome["numeric"] and outcome["trace"]
                else "numeric_mismatch"
                if not outcome["numeric"]
                else "fusion_not_observed: " + "; ".join(audit["violations"])
            )
            missing = [
                row
                for row in payload.get("required_witnesses", [])
                if not payload["witnesses"][row["node"]][row["event"]]
            ]
            if missing:
                outcome["reason"] = "pending_witness: " + ", ".join(row["event"] for row in missing)
            elif outcome["reason"] == "pass":
                _qualify(outcome, discharge)
    except Exception as error:  # A failed shard remains explicit and earns no admission.
        import traceback

        (directory / "failure.txt").write_text(traceback.format_exc())
        outcome["reason"] = f"{stage}: {type(error).__name__}: {error}"
    return outcome


def pinned_extension(target, explicit, options):
    from merlin.targetgen.spike_extension import resolve

    selected = resolve(target, default_library_dir=explicit.parent, default_extension_name=options["extension"])
    if selected.declared and (selected.extlib is None or selected.extlib.resolve() != explicit.resolve()):
        raise ValueError("declared extension differs from explicit campaign selection")
    if not explicit.is_absolute() or not explicit.is_file():
        raise ValueError("patched extension selection must be an existing absolute binary")
    return explicit


def verify_pass(payload, candidate, selection, directory):
    from merlin.targetgen.core_aten_device import load_execution_provider, selected_facts
    from merlin.targetgen.software_spec import load_software_spec

    from .fusion_cases import facts_for_cases

    if any(not payload["witnesses"][row["node"]][row["event"]] for row in payload.get("required_witnesses", [])):
        raise ValueError("fusion record lacks a required numerical witness")
    observed = json.loads((directory / "output.json").read_bytes())
    wanted = [
        np.asarray(r["values"], dtype={"i8": np.int8, "i32": np.int32}[r["dtype"]]).tobytes().hex()
        for r in payload["references"]
    ]
    if observed != dict(bytes=wanted, shapes=[r["shape"] for r in payload["references"]]):
        raise ValueError("fusion numeric evidence differs from the frozen exact reference")
    with selected_facts(selection["target"], Path(selection["facts"])) as raw:
        provider = load_execution_provider(
            selection["target"], Path(selection["provider"]), candidate_writable_roots=()
        )
        spec = load_software_spec(Path(selection["spec"]), target=selection["target"])
        policy = _selected_policy(spec, candidate)
        proof = provider.admit_numeric_schedule(
            payload["source"],
            package=Path(selection["package"]),
            numeric_policy=policy,
            readout_options=readout_options(selection, policy, raw, candidate),
        )
        if canonical(proof) != (directory / "admission.json").read_bytes():
            raise ValueError("fusion backend admission changed")
        placement, reason = prove_placement(payload["source"], proof)
        if reason:
            raise ValueError(reason)
        discharge = discharge_oracle_obligations(
            payload,
            proof,
            _catalog(directory),
            placement["symbol"],
            selection["extension_sha256"],
            relu_order=relu_order_registry(selection["target"]),
        )
        if canonical(discharge) != (directory / "oracle-obligations.json").read_bytes():
            raise ValueError("fusion oracle-obligation discharge changed")
        facts = facts_for_cases(selection["candidates"], raw)
        facts["raw"] = raw
        audit = _trace_audit(
            payload,
            candidate,
            placement,
            proof,
            directory,
            json.loads((directory / "build.json").read_bytes()),
            facts,
            selection["target"],
            retain=False,
        )
        if audit["status"] != "pass":
            raise ValueError("fusion retirement evidence no longer passes")
