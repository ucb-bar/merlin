"""Print a test case's quantization and the integer pipeline it implies, from the workspace's public inputs.

    python quant_params.py case NAME|DIR [--workspace .] [--prove] [--json]
    python quant_params.py ir FILE.mlir [--path] [--limit N] [--json]
    python quant_params.py target [--workspace .] [--json]

case    reads one public test case's capsule.yaml (NAME is found as */NAME/capsule.yaml under the
        workspace): operand dtypes, the declared epilogue stages in order, every scale (one per tensor,
        one per output column as equal-scale runs, a residual add's two multipliers and bound), the
        output dtype, the numeric policy and the accumulator bound, then the integer pipeline those
        imply. --prove adds the library's full-domain readout proofs: exact thresholds, an exact
        multiplier/shift when one exists, a one-correction fixed-point estimate, column partitions.
ir      parses a captured module with the library and, for every signed-integer contraction, prints the
        chain the library's output-fusion matcher reads after its accumulator (convert, scales with their
        numbers, bias, residual, rectifier, pool, quantize with scale/zero point/range). --path adds the
        cut the target's declared output path absorbs exactly, with every refusal. It also counts
        floating contractions with and without captured quantization evidence (a dequantized operand)
        and the quantize/dequantize ops with their zero points and ranges.
target  prints the output path from target_contract.yaml (the stage order the accelerator's narrowing
        readout applies, rounding, clamp, scale granularity, exits) and the quantization methodology of
        the public software spec.

Read-only. Library imports (ir, --prove) come from --library (default submission/compiler_library,
relative to the workspace) when it exists, else from the current interpreter's path.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

#: Top-level workspace directories that hold no test case of their own.
NOT_CASES = frozenset({"submission", "variants", "selfcheck_out", "qa", "ab"})
SCALE_KEYS = ("acc_scale", "lhs_scale", "rhs_scale", "bound_lsb", "requant_shift", "requant_round")
MODEL_KEYS = ("quant_scheme", "compile_dtype", "quant_recipe_agreement")
F32_EXACT_INT = 1 << 24


def f32(value: float) -> float:
    """``value`` rounded to IEEE binary32, as every readout stage computes."""
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def int_range(dtype: str | None) -> tuple[int, int] | None:
    """``(min, max)`` of an integer dtype name such as ``i8`` or ``u8``; None for other names."""
    if not dtype or dtype[0] not in "iu" or not dtype[1:].isdigit():
        return None
    bits = int(dtype[1:])
    return (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if dtype[0] == "i" else (0, (1 << bits) - 1)


def _yaml(path: Path):
    try:
        import yaml
    except ImportError as error:  # the workspace interpreter carries PyYAML; say so if this one does not
        raise SystemExit(f"PyYAML is required to read {path}: {error}") from error
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _use_library(workspace: Path, library: str) -> None:
    root = Path(library) if Path(library).is_absolute() else workspace / library
    if (root / "merlin").is_dir() and str(root) not in sys.path:
        sys.path.insert(0, str(root))


# ------------------------------------------------------------------------------------------- case


def find_case(workspace: Path, name: str) -> Path:
    """The capsule.yaml of a case given by directory or by name."""
    given = Path(name)
    if (given / "capsule.yaml").is_file():
        return given / "capsule.yaml"
    hits = [
        p
        for p in sorted(workspace.glob(f"*/{name}/capsule.yaml"))
        if p.parts[-3] not in NOT_CASES and not p.parts[-3].startswith(".")
    ]
    if len(hits) != 1:
        found = ", ".join(str(p.parent) for p in hits) or "none"
        raise SystemExit(f"case {name!r}: expected one */{name}/capsule.yaml under {workspace}, found {found}")
    return hits[0]


def column_runs(scales: list) -> list[dict]:
    """Contiguous runs of equal binary32 column scales: one readout scale setting per run."""
    runs: list[dict] = []
    for column, scale in enumerate(f32(s) for s in scales):
        if runs and runs[-1]["scale"] == scale:
            runs[-1]["columns"] += 1
        else:
            runs.append({"first_column": column, "columns": 1, "scale": scale})
    return runs


def _reduction(op: str, attrs: dict, inputs: dict) -> int | None:
    """The contraction's reduction length K, or None when the case declares no contraction."""
    weight = inputs.get(attrs.get("weight"))
    if weight is None:
        return None
    if all(type(attrs.get(k)) is int for k in ("ci", "kh", "kw")):
        return attrs["ci"] * attrs["kh"] * attrs["kw"]
    shape = weight.get("shape") or []
    return int(shape[-2]) if len(shape) >= 2 else None


def accumulator_bound(k: int, lhs: tuple[int, int], rhs: tuple[int, int]) -> dict:
    """Worst-case accumulator interval of K products, and whether i32 and an exact f32 conversion hold."""
    products = [a * b for a in lhs for b in rhs]
    lo, hi = k * min(0, min(products)), k * max(0, max(products))
    return {
        "terms": k,
        "min": lo,
        "max": hi,
        "fits_i32": -(1 << 31) <= lo and hi < (1 << 31),
        "exact_in_f32": max(-lo, hi) <= F32_EXACT_INT,
    }


def _stage_step(stage: str, attrs: dict, bias: str | None) -> str:
    if stage in ("bias_add", "bias"):
        return f"acc += {bias or 'bias'}[i32, one per output column]  (exact integer add in the accumulator)"
    if stage == "acc_scale":
        if "acc_scale_by_column" in attrs:
            runs = column_runs(attrs["acc_scale_by_column"])
            return (
                "y = round_half_even(f32(f32(acc) * scale[column]))  "
                f"({len(attrs['acc_scale_by_column'])} columns, {len(runs)} equal-scale runs)"
            )
        if "acc_scale" in attrs:
            return f"y = round_half_even(f32(f32(acc) * {f32(attrs['acc_scale'])!r}))  (one scale per tensor)"
        return "acc_scale declared without a multiplier: the case is malformed"
    if stage == "relu":
        return "y = max(y, 0)"
    if stage == "requant":
        shift = attrs.get("requant_shift")
        if type(shift) is not int:
            return "requant declared without requant_shift: the case is malformed"
        return f"y = (y + 2**{shift - 1}) >> {shift}  (arithmetic shift, ties round up)" if shift > 0 else "y = y"
    if stage == "maxpool":
        window = {k: attrs.get(k) for k in ("pool_size", "pool_stride", "pool_padding") if k in attrs}
        return f"y = max over the pool window {window}"
    return f"{stage}: not modelled here; read its arithmetic in command_buffer_abi.yaml"


def _narrow(dtype: str | None) -> str:
    bounds = int_range(dtype)
    if bounds is None:
        return f"output {dtype}: no integer narrowing"
    if bounds[1] >= (1 << 31) - 1:
        return f"store the raw {dtype} accumulator (no readout stage applies)"
    return f"saturate to [{bounds[0]}, {bounds[1]}] and store {dtype}"


def _policy(policy: dict) -> str:
    mode = policy.get("compare")
    if mode == "exact_int":
        return "exact_int: every element must equal the reference"
    if mode in ("bounded_int", "tolerance_float"):
        return f"{mode}: |observed - reference| <= {policy.get('atol', 0)} + {policy.get('rtol', 0)} * |reference|"
    return f"{mode}: read numeric_policy"


def pipeline(op: str, attrs: dict, inputs: dict, stages: list) -> list[str]:
    """The integer arithmetic a case's declaration implies, one step per line, in order."""
    steps: list[str] = []
    k = _reduction(op, attrs, inputs)
    if k is not None:
        lhs = inputs.get(attrs.get("lhs") or attrs.get("ifm"), {})
        weight = inputs[attrs["weight"]]
        steps.append(
            f"acc[i32] = sum over K={k} of {lhs.get('name', 'lhs')}[{lhs.get('dtype')}] * "
            f"{weight['name']}[{weight.get('dtype')}], accumulator starting at 0; y = acc below"
        )
    if "lhs_scale" in attrs and "rhs_scale" in attrs:
        steps.append(
            f"y = round_half_even(f32(f32(f32(a) * {f32(attrs['lhs_scale'])!r}) + "
            f"f32(f32(b) * {f32(attrs['rhs_scale'])!r})))  (one rounding; a form that rounds each operand "
            f"first may lie up to bound_lsb={attrs.get('bound_lsb')} steps away)"
        )
    steps.extend(_stage_step(stage, attrs, attrs.get("bias")) for stage in stages)
    steps.append(_narrow(attrs.get("output_dtype")))
    return steps


def case_summary(path: Path) -> dict:
    doc = _yaml(path) or {}
    operation = doc.get("operation") or {}
    attrs = dict(operation.get("attributes") or {})
    inputs = {row.get("name"): row for row in doc.get("inputs") or () if isinstance(row, dict)}
    stages = [str(s) for s in attrs.get("epilogue") or ()]
    scales = {key: attrs[key] for key in SCALE_KEYS if key in attrs}
    if "acc_scale_by_column" in attrs:
        scales["acc_scale_by_column"] = column_runs(attrs["acc_scale_by_column"])
    summary = {
        "case": doc.get("name", path.parent.name),
        "path": str(path),
        "kind": doc.get("kind"),
        "op": operation.get("op"),
        "inputs": [{k: row.get(k) for k in ("name", "role", "dtype", "shape")} for row in inputs.values()],
        "epilogue": stages,
        "scales": scales,
        "output_dtype": attrs.get("output_dtype"),
        "numeric_policy": doc.get("numeric_policy"),
        "policy": _policy(doc.get("numeric_policy") or {}),
        "stimulus_range": doc.get("stimulus_range"),
        "integer_partial_sum_bound": doc.get("integer_partial_sum_bound"),
        "pipeline": pipeline(operation.get("op"), attrs, inputs, stages),
    }
    model = {key: attrs[key] for key in MODEL_KEYS if key in attrs}
    if model:
        summary["model_quantization"] = model
    k = _reduction(operation.get("op"), attrs, inputs)
    lhs = inputs.get(attrs.get("lhs") or attrs.get("ifm"), {})
    weight = inputs.get(attrs.get("weight"), {})
    lhs_range, weight_range = int_range(lhs.get("dtype")), int_range(weight.get("dtype"))
    if k is not None and lhs_range and weight_range:
        summary["accumulator_bound"] = accumulator_bound(k, lhs_range, weight_range)
        summary["operand_ranges"] = [list(lhs_range), list(weight_range)]
    parts = [dict(row, origin="matmuls") for row in attrs.get("matmuls") or () if isinstance(row, dict)]
    if isinstance(attrs.get("prelude"), dict):
        parts.append(dict(attrs["prelude"], origin="prelude"))
    parts = [{k: row.get(k) for k in ("origin", "out", "epilogue", "output_dtype")} for row in parts]
    if parts:
        summary["sub_operations"] = parts
    screens = (doc.get("software_screen") or {}).get("numeric_screens") or ()
    if screens:
        summary["numeric_screens"] = [
            {k: s.get(k) for k in ("form", "status", "parameters", "max_error_lsb")} for s in screens
        ]
    return summary


def prove(summary: dict, attrs_scales: dict) -> dict:
    """Full-domain readout proofs from the library for the case's declared readout."""
    from merlin.llvmlower import integer_readout as IR
    from merlin.llvmlower.integer_producer_range import IntegerSumProductsRange

    out: dict = {}
    relu = "relu" in summary["epilogue"]
    bound, ranges = summary.get("accumulator_bound"), summary.get("operand_ranges")
    if bound and ranges:
        try:
            interval = IntegerSumProductsRange(bound["terms"], *ranges[0], *ranges[1]).interval()
            out["producer_interval"] = list(interval)
        except ValueError as error:
            out["producer_interval"] = f"refused: {error}"
    if int_range(summary.get("output_dtype")) != (-128, 127) or "acc_scale" not in summary["epilogue"]:
        out["readout"] = "no scaled i8 readout declared"
        return out
    if "acc_scale_by_column" in attrs_scales:
        scales = [run["scale"] for run in attrs_scales["acc_scale_by_column"] for _ in range(run["columns"])]
        try:
            part = IR.partition_column_scales(scales, len(scales), relu=relu)
            out["column_partition"] = {
                "partitions": part["partition_count"],
                "regions": [{k: r[k] for k in ("first_column", "columns", "scale")} for r in part["regions"]],
            }
        except ValueError as error:
            out["column_partition"] = f"refused: {error}"
        return out
    scale = f32(attrs_scales["acc_scale"])
    proof = IR.derive([scale], relu=relu)
    out["thresholds"] = {
        "count": len(proof["thresholds"]),
        "first": proof["thresholds"][0],
        "last": proof["thresholds"][-1],
        "output": [proof["output_min"], proof["output_max"]],
    }
    rational = IR.exact_rational_candidate([scale], relu=relu)
    out["exact_multiplier_shift"] = (
        {"multiplier": rational["multiplier"], "shift": rational["shift"]} if rational else None
    )
    fixed = IR.fixedpoint_candidate(proof)
    out["fixedpoint_estimate"] = (
        {k: fixed[k] for k in ("multiplier", "shift", "max_estimate_output_error")} if fixed else None
    )
    return out


def print_case(summary: dict) -> None:
    print(f"{summary['case']}  kind={summary['kind']} op={summary['op']}  ({summary['path']})")
    for row in summary["inputs"]:
        print(f"  input {row['name']}: role={row['role']} dtype={row['dtype']} shape={row['shape']}")
    print(f"  epilogue (in order): {summary['epilogue'] or 'none'}; output dtype {summary['output_dtype']}")
    for key, value in summary["scales"].items():
        if key == "acc_scale_by_column":
            print(f"  acc_scale_by_column: {len(value)} run(s)")
            for run in value[:16]:
                print(
                    f"    columns {run['first_column']}..{run['first_column'] + run['columns'] - 1}: {run['scale']!r}"
                )
        else:
            print(f"  {key}: {value!r}")
    for key in ("model_quantization", "accumulator_bound", "integer_partial_sum_bound", "sub_operations"):
        if summary.get(key):
            print(f"  {key}: {json.dumps(summary[key], sort_keys=True)}")
    print(f"  numeric policy: {summary['policy']}; stimulus range {summary['stimulus_range']}")
    for screen in summary.get("numeric_screens") or ():
        print(f"  numeric screen: {json.dumps(screen, sort_keys=True)}")
    print("  integer pipeline:")
    for number, step in enumerate(summary["pipeline"], 1):
        print(f"    {number}. {step}")
    if "proofs" in summary:
        print(f"  proofs: {json.dumps(summary['proofs'], sort_keys=True)}")


# --------------------------------------------------------------------------------------------- ir


def _numbers(numbers) -> object:
    if numbers is None:
        return None
    if isinstance(numbers, (int, float)):
        return {"per": "tensor", "value": float(numbers)}
    values = [float(v) for v in list(getattr(numbers, "tolist", lambda: numbers)())]
    distinct = sorted(set(values))
    if not distinct:
        return {"per": "channel", "n": 0}
    return {"per": "channel", "n": len(values), "distinct": len(distinct), "min": distinct[0], "max": distinct[-1]}


def _int_property(op, name: str):
    attr = op.properties.get(name) or op.attributes.get(name)
    data = getattr(getattr(attr, "value", None), "data", None)
    return data if isinstance(data, int) else None


def _quantizers(module, mq, values) -> dict:
    """Count quantize/dequantize ops by name with their zero points, scale kinds and integer ranges."""
    rows: dict[str, dict] = {}
    for op in module.walk():
        name = mq.op_name(op)
        if not name.startswith("quant_ext."):
            continue
        row = rows.setdefault(name, {"count": 0, "zero_points": set(), "ranges": set(), "scales": set()})
        row["count"] += 1
        operands = list(op.operands)
        if len(operands) > 2:
            zero = values.array(operands[2])
            row["zero_points"].update(["runtime"] if zero is None else [int(v) for v in zero.reshape(-1).tolist()])
        if len(operands) > 1:
            scale = values.array(operands[1])
            row["scales"].add("runtime" if scale is None else "per_tensor" if scale.size == 1 else "per_channel")
        row["ranges"].add((_int_property(op, "quant_min"), _int_property(op, "quant_max")))
    return {
        name: {k: sorted(v, key=str) if isinstance(v, set) else v for k, v in row.items()} for name, row in rows.items()
    }


def _is_product_reduction(op, mq) -> bool:
    """A generic that reduces products of two inputs (a contraction), not a max/sum window."""
    iterators = getattr(op, "iterator_types", None)
    kinds = [str(getattr(it, "data", it)).lower() for it in getattr(iterators, "data", ())]
    if not any("reduction" in kind for kind in kinds) or len(list(op.operands)) < 3 or not op.regions:
        return False
    return any(mq.op_name(inner) in ("arith.mulf", "arith.muli") for inner in op.regions[0].walk())


def _float_contractions(module, mq, quant_scope) -> dict:
    """Floating contractions split by whether an input traces to a captured dequantize."""
    out = {"with_captured_quantization": [], "without": []}
    for op in module.walk():
        name = mq.op_name(op)
        if not name.startswith("linalg.") or not op.results:
            continue
        named = any(word in name for word in ("matmul", "matvec", "vecmat", "conv"))
        if not named and not (name == "linalg.generic" and _is_product_reduction(op, mq)):
            continue
        _, dtype = mq.type_shape_dtype(op.results[0].type)
        if not str(dtype).startswith(("f", "bf")):
            continue
        key = "with_captured_quantization" if quant_scope.captured_quantized_contraction(op) else "without"
        out[key].append(mq.attr_str(op, "prov.region_id") or name)
    return out


def ir_summary(path: Path, *, contract: Path | None, limit: int) -> dict:
    from merlin.common import mlir_query as mq
    from merlin.llvmlower import quant_scope
    from merlin.llvmlower.integer_gemm import match_integer_gemm
    from merlin.xdsl_dialects.lowering import output_fusion as OF

    module = mq.parse(path.read_text(encoding="utf-8"))
    values = OF.Values(path)
    declared = None
    if contract is not None:
        from merlin.targetgen import output_path as OP

        found = find_output_path(_yaml(contract) or {})
        declared = OP.validate(found) if found else None
    chains = []
    for op in module.walk():
        dims = match_integer_gemm(op)
        if dims is None:
            continue
        chain = OF.source_chain(op, values)
        row = {
            "region": mq.attr_str(op, "prov.region_id") or mq.op_name(op),
            "dimensions": {k: getattr(dims, k) for k in ("batch", "m", "n", "k")},
            "channels": chain.channels,
            "stages": [
                {
                    "kind": s.kind,
                    "numbers": _numbers(s.numbers),
                    **{k: v for k, v in s.detail.items() if k != "operand"},
                }
                for s in chain.stages
            ],
            "unread": chain.unread,
        }
        if contract is not None:
            if declared is None:
                row["fusion"] = "the contract declares no output path"
            else:
                # The fusion pass bounds the accumulator by (M, K, N, rank) of the contraction.
                extents = {id(op): (dims.batch * dims.m, dims.k, dims.n, 3 if dims.batch > 1 else 2)}
                decision = OF.decide(chain, declared, values, extents)
                proof = decision.proof or {}
                row["fusion"] = {
                    "cut": decision.cut,
                    "absorbed": decision.stages,
                    "refusals": [{k: r.get(k) for k in ("cut", "code", "reason")} for r in decision.refusals],
                    **{k: proof[k] for k in ("commands", "scale_reconfigurations", "readout") if k in proof},
                }
                if "offsets" in proof:  # the integer bias each channel preloads into the accumulator
                    row["fusion"]["preload_offsets"] = _numbers(proof["offsets"])
                if "device_scale" in proof:  # the one f32 multiplier per channel the readout applies
                    row["fusion"]["device_scale"] = _numbers(proof["device_scale"])
        chains.append(row)
    return {
        "path": str(path),
        "integer_contractions": len(chains),
        "chains": chains[:limit],
        "chains_omitted": max(0, len(chains) - limit),
        "quantizers": _quantizers(module, mq, values),
        "float_contractions": _float_contractions(module, mq, quant_scope),
    }


def print_ir(summary: dict) -> None:
    print(f"{summary['path']}: {summary['integer_contractions']} signed-integer contraction(s)")
    for row in summary["chains"]:
        dims = row["dimensions"]
        print(f"  {row['region']}: i8 x i8 -> i32, batch={dims['batch']} M={dims['m']} N={dims['n']} K={dims['k']}")
        for stage in row["stages"]:
            extra = {k: v for k, v in stage.items() if k != "kind" and v is not None}
            print(f"    -> {stage['kind']} {json.dumps(extra, sort_keys=True) if extra else ''}".rstrip())
        if row["unread"]:
            print(f"    (chain stops: {row['unread']})")
        fusion = row.get("fusion")
        if isinstance(fusion, dict):
            taken = {k: v for k, v in fusion.items() if k != "refusals"}
            print(f"    output path takes: {json.dumps(taken, sort_keys=True)}")
            for reason in dict.fromkeys(f"{r['code']}: {r['reason']}" for r in fusion["refusals"]):
                cuts = [r["cut"] for r in fusion["refusals"] if f"{r['code']}: {r['reason']}" == reason]
                print(f"      refused {'/'.join(cuts)}: {reason}")
        elif fusion:
            print(f"    output path: {fusion}")
    if summary["chains_omitted"]:
        print(f"  ... {summary['chains_omitted']} more (raise --limit)")
    for name, row in summary["quantizers"].items():
        print(f"  {name}: {json.dumps(row, sort_keys=True)}")
    floats = summary["float_contractions"]
    print(
        f"  floating contractions with captured quantization evidence: {len(floats['with_captured_quantization'])} "
        f"{floats['with_captured_quantization'][:8]}"
    )
    print(f"  floating contractions without it: {len(floats['without'])} {floats['without'][:8]}")


# ----------------------------------------------------------------------------------------- target


def find_output_path(document):
    """The first mapping declaring the output path schema, searched depth first, or None."""
    stack = [document]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if node.get("schema") == "output_path_v1" and "readout" in node:
                return node
            stack.extend(reversed(list(node.values())))
        elif isinstance(node, list):
            stack.extend(reversed(node))
    return None


def target_summary(workspace: Path) -> dict:
    out: dict = {}
    contract = workspace / "target_contract.yaml"
    out["output_path"] = find_output_path(_yaml(contract)) if contract.is_file() else None
    for spec in (workspace / "public_declarations" / "software-spec.yaml", workspace / "software-spec.yaml"):
        if spec.is_file():
            quantization = dict((_yaml(spec) or {}).get("quantization") or {})
            method = dict(quantization.get("methodology") or {})
            variants = method.pop("model_variants", None) or ()
            method["model_variants"] = [
                {k: v.get(k) for k in ("model", "name", "status", "changes")} for v in variants if isinstance(v, dict)
            ]
            out["software_spec"] = {"path": str(spec), "formats": quantization.get("formats"), "methodology": method}
            break
    return out


def print_target(summary: dict) -> None:
    path = summary.get("output_path")
    if not path:
        print("target_contract.yaml declares no output path")
    else:
        print(f"output path: accumulator {path.get('accumulator')}, exits {path.get('exits')}")
        for stage in path.get("accumulator_stages") or ():
            print(f"  accumulator stage: {json.dumps(stage, sort_keys=True)}")
        for number, stage in enumerate(path.get("readout") or (), 1):
            print(f"  readout {number}: {json.dumps(stage, sort_keys=True)}")
    spec = summary.get("software_spec")
    if spec:
        print(f"software spec {spec['path']}:")
        print(f"  formats: {json.dumps(spec['formats'], sort_keys=True)}")
        for key, value in spec["methodology"].items():
            print(f"  {key}: {json.dumps(value, sort_keys=True)}")


# ------------------------------------------------------------------------------------------- main


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    case = sub.add_parser("case", help="one test case's declared quantization and integer pipeline")
    case.add_argument("name")
    case.add_argument("--prove", action="store_true", help="add the library's full-domain readout proofs")
    ir = sub.add_parser("ir", help="captured quantization chains in an MLIR module")
    ir.add_argument("file")
    ir.add_argument("--path", action="store_true", help="decide each chain against target_contract.yaml")
    ir.add_argument(
        "--contract",
        default="",
        help="contract file (implies --path; default WORKSPACE/target_contract.yaml)",
    )
    ir.add_argument("--limit", type=int, default=20)
    sub.add_parser("target", help="the output path and the software-spec quantization methodology")
    for each in (case, ir, sub.choices["target"]):
        each.add_argument("--workspace", default=".")
        each.add_argument("--library", default="submission/compiler_library")
        each.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    workspace = Path(args.workspace)
    _use_library(workspace, args.library)
    if args.mode == "case":
        path = find_case(workspace, args.name)
        summary = case_summary(path)
        if args.prove:
            summary["proofs"] = prove(summary, summary["scales"])
        show = print_case
    elif args.mode == "ir":
        contract = None
        if args.path or args.contract:
            contract = Path(args.contract) if args.contract else workspace / "target_contract.yaml"
        summary = ir_summary(Path(args.file), contract=contract, limit=max(1, args.limit))
        show = print_ir
    else:
        summary = target_summary(workspace)
        show = print_target
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    else:
        show(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
