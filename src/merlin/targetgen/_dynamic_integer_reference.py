"""Independent bounded-exact reference for dynamic-only signed INT8 FX products.

No contraction primitive executes here: ascending-K outer products accumulate in
int64 and every prefix is checked against i32. This domain deliberately refuses
modular-overflow programs; it does not change their semantics or grant support.
Host operations and the original graph/output structure remain unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "merlin.dynamic_integer_reference.v1"


def run_reference(model, inputs, *, max_macs=64 << 20):
    import torch
    from torch.fx import Interpreter

    if type(max_macs) is not int or max_macs < 1:
        raise ValueError("independent reference needs a positive finite MAC budget")
    calls, work = [], 0
    integer_mm = torch.ops.aten._int_mm.default
    floating = {torch.ops.aten.matmul.default, torch.ops.aten.mm.default, torch.ops.aten.bmm.default,
                torch.ops.aten.linear.default, torch.ops.aten.conv2d.default, torch.ops.aten.convolution.default}

    class Reference(Interpreter):
        def run_node(self, node):
            nonlocal work
            if node.op != "call_function":
                return super().run_node(node)
            if node.target in floating:
                raise ValueError("dynamic-only reference refuses unaccounted floating contractions")
            if node.target != integer_mm:
                return super().run_node(node)
            args, kwargs = self.fetch_args_kwargs_from_env(node)
            if len(args) != 2 or kwargs:
                raise ValueError("integer reference requires the closed two-operand signed dot")
            lhs, rhs = args
            if (not all(isinstance(x, torch.Tensor) and x.dtype == torch.int8 and x.device.type == "cpu"
                        and x.ndim == 2 for x in args) or lhs.shape[1] != rhs.shape[0]
                    or min(*lhs.shape, *rhs.shape) < 1):
                raise ValueError("integer reference requires positive rank-two CPU signed-i8 operands")
            m, k, n = int(lhs.shape[0]), int(lhs.shape[1]), int(rhs.shape[1])
            work += m * k * n
            if work > max_macs:
                raise ValueError("independent reference MAC budget exceeded")
            # Worst-case bound prevents int64 overflow before checking actual i32 prefixes.
            if k * 128 * 128 > (1 << 31) - 1:
                raise ValueError("independent bounded-exact reference cannot prove the i32 reduction domain")
            left, right = lhs.to(torch.int64), rhs.to(torch.int64)
            acc = torch.zeros((m, n), dtype=torch.int64)
            maximum = 0
            for index in range(k):
                acc += left[:, index:index + 1] * right[index:index + 1, :]
                maximum = max(maximum, int(acc.abs().max()))
                if bool(((acc < -(1 << 31)) | (acc > (1 << 31) - 1)).any()):
                    raise ValueError("independent reference encountered i32 partial-sum overflow")
            def identity(x):
                return {"shape": list(x.shape), "dtype": "i8",
                        "sha256": hashlib.sha256(x.detach().contiguous().numpy().tobytes()).hexdigest()}
            calls.append({"node": node.name, "lhs": identity(lhs), "rhs": identity(rhs),
                          "mkn": [m, k, n], "max_abs_partial_sum": maximum,
                          "reduction_order": "k_ascending", "initial_sum": 0})
            return acc.to(torch.int32)

    with torch.no_grad():
        output = Reference(model).run(*tuple(inputs))
    if not calls:
        raise ValueError("dynamic integer reference executed no signed dot")
    return output, {"schema": SCHEMA, "numeric_domain": "bounded_exact_signed_i32",
                    "annotated_contractions": 0, "integer_mm_emitted": 0,
                    "reference_integer_mm_count": len(calls), "calls": calls,
                    "accumulator_bound_checked": True, "reference_macs": work}


def validate_receipt(receipt, *, annotated, dynamic_calls, exported):
    """Check the complete dynamic-only execution domain, not a PT2E exemption."""
    if (not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA
            or receipt.get("numeric_domain") != "bounded_exact_signed_i32"
            or annotated != 0 or type(annotated) is not int
            or type(receipt.get("annotated_contractions")) is not int or receipt["annotated_contractions"] != 0
            or type(receipt.get("integer_mm_emitted")) is not int or receipt["integer_mm_emitted"] != 0
            or receipt.get("accumulator_bound_checked") is not True
            or any(type(x) is not int or x < 1 for x in (dynamic_calls, exported))
            or receipt.get("reference_integer_mm_count") != dynamic_calls or dynamic_calls != exported
            or receipt.get("exported_integer_mm_count") != exported):
        raise ValueError("dynamic independent reference census/domain is incomplete")
    calls = receipt.get("calls")
    if not isinstance(calls, list) or len(calls) != exported:
        raise ValueError("dynamic independent reference lacks every executed dot")
    seen, macs = set(), 0
    for call in calls:
        if (not isinstance(call, dict) or not isinstance(call.get("node"), str)
                or not call["node"] or call["node"] in seen):
            raise ValueError("dynamic independent reference has ambiguous dot identities")
        seen.add(call["node"])
        dims = call.get("mkn")
        if not isinstance(dims, list) or len(dims) != 3 or any(type(x) is not int or x < 1 for x in dims):
            raise ValueError("dynamic independent reference has invalid geometry")
        m, k, n = dims
        if (k * 128 * 128 > (1 << 31) - 1 or call.get("initial_sum") != 0
                or call.get("reduction_order") != "k_ascending"
                or type(call.get("max_abs_partial_sum")) is not int
                or not 0 <= call["max_abs_partial_sum"] <= (1 << 31) - 1):
            raise ValueError("dynamic independent reference lacks bounded ordered accumulation")
        for key, shape in (("lhs", [m, k]), ("rhs", [k, n])):
            operand = call.get(key)
            if (not isinstance(operand, dict) or operand.get("shape") != shape or operand.get("dtype") != "i8"
                    or not isinstance(operand.get("sha256"), str) or len(operand["sha256"]) != 64
                    or any(c not in "0123456789abcdef" for c in operand["sha256"])):
                raise ValueError("dynamic independent reference lacks exact signed operands")
        macs += m * k * n
    if receipt.get("reference_macs") != macs:
        raise ValueError("dynamic independent reference work differs from its dots")
    limit = receipt.get("reference_work_limit", 64 << 20)
    if type(limit) is not int or not 1 <= limit <= 1 << 40 or macs > limit:
        raise ValueError("dynamic independent reference exceeds its selected work limit")
    agreement = receipt.get("golden_agreement")
    if (not isinstance(agreement, dict) or agreement.get("reference") != "dynamic_integer"
            or agreement.get("status") != "passed" or agreement.get("finite") is not True
            or type(agreement.get("samples")) is not int or agreement["samples"] < 1
            or not isinstance(agreement.get("outputs"), list) or not agreement["outputs"]):
        raise ValueError("dynamic independent reference lacks complete exact output agreement")
    for row in [agreement, *agreement["outputs"]]:
        if (not isinstance(row, dict) or row.get("finite") is not True
                or any(type(row.get(k)) not in (int, float) or row[k] != 0
                       for k in ("atol", "rtol", "max_abs", "max_rel"))
                or (row is not agreement and row.get("within_tolerance") is not True)):
            raise ValueError("dynamic independent reference output is not exact")


def compare_outputs(before, after, *, atol: float, rtol: float, output_abi) -> dict:
    """Compare two independently supplied executions on the capture input."""
    import torch

    left, left_abi = output_abi(before)
    right, right_abi = output_abi(after)
    rows = []
    compatible = left_abi == right_abi and bool(left)
    for original, rewritten in zip(left, right):
        lhs = original.detach().to(torch.float64)
        rhs = rewritten.detach().to(torch.float64)
        same_shape = lhs.shape == rhs.shape
        finite = bool(torch.isfinite(lhs).all() and torch.isfinite(rhs).all()) if same_shape else False
        if same_shape and finite and lhs.numel():
            delta = (lhs - rhs).abs()
            max_abs = float(delta.max())
            max_rel = float((delta / lhs.abs().clamp_min(1e-12)).max())
            within = bool(torch.all(delta <= atol + rtol * lhs.abs()))
        elif same_shape and finite:
            max_abs = max_rel = 0.0
            within = True
        else:
            max_abs = max_rel = 0.0
            within = False
        rows.append(
            {
                "max_abs": max_abs,
                "max_rel": max_rel,
                "within_tolerance": within,
                "finite": finite,
                "atol": atol,
                "rtol": rtol,
                "shape": list(original.shape),
            }
        )
    passed = compatible and len(rows) == len(left) and all(row["within_tolerance"] for row in rows)
    return {
        "status": "passed" if passed else "failed",
        "samples": 1,
        "atol": atol,
        "rtol": rtol,
        "max_abs": max((row["max_abs"] for row in rows), default=0.0),
        "max_rel": max((row["max_rel"] for row in rows), default=0.0),
        "outputs": rows,
        "finite": compatible and all(row["finite"] for row in rows),
    }


def capture_reference(model, inputs, out, *, output_abi, to_native, recipe=None):
    """Execute the independent reference and retain every original output and exact agreement."""
    import torch

    limit = reference_work_limit(recipe)
    output, receipt = run_reference(model, inputs, max_macs=limit)
    if "dynamic_integer_reference_limits" in (recipe or {}):
        receipt["reference_work_limit"] = limit
    with torch.no_grad():
        actual = model(*inputs)
    agreement = compare_outputs(output, actual, atol=0.0, rtol=0.0, output_abi=output_abi)
    leaves, abi = output_abi(output)
    raw = (json.dumps({"schema": "merlin.capture.integer_reference.v1", "output_abi": abi,
                       "outputs": [to_native(value) for value in leaves]},
                      sort_keys=True, separators=(",", ":")) + "\n").encode()
    (out / "integer-reference.json").write_bytes(raw)
    source = Path(__file__).resolve()
    agreement.update(reference="dynamic_integer",
        source={"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
        output={"path": "integer-reference.json", "sha256": hashlib.sha256(raw).hexdigest()})
    receipt["golden_agreement"] = agreement
    return receipt


def bind_reference_source(out):
    """Bind the executed reference helper in the ordinary producer receipt."""
    receipt_path = out / "capture_receipt.json"
    producer = json.loads(receipt_path.read_bytes())
    producer["tool"].setdefault("reference_sources", {})[
        "merlin/targetgen/_dynamic_integer_reference.py"
    ] = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(producer, sort_keys=True, indent=2) + "\n")


def reference_work_limit(recipe):
    """Read an explicit source-selected work budget without changing the numeric domain."""
    key = "dynamic_integer_reference_limits"
    if key not in (recipe or {}):
        return 64 << 20
    limits = recipe[key]
    if (type(limits) is not dict or set(limits) != {"schema", "max_macs"}
            or limits["schema"] != "merlin.dynamic_integer_reference_limits.v1"
            or (recipe or {}).get("software_numerical_engine") != "integer_reference"
            or type(limits["max_macs"]) is not int or not 1 <= limits["max_macs"] <= 1 << 40):
        raise ValueError("dynamic integer reference requires a finite explicit selected MAC budget")
    return limits["max_macs"]
