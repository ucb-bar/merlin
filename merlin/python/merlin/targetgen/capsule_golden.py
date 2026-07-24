"""Independent numeric golden for capsules + exact-equality comparison.

The golden is computed from the **capsule's declared operation** (not from the package's emitted
command buffer), so a wrong command buffer is caught by ``golden != reference(cb)``. The arithmetic
reuses the authoritative, dependency-free :class:`~merlin.runtime.tensor.Tensor` primitives (exact
int matmul, round-half-even f32 ``acc_scale``, saturating i8 cast) so the golden is bit-identical in
*semantics* to the reference/simulate/oracle paths while being structurally independent.

Leaves are materialized via :meth:`Tensor.deterministic` — the SAME function the command-buffer
materializer and the device harness use — so L0 (this golden) cannot silently diverge from L2/L3
on leaf data (the single-source-of-truth rule).

torch is not available in this environment; goldens are labeled ``merlin_tensor_int`` (numpy is used
only for the im2col gather). A capsule may cross-check against torch in ``model_slice_export`` when
torch is present.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from merlin.runtime.tensor import Tensor


# --------------------------------------------------------------------------------------------
# leaf materialization (single source of truth)
# --------------------------------------------------------------------------------------------
def materialize_capsule_leaves(capsule: dict) -> dict[str, Tensor]:
    """Materialize the capsule's declared leaf tensors deterministically by name."""
    env: dict[str, Tensor] = {}
    for spec in capsule.get("inputs", []):
        if spec.get("role") in ("input", "weight", "bias"):
            env[spec["name"]] = Tensor.deterministic(
                spec["name"], tuple(spec["shape"]), spec.get("dtype", "i8"))
    return env


# --------------------------------------------------------------------------------------------
# im2col (shared by golden + runner harness for conv2d)
# --------------------------------------------------------------------------------------------
from merlin.runtime.commandbuffer import conv_im2col, conv_out_dims  # noqa: E402  (single source of truth)


def im2col(ifm: Tensor, ci: int, kh: int, kw: int, *, stride, padding, dilation,
           layout: str = "nhwc") -> Tensor:
    """Thin wrapper over the runtime's canonical :func:`conv_im2col` (shared with the harness)."""
    return conv_im2col(ifm, kh=kh, kw=kw, ci=ci, stride=stride, padding=padding,
                       dilation=dilation, layout=layout)


# --------------------------------------------------------------------------------------------
# epilogue application (matches runtime/reference.py)
# --------------------------------------------------------------------------------------------
def _apply_epilogue(t: Tensor, attrs: dict, env: dict[str, Tensor]) -> Tensor:
    for stage in attrs.get("epilogue", []):
        if stage in ("bias_add", "bias"):
            bname = attrs.get("bias")
            if bname:
                t = t.add_bias(env[bname])
        elif stage == "requant":
            t = t.requant(int(attrs.get("requant_shift", 4)))
        elif stage == "acc_scale":
            t = t.requant_acc_scale(float(attrs.get("acc_scale", 1.0)))
        elif stage == "relu":
            t = t.relu()
    if attrs.get("output_dtype", "i32") == "i8":
        t = t.to_i8()
    return t


def _transpose2d(t: Tensor) -> Tensor:
    r, c = t.shape
    out = [0] * (r * c)
    for i in range(r):
        for j in range(c):
            out[j * r + i] = t.data[i * c + j]
    return Tensor((c, r), out, t.dtype)


# --------------------------------------------------------------------------------------------
# golden dispatch
# --------------------------------------------------------------------------------------------
def golden(capsule: dict) -> dict[str, list]:
    """Compute the capsule's expected outputs (name -> nested list)."""
    env = materialize_capsule_leaves(capsule)
    op = capsule["operation"]["op"]
    attrs = capsule["operation"].get("attributes", {})
    out_name = attrs.get("out", "Y0")

    def _pick(role: str) -> str:
        for s in capsule["inputs"]:
            if s.get("role") == role:
                return s["name"]
        raise KeyError(f"no input with role {role!r}")

    if op in ("matmul", "linear"):
        lhs = env[attrs.get("lhs", _pick("input"))]
        w = env[attrs.get("weight", _pick("weight"))]
        t = lhs.matmul(w)
        t = _apply_epilogue(t, attrs, env)
        return {out_name: t.to_list()}

    if op == "movement":
        src = env[attrs.get("src", _pick("input"))]
        return {out_name: src.to_list()}

    if op == "conv2d":
        ifm = env[attrs["ifm"]]
        w = env[attrs["weight"]]              # packed [Kh*Kw*Ci, Co]
        ci = int(attrs["ci"]); kh = int(attrs["kh"]); kw = int(attrs["kw"])
        cols = im2col(ifm, ci, kh, kw, stride=tuple(attrs.get("stride", [1, 1])),
                      padding=tuple(attrs.get("padding", [0, 0, 0, 0])),
                      dilation=tuple(attrs.get("dilation", [1, 1])),
                      layout=attrs.get("layout", "nhwc"))
        t = cols.matmul(w)
        t = _apply_epilogue(t, attrs, env)
        return {out_name: t.to_list()}

    if op == "attention_qk":
        q = env[attrs["q"]]; k = env[attrs["k"]]
        t = q.matmul(_transpose2d(k))         # Q @ K^T
        t = _apply_epilogue(t, attrs, env)
        return {out_name: t.to_list()}

    if op == "attention_pv":
        p = env[attrs["p"]]; v = env[attrs["v"]]
        t = p.matmul(v)
        t = _apply_epilogue(t, attrs, env)
        return {out_name: t.to_list()}

    if op == "resident_reuse":
        # one resident weight, multiple matmuls (each with its own lhs/epilogue/out)
        w = env[attrs["weight"]]
        outs: dict[str, list] = {}
        for spec in attrs["matmuls"]:
            t = env[spec["lhs"]].matmul(w)
            sub = {"epilogue": spec.get("epilogue", []),
                   "output_dtype": spec.get("output_dtype", "i32"),
                   "acc_scale": spec.get("acc_scale", attrs.get("acc_scale", 1.0))}
            t = _apply_epilogue(t, sub, env)
            outs[spec["out"]] = t.to_list()
        return outs

    raise ValueError(f"golden: unsupported operation {op!r}")


# --------------------------------------------------------------------------------------------
# comparison + numeric report
# --------------------------------------------------------------------------------------------
def _flat(nested) -> list:
    out: list = []
    if nested and isinstance(nested[0], list):
        for r in nested:
            out.extend(r)
    else:
        out.extend(nested)
    return out


def compare(expected: dict[str, list], observed: dict[str, list], policy: dict) -> dict:
    """Exact-int (or tolerance-float) comparison; returns a numeric_report dict."""
    mode = policy.get("compare", "exact_int")
    rep: dict[str, Any] = {"policy": mode, "golden_source": "merlin_tensor_int",
                           "status": "pass", "mismatch_count": 0,
                           "max_abs_error": 0, "max_rel_error": 0.0,
                           "first_mismatch": None, "per_output": {}}
    total_mismatch = 0
    for name, exp in expected.items():
        ef = _flat(exp)
        if name not in observed:
            rep["status"] = "fail"
            rep["per_output"][name] = {"status": "fail", "reason": "missing from observed"}
            total_mismatch += len(ef)
            continue
        of = _flat(observed[name])
        if len(ef) != len(of):
            rep["status"] = "fail"
            rep["per_output"][name] = {"status": "fail",
                                       "reason": f"length {len(of)} != {len(ef)}"}
            total_mismatch += abs(len(ef) - len(of)) + 1
            continue
        mism = 0
        maxabs = 0
        maxrel = 0.0
        first = None
        for idx, (a, b) in enumerate(zip(ef, of)):
            if mode == "exact_int":
                bad = int(a) != int(b)
                d = abs(int(a) - int(b))
            else:
                rtol = float(policy.get("rtol", 0.0)); atol = float(policy.get("atol", 0.0))
                d = abs(float(a) - float(b))
                bad = d > (atol + rtol * abs(float(a)))
            if bad:
                mism += 1
                maxabs = max(maxabs, d)
                # max relative error over the DIVERGING elements. Undefined when the expected value is 0
                # (division by zero) — max_abs_error covers that case; we simply don't fold it into maxrel.
                den = abs(float(a))
                if den > 0.0:
                    maxrel = max(maxrel, d / den)
                if first is None:
                    first = {"output": name, "index": idx, "expected": a, "observed": b}
        rep["per_output"][name] = {"status": "pass" if mism == 0 else "fail",
                                   "mismatch_count": mism, "max_abs_error": maxabs,
                                   "max_rel_error": maxrel}
        if mism:
            rep["status"] = "fail"
            rep["max_abs_error"] = max(rep["max_abs_error"], maxabs)
            rep["max_rel_error"] = max(rep["max_rel_error"], maxrel)
            if rep["first_mismatch"] is None:
                rep["first_mismatch"] = first
        total_mismatch += mism
    rep["mismatch_count"] = total_mismatch
    return rep


def write_numeric_report(path: str | Path, report: dict) -> None:
    import yaml
    Path(path).write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
