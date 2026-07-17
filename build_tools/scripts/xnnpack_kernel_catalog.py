#!/usr/bin/env python
"""Enumerate the FULL XNNPACK RVV microkernel surface and classify each kernel by whether Merlin
has (or can have) a codegen equivalent — the "explore ALL XNNPACK kernels" catalog (P3).

The head-to-head comparison (expert kernel vs our codegen) only makes sense for kernels we can lower
an equivalent for. This catalog is the honest inventory: it lists EVERY RVV ukernel (so nothing is
silently ignored) and tags each `mapped` (a Merlin op emits the same computation — GEMM/IGEMM,
int8 GEMM, dwconv, activations, elementwise binary, sparse matmul) or `expert-only` (f16, pooling,
convert, pack/transpose, reductions, transcendental primitives with no Merlin op). Writes
``out/artifacts/ceiling/kernel_catalog.{json,md}``.

Run:  .venv/bin/python build_tools/scripts/xnnpack_kernel_catalog.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "merlin" / "python"))
from merlin.common.paths import artifacts_dir  # noqa: E402
from merlin.runtime.backends import xnnpack_board as xb  # noqa: E402

# family prefix -> (merlin_op, status). status: "mapped" = a Merlin op emits the same compute;
# "partial" = Merlin has the op but a known lowering gap; "expert-only" = no Merlin codegen equiv.
# Keyed by the leading token(s) of the family directory (e.g. "f32-gemm", "qd8-f32-qc8w-gemm").
_MAP: dict[str, tuple[str, str]] = {
    "f32-gemm": ("linalg.matmul (f32)", "mapped"),
    "f32-igemm": ("conv-as-matmul (f32)", "mapped"),
    "f16-gemm": ("matmul (f16)", "expert-only"),      # no f16 codegen path today
    "f16-igemm": ("conv (f16)", "expert-only"),
    "qd8-f32-qc8w-gemm": ("int8 matmul W8A8 (vwmacc)", "mapped"),
    "qd8-f32-qc4w-gemm": ("int4 matmul", "expert-only"),
    "qd8-f32-qc8w-igemm": ("int8 conv-as-matmul", "mapped"),
    "qs8-qc8w-gemm": ("int8 matmul", "mapped"),
    "qs8-qc8w-igemm": ("int8 conv-as-matmul", "mapped"),
    "qs8-gemm": ("int8 matmul", "mapped"),
    "qs8-igemm": ("int8 conv-as-matmul", "mapped"),
    "qu8-gemm": ("uint8 matmul", "partial"),
    "qu8-igemm": ("uint8 conv", "partial"),
    "f32-dwconv": ("depthwise conv (f32)", "partial"),   # Merlin conv path exists; depthwise prim is a gap
    "f32-dwconv2d-chw": ("depthwise conv chw", "partial"),
    "f32-spmm": ("sparse matmul", "partial"),
    "f32-vgelu": ("gelu activation", "mapped"),
    "f32-vapproxgelu": ("approx gelu", "mapped"),
    "f32-vsigmoid": ("sigmoid activation", "mapped"),
    "f32-vtanh": ("tanh activation", "mapped"),
    "f32-vhswish": ("hardswish activation", "mapped"),
    "f32-vlrelu": ("leaky-relu activation", "mapped"),
    "f32-velu": ("elu activation", "mapped"),
    "f32-vunary": ("unary elementwise (f32)", "mapped"),
    "f32-vbinary": ("binary elementwise add/mul (f32)", "mapped"),
    "f32-vsqrt": ("sqrt", "partial"),
    "f32-vrsqrt": ("rsqrt (rmsnorm)", "mapped"),
    "f32-vlog": ("log", "partial"),
    "f32-vexp": ("exp (softmax)", "partial"),
    "f32-vsin": ("sin", "expert-only"),
    "f32-vcos": ("cos", "expert-only"),
    "f32-vcopysign": ("copysign", "expert-only"),
    "f32-vclamp": ("clamp/relu", "mapped"),
    "f32-vrnd": ("round", "expert-only"),
    "f32-rminmax": ("reduce min/max", "partial"),
    "f32-rsum": ("reduce sum", "partial"),
    "f32-maxpool": ("maxpool", "expert-only"),
    "f32-avgpool": ("avgpool", "expert-only"),
    "f32-argmaxpool": ("argmaxpool", "expert-only"),
    "x32-packw": ("weight pack", "expert-only"),        # an internal primitive, not a model op
    "x32-transposec": ("transpose", "partial"),
}
# dtype prefixes that are categorically expert-only for Merlin (no codegen path): f16 + quantized
# convert + 8-bit clamp/pool. A family not in _MAP falls back to a dtype-based default.
_EXPERT_DTYPES = ("f16-", "s8-", "u8-", "qs8-", "qu8-", "qc8w-")


def _family(rel: Path) -> str:
    """The family dir under src/ (e.g. 'f32-gemm', 'qd8-f32-qc8w-gemm')."""
    return rel.parts[0] if rel.parts else rel.stem


def _classify(family: str) -> tuple[str, str]:
    if family in _MAP:
        return _MAP[family]
    if family.startswith("f16-") or family.startswith("qd8-f16"):
        return ("(f16 — no Merlin codegen)", "expert-only")
    if any(family.startswith(p) for p in _EXPERT_DTYPES):
        return ("(quantized/8-bit primitive)", "expert-only")
    return ("(unmapped)", "expert-only")


def build_catalog(xnn_src: Path) -> dict:
    rows = []
    for c in sorted(xnn_src.rglob("*rvv*.c")) + sorted(xnn_src.rglob("*rvv*.c.in")):
        rel = c.relative_to(xnn_src)
        fam = _family(rel)
        op, status = _classify(fam)
        dtype = fam.split("-")[0]
        rows.append({"family": fam, "dtype": dtype, "kernel_file": str(rel),
                     "template": c.suffix == ".in", "merlin_op": op, "status": status})
    by_status = Counter(r["status"] for r in rows)
    by_family = Counter(r["family"] for r in rows)
    return {"n_kernels": len(rows), "by_status": dict(by_status),
            "n_families": len(by_family), "rows": rows}


def _write_md(path: Path, cat: dict) -> None:
    st = cat["by_status"]
    lines = [
        "# XNNPACK RVV microkernel catalog (Merlin codegen coverage)",
        "",
        f"**{cat['n_kernels']} RVV microkernels** across **{cat['n_families']} families**. "
        f"mapped={st.get('mapped',0)} · partial={st.get('partial',0)} · "
        f"expert-only={st.get('expert-only',0)}.",
        "",
        "`mapped` = a Merlin op emits the same computation (head-to-head comparable). "
        "`partial` = Merlin has the op but a known lowering gap. `expert-only` = no Merlin codegen "
        "equivalent (f16, pooling, convert, pack/transpose, some transcendentals).",
        "",
        "## Families (grouped, mapped first)",
        "",
        "| family | dtype | #kernels | merlin op | status |",
        "|---|---|---|---|---|",
    ]
    fam_rows: dict[str, dict] = {}
    for r in cat["rows"]:
        f = fam_rows.setdefault(r["family"], {"dtype": r["dtype"], "op": r["merlin_op"],
                                              "status": r["status"], "n": 0})
        f["n"] += 1
    order = {"mapped": 0, "partial": 1, "expert-only": 2}
    for fam, f in sorted(fam_rows.items(), key=lambda kv: (order[kv[1]["status"]], kv[0])):
        lines.append(f"| `{fam}` | {f['dtype']} | {f['n']} | {f['op']} | {f['status']} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    xnn = xb._xnnpack_repo() / "src"
    if not xnn.is_dir():
        print(f"XNNPACK src not found at {xnn} (set MERLIN_XNNPACK_REPO)", file=sys.stderr)
        return 1
    cat = build_catalog(xnn)
    out = artifacts_dir() / "ceiling"
    out.mkdir(parents=True, exist_ok=True)
    (out / "kernel_catalog.json").write_text(json.dumps(cat, indent=2))
    _write_md(out / "kernel_catalog.md", cat)
    st = cat["by_status"]
    print(f"{cat['n_kernels']} kernels / {cat['n_families']} families -> {out}/kernel_catalog.md")
    print(f"  mapped={st.get('mapped',0)} partial={st.get('partial',0)} "
          f"expert-only={st.get('expert-only',0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
