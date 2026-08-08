"""Run a matmul interface on a self-hosted-ISA (``external_backend``) target's program oracle with the REAL
operands INJECTED — a TARGET-AGNOSTIC tool.

Nothing here is target-specific: the kernel is emitted by the target's GENERATED OOT package (the same
``run_entrypoints`` the grader uses), the operand layout + golden come from the target's ``model_ext``, and
the RTL cosim is DERIVED from the target by mlc's registry (``program_oracle``). This module only threads the
interface, injects the operands onto the command buffer's leaf tensors (by the harness-stamped DRAM layout),
and reads the output back. It carries no target-name literal and fails closed (returns ``None``) on any gap.
"""
from __future__ import annotations

import base64
import tempfile
from pathlib import Path


def _encode_operand(values, dtype: str) -> bytes | None:
    """Encode a nested-list operand to raw device bytes for the cb's DECLARED dtype (derived from the dtype
    token, not the target). Returns None for a dtype with no registered byte codec here — fail closed."""
    import numpy as np
    a = np.array(values, dtype=np.float64)
    if dtype in ("i8", "int8"):
        return np.clip(np.rint(a), -128, 127).astype("<i1").tobytes()
    if dtype in ("u8", "uint8"):
        return np.clip(np.rint(a), 0, 255).astype("<u1").tobytes()
    if dtype in ("i32", "int32"):
        return np.rint(a).astype("<i4").tobytes()
    if dtype in ("f32", "float32"):
        return a.astype("<f4").tobytes()
    try:                                             # fp8 / bf16 / fp16 via the derived float codec
        from merlin.targetgen.rtl.fp8_codec import encode_bytes as _fp_encode
        return _fp_encode(a, dtype)
    except Exception:                                # noqa: BLE001 — no codec for this dtype: fail closed
        return None


def matmul_on_program_oracle(target: str, interface_mlir: str, A, W, *, model_ext: str,
                             package: str | None, timeout: int = 900) -> list | None:
    """Emit the target's matmul kernel from ``interface_mlir`` via its generated package, inject ``A``/``W``
    onto the command buffer, run the mlc-derived program oracle, and return the output tensor (nested list)
    or ``None`` (fail closed). Target-agnostic — the target only enters as the parameter that selects its own
    generated package + mlc cosim."""
    if package is None:
        return None
    from . import capsule_common as CC
    from . import program_oracle as PO
    from .benchharness import runs_root
    from .capsule_common import make_run_paths

    with tempfile.TemporaryDirectory(prefix="mesh_prog_") as td:
        tdp = Path(td)
        # a minimal capsule dir the shared entrypoint runner accepts (interface only; op = matmul).
        cdir = tdp / "cap"
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "capsule.interface.mlir").write_text(interface_mlir, encoding="utf-8")
        capsule = {"name": "mesh_layer", "kind": "op", "interface_mlir": "capsule.interface.mlir",
                   "operation": {"op": "matmul", "attributes": {}}, "__dir__": str(cdir),
                   "required_oracle_tiers": ["L3"]}
        paths = make_run_paths(runs_root(target, "mesh_prog"), "mesh_layer", suite="mesh",
                               target=target, dtype="prog", benchmark="mesh_layer")
        try:
            _pkg, cb, kernel_text = CC.run_entrypoints(None, package, capsule, paths, timeout=timeout)
        except Exception:                            # noqa: BLE001 — package can't emit this kernel: honest None
            return None
        if cb is None or not kernel_text:
            return None

        # inject the REAL operands onto the cb's leaf tensors (the harness-stamped DRAM layout carries the
        # base; we only supply the bytes, encoded for each tensor's declared dtype).
        operands = {"A0": A, "W": W}
        for tname, tspec in (cb.get("tensors") or {}).items():
            if tspec.get("role") in ("input", "weight", "bias") and tname in operands:
                raw = _encode_operand(operands[tname], tspec.get("dtype", "i8"))
                if raw is None:
                    return None
                tspec["preload_b64"] = base64.b64encode(raw).decode()

        run = PO.program_oracle_adapter(target, model_ext=model_ext)
        try:
            res = run(cb, kernel_text, tdp / "oracle", timeout)
        except PO.OracleUnavailable:
            return None
        outs = res.get("outputs") or {}
        return next(iter(outs.values()), None)
