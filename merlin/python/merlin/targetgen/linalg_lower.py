"""Lower a linalg-on-tensors workload inventory to a Merlin command buffer (REFERENCE backend only).

The linalg-on-tensors grammar has a granted *reader* (:mod:`.contract.linalg_iface`) that a package
uses to parse; authoring the lowering to the target command buffer is the package's own job. This module
is the REFERENCE backend's lowering — the known-good baseline, NOT an agent tool — so it lives outside the
granted set (like :mod:`merlin.runtime.reference`). It maps the inventory the reader produces to the
command-buffer opcodes the reference emitter (:mod:`merlin.targets.muon.backend.muon_codegen_mlir`)
supports, fail-closing on any pattern the emitter cannot yet build.

Currently supported patterns (each verified against the reference emitter):
- a single equal-shape elementwise ``add``/``mul`` (residual-add / elementwise product) -> ``VECTOR_MAP``;
- a single row-broadcast elementwise ``add``/``mul`` (a length-n rhs over an (m,n) lhs; standalone
  bias-add / per-channel scale) -> ``VECTOR_MAP`` (the emitter detects the broadcast from the shapes).

The dtype is set to ``f32`` (the reference emitter's compute domain); whether an fp32 kernel meets a
bf16/f16 capsule's tolerance is a separate fidelity question settled by the cyclotron oracle.
"""
from __future__ import annotations

from typing import Any


class LinalgLowerError(RuntimeError):
    """Raised when the inventory is not a pattern the reference emitter can build (fail-closed)."""


def _combine_from_body(body_ops: list[str]) -> str:
    if any(b.endswith("addf") for b in body_ops):
        return "add"
    if any(b.endswith("mulf") for b in body_ops):
        return "mul"
    raise LinalgLowerError(f"elementwise body {body_ops} is neither an add nor a mul")


def lower_linalg_to_cb(parsed: dict[str, Any], *, target: str) -> dict[str, Any]:
    """Lower a :func:`merlin.targetgen.contract.linalg_iface.parse_linalg_mlir` inventory to a command
    buffer, or raise :class:`LinalgLowerError` for an unsupported pattern."""
    ops = parsed.get("ops", [])
    args = parsed.get("args", [])
    if not ops:
        raise LinalgLowerError("empty linalg workload (no payload ops)")

    argname = {a["index"]: f"arg{a['index']}" for a in args}
    argshape = {a["index"]: list(a["shape"]) for a in args}

    # --- single elementwise map (equal-shape or row-broadcast) ---------------------------------------
    if len(ops) == 1 and ops[0]["family"] in ("elementwise", "elementwise_map"):
        op = ops[0]
        ins = op["ins"]
        if len(ins) != 2 or any(s["source"][0] != "arg" for s in ins):
            raise LinalgLowerError(
                "reference lowering supports a two-operand elementwise map over func args only "
                f"(got {[s['source'] for s in ins]})")
        combine = _combine_from_body(op["body_ops"])
        i0, i1 = ins[0]["source"][1], ins[1]["source"][1]
        s0, s1 = argshape[i0], argshape[i1]
        # accept equal-shape, or a length-n rhs broadcast over an (m,n) lhs (the emitter handles both)
        if s0 != s1 and not (len(s0) == 2 and s1 == [s0[1]]):
            raise LinalgLowerError(f"elementwise operands {s0}/{s1} are neither equal nor a row broadcast")
        rhs_role = "bias" if len(s1) == 1 else "input"
        out = "out"
        tensors = {
            argname[i0]: {"shape": s0, "dtype": "f32", "role": "input"},
            argname[i1]: {"shape": s1, "dtype": "f32", "role": rhs_role},
            out: {"shape": list(op["results"][0]["shape"]), "dtype": "f32", "role": "output"},
        }
        cmd = {"opcode": "VECTOR_MAP",
               "operands": {"lhs": argname[i0], "rhs": argname[i1], "dst": out},
               "attributes": {"combine": combine}}
        return {"abi_version": "0.1", "target": target, "tensors": tensors,
                "commands": [cmd], "outputs": [out]}

    raise LinalgLowerError(
        f"reference lowering does not yet support this linalg pattern "
        f"({[o['op'] for o in ops]}); supported: single elementwise add/mul")
