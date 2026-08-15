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

    # --- a single 2-D matmul, optionally followed by a row-broadcast bias add -------------------------
    if ops[0].get("family") == "contraction":
        mm = ops[0]
        ext = mm.get("extents", {})
        if "batch" in ext or "k_rhs" in ext or not {"m", "k", "n"} <= ext.keys():
            raise LinalgLowerError(f"reference lowering supports a plain 2-D matmul only (extents {ext})")
        m_ins = mm["ins"]
        if len(m_ins) < 2 or m_ins[0]["source"][0] != "arg" or m_ins[1]["source"][0] != "arg":
            raise LinalgLowerError("matmul operands must both be func args (no fused producer)")
        act_i, w_i = m_ins[0]["source"][1], m_ins[1]["source"][1]

        bias_i = None
        epilogue: list[str] = []
        if len(ops) == 2:
            add = ops[1]
            if add["family"] not in ("elementwise", "elementwise_map") or "add" not in _combine_from_body(add["body_ops"]):
                raise LinalgLowerError(f"the op after a matmul is not a bias add ({add['op']})")
            ains = add["ins"]
            srcs = [s["source"] for s in ains]
            if ("op", mm["id"]) not in srcs:
                raise LinalgLowerError("the trailing add does not consume the matmul result")
            other = [s for s in srcs if s != ("op", mm["id"])]
            if len(other) != 1 or other[0][0] != "arg":
                raise LinalgLowerError("bias add must combine the matmul result with a func-arg bias")
            bias_i = other[0][1]
            epilogue = ["bias_add"]
        elif len(ops) != 1:
            raise LinalgLowerError(f"unsupported matmul composition ({[o['op'] for o in ops]})")

        out = "out"
        tensors = {
            argname[act_i]: {"shape": argshape[act_i], "dtype": "f32", "role": "input"},
            argname[w_i]: {"shape": argshape[w_i], "dtype": "f32", "role": "weight"},
            out: {"shape": list(ops[-1]["results"][0]["shape"]), "dtype": "f32", "role": "output"},
        }
        commit_ops = {"src": "acc", "dst": out}
        if bias_i is not None:
            tensors[argname[bias_i]] = {"shape": argshape[bias_i], "dtype": "f32", "role": "bias"}
            commit_ops["bias"] = argname[bias_i]
        commands = [
            {"opcode": "RES_PACK", "operands": {"src": argname[w_i], "dst": "Wp"},
             "attributes": {"layout": "packed_rhs"}},
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": argname[act_i], "rhs": "Wp", "dst": "acc"}},
            {"opcode": "COMMIT", "operands": commit_ops,
             "attributes": {"epilogue": epilogue, "output_dtype": "f32"}},
        ]
        return {"abi_version": "0.1", "target": target, "tensors": tensors,
                "commands": commands, "outputs": [out]}

    raise LinalgLowerError(
        f"reference lowering does not yet support this linalg pattern "
        f"({[o['op'] for o in ops]}); supported: single elementwise add/mul, single 2-D matmul (+bias)")
