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
        combine = _combine_from_body(op["body_ops"])
        if len(ins) != 2:
            raise LinalgLowerError(f"reference lowering supports a two-operand elementwise map (got {len(ins)})")

        # A <op> c : one arg operand and one compile-time scalar (a splat of an arith.constant) -> a
        # scalar map with the constant baked into the kernel (per-tensor scale / scalar bias).
        arg_ins = [s for s in ins if s["source"][0] == "arg"]
        const_ins = [s for s in ins if "const_value" in s]
        if len(arg_ins) == 1 and len(const_ins) == 1:
            ai = arg_ins[0]["source"][1]
            out = "out"
            tensors = {argname[ai]: {"shape": argshape[ai], "dtype": "f32", "role": "input"},
                       out: {"shape": list(op["results"][0]["shape"]), "dtype": "f32", "role": "output"}}
            cmd = {"opcode": "VECTOR_MAP", "operands": {"lhs": argname[ai], "dst": out},
                   "attributes": {"combine": combine, "scalar": const_ins[0]["const_value"]}}
            # the emitted kernel takes (lhs, dst) — the func-arg order the linalg_positional harness feeds
            return {"abi_version": "0.1", "target": target, "tensors": tensors,
                    "commands": [cmd], "outputs": [out],
                    "interface": "linalg_positional", "arg_order": [argname[ai], out]}

        if any(s["source"][0] != "arg" for s in ins):
            raise LinalgLowerError(
                "reference lowering supports a two-arg elementwise map or arg-times-constant only "
                f"(got {[s['source'] for s in ins]})")
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
        # the emitted kernel takes (lhs, rhs, dst) — the func-arg order the linalg_positional harness feeds
        return {"abi_version": "0.1", "target": target, "tensors": tensors,
                "commands": [cmd], "outputs": [out],
                "interface": "linalg_positional", "arg_order": [argname[i0], argname[i1], out]}

    # --- two chained 2-D matmuls: A@W1 then (that result)@W2 -----------------------------------------
    if (len(ops) == 2 and ops[0].get("family") == "contraction" and ops[1].get("family") == "contraction"):
        mm0, mm1 = ops[0], ops[1]
        for mm in (mm0, mm1):
            ext = mm.get("extents", {})
            if "batch" in ext or not {"m", "k", "n"} <= ext.keys():
                raise LinalgLowerError(f"chained matmul supports plain 2-D matmuls only (extents {ext})")
        s0, s1 = [s["source"] for s in mm0["ins"][:2]], [s["source"] for s in mm1["ins"][:2]]
        if any(s[0] != "arg" for s in s0):
            raise LinalgLowerError("first matmul operands must be func args")
        if ("op", mm0["id"]) not in s1:
            raise LinalgLowerError("the second matmul does not consume the first matmul result")
        w2_src = [s for s in s1 if s != ("op", mm0["id"])]
        if len(w2_src) != 1 or w2_src[0][0] != "arg":
            raise LinalgLowerError("the second matmul's weight must be a func arg")
        act_i, w1_i, w2_i = s0[0][1], s0[1][1], w2_src[0][1]
        out = "out"
        tensors = {
            argname[act_i]: {"shape": argshape[act_i], "dtype": "f32", "role": "input"},
            argname[w1_i]: {"shape": argshape[w1_i], "dtype": "f32", "role": "weight"},
            argname[w2_i]: {"shape": argshape[w2_i], "dtype": "f32", "role": "weight"},
            out: {"shape": list(mm1["results"][0]["shape"]), "dtype": "f32", "role": "output"},
        }
        commands = [
            {"opcode": "RES_PACK", "operands": {"src": argname[w1_i], "dst": "W1p"},
             "attributes": {"layout": "packed_rhs"}},
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": argname[act_i], "rhs": "W1p", "dst": "acc0"}},
            {"opcode": "COMMIT", "operands": {"src": "acc0", "dst": "H"},
             "attributes": {"epilogue": [], "output_dtype": "f32"}},
            {"opcode": "RES_PACK", "operands": {"src": argname[w2_i], "dst": "W2p"},
             "attributes": {"layout": "packed_rhs"}},
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "H", "rhs": "W2p", "dst": "acc1"}},
            {"opcode": "COMMIT", "operands": {"src": "acc1", "dst": out},
             "attributes": {"epilogue": [], "output_dtype": "f32"}},
        ]
        return {"abi_version": "0.1", "target": target, "tensors": tensors,
                "commands": commands, "outputs": [out]}

    # --- a single batched matmul (batch,m,k)@(batch,k,n) ---------------------------------------------
    if len(ops) == 1 and ops[0].get("family") == "contraction" and "batch" in ops[0].get("extents", {}):
        mm = ops[0]
        ins = mm["ins"]
        if len(ins) < 2 or ins[0]["source"][0] != "arg" or ins[1]["source"][0] != "arg":
            raise LinalgLowerError("batched matmul operands must both be func args")
        if len(ins[0]["shape"]) != 3 or len(ins[1]["shape"]) != 3:
            raise LinalgLowerError(f"batched matmul needs 3-D operands (got {ins[0]['shape']}/{ins[1]['shape']})")
        a_i, w_i = ins[0]["source"][1], ins[1]["source"][1]
        out = "out"
        tensors = {
            argname[a_i]: {"shape": argshape[a_i], "dtype": "f32", "role": "input"},
            argname[w_i]: {"shape": argshape[w_i], "dtype": "f32", "role": "weight"},
            out: {"shape": list(mm["results"][0]["shape"]), "dtype": "f32", "role": "output"},
        }
        cmd = {"opcode": "BATCHED_MATMUL",
               "operands": {"a": argname[a_i], "w": argname[w_i], "dst": out}}
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

    # --- a decomposed row LayerNorm, recognized by the layer_norm provenance -------------------------
    if any(o.get("op") == "layer_norm" for o in ops):
        # src = the sole 2-D func arg; gamma/beta = the 1-D args used in the final mul / add;
        # eps = the constant added just before the rsqrt.
        arg2d = [a["index"] for a in args if len(a["shape"]) == 2]
        arg1d = [a["index"] for a in args if len(a["shape"]) == 1]
        if len(arg2d) != 1 or len(arg1d) != 2:
            raise LinalgLowerError("layernorm expects one 2-D input and two 1-D (gamma, beta) args")
        x_i = arg2d[0]
        gamma_i = beta_i = None
        for o in ops:
            for inp in o["ins"]:
                if inp["source"][0] == "arg" and inp["source"][1] in arg1d:
                    if any(b.endswith("mulf") for b in o["body_ops"]):
                        gamma_i = inp["source"][1]
                    elif any(b.endswith("addf") for b in o["body_ops"]):
                        beta_i = inp["source"][1]
        if gamma_i is None or beta_i is None:
            raise LinalgLowerError("could not identify the layernorm gamma (mul) and beta (add) args")
        eps = 1e-5
        for o in ops:
            if any("rsqrt" in b for b in o["body_ops"]):
                src = o["ins"][0]["source"] if o["ins"] else None
                if src and src[0] == "op":
                    for inp in ops[src[1]]["ins"]:
                        if "const_value" in inp:
                            eps = inp["const_value"]
        out = "out"
        result_shape = list(ops[-1]["results"][0]["shape"]) if ops[-1].get("results") else argshape[x_i]
        tensors = {
            argname[x_i]: {"shape": argshape[x_i], "dtype": "f32", "role": "input"},
            argname[gamma_i]: {"shape": argshape[gamma_i], "dtype": "f32", "role": "weight"},
            argname[beta_i]: {"shape": argshape[beta_i], "dtype": "f32", "role": "bias"},
            out: {"shape": result_shape, "dtype": "f32", "role": "output"},
        }
        cmd = {"opcode": "LAYERNORM",
               "operands": {"src": argname[x_i], "gamma": argname[gamma_i],
                            "beta": argname[beta_i], "dst": out},
               "attributes": {"eps": eps}}
        return {"abi_version": "0.1", "target": target, "tensors": tensors,
                "commands": [cmd], "outputs": [out]}

    raise LinalgLowerError(
        f"reference lowering does not yet support this linalg pattern "
        f"({[o['op'] for o in ops]}); supported: single elementwise add/mul, matmul (+bias/chain/batch), "
        f"layernorm")
