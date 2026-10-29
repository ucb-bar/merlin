"""Complete public attention semantics; selected mixed-numeric admission pending.

This is a semantic source, not FlashAttention or an accelerator schedule. QK is
exact i8/i32; softmax and PV use the stated FP32 semantic diagnostic. No missing
probability quantization, floating tolerance or target effect contract is filled
in. Only O and carried KV are live-outs, allowing future legal score elimination.
"""

from __future__ import annotations

import math

import numpy as np

from . import development_sources as D

SCHEMA = "merlin.phase0.public_attention_semantic_source.v1"
CONTRACT = "signed_i8_i32_qk_then_f32_scaled_masked_softmax_pv_semantic_diagnostic"


def declaration(name, *, heads, kv, width, prefix, queries, mask="causal", value_width=None):
    return {
        "schema": SCHEMA,
        "scope": "public_development",
        "name": name,
        "pattern": "complete_attention_semantics",
        "geometry": dict(
            query_heads=heads,
            kv_heads=kv,
            head_dim=width,
            value_dim=width if value_width is None else value_width,
            prefix_tokens=prefix,
            query_tokens=queries,
            mask=mask,
        ),
        "arithmetic": {"contract": CONTRACT},
        "max_logical_elements": 2**32,
    }


def describe(d):
    if (
        not isinstance(d, dict)
        or set(d) != {"schema", "scope", "name", "pattern", "geometry", "arithmetic", "max_logical_elements"}
        or d["schema"] != SCHEMA
        or d["scope"] != "public_development"
        or d["pattern"] != "complete_attention_semantics"
        or not D.workload_roster._label(d["name"])
        or d["arithmetic"] != {"contract": CONTRACT}
    ):
        raise ValueError("select an explicit public complete-attention semantic diagnostic")
    g = d["geometry"]
    if (
        not isinstance(g, dict)
        or set(g) != {"query_heads", "kv_heads", "head_dim", "value_dim", "prefix_tokens", "query_tokens", "mask"}
        or any(
            type(g[k]) is not int or not 1 <= g[k] <= 65536
            for k in ("query_heads", "kv_heads", "head_dim", "value_dim")
        )
        or not 1 <= g["query_heads"] <= 32
        or g["query_heads"] % g["kv_heads"]
        or type(g["prefix_tokens"]) is not int
        or not 0 <= g["prefix_tokens"] <= 65536
        or not isinstance(g["query_tokens"], list)
        or not 1 <= len(g["query_tokens"]) <= 4
        or any(type(q) is not int or not 1 <= q <= 65536 for q in g["query_tokens"])
        or g["mask"] not in {"causal", "explicit_boolean", "none"}
        or type(d["max_logical_elements"]) is not int
        or not 1 <= d["max_logical_elements"] <= 2**32
    ):
        raise ValueError("complete attention needs bounded GQA, mask and carried-KV geometry")
    if g["head_dim"] * 128**2 > 2**31 - 1:
        raise ValueError("exact QK can overflow i32")
    heads, kv, width, value = (g[k] for k in ("query_heads", "kv_heads", "head_dim", "value_dim"))
    length, work_i, work_f, scores = g["prefix_tokens"], 0, 0, 0

    def tensor(name, shape, dtype="int8"):
        return dict(name=name, shape=shape, dtype=dtype)

    inputs = [tensor("prefix_k", [kv, length, width]), tensor("prefix_v", [kv, length, value])] if length else []
    outputs = []
    for step, q in enumerate(g["query_tokens"]):
        length += q
        inputs += [
            tensor(f"q{step}", [heads, q, width]),
            tensor(f"k{step}", [kv, q, width]),
            tensor(f"v{step}", [kv, q, value]),
        ]
        if g["mask"] == "explicit_boolean":
            inputs.append(tensor(f"allowed{step}", [q, length], "bool"))
        outputs.append(tensor(f"attention{step}", [heads, q, value], "float32"))
        scores += heads * q * length
        work_i += heads * q * length * width
        work_f += heads * q * length * value
    outputs += [tensor("final_k", [kv, length, width]), tensor("final_v", [kv, length, value])]
    elements = sum(math.prod(row["shape"]) for row in inputs + outputs)
    if elements > d["max_logical_elements"]:
        raise ValueError("complete attention ABI exceeds its finite source ceiling")
    return dict(
        declaration=d,
        declaration_sha256=D._digest(d),
        inputs=inputs,
        outputs=outputs,
        logical_elements=elements,
        scalar_products=work_i + work_f,
        integer_scalar_products=work_i,
        float_scalar_products=work_f,
        contractions=heads * len(g["query_tokens"]),
        logical_contractions=2 * heads * len(g["query_tokens"]),
        pending_numeric_policy={
            "compare": "per_output",
            "outputs": {
                row["name"]: (
                    {"compare": "UNKNOWN", "dtype": "float32", "reason": "selected float/PV policy unavailable"}
                    if row["dtype"] == "float32"
                    else {"compare": "exact_int", "dtype": row["dtype"]}
                )
                for row in outputs
            },
        },
        pending_contract=[
            "selected score scaling and floating equivalence policy",
            "selected probability/value-to-device conversion; no quantization invented",
            "target softmax/PV execution and completion contracts",
        ],
        score_pressure={
            "logical_score_elements": scores,
            "materialized_fp32_score_bytes": scores * 4,
            "intermediate_scores_required_liveouts": False,
            "physical_peak_bytes": "UNKNOWN",
        },
        optimization_mechanisms=[
            "score_materialization_elimination",
            "online_softmax_if_numerically_legal",
            "QK_PV_tiling",
            "KV_reuse",
            "intermediate_lifetime",
            "GQA_layout",
        ],
        status="semantic_source_only_selected_attention_contract_pending",
        selected_target_attention_contract_sha256=None,
        causal_rule="key_index <= carried_prefix_length + query_index_in_current_call",
    )


def render(d):
    info, g = describe(d), d["geometry"]
    names = [row["name"] for row in info["inputs"]]
    lines = [
        '"""Public complete-attention semantics; no selected numerical or device admission."""',
        "import math",
        "import torch",
        f"# Declaration: {info['declaration_sha256']}",
        "class Program(torch.nn.Module):",
        "    def forward(self," + ",".join(names) + "):",
    ]
    if g["prefix_tokens"]:
        lines.append("        cache_k, cache_v = prefix_k, prefix_v")
    length = g["prefix_tokens"]
    for step, q in enumerate(g["query_tokens"]):
        before, length = length, length + q
        if before:
            lines += [
                f"        cache_k = torch.cat((cache_k,k{step}),dim=1)",
                f"        cache_v = torch.cat((cache_v,v{step}),dim=1)",
            ]
        else:
            lines.append(f"        cache_k, cache_v = k{step}, v{step}")
        if g["mask"] == "causal":
            lines.append(
                f"        allowed{step} = torch.arange({length})[None,:] <= ({before} + torch.arange({q})[:,None])"
            )
        for head in range(g["query_heads"]):
            kv = head // (g["query_heads"] // g["kv_heads"])
            suffix = f"{step}_{head}"
            lines += [
                f"        score_{suffix} = torch._int_mm(q{step}[{head}],cache_k[{kv}].transpose(0,1).contiguous())",
                f"        scaled_{suffix} = score_{suffix}.to(torch.float32) / math.sqrt({g['head_dim']})",
            ]
            if g["mask"] != "none":
                lines.append(
                    f"        scaled_{suffix} = torch.where(allowed{step},scaled_{suffix},"
                    "torch.finfo(torch.float32).min)"
                )
            lines += [
                f"        probability_{suffix} = torch.softmax(scaled_{suffix},dim=-1)",
                f"        out_{suffix} = probability_{suffix} @ cache_v[{kv}].to(torch.float32)",
            ]
        lines.append(
            f"        attention{step} = torch.stack(("
            + ",".join(f"out_{step}_{h}" for h in range(g["query_heads"]))
            + ",),dim=0)"
        )
    lines += [
        "        final_k, final_v = cache_k, cache_v",
        "        return (" + ",".join(row["name"] for row in info["outputs"]) + ",)",
        "def get_model_and_inputs():",
    ]
    for ordinal, row in enumerate(info["inputs"]):
        n = math.prod(row["shape"])
        expression = (
            f"(((torch.arange({n},dtype=torch.int64)*{1 + 2 * ordinal % 16}+{ordinal})%17)-8)"
            f".to(torch.int8).reshape({row['shape']!r})"
        )
        if row["dtype"] == "bool":
            expression = (
                f"((torch.arange({n}).reshape({row['shape']!r})%5)!=0) | (torch.arange({row['shape'][1]})[None,:]==0)"
            )
        lines.append(f"    {row['name']} = {expression}")
    lines.append("    return Program().eval(), (" + ",".join(names) + ",)")
    return "\n".join(lines) + "\n"


def inputs(d):
    values = []
    for ordinal, row in enumerate(describe(d)["inputs"]):
        n = math.prod(row["shape"])
        value = ((np.arange(n, dtype=np.int64) * (1 + 2 * ordinal % 16) + ordinal) % 17) - 8
        value = value.astype(np.int8).reshape(row["shape"])
        if row["dtype"] == "bool":
            value = ((np.arange(n).reshape(row["shape"]) % 5) != 0) | (np.arange(row["shape"][1])[None, :] == 0)
        values.append(value)
    return values


def evaluate(d, values):
    info, g = describe(d), d["geometry"]
    if len(values) != len(info["inputs"]):
        raise ValueError("attention reference requires the complete independent input roster")
    if any(
        not isinstance(value, np.ndarray) or list(value.shape) != row["shape"] or value.dtype != np.dtype(row["dtype"])
        for row, value in zip(info["inputs"], values, strict=True)
    ):
        raise ValueError("attention reference input shape or storage differs from its exact source ABI")
    source = dict(zip((row["name"] for row in info["inputs"]), values, strict=True))
    kv, width, value = (g[k] for k in ("kv_heads", "head_dim", "value_dim"))
    cache_k = source["prefix_k"].copy() if g["prefix_tokens"] else np.empty((kv, 0, width), dtype=np.int8)
    cache_v = source["prefix_v"].copy() if g["prefix_tokens"] else np.empty((kv, 0, value), dtype=np.int8)
    outputs = []
    for step, q in enumerate(g["query_tokens"]):
        before = cache_k.shape[1]
        cache_k = np.concatenate((cache_k, source[f"k{step}"]), axis=1)
        cache_v = np.concatenate((cache_v, source[f"v{step}"]), axis=1)
        length = cache_k.shape[1]
        out = []
        for head in range(g["query_heads"]):
            kv = head // (g["query_heads"] // g["kv_heads"])
            score = source[f"q{step}"][head].astype(np.int64) @ cache_k[kv].astype(np.int64).T
            scaled = score.astype(np.float32) / np.float32(math.sqrt(g["head_dim"]))
            if g["mask"] != "none":
                allowed = (
                    np.arange(length)[None, :] <= before + np.arange(q)[:, None]
                    if g["mask"] == "causal"
                    else source[f"allowed{step}"]
                )
                scaled = np.where(allowed, scaled, np.finfo(np.float32).min)
            exponent = np.exp(scaled - scaled.max(-1, keepdims=True), dtype=np.float32)
            probability = exponent / exponent.sum(-1, keepdims=True, dtype=np.float32)
            out.append(probability @ cache_v[kv].astype(np.float32))
        outputs.append(np.stack(out, axis=0))
    return outputs + [cache_k, cache_v]
