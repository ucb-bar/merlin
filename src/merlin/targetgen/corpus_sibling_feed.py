"""Two sibling contractions over one input whose readouts merge and feed a third contraction.

``X[M,K] @ WG[K,H]`` and ``X[M,K] @ WU[K,H]`` read the SAME activation; each commits through its own
readout into the operand format; the two committed tensors meet in one residual-style add; the sum
feeds ``@ WD[H,N]``. This is the topology of a gated feed-forward block (gate and up projections that
share their input, merged, then projected down), written with the target-neutral interface commands
that already have reviewed lanes: ``matmul``, ``commit`` and ``residual_add``.

Every region is separate and nothing declares how they are placed or fused. A backend may load ``X``
once for both siblings, keep the merged tensor resident for the down projection, or run every region
on its own; the case is timed as one call, so the difference shows up only as cycles. The merge is
the declared scaled add, not a captured model's float gating product, which stays outside the case.
"""

from __future__ import annotations

from typing import Any


def build(entry: dict, binding) -> tuple[dict, str]:
    from merlin.runtime.commandbuffer import residual_add_params

    from . import corpus_spec as CS

    D = binding.tile_dim
    M = int(entry.get("M", entry.get("M_tiles", 1) * D))
    K = int(entry.get("K", entry.get("K_tiles", 1) * D))
    H = int(entry.get("H", entry.get("H_tiles", 1) * D))
    N = int(entry.get("N", entry.get("N_tiles", 1) * D))
    if min(M, K, H, N) < 1:
        raise ValueError(f"{entry.get('name', 'sibling_feed')}: extents must be positive, got {M, K, H, N}")
    x, wg, wu, wd = (entry.get(key, default) for key, default in
                     (("lhs", "X"), ("gate_weight", "WG"), ("up_weight", "WU"), ("down_weight", "WD")))
    gate, up, merged, out = (entry.get(key, default) for key, default in
                             (("gate", "G"), ("up", "U"), ("merged", "Hm"), ("out", "Y0")))
    names = [x, wg, wu, wd, gate, up, merged, out]
    if len(set(names)) != len(names) or not all(isinstance(n, str) and n.isidentifier() for n in names):
        raise ValueError(f"{entry.get('name', 'sibling_feed')}: tensor names must be distinct identifiers")
    idt, adt = binding.cap_dtype(binding.operand_dtype), binding.cap_dtype(binding.accum_dtype)
    midt, madt = binding.mlir_dtype(binding.operand_dtype), binding.mlir_dtype(binding.accum_dtype)
    scales = {}
    for role in ("gate", "up"):
        value = entry.get(f"{role}_acc_scale")
        if type(value) not in (int, float) or isinstance(value, bool) or not value > 0:
            raise ValueError(
                f"{entry.get('name', 'sibling_feed')}: the {role} readout needs an explicit positive "
                f"`{role}_acc_scale`; a readout without one would be applied at an assumed 1.0"
            )
        scales[role] = float(value)
    add_epilogue = list(entry.get("add_epilogue", []))
    lhs_scale, rhs_scale, bound = residual_add_params(
        {"lhs_scale": entry.get("lhs_scale"), "rhs_scale": entry.get("rhs_scale"),
         "bound_lsb": entry.get("bound_lsb"), "epilogue": add_epilogue},
        op=str(entry.get("name", "sibling_feed")),
    )
    attrs: dict[str, Any] = {
        "lhs": x, "gate_weight": wg, "up_weight": wu, "down_weight": wd,
        "gate": gate, "up": up, "merged": merged, "out": out,
        "M": M, "K": K, "H": H, "N": N,
        "gate_acc_scale": scales["gate"], "up_acc_scale": scales["up"],
        "readout_epilogue": ["acc_scale"], "readout_dtype": idt,
        "add_epilogue": add_epilogue, "lhs_scale": lhs_scale, "rhs_scale": rhs_scale, "bound_lsb": bound,
        "output_dtype": adt,
        "semantic": "sibling_contractions_merge_then_contraction",
    }
    if bound != 0:
        # The merged tensor is an operand of the down projection, so any tolerance on the add would be
        # multiplied by the down weights into the output. Only an exact merge keeps the output exact.
        raise ValueError(f"{entry.get('name', 'sibling_feed')}: the merge feeds a contraction and must be exact "
                         f"(bound_lsb 0), got {bound}")
    numeric_policy = CS._numeric_policy(binding, binding.accum_dtype, None)
    cap = {
        "name": entry["name"],
        "kind": entry["kind"],
        "source_role": entry["source_role"],
        "source_reference": entry["source_reference"],
        "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [
            {"name": x, "role": "input", "shape": [M, K], "dtype": idt},
            {"name": wg, "role": "weight", "shape": [K, H], "dtype": idt},
            {"name": wu, "role": "weight", "shape": [K, H], "dtype": idt},
            {"name": wd, "role": "weight", "shape": [H, N], "dtype": idt},
        ],
        "operation": {"op": "sibling_feed", "attributes": attrs},
        "numeric_policy": numeric_policy,
        "expected": {
            "instruction_classes": binding.classes_for(op="matmul", output_dtype=idt, epilogue=["acc_scale"],
                                                       movement=False),
            "modes": dict(entry["modes"]) if "modes" in entry else {"matmul": True, "acc_scale": True},
        },
        "required_oracle_tiers": list(binding.tiers),
        "vcs": "optional",
        "firesim": "optional",
    }
    add_epi = ", ".join(f'"{e}"' for e in add_epilogue)
    L = CS._iface_prelude(binding.target, entry.get("comment", ""))
    L += [
        f'  %{x} = merlin_iface.tensor {{name = "{x}", role = "input"}} : tensor<{M}x{K}x{midt}>',
        f'  %{wg} = merlin_iface.tensor {{name = "{wg}", role = "weight"}} : tensor<{K}x{H}x{midt}>',
        f'  %{wu} = merlin_iface.tensor {{name = "{wu}", role = "weight"}} : tensor<{K}x{H}x{midt}>',
        f'  %{wd} = merlin_iface.tensor {{name = "{wd}", role = "weight"}} : tensor<{H}x{N}x{midt}>',
    ]
    for weight, rows, cols in ((wg, K, H), (wu, K, H), (wd, H, N)):
        L.append(f'  %{weight}_res = merlin_iface.resident_pack %{weight} {{layout = "packed_rhs"}} '
                 f": (tensor<{rows}x{cols}x{midt}>) -> !merlin_iface.resident")
    for role, value, weight in (("gate", gate, wg), ("up", up, wu)):
        L += [
            f"  %{value}_acc = merlin_iface.matmul %{x}, %{weight}_res "
            f": (tensor<{M}x{K}x{midt}>, !merlin_iface.resident) -> !merlin_iface.acc<{madt}>",
            f'  %{value} = merlin_iface.commit %{value}_acc {{name = "{value}", epilogue = ["acc_scale"], '
            f'output_dtype = "{idt}", acc_scale = {scales[role]} : f32}} '
            f": (!merlin_iface.acc<{madt}>) -> tensor<{M}x{H}x{midt}>",
        ]
    L += [
        f'  %{merged} = merlin_iface.residual_add %{gate}, %{up} {{name = "{merged}", '
        f"epilogue = [{add_epi}], lhs_scale = {lhs_scale} : f32, rhs_scale = {rhs_scale} : f32, "
        f'bound_lsb = {bound} : i64, output_dtype = "{idt}"}} '
        f": (tensor<{M}x{H}x{midt}>, tensor<{M}x{H}x{midt}>) -> tensor<{M}x{H}x{midt}>",
        f"  %{out}_acc = merlin_iface.matmul %{merged}, %{wd}_res "
        f": (tensor<{M}x{H}x{midt}>, !merlin_iface.resident) -> !merlin_iface.acc<{madt}>",
        f'  %{out} = merlin_iface.commit %{out}_acc {{name = "{out}", epilogue = [], output_dtype = "{adt}"}} '
        f": (!merlin_iface.acc<{madt}>) -> tensor<{M}x{N}x{madt}>",
    ]
    L += [f"  merlin_iface.evict %{w}_res : (!merlin_iface.resident) -> ()" for w in (wg, wu, wd)]
    L.append("}")
    return cap, "\n".join(L) + "\n"


def merged_operand(capsule: dict, env: dict, *, apply_epilogue, narrow):
    """The down projection's operand: both sibling readouts, merged by the declared add."""
    from merlin.runtime.commandbuffer import residual_add_params

    attrs = capsule["operation"]["attributes"]
    x = env[attrs["lhs"]]
    committed = []
    for role, weight in (("gate", "gate_weight"), ("up", "up_weight")):
        stage = {"epilogue": ["acc_scale"], "acc_scale": attrs[f"{role}_acc_scale"]}
        committed.append(narrow(apply_epilogue(x.matmul(env[attrs[weight]]), stage, env), attrs["readout_dtype"]))
    lhs_scale, rhs_scale, _ = residual_add_params(
        {**attrs, "epilogue": list(attrs.get("add_epilogue") or [])}, op=f"sibling_feed {attrs['out']!r}"
    )
    merged = narrow(committed[0].residual_add_scaled(committed[1], lhs_scale, rhs_scale), attrs["readout_dtype"])
    return merged.relu() if "relu" in (attrs.get("add_epilogue") or ()) else merged


def golden(capsule: dict, env: dict, *, apply_epilogue, narrow) -> dict[str, list]:
    """The case's outputs from the same readout, add and contraction arithmetic the other ops use."""
    attrs = capsule["operation"]["attributes"]
    merged = merged_operand(capsule, env, apply_epilogue=apply_epilogue, narrow=narrow)
    return {attrs["out"]: merged.matmul(env[attrs["down_weight"]]).to_list()}
