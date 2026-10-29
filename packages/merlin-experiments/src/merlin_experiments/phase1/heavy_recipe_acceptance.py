"""Preflight check ``heavy-pass-recipes``: every heavy-pass recipe runs against the staged library.

Each script of the ``heavy-pass-recipes`` author skill runs in the author's import roots (the staged
``submission/compiler_library`` first, the workspace last; no harness module), on the workspace's RTL
facts or the scripts' labelled synthetic body. A script exits nonzero when an emitted program does not
reproduce its source result, so a zero exit is the numerical check; this check then reads the printed
JSON for the property each recipe exists to show, and refuses any ``merlin`` module loaded from
outside the staged library.
"""

from __future__ import annotations

SKILL = "heavy-pass-recipes"
NAME = "heavy-pass-recipes"


def _dense_readout(out: dict) -> str | None:
    variants = out.get("variants") or {}
    if not out.get("accumulator_bound"):
        return "the derived shape does not overflow the accumulator"
    overlap, ready, base = (variants.get(k) or {} for k in ("readout_overlap", "ready_readout", "segmented"))
    if not overlap.get("disjoint_readout_epochs") or not ready.get("causal_ready_epochs"):
        return "readout overlap or ready readout realized no overlapped epoch"
    if not overlap.get("epoch_drains", 0) < base.get("epoch_drains", 0):
        return "readout overlap did not remove an epoch drain"
    return None


def _long_k(out: dict) -> str | None:
    segmented = out.get("segmented")
    if not str(out.get("retained_full_k", "")).startswith("declined"):
        return "the derived K does not exceed the full-K panel-pair bound"
    if not isinstance(segmented, dict) or not segmented.get("overlapped_compute_pairs"):
        return f"the segmented pipeline did not realize overlapped compute: {segmented}"
    return None


def _conv(out: dict) -> str | None:
    routes = out.get("routes") or {}
    unique = {name: [c for c in route.get("candidates", ()) if c.get("counts")] for name, route in routes.items()}
    if set(unique) != {"border_rows", "halo_source"} or not all(unique.values()):
        return f"convolution routes realized no candidate: {sorted(unique)}"
    if any(c["counts"]["input_one_row"] or c["counts"]["input_zero_rows"] for c in unique["halo_source"]):
        return "the halo route still issues one-row or zero transfers"
    gather = out.get("one_row_gather_input_transfers") or 0
    if any(not c["counts"]["input_multi_row"] or c["input_transfers"] >= gather for c in unique["border_rows"]):
        return "a convolution candidate issues no fewer input transfers than a per-pixel gather"
    weights = {c["resident_weights"]: c["counts"]["weight_loads"] for c in unique["border_rows"]}
    if not weights.get(True, 0) < weights.get(False, 0):
        return "resident weights did not reduce weight loads"
    return None


def _conv_epochs(out: dict) -> str | None:
    routes = {(row.get("route"), row.get("compute_pairs")): row for row in out.get("routes") or ()}
    if not any(route == "synchronous" for route, _ in routes):
        return "the synchronous convolution route was not realized"
    if out.get("epoch_law") != "admitted":
        return None  # an UNKNOWN ordering flag keeps the drains; the flag table says which
    base = next((row for (route, _), row in routes.items() if route == "drain_per_compute"), None)
    epochs = [row for (route, _), row in routes.items() if route == "epochs"]
    if base is None or not epochs or min(row["drains"] for row in epochs) >= base["drains"]:
        return "admitted compute epochs did not remove a drain"
    return None


def _dense_epochs(out: dict) -> str | None:
    cases = out.get("cases") or ()
    if not cases or any(set(c.get("routes") or {}) != {"drained", "epochs"} for c in cases):
        return "the drained and epoch routes were not both realized"
    if out.get("epoch_law") != "admitted":
        return None  # an UNKNOWN ordering flag keeps its drains; the flag table says which
    if any(c["comparison"]["drains_after"] >= c["comparison"]["drains_before"] for c in cases):
        return "an admitted epoch law removed no drain"
    return None


def _spatial(out: dict) -> str | None:
    storage, effects = out.get("storage") or {}, out.get("effects") or {}
    if (out.get("discovery") or {}).get("selected_row") != "bounded_spatial_packing_and_rolling":
        return "the composition catalog row was not discovered"
    if storage.get("full_intermediate_image_bytes") != 0 or not effects.get("stores"):
        return "the chain materialized an intermediate image or stored nothing"
    return None


def _policy(out: dict) -> str | None:
    statuses = {row["class"]: row["status"] for row in out.get("policy") or ()}
    if sorted(statuses.values()) != ["empirically_supported_declared_panel", "incumbent_retained"]:
        return f"the demo policy did not show one switch and one retained incumbent: {statuses}"
    runs = out.get("executions") or ()
    if not any(r["variant"] != "compact" for r in runs):
        return "no shape executed the selected library pipeline"
    if not any(
        r["variant"] == "compact" and ["segmented", "unsupported"] in [list(d) for d in r["decisions"]] for r in runs
    ):
        return "no shape showed the legality fallback to the incumbent"
    return None


RECIPES = (
    ("dense_readout_recipe.py", _dense_readout),
    ("long_k_recipe.py", _long_k),
    ("conv_window_recipe.py", _conv),
    ("conv_epoch_recipe.py", _conv_epochs),
    ("dense_epoch_recipe.py", _dense_epochs, ["--small"]),
    ("spatial_chain_recipe.py", _spatial),
    ("library_pipeline_policy.py", _policy),
)


def check_heavy_pass_recipes(ctx):
    from .author_acceptance import FAIL, PASS, Check, _run_skill_script, _stdout_json, _tail

    detail, foreign, loaded = {}, set(), 0
    for script, verify, *args in RECIPES:
        proc, result = _run_skill_script(ctx, SKILL, script, args[0] if args else [])
        out = _stdout_json(proc) or {}
        detail[script] = {"child": result, "facts": out.get("facts")}
        if result is None or result["code"] != 0:
            return Check(NAME, FAIL, f"{script} did not verify: {_tail(proc)}", detail)
        problem = verify(out)
        if problem:
            return Check(NAME, FAIL, f"{script}: {problem}", detail)
        foreign.update(result["foreign"])
        loaded = max(loaded, result["loaded"])
    if foreign:
        return Check(NAME, FAIL, f"merlin modules came from outside the staged library: {sorted(foreign)[:4]}", detail)
    facts = sorted({str(row["facts"]) for row in detail.values()})
    return Check(
        NAME,
        PASS,
        f"{len(RECIPES)} recipes verified on host integers from {facts} capacities against "
        f"the staged library (up to {loaded} modules)",
        detail,
    )
