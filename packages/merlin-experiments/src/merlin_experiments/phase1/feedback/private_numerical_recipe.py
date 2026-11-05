"""Recompute an explicitly selected numerical policy from original Phase 0 recipes.

This records transformation-input correspondence only. It supplies no capture,
source-execution, numerical-accuracy or hardware qualification.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import PurePosixPath

from merlin.common.digest import is_sha256
from merlin.targetgen import quant_methodology as M
from merlin.targetgen import quant_recipe as Q
from merlin.targetgen import quant_recipe_policy as P


def recompute(selected: bytes, *, model: str, target: str, index, artifacts, methodology, output_path) -> dict:
    """Require exact canonical selected bytes derived from an indexed original model variant."""
    body = json.loads(selected)
    selection = body.get("numerical_selection") if isinstance(body, Mapping) else None
    identities = {"source_recipe_sha256", "source_methodology_sha256", "source_output_path_sha256"}
    if (
        not isinstance(model, str)
        or not model
        or not isinstance(selection, Mapping)
        or set(selection) != {"schema", "policy", *identities}
        or selection.get("schema") != "numerical_recipe_selection_v1"
        or not all(is_sha256(selection.get(key)) for key in identities)
        or body.get("target") != target
        or body.get("recipe_sha256") != Q.digest(body)
    ):
        raise ValueError("selected numerical recipe lacks its closed content-bound policy selection")
    policy, method = P.validate(selection["policy"]), M.validate(methodology)
    method_sha = M.digest({key: value for key, value in method.items() if key != "model_variants"})
    path_sha = M.digest(output_path)
    if selection["source_methodology_sha256"] != method_sha or selection["source_output_path_sha256"] != path_sha:
        raise ValueError("selected numerical recipe differs from current methodology or output path")
    variants = index.get("model_variants")
    if not isinstance(variants, list):
        raise ValueError("numerical recipe derivation lacks original indexed model variants")
    matches = []
    for row in variants:
        if not isinstance(row, Mapping) or row.get("model") != model:
            continue
        name = row.get("path")
        path = PurePosixPath(name) if isinstance(name, str) else PurePosixPath()
        raw = artifacts.get(name) if isinstance(name, str) else None
        if (
            not name
            or path.is_absolute()
            or ".." in path.parts
            or path.parts[:2] != ("software", "quantization-recipes")
            or not isinstance(raw, bytes)
            or hashlib.sha256(raw).hexdigest() != row.get("sha256")
            or path.stem != row.get("sha256")
        ):
            raise ValueError("original numerical recipe variant lacks its indexed byte identity")
        source = json.loads(raw)
        if (
            not isinstance(source, Mapping)
            or source.get("schema") != Q.SCHEMA
            or source.get("status") != Q.DERIVED
            or source.get("target") != target
            or source.get("recipe_sha256") != row.get("recipe_sha256")
            or Q.digest(source) != row["recipe_sha256"]
            or (source.get("methodology") or {}).get("model") != model
        ):
            raise ValueError("original numerical recipe variant differs from its semantic identity")
        # Both are existing explicit P.select inputs. The preselected source digest,
        # never a model name or observed result, selects the actual preparation.
        prepared = Q.with_native_output_quantization(source, **policy["native_outputs"])
        for key in ("accumulator_residual", "integer_mean", "projection_input_reuse"):
            if key in policy["integer_extensions"]:
                prepared = Q.with_integer_extension(prepared, key=key, policy=policy["integer_extensions"][key])
        seeds = {source["recipe_sha256"]: ("indexed_recipe", source)}
        seeds.setdefault(prepared["recipe_sha256"], ("native_output_extensions_v1", prepared))
        choice = seeds.get(selection["source_recipe_sha256"])
        if choice is None:
            continue
        preparation, seed = choice
        rebuilt = P.select(seed, method, output_path, policy=policy, model=model)
        exact = (json.dumps(rebuilt, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        if exact != selected:
            raise ValueError("selected numerical recipe differs from the complete recomputed recipe bytes")
        matches.append(
            {
                "schema": "merlin.phase1.numerical_recipe_derivation.v1",
                "indexed_source": {"path": name, "sha256": row["sha256"], "recipe_sha256": row["recipe_sha256"]},
                "model": model,
                "preparation": preparation,
                "policy_sha256": Q.digest(policy),
                **{key: selection[key] for key in identities},
                "selected_file_sha256": hashlib.sha256(selected).hexdigest(),
                "selected_recipe_sha256": rebuilt["recipe_sha256"],
                "scope": "recomputed selected numerical policy; transformation inputs only, no admission",
            }
        )
    if len(matches) != 1:
        raise ValueError("selected numerical recipe has no unique original indexed derivation")
    return matches[0]
