"""Select exactly one verified software-scoped capture recipe from frozen evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _selected_capture_recipe(evidence_root: Path, *, target: str, operand_dtype: str, accumulator_dtype: str) -> dict:
    """Read exactly one frozen SW-scoped recipe for a live integer model capture.

    Hardware-only recipe derivation is intentionally not a fallback: it can quantize families
    the selected software spec explicitly left on the host.
    """
    from merlin.common import quant_formats
    from merlin.targetgen import quant_recipe

    index_path = evidence_root / "software" / "quantization-recipes.json"
    index = json.loads(index_path.read_bytes())
    if index.get("schema") != "merlin.phase0.capture_recipes.v1" or index.get("target") != target:
        raise ValueError("frozen capture-recipe index has wrong schema or target")
    wanted_operand = quant_formats.get(operand_dtype).name
    wanted_accumulator = quant_formats.get(accumulator_dtype).name
    matches = []
    for row in index.get("recipes") or []:
        rel = Path(str(row.get("path") or ""))
        if rel.is_absolute() or not rel.parts or ".." in rel.parts or rel.parts[0] != "software":
            raise ValueError("frozen capture recipe has an unsafe relative path")
        path = evidence_root / rel
        if path.is_symlink() or not path.is_file():
            raise ValueError("frozen capture recipe is absent or symlinked")
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != row.get("sha256") or path.stem != actual:
            raise ValueError("frozen capture recipe bytes differ from their index")
        recipe = json.loads(raw)
        if recipe.get("target") != target or recipe.get("status") != quant_recipe.DERIVED:
            raise ValueError("frozen capture recipe has wrong target or unresolved status")
        digest = quant_recipe.digest(recipe)
        if recipe.get("recipe_sha256") != digest or row.get("recipe_sha256") != digest:
            raise ValueError("frozen capture recipe semantic digest differs from its index")
        if (
            quant_formats.get(recipe["weight"]["dtype"]).name == wanted_operand
            and quant_formats.get(recipe["activation"]["dtype"]).name == wanted_operand
            and quant_formats.get(recipe["accumulator_dtype"]).name == wanted_accumulator
        ):
            matches.append(recipe)
    if len(matches) != 1:
        raise ValueError(
            f"frozen evidence has {len(matches)} unambiguous SW-scoped recipes for "
            f"{wanted_operand}/{wanted_accumulator}; refusing a hardware-only default"
        )
    return matches[0]
