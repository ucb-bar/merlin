"""Chipyard hardware-recipe loader.

Recipes live at `build_tools/hardware/*.yaml`. Each recipe describes a
chipyard checkout state (branch, SHA, submodules) plus optional firesim
metadata. `./merlin chipyard <action>` typically takes `--recipe <name>`
matching a yaml file's stem.
"""

from __future__ import annotations

import yaml

import utils

HARDWARE_DIR = utils.REPO_ROOT / "build_tools" / "hardware"


def list_recipes() -> list[dict]:
    recipes = []
    for f in sorted(HARDWARE_DIR.glob("*.yaml")):
        with f.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
            data["_file"] = str(f)
            recipes.append(data)
    return recipes


def load_recipe(name: str) -> dict | None:
    for f in HARDWARE_DIR.glob("*.yaml"):
        with f.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
            data["_file"] = str(f)
            if f.stem == name or data.get("name") == name:
                return data
    return None


def require_recipe(name: str) -> dict | None:
    recipe = load_recipe(name)
    if not recipe:
        utils.eprint(f"Recipe not found: {name}")
        utils.eprint(f"Available: {', '.join(r['name'] for r in list_recipes())}")
    return recipe


def _recipe_mode(recipe: dict) -> str:
    return recipe.get("mode", "bare-metal")
