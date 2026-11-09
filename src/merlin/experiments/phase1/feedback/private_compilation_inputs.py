"""Recheck producer-owned compilation inputs at the private linked-image gate."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCOPE = "completed explicit compiler/link inputs rehashed; transitive toolchain closure and symbol suppliers unproved"


def verify(output: Mapping[str, Any], elf: Path, *, required: bool) -> dict[str, Any]:
    from merlin.llvmlower.compilation_recipe import FILENAME, verify_completed_recipe

    binding = output.get("compilation_recipe")
    if binding is None and not required:
        return {"status": "unavailable_in_prebuilt_receipt", "scope": "diagnostic only; command inputs unverified"}
    if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
        raise ValueError("compiled program has no producer-bound compilation recipe")
    expected = elf.parent / FILENAME
    if binding.get("path") != str(expected):
        raise ValueError("compilation recipe is not the selected linked build's record")
    record = verify_completed_recipe(expected, executable=elf, expected_recipe_sha256=binding.get("sha256"))
    link = record["commands"][-1]
    return {
        "status": "completed_compilation_inputs_verified",
        "recipe_path": str(expected),
        "recipe_sha256": binding["sha256"],
        "elf_path": str(elf),
        "elf_sha256": record["executable"]["sha256"],
        "n_commands": len(record["commands"]),
        "link_executable": link["executable"],
        "link_inputs": link["inputs"],
        "scope": SCOPE,
    }


def complete(entry: Mapping[str, Any]) -> bool:
    binding = entry.get("compilation_recipe")
    if not isinstance(binding, Mapping):
        return False
    if (
        binding.get("status") != "completed_compilation_inputs_verified"
        or binding.get("scope") != SCOPE
        or binding.get("elf_sha256") != entry.get("elf_sha256")
        or any(
            not isinstance(binding.get(key), str) or not Path(binding[key]).is_absolute()
            for key in ("recipe_path", "elf_path")
        )
        or not isinstance(binding.get("recipe_sha256"), str)
        or len(binding["recipe_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in binding["recipe_sha256"])
        or type(binding.get("n_commands")) is not int
        or binding["n_commands"] < 1
        or not isinstance(binding.get("link_inputs"), list)
        or not binding["link_inputs"]
    ):
        return False
    identities = [binding.get("link_executable"), *binding["link_inputs"]]
    if not all(
        isinstance(identity, Mapping)
        and isinstance(identity.get("path"), str)
        and Path(identity["path"]).is_absolute()
        and isinstance(identity.get("sha256"), str)
        and len(identity["sha256"]) == 64
        and all(character in "0123456789abcdef" for character in identity["sha256"])
        and type(identity.get("bytes")) is int
        and identity["bytes"] >= 0
        for identity in identities
    ):
        return False
    try:
        fresh = verify(
            {"compilation_recipe": {"path": binding["recipe_path"], "sha256": binding["recipe_sha256"]}},
            Path(binding["elf_path"]),
            required=True,
        )
    except (OSError, ValueError):
        return False
    return fresh == dict(binding)
