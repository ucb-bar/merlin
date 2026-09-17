"""The target roster, DERIVED from the registries that declare a target -- never a literal list.

The no-target-name gate used to hunt a hand-written set of seven names, so every target registered
after that set was written was invisible to it: a literal naming one passed the gate by construction.
The roster is now the union of every place the repository DECLARES a target:

  * ``name:`` in ``merlin/targets/*/contracts/target_contract.yaml`` (curated reference targets);
  * ``target:`` in ``merlin/experiments/capsule_bench/targets/*/target_experiment.yaml``;
  * every ``targets:`` list in ``merlin/contract/hardware_pins.yaml``;
  * the stem of every ``merlin/contract/capsules/profiles/<target>[.hidden|.synth].yaml``.

Registering a target in any of them extends the gate with zero edits. Stdlib only (no PyYAML, no
merlin import) and structural -- column-0 ``key: value`` lines and flow/block lists, no regex -- so it
runs in the dependency-free docs CI job. It FAILS CLOSED: a roster that derives nothing raises,
because a gate fed an empty name set would pass every file.
"""

from __future__ import annotations

from pathlib import Path

CONTRACTS = "merlin/targets"
EXPERIMENTS = "merlin/experiments/capsule_bench/targets"
PINS = "merlin/contract/hardware_pins.yaml"
PROFILES = "merlin/contract/capsules/profiles"


class RosterError(RuntimeError):
    """No target could be derived: the gate must refuse rather than scan for nothing."""


def _clean(value: str) -> str:
    return value.split("#", 1)[0].strip().strip("'\"")


def _top_level_scalar(text: str, key: str) -> str | None:
    """Value of a column-0 ``key:`` line. A nested key of the same name is not the document's identity."""
    prefix = f"{key}:"
    for line in text.splitlines():
        if line.startswith(prefix):
            return _clean(line[len(prefix) :]) or None
    return None


def _list_value(lines: list[str], i: int) -> list[str]:
    """Items of the list-valued key on ``lines[i]``: a flow list on the same line, else a block list below."""
    head = _clean(lines[i].split(":", 1)[1])
    if head.startswith("[") and head.endswith("]"):
        return [_clean(x) for x in head[1:-1].split(",") if _clean(x)]
    if head:
        return [head]
    items: list[str] = []
    for line in lines[i + 1 :]:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if not s.startswith("-"):
            break
        items.append(_clean(s[1:]))
    return [x for x in items if x]


def sources(root: Path) -> dict[str, set[str]]:
    """Target names per declaring registry (kept separate so a report can say where a name came from)."""
    root = Path(root)
    out: dict[str, set[str]] = {"contracts": set(), "experiments": set(), "hardware_pins": set(), "profiles": set()}
    for p in sorted((root / CONTRACTS).glob("*/contracts/target_contract.yaml")):
        name = _top_level_scalar(p.read_text(encoding="utf-8"), "name")
        if name:
            out["contracts"].add(name)
    for p in sorted((root / EXPERIMENTS).glob("*/target_experiment.yaml")):
        name = _top_level_scalar(p.read_text(encoding="utf-8"), "target")
        if name:
            out["experiments"].add(name)
    pins = root / PINS
    if pins.is_file():
        lines = pins.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("targets:"):
                out["hardware_pins"].update(_list_value(lines, i))
    profiles = root / PROFILES
    if profiles.is_dir():
        for p in profiles.glob("*.yaml"):
            if not p.name.startswith("_"):
                out["profiles"].add(p.name.split(".", 1)[0])
    return out


def target_names(root: Path) -> set[str]:
    """Every declared target name. Raises :class:`RosterError` when none is declared."""
    found: set[str] = set().union(*sources(root).values())
    if not found:
        raise RosterError(
            f"no target is declared under {root} (looked in {CONTRACTS}, {EXPERIMENTS}, "
            f"{PINS}, {PROFILES}); refusing to gate on an empty name set"
        )
    return found


if __name__ == "__main__":
    import sys

    here = Path(__file__).resolve().parents[2]
    for registry, names in sources(here).items():
        print(f"{registry:14s} {', '.join(sorted(names))}")
    print(f"{'roster':14s} {', '.join(sorted(target_names(here)))}")
    sys.exit(0)
