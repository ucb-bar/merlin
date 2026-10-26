"""How often a test case's command stream drains the whole device, and what that leaves to overlap.

Measured on one attempt: the author's compiler emitted a full-device fence after every output tile's
readout ("before following accumulator recycling") by default. On one large GEMM that was 131 fences
for 128 output tiles, and the case ran about 4x slower than a hand-tuned kernel and the target's own
library: a fence drains every in-flight load, compute and store, so the next tile's operand stream
cannot start while the current tile computes. No feedback said so; the author never opened the
per-command debugger and never compared its own alternative policy. This module states it per case,
from evidence the self-check and the grade already hold:

* **static** -- the case's decoded instruction stream (``instruction_trace.json`` beside its
  ``command_buffer.json``, written by the same decoder ``isa_tools.py disasm`` uses). Each instruction
  class takes the roles the target's own compute-endpoint declaration binds to that name in its
  RTL-derived decode table (:func:`merlin.compile.analysis.compute_endpoints.endpoints_for`); a class
  the decoder identifies structurally as an architectural fence takes the shared
  ``synchronization`` family (:func:`merlin.targetgen.semantic_families.from_isa_class`). Fences cut
  the stream into epochs; per epoch it counts commands, output-tile readouts and loads, and a load
  counts as overlapped when a compute was issued earlier in the same epoch (still outstanding: no
  fence drained it). That share is the **static overlap ratio**: an ordering property of the emitted
  stream (instruction sites, not executions), not a timing measurement.
* **dynamic** -- the timing model's own command census of the run (fences, stores, loads, computes
  and the modeled cycles spent waiting on fences), when the response row carries it.

Nothing here names a target, an opcode or a value; a class with no declared role is counted as such.
Everything is advisory: the block changes no status or count and never fails a check.
"""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA = "merlin.fence_overlap.v1"
TRACE = "instruction_trace.json"
BUFFER = "command_buffer.json"
#: Roles of the closed vocabulary (:mod:`merlin.compile.analysis.instruction_roles`) read here.
LOAD_ROLES = frozenset({"operand_load", "dma"})
COMPUTE_ROLES = frozenset({"accumulate"})
READOUT_ROLES = frozenset({"readout", "commit"})
SYNC_ROLE = "sync"
#: Fewer fences than this, or fewer output tiles, is not called a fence after every tile.
MIN_FENCES = 3
MIN_TILES = 2
#: Output tiles per fence at or below which the stream reads as fenced after every tile.
PER_TILE = 1.5
_MAX_TRACE_BYTES = 32 << 20
_ENDPOINTS: dict[str, dict | None] = {}


def _load(path: Path):
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_TRACE_BYTES:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _endpoint_roles(target: str | None) -> dict | None:
    """``{class name: frozenset(roles)}`` from the target's endpoint declaration, or None (cached)."""
    if not isinstance(target, str) or not target:
        return None
    if target not in _ENDPOINTS:
        try:
            from merlin.compile.analysis.compute_endpoints import endpoints_for

            endpoints = endpoints_for(target)
            table: dict[str, set] = {}
            for endpoint in endpoints:
                for role, names in (endpoint.roles or {}).items():
                    for name in names:
                        table.setdefault(str(name), set()).add(role)
            _ENDPOINTS[target] = {name: frozenset(roles) for name, roles in table.items()} or None
        except Exception:  # noqa: BLE001 -- an unavailable declaration is UNKNOWN, never a guess
            _ENDPOINTS[target] = None
    return _ENDPOINTS[target]


def _is_fence(instruction: dict) -> bool:
    """An architectural fence the decoder identified by structure (no endpoint selector)."""
    from merlin.targetgen.semantic_families import from_isa_class

    return instruction.get("funct") is None and from_isa_class(str(instruction.get("class") or "")) == (
        "synchronization"
    )


def static_profile(trace: dict, roles: dict) -> dict:
    """Fence-delimited epochs of a decoded stream, with the static overlap ratio (module docstring)."""
    instructions = [row for row in trace.get("instructions") or () if isinstance(row, dict)]
    shared = [row for row in instructions if _is_fence(row)]
    epochs, current = [], None
    loads = overlapped = after_readout = readouts = computes = fences = unroled = 0

    def close():
        if current and current["commands"]:
            epochs.append(current)

    for row in instructions:
        found = roles.get(str(row.get("class"))) or frozenset()
        fence = _is_fence(row) if shared else SYNC_ROLE in found
        if fence:
            fences += 1
            close()
            current = None
            continue
        if current is None:
            current = {"commands": 0, "readouts": 0, "loads": 0, "compute_seen": False, "readout_seen": False}
        current["commands"] += 1
        if not found and row.get("funct") is not None:
            unroled += 1
        if found & LOAD_ROLES:
            loads += 1
            current["loads"] += 1
            overlapped += current["compute_seen"]
            after_readout += current["readout_seen"]
        if found & COMPUTE_ROLES:
            computes += 1
            current["compute_seen"] = True
        if found & READOUT_ROLES:
            readouts += 1
            current["readouts"] += 1
            current["readout_seen"] = True
    close()
    # Fences before the first command or after the last one bracket the program; the rest separate work.
    flags = [
        (_is_fence(row) if shared else SYNC_ROLE in (roles.get(str(row.get("class"))) or ())) for row in instructions
    ]
    work = [index for index, fence in enumerate(flags) if not fence]
    boundary = (sum(flags[: work[0]]) + sum(flags[work[-1] + 1 :])) if work else fences
    sizes = sorted(epoch["commands"] for epoch in epochs)
    with_readout = sorted(epoch["readouts"] for epoch in epochs if epoch["readouts"])
    summary = (trace.get("summary") or {}).get("counted_loops") or {}
    return {
        "basis": "static instruction sites of the decoded stream, in program order",
        "multiplicity": summary.get("instruction_multiplicity", "static_sites"),
        "instructions": len(instructions),
        "fences": fences,
        "boundary_fences": boundary,
        "epochs": len(epochs),
        "commands_per_epoch": {"median": _median(sizes), "max": sizes[-1] if sizes else None},
        "output_tiles": readouts,
        "output_tiles_per_fence": round(readouts / fences, 2) if fences else None,
        "epochs_with_readout": len(with_readout),
        "median_tiles_per_readout_epoch": _median(with_readout),
        "loads": loads,
        "computes": computes,
        "loads_while_compute_outstanding": overlapped,
        "loads_after_earlier_readout_in_epoch": after_readout,
        "static_overlap_ratio": round(overlapped / loads, 3) if loads else None,
        "commands_without_declared_role": unroled,
    }


def dynamic_profile(performance) -> dict | None:
    """The timing model's command census of the run: fences, stores, loads, computes, fence waits.

    The class grouping is the one the timing model's scoped census applies
    (:mod:`merlin_timing.scoped_resources`): ``compute`` and ``compute_*`` are computes, ``store``,
    ``pool_store`` and their ``_*`` refinements are output stores.
    """
    if not isinstance(performance, dict):
        return None
    counts = performance.get("feature_counts") if isinstance(performance.get("feature_counts"), dict) else {}
    commands = (performance.get("scoped_resource_feedback") or {}).get("commands") or {}

    def count(value):
        return value if type(value) is int and value >= 0 else None

    fences = count(commands.get("fences"))
    if fences is None:
        fences = count(counts.get("target.fences"))
    classes = {
        key.removeprefix("target.class."): count(value)
        for key, value in counts.items()
        if isinstance(key, str) and key.startswith("target.class.")
    }
    classes = {key: value for key, value in classes.items() if value is not None}
    if fences is None or not classes:
        return None

    def group(*heads):
        return sum(
            value for key, value in classes.items() if key in heads or key.startswith(tuple(f"{h}_" for h in heads))
        )

    stores = count(commands.get("store")) if commands.get("store") is not None else group("store", "pool_store")
    found = {
        "basis": "timing model command census (executed commands)",
        "fences": fences,
        "output_stores": stores,
        "loads": group("load"),
        "computes": group("compute"),
    }
    if fences:
        found["output_stores_per_fence"] = round(stores / fences, 2)
    waits = [
        value
        for key, value in counts.items()
        if isinstance(key, str)
        and type(value) in (int, float)
        and "fence" in key.split(".")[-1].split("_")
        and key.endswith("_wait_cycles")
    ]
    cycles = performance.get("estimated_whole_soc_cycles")
    if waits:
        found["modeled_fence_wait_cycles"] = sum(waits)
        if type(cycles) in (int, float) and cycles > 0:
            found["fence_wait_share_of_cycles"] = round(sum(waits) / cycles, 3)
    return found


def _median(values):
    if not values:
        return None
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def case_dir(name: str, workspace: Path, artifact_root=None) -> Path | None:
    """The folder holding a case's copied command buffer, or None."""
    if not name or "/" in name:
        return None
    root = Path(artifact_root or "selfcheck_out")
    folder = (root if root.is_absolute() else Path(workspace) / root) / name
    return folder if (folder / TRACE).is_file() or (folder / BUFFER).is_file() else None


def analyze(folder: Path | None = None, performance=None, *, target: str | None = None) -> dict | None:
    """The block for one case (module docstring), or None when neither source carries anything."""
    profile = {"schema": SCHEMA, "advisory_only": True}
    if folder is not None:
        trace = _load(Path(folder) / TRACE)
        if target is None:
            buffer = _load(Path(folder) / BUFFER)
            target = buffer.get("target") if isinstance(buffer, dict) else None
        roles = _endpoint_roles(target) if isinstance(trace, dict) else None
        if isinstance(trace, dict) and roles is not None:
            profile["static"] = static_profile(trace, roles)
        elif isinstance(trace, dict):
            profile["static_unavailable"] = "no compute-endpoint role declaration resolved for this target"
    dynamic = dynamic_profile(performance)
    if dynamic is not None:
        profile["dynamic"] = dynamic
    if "static" not in profile and dynamic is None:
        return None
    profile["fence_per_tile"] = fence_per_tile(profile)
    profile["reading"] = reading(profile)
    return profile


def _numbers(profile: dict) -> tuple[int | None, int | None, str]:
    dynamic, static = profile.get("dynamic") or {}, profile.get("static") or {}
    if dynamic.get("fences") is not None:
        return dynamic["fences"], dynamic.get("output_stores"), "executed"
    return static.get("fences"), static.get("output_tiles"), "static sites"


def fence_per_tile(profile: dict) -> bool:
    """Whether the stream drains the device after nearly every output tile.

    From the executed census when there is one: at least :data:`MIN_FENCES` fences besides those
    bracketing the program (counted in the static stream; none subtracted without it), at least
    :data:`MIN_TILES` output stores, and at most :data:`PER_TILE` stores per fence. From the static
    stream alone: at least :data:`MIN_TILES` fence-delimited epochs hold readouts, one tile each.
    """
    dynamic, static = profile.get("dynamic") or {}, profile.get("static") or {}
    if dynamic.get("fences") is not None:
        interior = dynamic["fences"] - (static.get("boundary_fences") or 0)
        stores = dynamic.get("output_stores") or 0
        return bool(interior >= MIN_FENCES and stores >= MIN_TILES and stores / interior <= PER_TILE)
    return bool(
        (static.get("epochs_with_readout") or 0) >= MIN_TILES
        and (static.get("median_tiles_per_readout_epoch") or 0) <= 1
    )


def reading(profile: dict) -> str:
    """One plain sentence: fence density, tiles per fence and the static overlap ratio."""
    fences, tiles, kind = _numbers(profile)
    static = profile.get("static") or {}
    ratio = static.get("static_overlap_ratio")
    overlap = (
        f"{round(100 * ratio)}% of loads issue while compute is outstanding in the same epoch (static order)"
        if ratio is not None
        else None
    )
    per = f"{round(tiles / fences, 2)} tile/fence" if fences and tiles is not None else None
    if profile.get("fence_per_tile"):
        text = (
            f"a full-device fence after every output tile ({fences} fences, {per}, {kind}): loads cannot "
            "overlap compute across tiles"
        )
        if static.get("loads") and not static.get("loads_after_earlier_readout_in_epoch"):
            text += "; no load follows an earlier tile's readout inside an epoch"
    elif fences:
        text = f"{fences} fences ({kind})" + (f", {per}" if per else "")
    else:
        text = f"no full-device fence ({kind})"
    if overlap:
        text += f"; {overlap}"
    wait = (profile.get("dynamic") or {}).get("fence_wait_share_of_cycles")
    if wait is not None:
        text += f"; the timing model spends {round(100 * wait, 1)}% of cycles waiting on fences"
    return text


def mechanisms(profile: dict, selection) -> dict:
    """Catalogue mechanisms for a fence-bound case (:func:`.mechanism_hints.fence_hints`)."""
    from . import mechanism_hints

    available, unconfirmed = mechanism_hints.fence_hints(
        selection, why="a full-device fence after nearly every output tile serializes tiles"
    )
    found = {}
    if available:
        found["suggested_mechanisms"] = [
            {key: hint.get(key) for key in ("id", "api", "facet", "workload_check")} for hint in available
        ]
    if unconfirmed:
        found["related_capability_unconfirmed"] = unconfirmed
    return found


def compact(profile: dict) -> dict:
    """The few numbers a ranked entry or a console column needs."""
    fences, tiles, kind = _numbers(profile)
    static = profile.get("static") or {}
    out = {
        "fences": fences,
        "output_tiles": tiles,
        "count_kind": kind,
        "tiles_per_fence": round(tiles / fences, 2) if fences and tiles is not None else None,
        "static_overlap_ratio": static.get("static_overlap_ratio"),
        "fence_per_tile": profile.get("fence_per_tile"),
        "reading": profile.get("reading"),
    }
    return {key: value for key, value in out.items() if value is not None}


def for_entry(
    name: str, workspace, performance=None, *, artifact_root=None, selection=None, target: str | None = None
) -> dict | None:
    """``fence_overlap`` for one ranked entry, with mechanisms when the case is fence-bound; never raises."""
    try:
        profile = analyze(case_dir(name, Path(workspace), artifact_root), performance, target=target)
        if profile is None:
            return None
        block = compact(profile)
        if profile.get("static"):
            block["static"] = {
                key: profile["static"].get(key)
                for key in (
                    "multiplicity",
                    "epochs",
                    "commands_per_epoch",
                    "median_tiles_per_readout_epoch",
                    "loads",
                    "loads_while_compute_outstanding",
                    "loads_after_earlier_readout_in_epoch",
                )
            }
        if profile.get("dynamic"):
            block["dynamic"] = profile["dynamic"]
        if profile.get("fence_per_tile"):
            block.update(mechanisms(profile, selection))
        return block
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None


#: Fence-bound cases listed in one summary.
SUMMARY_LIMIT = 6
SUMMARY_NOTE = (
    "Test cases whose command stream drains the whole device after nearly every output tile: the next "
    "tile's loads wait for the current tile's compute and readout. Fences, tiles per fence and the "
    "static overlap ratio (loads issued while compute is outstanding in the same fence epoch, from the "
    "emitted order) are listed per case; compare a variant or a policy value with fewer fences on the "
    "same cases by cycles (skill ab-library-pass; tuning_knob_candidates names switches your compiler "
    "already has)."
)


def summary(blocks: dict, selection=None) -> dict | None:
    """The fence-bound cases among ``{test case: for_entry block}``, most fences first; None if none."""
    bound = [(name, block) for name, block in blocks.items() if isinstance(block, dict) and block.get("fence_per_tile")]
    if not bound:
        return None
    bound.sort(key=lambda item: (-(item[1].get("fences") or 0), str(item[0])))
    found = {
        "schema": SCHEMA,
        "advisory_only": True,
        "note": SUMMARY_NOTE,
        "count": len(bound),
        "cases": [
            {
                "test_case": name,
                **{
                    key: block.get(key)
                    for key in (
                        "fences",
                        "output_tiles",
                        "tiles_per_fence",
                        "count_kind",
                        "static_overlap_ratio",
                        "reading",
                    )
                    if block.get(key) is not None
                },
            }
            for name, block in bound[:SUMMARY_LIMIT]
        ],
    }
    found.update(mechanisms({}, selection))
    return found


def main(argv=None) -> int:
    """``python -m ...fence_overlap CASE_DIR [--target T]``: print the full block for one case folder."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--target", default=None)
    args = parser.parse_args(argv)
    print(json.dumps(analyze(args.folder, target=args.target), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
