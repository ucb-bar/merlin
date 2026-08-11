"""Derive the Radiance facts this package is constrained by, from Radiance's own sources.

Run once to (re)generate ``inputs/derived_facts.yaml`` and the derived fields of the package
contract. Nothing here is a literal: the scratchpad geometry comes from the RTL configuration the
hardware is elaborated from, the SIMT geometry from the committed capability manifest. Every fact
FAILS CLOSED — an underivable fact is recorded as absent and the script refuses rather than
substituting a default.

Two facts that look like one
----------------------------
An earlier version of this script derived ``resident_storage_bytes`` as ``1 << SMEM_LOG_SIZE`` from
the kernel headers. That is **not a capacity** — the same header uses the value as the base of the
*next* aperture (``IO_BASE_ADDR = SMEM_BASE_ADDR + (1 << SMEM_LOG_SIZE)``), i.e. it is the address
window reserved for shared memory, four times the memory that actually exists. Anything that sized a
tile against it would have overrun the scratchpad by 4x. The two are now separate facts with separate
names, and the capacity comes from the RTL config key the hardware is generated from
(``RadianceSharedMemKey.size``), cross-checkable against the kernel headers' own
``MU_SMEM_SIZE_BYTES``.

This is the distinction a weight-stationary systolic target keeps by hand: its generated parameter
header carries ``BANK_NUM``/``BANK_ROWS`` (capacity) and ``ADDR_LEN`` (address width) as different
constants, and its tiling helpers size against the former. Same discipline, different hardware.

This script lives inside the package because it is legitimately *about* Radiance. Core library code
may not name a target; a per-target package may.

    python out/artifacts/targets/radiance/hand_v0/derive_facts.py

Source trees are located through the environment (``MERLIN_CHIPYARD``, ``MERLIN_RADIANCE_KERNELS``),
never through an absolute path baked into this file: a checkout-specific path in a tracked file is
both wrong for every other clone and not publishable.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _bootstrap_import_path() -> None:
    """Put merlin on ``sys.path`` (this script runs standalone, outside the package's own imports)."""
    for candidate in (Path.cwd() / "merlin" / "python", HERE.parents[3] / "merlin" / "python"):
        if (candidate / "merlin").is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            return


# ---------------------------------------------------------------- structural source parsing (no regex)
def defines(header_text: str) -> dict[str, str]:
    """``#define NAME VALUE`` pairs, parsed structurally (no regex, per the repo mandate)."""
    out: dict[str, str] = {}
    for raw in header_text.splitlines():
        line = raw.strip()
        if not line.startswith("#define"):
            continue
        rest = line[len("#define"):].strip()
        if not rest:
            continue
        name, _, value = rest.partition(" ")
        out[name.strip()] = value.strip()
    return out


def as_int(token: str) -> int | None:
    """An integer literal or a shift expression (``128 << 10``). Returns None when unresolvable.

    Shifts are how both the RTL config and the kernel headers spell sizes, so refusing to evaluate
    them would make every size fact underivable.
    """
    token = token.strip()
    if token.startswith("x\"") and token.endswith("\""):     # Scala's x"ff000000" hex literal
        token = "0x" + token[2:-1]
    if "<<" in token:
        left, _, right = token.partition("<<")
        lhs, rhs = as_int(left), as_int(right)
        return None if lhs is None or rhs is None else lhs << rhs
    try:
        return int(token, 0)
    except ValueError:
        return None


def _matching_paren(text: str, open_index: int) -> int:
    """Index of the ``)`` closing the ``(`` at ``open_index``, or -1. Depth-counted, not matched."""
    depth = 0
    for i in range(open_index, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _split_top_level(text: str, sep: str = ",") -> list[str]:
    """Split on ``sep`` at paren/bracket depth 0 (so nested calls stay whole)."""
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == sep and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return parts


def named_args(text: str, marker: str) -> dict[str, str] | None:
    """The ``name = value`` arguments of the first call whose declaration starts with ``marker``.

    ``marker`` is matched literally and must end at (or just before) the call's ``(``; the argument
    list is then read by depth-counting to the matching ``)``, so a nested call or a trailing
    argument list on another line does not truncate it.
    """
    idx = text.find(marker)
    if idx == -1:
        return None
    open_index = text.find("(", idx)
    if open_index == -1:
        return None
    close_index = _matching_paren(text, open_index)
    if close_index == -1:
        return None
    out: dict[str, str] = {}
    for part in _split_top_level(text[open_index + 1:close_index]):
        name, sep, value = part.partition("=")
        if sep:
            out[name.strip()] = value.strip()
    return out


def _declaration_block(text: str, declaration: str) -> str | None:
    """The source from ``declaration`` up to the next top-level ``class``/``object`` declaration."""
    idx = text.find(declaration)
    if idx == -1:
        return None
    rest = text[idx + len(declaration):]
    ends = [rest.find(f"\n{kw} ") for kw in ("class", "object")]
    cut = min((e for e in ends if e != -1), default=-1)
    return text[idx:] if cut == -1 else text[idx:idx + len(declaration) + cut]


# ---------------------------------------------------------------- the facts
def shared_memory_from_rtl(scala_text: str, config_class: str) -> dict[str, int | str]:
    """The scratchpad's real geometry, from the RTL config the hardware is elaborated from.

    Chain of evidence, each link read from the source rather than assumed:
    ``class <config_class>`` -> its cluster's ``smemConfig = <Key>`` -> ``object <Key> extends
    RadianceSharedMemKey(size =, numBanks =, numWords =, wordSize =)``. Capacity is the declared
    ``size``; the bank geometry is derived from it exactly as the generator does
    (``smemWidth = numWords * wordSize``, ``smemDepth = size / smemWidth / numBanks``).

    Raises when any link is missing — a scratchpad size is not a thing to guess.
    """
    block = _declaration_block(scala_text, f"class {config_class} extends Config(")
    if block is None:
        raise ValueError(f"RTL config class {config_class!r} not found in the Radiance configs")
    cluster = named_args(block, "WithRadianceCluster")
    if cluster is None or "smemConfig" not in cluster:
        raise ValueError(f"{config_class} declares no WithRadianceCluster(smemConfig = ...)")
    key_name = cluster["smemConfig"]
    key = named_args(scala_text, f"object {key_name} extends RadianceSharedMemKey")
    if key is None:
        raise ValueError(f"shared-memory key object {key_name!r} not found")
    size, banks = as_int(key.get("size", "")), as_int(key.get("numBanks", ""))
    words, word_size = as_int(key.get("numWords", "")), as_int(key.get("wordSize", "4"))
    missing = [n for n, v in (("size", size), ("numBanks", banks), ("numWords", words),
                              ("wordSize", word_size)) if v is None]
    if missing:
        raise ValueError(f"{key_name} does not resolve {missing} to integers")
    row_bytes = words * word_size
    return {
        "capacity_bytes": size,
        "banks": banks,
        "row_bytes": row_bytes,
        "depth_rows": size // row_bytes // banks,
        "config_class": config_class,
        "key_object": key_name,
    }


def shared_memory_aperture(header_text: str) -> int:
    """The ADDRESS WINDOW reserved for shared memory: ``1 << SMEM_LOG_SIZE``.

    Not a capacity — see the module docstring. Kept as a fact because the address map needs it; it
    must never be the number a tile is sized against.
    """
    d = defines(header_text)
    if "SMEM_LOG_SIZE" not in d:
        raise ValueError("SMEM_LOG_SIZE not defined in header (cannot derive; fail closed)")
    log_size = as_int(d["SMEM_LOG_SIZE"])
    if log_size is None:
        raise ValueError(f"SMEM_LOG_SIZE resolves to unresolvable token {d['SMEM_LOG_SIZE']!r}")
    return 1 << log_size


def software_capacity_claim(header_text: str) -> int | None:
    """What the target's own kernel library believes the capacity is (``MU_SMEM_SIZE_BYTES``).

    A second, independent witness to the RTL config's ``size``. Disagreement is reported, never
    averaged or silently preferred — two sources that disagree about a capacity mean one of them is
    describing different hardware, and picking one would hide that.
    """
    return as_int(defines(header_text).get("MU_SMEM_SIZE_BYTES", ""))


def abstraction_features(contract: dict) -> tuple[list[str], dict[str, str]]:
    """The abstraction-surface features the staged pipeline checks, each JUSTIFIED by a contract fact.

    The contract lists what Radiance *is* (simt, cvfpu, tensor_core, microscaling); the staged
    pipeline asks what it *offers* (resident_packed_tensor, accumulator_commit, command_buffer,
    metrics). Those are two different vocabularies, and the second must be derived from the first
    rather than typed in — a feature string with no justifying fact is a fabricated capability, and
    it would be believed by every check downstream.
    """
    features = list(contract.get("features") or [])
    provenance: dict[str, str] = {}
    memory = contract.get("memory_model") or {}
    units = contract.get("compute_units") or []
    runtime = contract.get("runtime") or {}
    promises = list(contract.get("hardware_promises") or [])
    runtime_promises = list(contract.get("runtime_promises") or [])

    def offer(feature: str, justification: str | None) -> None:
        if justification and feature not in features:
            features.append(feature)
            provenance[feature] = justification

    offer("resident_packed_tensor",
          "memory_model.resident: true" if memory.get("resident") else None)
    accumulates = any(u.get("accumulate") for u in units if isinstance(u, dict))
    offer("accumulator_commit",
          "compute_units[].accumulate declares an accumulator dtype"
          if accumulates else ("hardware_promises: fp32_accumulate"
                               if "fp32_accumulate" in promises else None))
    offer("command_buffer",
          f"runtime.default_backend: {runtime.get('default_backend')}"
          if runtime.get("default_backend") else None)
    offer("metrics", "runtime_promises: metrics" if "metrics" in runtime_promises else None)
    return features, provenance


def main() -> int:
    import yaml

    _bootstrap_import_path()
    from merlin.runtime.backends import muon
    from merlin.targetgen import capability_manifests as cm
    from merlin.targetgen.rtl.muon_introspect import VCS_CONFIG

    header_path = muon.radiance_kernels_root() / "lib" / "include" / "VX_config.h"
    intrinsics_path = muon.radiance_kernels_root() / "lib" / "include" / "mu_intrinsics.h"
    if not header_path.is_file():
        print(f"Radiance kernel header not found: {header_path}\n"
              "set MERLIN_RADIANCE_KERNELS to the radiance-kernels checkout", file=sys.stderr)
        return 1
    header = header_path.read_text(encoding="utf-8")

    # The RTL configuration the hardware is elaborated from. VCS_CONFIG is imported rather than named
    # here so this package and the RTL introspect cannot drift onto different configs.
    scala_path = (muon.chipyard_root() / "generators" / "radiance" / "chipyard"
                  / "RadianceConfigs.scala")
    if not scala_path.is_file():
        print(f"Radiance RTL config not found: {scala_path}\n"
              "set MERLIN_CHIPYARD to the chipyard checkout", file=sys.stderr)
        return 1
    smem = shared_memory_from_rtl(scala_path.read_text(encoding="utf-8"), VCS_CONFIG)

    aperture = shared_memory_aperture(header)
    claimed = software_capacity_claim(intrinsics_path.read_text(encoding="utf-8")) \
        if intrinsics_path.is_file() else None
    if claimed is not None and claimed != smem["capacity_bytes"]:
        print(f"REFUSING: the RTL config says the scratchpad is {smem['capacity_bytes']} bytes but "
              f"the target's own kernel library says {claimed} (MU_SMEM_SIZE_BYTES). One of them "
              "describes different hardware; resolve it rather than picking one.", file=sys.stderr)
        return 1

    manifest = cm.manifest_for("radiance")
    simt = (manifest.get("capabilities", {}) or {}).get("simt", {}) or {}
    lanes, warps = simt.get("lanes_per_warp"), simt.get("warps_per_core")
    for name, value in (("lanes_per_warp", lanes), ("warps_per_core", warps)):
        if value is None:
            print(f"capability manifest declares no capabilities.simt.{name}", file=sys.stderr)
            return 1

    base_path = (HERE.parents[1] / "radiance_oot" / "contracts" / "target_contract.yaml")
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    features, feature_provenance = abstraction_features(base)

    facts = {
        "generator": "radiance-derive-facts-v2",
        "resident_storage_bytes": int(smem["capacity_bytes"]),
        "smem_aperture_bytes": int(aperture),
        "smem_banks": int(smem["banks"]),
        "smem_row_bytes": int(smem["row_bytes"]),
        "smem_depth_rows": int(smem["depth_rows"]),
        "lanes_per_warp": int(lanes),
        "warps_per_core": int(warps),
        "abstraction_features": features,
        "provenance": {
            "resident_storage_bytes":
                f"{smem['key_object']}.size in RadianceConfigs.scala, reached from "
                f"class {smem['config_class']} -> WithRadianceCluster(smemConfig=...); "
                f"corroborated by MU_SMEM_SIZE_BYTES in mu_intrinsics.h"
                + ("" if claimed is not None else " (header witness absent)"),
            "smem_aperture_bytes":
                f"1 << SMEM_LOG_SIZE from {header_path.name} — the ADDRESS WINDOW, not the capacity "
                "(the same header derives IO_BASE_ADDR from it); never size a tile against this",
            "smem_banks": f"{smem['key_object']}.numBanks",
            "smem_row_bytes": f"{smem['key_object']}.numWords * wordSize",
            "smem_depth_rows": "size / row_bytes / banks (the generator's own division)",
            "lanes_per_warp": "capabilities.simt.lanes_per_warp from the committed manifest",
            "warps_per_core": "capabilities.simt.warps_per_core from the committed manifest",
            "features": feature_provenance,
        },
    }
    out = HERE / "inputs" / "derived_facts.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(facts, sort_keys=True), encoding="utf-8")

    capabilities = base.setdefault("capabilities", {})
    capabilities["resident_storage_bytes"] = int(smem["capacity_bytes"])
    capabilities["smem_aperture_bytes"] = int(aperture)
    capabilities.setdefault("simt", {})["warps_per_core"] = int(warps)
    base["features"] = features
    # This package supplies its own runtime backend (``backend.py``), so the core carries no
    # name -> module map for it: merlin.runtime.backends.base loads whatever ``plugin.backend`` names.
    base.setdefault("plugin", {})["backend"] = "backend.py"
    base["derived_from"] = {
        "source": str(base_path.relative_to(HERE.parents[2].parent))
        if str(base_path).startswith(str(HERE.parents[2].parent)) else base_path.name,
        "generator": facts["generator"],
        "added": {
            "capabilities.resident_storage_bytes": facts["provenance"]["resident_storage_bytes"],
            "capabilities.smem_aperture_bytes": facts["provenance"]["smem_aperture_bytes"],
            "capabilities.simt.warps_per_core": facts["provenance"]["warps_per_core"],
            "plugin.backend": "this package's own command-buffer backend",
            "features": feature_provenance,
        },
    }
    contract_path = HERE / "contracts" / "target_contract.yaml"
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(
        "# GENERATED by derive_facts.py for the radiance/hand_v0 package. The base is Radiance's\n"
        "# committed OOT contract; the scratchpad geometry, the SIMT warp count, the\n"
        "# abstraction-surface features and the backend plugin path are DERIVED from its own sources\n"
        "# (see inputs/derived_facts.yaml for the per-fact provenance).\n"
        "# Do not hand-edit: re-run derive_facts.py.\n"
        + yaml.safe_dump(base, sort_keys=True), encoding="utf-8")
    print(f"wrote {out}: capacity={smem['capacity_bytes']} "
          f"({smem['banks']} banks x {smem['row_bytes']} B x {smem['depth_rows']} rows) "
          f"aperture={aperture} lanes={lanes} warps={warps}")
    print(f"wrote {contract_path}: features={features}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
