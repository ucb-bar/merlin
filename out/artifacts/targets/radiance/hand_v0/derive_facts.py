"""Derive the Radiance facts this package is constrained by, from Radiance's own sources.

Run once to (re)generate ``inputs/derived_facts.yaml`` and the package contract's
``capabilities.resident_storage_bytes``. Nothing here is a literal: the scratchpad capacity comes
from the kernel headers' own ``SMEM_LOG_SIZE``, and the SIMT warp width from the committed
capability manifest. Both FAIL CLOSED — an underivable fact is recorded as absent and the package
refuses to load rather than substituting a default.

This script lives inside the package because it is legitimately *about* Radiance. Core library code
may not name a target; a per-target package may.

    python out/artifacts/targets/radiance/hand_v0/derive_facts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Radiance's kernel headers. Absent in a fresh clone, which is why the DERIVED VALUES are committed
# to inputs/derived_facts.yaml with their provenance rather than re-derived at load time.
KERNEL_HEADER_CANDIDATES = (
    Path("/scratch/agustin/projects/radiance-kernels/lib/include/VX_config.h"),
)


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
    token = token.strip()
    try:
        return int(token, 0)
    except ValueError:
        return None


def shared_memory_bytes(header_text: str) -> int:
    """The scratchpad capacity: ``1 << SMEM_LOG_SIZE``. Raises when it cannot be resolved."""
    d = defines(header_text)
    if "SMEM_LOG_SIZE" not in d:
        raise ValueError("SMEM_LOG_SIZE not defined in header (cannot derive; fail closed)")
    log_size = as_int(d["SMEM_LOG_SIZE"])
    if log_size is None:
        raise ValueError(f"SMEM_LOG_SIZE resolves to unresolvable token {d['SMEM_LOG_SIZE']!r}")
    return 1 << log_size


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

    header = next((p for p in KERNEL_HEADER_CANDIDATES if p.is_file()), None)
    if header is None:
        print(f"no Radiance kernel header found among {[str(p) for p in KERNEL_HEADER_CANDIDATES]}",
              file=sys.stderr)
        return 1
    text = header.read_text(encoding="utf-8")
    smem = shared_memory_bytes(text)

    sys.path.insert(0, str(Path.cwd() / "merlin" / "python"))
    from merlin.targetgen import capability_manifests as cm
    manifest = cm.manifest_for("radiance")
    lanes = (manifest.get("capabilities", {}) or {}).get("simt", {}).get("lanes_per_warp")
    if lanes is None:
        print("capability manifest declares no capabilities.simt.lanes_per_warp", file=sys.stderr)
        return 1

    base_path = (HERE.parents[1] / "radiance_oot" / "contracts" / "target_contract.yaml")
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    features, feature_provenance = abstraction_features(base)

    facts = {
        "generator": "radiance-derive-facts-v1",
        "resident_storage_bytes": int(smem),
        "lanes_per_warp": int(lanes),
        "abstraction_features": features,
        "provenance": {
            "resident_storage_bytes": f"1 << SMEM_LOG_SIZE from {header.name} (kernel headers)",
            "lanes_per_warp": "capabilities.simt.lanes_per_warp from the committed manifest",
            "features": feature_provenance,
        },
    }
    out = HERE / "inputs" / "derived_facts.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(facts, sort_keys=True), encoding="utf-8")

    base.setdefault("capabilities", {})["resident_storage_bytes"] = int(smem)
    base["features"] = features
    base["derived_from"] = {
        "source": str(base_path.relative_to(HERE.parents[2].parent))
        if str(base_path).startswith(str(HERE.parents[2].parent)) else base_path.name,
        "generator": facts["generator"],
        "added": {
            "capabilities.resident_storage_bytes": facts["provenance"]["resident_storage_bytes"],
            "features": feature_provenance,
        },
    }
    contract_path = HERE / "contracts" / "target_contract.yaml"
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(
        "# GENERATED by derive_facts.py for the radiance/hand_v0 package. The base is Radiance's\n"
        "# committed OOT contract; resident_storage_bytes and the abstraction-surface features are\n"
        "# DERIVED from its own facts (see inputs/derived_facts.yaml for the per-fact provenance).\n"
        "# Do not hand-edit: re-run derive_facts.py.\n"
        + yaml.safe_dump(base, sort_keys=True), encoding="utf-8")
    print(f"wrote {out}: resident_storage_bytes={smem} lanes_per_warp={lanes}")
    print(f"wrote {contract_path}: features={features}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
