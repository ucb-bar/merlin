"""Generate the 4-arm experiment bundle manifests for ANY target from its ``target_experiment.yaml``.

The 4-arm ladder (raw C++ scaffold → C++ + Merlin infra → xDSL + the CCA where/how spine → + CIRCT RTL
checks) is a fixed METHODOLOGY. Each arm's toolset is (a) a per-rung block of TARGET-AGNOSTIC merlin
module paths — identical literal strings for every target — plus (b) a small TARGET/EXPERIMENT-specific
block that comes from the descriptor (ISA headers, hwbringup set, corpus) or is DERIVED from the target
name (the rtl_facts pin, the irdl pin, the prior-backend deny surfaces). This generator emits (a)+(b), so
a new accelerator drops a descriptor + registers its RTL with mlc and gets the whole ladder — no
hand-authored, gemmini-overfit YAML.

Faithful to the hand-authored gemmini bundles (verified by ``test_generate_bundles`` — the generated
allow/deny path SETS match, and verify_no_cheat + the sandbox stay green).
"""
from __future__ import annotations

from typing import Any

from .target_experiment import TargetExperiment

_PY = "merlin/python/merlin/"  # the agnostic merlin package prefix (identical for every target)

# --- per-rung AGNOSTIC tool blocks (literal merlin module paths; NO target name) ---------------------
# arm2 adds the generic C++ OOT generators.
_CPP_ALLOW = [f"{_PY}targetgen/generate/{m}.py" for m in ("mlir_scaffold", "llvm_plan", "target_repo")]
_CPP_DENY_AGN = [f"{_PY}targetgen/rtl/{m}.py" for m in
                 ("gen_iface_irdl", "gen_isa_module", "gen_rtl_digest", "gen_numeric_facts")] + [
    f"{_PY}targetgen/oot_starterkit/", f"{_PY}targetgen/synthesize/", f"{_PY}xdsl_dialects/",
    f"{_PY}targetgen/generate/xdsl.py", f"{_PY}targetgen/generate/runtime_adapter.py",
    f"{_PY}runtime/reference.py", f"{_PY}runtime/simulator.py", f"{_PY}xdsl_dialects/lowering/"]
# arm3 adds the xDSL authoring kit + the CCA compiler-modification spine.
_XDSL_ALLOW = [f"{_PY}targetgen/synthesize/", f"{_PY}targetgen/generate/", f"{_PY}xdsl_dialects/",
               f"{_PY}targetgen/contract/interface_emit.py", f"{_PY}targetgen/oot_starterkit/"] + [
    f"{_PY}kernels/{m}.py" for m in ("cca", "cca_compare", "cca_contract", "action_catalog", "microkernel")] + [
    f"{_PY}targetgen/rtl_backend.py"]
# oracle-callable routes denied in the xDSL/CIRCT arms (arm3/arm4).
_ORACLE_DENY = [f"{_PY}runtime/reference.py", f"{_PY}runtime/simulator.py",
                f"{_PY}targetgen/generate/runtime_adapter.py", f"{_PY}xdsl_dialects/lowering/"]
# arm4 adds the CIRCT RTL-facts generators (agnostic module set).
_RTLCHECKS_ALLOW = [f"{_PY}targetgen/rtl/"]


def _shared_allow(te: TargetExperiment) -> list[dict]:
    """The target/experiment-parameterized allow block present in EVERY arm."""
    exp = te.exp_name
    out = [{"path": "merlin/contract/", "mode": "ro", "note": "frozen ABI v0.1"},
           {"path": te.corpus_rel(), "mode": "ro", "note": "capsule corpus"}]
    out += [{"path": s, "mode": "ro"} for s in te.corpus_siblings()]
    out += [{"path": h, "mode": "ro", "note": "ISA header (shared hardware spec)"} for h in te.isa_headers]
    out += [{"path": f"experiments/{exp}/task/", "mode": "ro"},
            {"path": "third_party/llvm-install/", "mode": "ro", "note": "LLVM/MLIR 23 toolchain"}]
    if te.hwbringup_set:
        out.append({"path": te.hwbringup_set, "as": te.target, "mode": "ro",
                    "note": "shared hardware spec: RTL + ISA headers + README + example (ALL arms)"})
    out.append({"path": f"experiments/{exp}/scripts/agent_selfcheck.py", "as": "agent_selfcheck.py",
                "mode": "ro", "note": "redacted self-check"})
    return out


def _shared_deny(te: TargetExperiment) -> list[dict]:
    """The target/experiment-parameterized deny block present in EVERY arm (answer surfaces)."""
    exp = te.exp_name
    # answer surfaces live under the out/ generated root (the hand-authored bundles used a stale prefix
    # missing the out/ root; the real backends are under out/artifacts/, matching the launcher's lock).
    out = [{"path": f"out/artifacts/targets/{te.target}/{b}/",
            "reason": "prior backend / exemplar (answer surface)"} for b in te.prior_backends]
    if te.hidden_corpus():
        out.append({"path": te.hidden_corpus(), "reason": "hidden capsules + goldens"})
    out += [{"path": f"experiments/{exp}/input_bundles/grader_private_v0/", "reason": "grader-private"},
            {"path": f"experiments/{exp}/runs/", "reason": "prior submissions"}]
    return out


def _arm_manifest(te: TargetExperiment, arm: str, bundle_id: str) -> dict[str, Any]:
    """Assemble one arm's manifest = shared target/exp block + the per-rung agnostic tool block."""
    allow = _shared_allow(te)
    deny = _shared_deny(te)
    if arm == "raw_baseline":
        deny = [{"path": "merlin/", "reason": "Merlin internals (no tools for the raw arm)"}] + deny
    elif arm == "cpp_merlininfra":
        allow += [{"path": p, "mode": "ro", "note": "ALLOWED tool: generic C++ OOT generator"} for p in _CPP_ALLOW]
        deny = ([{"path": p, "reason": "denied tool (kept a strict subset of the xDSL arm)"} for p in _CPP_DENY_AGN]
                + [{"path": te.irdl_pin, "reason": "IRDL spec (xDSL arm only)"},
                   {"path": te.rtl_facts_pin, "reason": "RTL facts (CIRCT arm only)"}] + deny)
    elif arm == "merlin_assisted":
        allow += [{"path": p, "mode": "ro", "note": "ALLOWED tool: xDSL kit / CCA spine"} for p in _XDSL_ALLOW]
        deny = ([{"path": f"{_PY}targetgen/rtl/", "reason": "CIRCT RTL generators (CIRCT arm only)"},
                 {"path": te.rtl_facts_pin, "reason": "RTL facts (CIRCT arm only)"}]
                + [{"path": p, "reason": "oracle-callable route"} for p in _ORACLE_DENY] + deny)
    elif arm == "merlin_rtlchecks":
        allow += [{"path": p, "mode": "ro", "note": "ALLOWED tool: xDSL kit / CCA spine"} for p in _XDSL_ALLOW]
        allow += [{"path": p, "mode": "ro", "note": "ALLOWED (CIRCT arm): RTL-facts generators"} for p in _RTLCHECKS_ALLOW]
        allow += [{"path": te.rtl_facts_pin, "mode": "ro", "note": "ALLOWED (CIRCT arm): RTL-extracted facts"}]
        deny = [{"path": p, "reason": "oracle-callable route"} for p in _ORACLE_DENY] + deny
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return {"bundle_id": bundle_id, "arm": arm, "task": f"{te.target}-mlir-oot-capsule",
            "description": f"{arm} arm for the {te.target} target (generated from target_experiment.yaml)",
            "allowed": allow, "denied": deny, "integrity_required": True}


# arm -> the bundle-id stem (the launcher appends the variant suffix).
_ARMS = {"raw_baseline": "raw_baseline", "cpp_merlininfra": "cpp_merlininfra",
         "merlin_assisted": "merlin_assisted", "merlin_rtlchecks": "merlin_assisted_rtlchecks"}


def generate_bundles(te: TargetExperiment, *, variant: str = "hwbringup_v0") -> dict[str, dict]:
    """The 4-arm bundle manifests for ``te``, keyed by bundle_id. Target-agnostic: the same code emits
    them for any target from its descriptor + derived paths (no hand-authored YAML)."""
    return {f"{stem}_{variant}": _arm_manifest(te, arm, f"{stem}_{variant}")
            for arm, stem in _ARMS.items()}


def _dump_manifest(manifest: dict[str, Any]) -> str:
    """Serialize one bundle manifest to YAML (a generated header + the manifest body). Key order is
    preserved so the file reads like the hand-authored ones; consumers ``yaml.safe_load`` it."""
    import yaml
    header = (f"# GENERATED by merlin.targetgen.generate_bundles for target {manifest.get('task')!r}.\n"
              f"# Do not hand-edit: regenerate from target_experiment.yaml (the 4-arm ladder is a fixed\n"
              f"# methodology; the target/experiment-specific paths come from the descriptor).\n")
    return header + yaml.safe_dump(manifest, sort_keys=False, default_flow_style=False, width=120)


def materialize_bundles(te: TargetExperiment, dest, *,
                        variants: tuple[str, ...] = ("hwbringup_v0",)) -> list["Path"]:
    """Write every generated bundle's ``input_bundle_manifest.yaml`` under ``dest/<bundle_id>/``, for each
    requested ``variant``. Target-agnostic: works for any descriptor. Returns the written manifest paths.

    ``dest`` is typically ``experiments/<exp>/input_bundles`` — the same tracked location the launcher and
    ``require_scaffolding`` read (bundles are curated inputs, not ``out/`` generated output)."""
    from pathlib import Path
    dest = Path(dest)
    written: list[Path] = []
    for variant in variants:
        for bundle_id, manifest in generate_bundles(te, variant=variant).items():
            bdir = dest / bundle_id
            bdir.mkdir(parents=True, exist_ok=True)
            out = bdir / "input_bundle_manifest.yaml"
            out.write_text(_dump_manifest(manifest))
            written.append(out)
    return written


def _main(argv: list[str] | None = None) -> int:
    """CLI: materialize the 4-arm bundle manifests for a target's descriptor.

    Usage: python -m merlin.targetgen.generate_bundles --descriptor <target_experiment.yaml> \\
               [--dest <input_bundles dir>] [--variants hwbringup_v0,hwbringup_nokernel_v0]

    ``--dest`` defaults to ``<descriptor dir>/input_bundles`` (beside the descriptor)."""
    import argparse
    from pathlib import Path

    from .target_experiment import load_target_experiment

    ap = argparse.ArgumentParser(description="Materialize the 4-arm bundle manifests from a descriptor.")
    ap.add_argument("--descriptor", required=True, help="path to a target_experiment.yaml")
    ap.add_argument("--dest", default=None, help="output input_bundles dir (default: beside the descriptor)")
    ap.add_argument("--variants", default="hwbringup_v0",
                    help="comma-separated bundle-id variant suffixes (default: hwbringup_v0)")
    a = ap.parse_args(argv)

    te = load_target_experiment(a.descriptor)
    dest = Path(a.dest) if a.dest else Path(a.descriptor).parent / "input_bundles"
    variants = tuple(v.strip() for v in a.variants.split(",") if v.strip())
    written = materialize_bundles(te, dest, variants=variants)
    print(f"materialized {len(written)} bundle manifests under {dest} (target={te.target}):")
    for p in written:
        print(f"  {p.parent.name}/{p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
