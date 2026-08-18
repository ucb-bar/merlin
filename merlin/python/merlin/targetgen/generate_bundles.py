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
# Shared, answer-free INFRASTRUCTURE every granted merlin tool imports. `targetgen/rtl/facts.py` opens
# with `from merlin.common.paths import artifacts_dir, targets_dir`, so without this grant the arm-4
# RTL-facts generators die in the sandbox with ModuleNotFoundError: No module named 'merlin.common'.
# Measured across three live runs (codex 6 hits, codex2 6, nemotron 5): every model tried the granted
# generators, failed, and either worked around them or stopped reaching for them — so the arm-4
# treatment was partly unavailable to all of them and the arm-4-vs-arm-3 contrast was understated.
# merlin/common holds path/yaml/schema helpers; it is not an oracle, a grader or an answer surface
# (answer_surfaces lists neither), so granting it widens no moat.
_INFRA_ALLOW = [f"{_PY}common/"]
# arm3 adds the xDSL authoring kit + the CCA compiler-modification spine.
_XDSL_ALLOW = _INFRA_ALLOW + [f"{_PY}targetgen/synthesize/", f"{_PY}targetgen/generate/", f"{_PY}xdsl_dialects/",
               f"{_PY}targetgen/contract/interface_emit.py", f"{_PY}targetgen/contract/linalg_iface.py",
               f"{_PY}targetgen/oot_starterkit/"] + [
    f"{_PY}kernels/{m}.py" for m in ("cca", "cca_compare", "cca_contract", "action_catalog", "microkernel")] + [
    f"{_PY}targetgen/rtl_backend.py"]
# oracle-callable routes denied in the xDSL/CIRCT arms (arm3/arm4).
_ORACLE_DENY = [f"{_PY}runtime/reference.py", f"{_PY}runtime/simulator.py",
                f"{_PY}targetgen/generate/runtime_adapter.py", f"{_PY}xdsl_dialects/lowering/"]
# arm4 adds the CIRCT RTL-facts generators (agnostic module set).
_RTLCHECKS_ALLOW = [f"{_PY}targetgen/rtl/"]
# arm5 adds the EQUIVALENCE SEAM and nothing else: the e-graph over real IR plus the persistent
# equivalence store. The treatment under test is the seam itself — the agent registers its own
# implementation as an alternative in an e-class and the extractor chooses — so arm5 differs from the
# xDSL arm in exactly this one declared way, and the moat check asserts no other arm can reach it.
_EQSAT_ALLOW = [f"{_PY}targetgen/contraction_egraph.py", f"{_PY}targetgen/persistent_equivalence.py"]


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
    elif arm == "merlin_eqsat":
        allow += [{"path": p, "mode": "ro", "note": "ALLOWED tool: xDSL kit / CCA spine"} for p in _XDSL_ALLOW]
        allow += [{"path": p, "mode": "ro", "note": "ALLOWED (eqsat arm): the equivalence seam"}
                  for p in _EQSAT_ALLOW]
        # Same denials as the xDSL arm it is compared against: an arm that also gained the RTL facts
        # would differ in TWO ways and its result would not attribute to the seam.
        deny = ([{"path": f"{_PY}targetgen/rtl/", "reason": "CIRCT RTL generators (CIRCT arm only)"},
                 {"path": te.rtl_facts_pin, "reason": "RTL facts (CIRCT arm only)"}]
                + [{"path": p, "reason": "oracle-callable route"} for p in _ORACLE_DENY] + deny)
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return {"bundle_id": bundle_id, "arm": arm, "task": f"{te.target}-mlir-oot-capsule",
            "description": f"{arm} arm for the {te.target} target (generated from target_experiment.yaml)",
            "allowed": allow, "denied": deny, "integrity_required": True}


# arm -> the bundle-id stem (the launcher appends the variant suffix).
_ARMS = {"raw_baseline": "raw_baseline", "cpp_merlininfra": "cpp_merlininfra",
         "merlin_assisted": "merlin_assisted", "merlin_rtlchecks": "merlin_assisted_rtlchecks",
         # arm5's stem CONTAINS "merlin_assisted" on purpose: generate_prompt._is_assisted_arm is a
         # substring test, so the arm inherits the assisted seam menu with no prompt edit.
         "merlin_eqsat": "merlin_assisted_eqsat"}


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


def _grant_txt(manifest: dict[str, Any], key: str) -> str:
    """The bundle's allow/deny path list as a newline file (the ``allowed_files.txt`` / ``denied_files.txt``
    the sandbox + graders read) — DERIVED verbatim from the manifest's ``allowed``/``denied`` path set."""
    paths = [e["path"] for e in manifest.get(key, []) if isinstance(e, dict) and e.get("path")]
    return "\n".join(paths) + ("\n" if paths else "")


def _materialize_prompt_and_grants(te: TargetExperiment, bdir, bundle_id: str, variant: str,
                                   manifest: dict[str, Any], cap, written: list) -> None:
    """Emit the derivable, non-manifest bundle files IDEMPOTENTLY (never overwrite a file that already
    exists, so hand-authored bundles — e.g. gemmini's committed prompts — are untouched):
      * ``STARTER_PROMPT.md`` — the target-general task prompt (``generate_prompt.render_prompt``); passed
        the bundle STEM as the arm so the assisted/CIRCT arms get their seam menu (``_is_assisted_arm``
        keys on the ``merlin_assisted`` substring, which the stem — not the short arm key — carries).
      * ``allowed_files.txt`` / ``denied_files.txt`` — the manifest's allow/deny path lists.
    ``cap`` is the target's capability manifest (or ``None`` if it could not be loaded — then the prompt is
    skipped with the grants still written)."""
    from pathlib import Path
    bdir = Path(bdir)
    stem = bundle_id[: -(len(variant) + 1)] if bundle_id.endswith("_" + variant) else bundle_id
    # hwbringup bundles are the REALISTIC experiment's info set; anything else renders at full scope.
    experiment = "realistic" if variant.startswith("hwbringup") else "full"

    def _w(name: str, text: str) -> None:
        p = bdir / name
        if not p.exists():
            p.write_text(text)
            written.append(p)

    if cap is not None:
        from .generate_prompt import render_prompt
        _w("STARTER_PROMPT.md", render_prompt(te, cap, experiment, stem))
    _w("allowed_files.txt", _grant_txt(manifest, "allowed"))
    _w("denied_files.txt", _grant_txt(manifest, "denied"))


def materialize_bundles(te: TargetExperiment, dest, *,
                        variants: tuple[str, ...] = ("hwbringup_v0",)) -> list["Path"]:
    """Write every generated bundle under ``dest/<bundle_id>/`` for each requested ``variant``:
    ``input_bundle_manifest.yaml`` (always, overwritten — the manifest is fully generated) plus the
    derivable non-manifest files (``STARTER_PROMPT.md`` + ``allowed/denied_files.txt``) written only when
    ABSENT so hand-authored bundles stay untouched. Target-agnostic. Returns the written paths.

    ``dest`` is typically ``experiments/<exp>/input_bundles`` — the same tracked location the launcher and
    ``require_scaffolding`` read (bundles are curated inputs, not ``out/`` generated output)."""
    from pathlib import Path
    dest = Path(dest)
    # The capability manifest (needed to render STARTER_PROMPT.md) is target-level; load it once and
    # degrade honestly if unavailable (e.g. mlc absent) — manifests + grant files are still written.
    try:
        from .target_experiment import declared_vs_resolved_contract, load_capability_manifest
        # When the registry resolves nothing for this target, fall back to the contract the DESCRIPTOR
        # declares. Without this a descriptor could name its contract, have that file sit right there on
        # disk, and still render no prompt — which is how a target reached "bundles generated" with three
        # of its four STARTER_PROMPT.md missing and the anti-cheat gate failing on their absence.
        declared, resolved, verdict = declared_vs_resolved_contract(te)
        explicit = declared if verdict == "declared_only" else None
        if explicit:
            print(f"  note: registry resolves no contract for {te.target!r}; using the descriptor's "
                  f"declared {te.declared_contract}")
        cap = load_capability_manifest(te.target, contract_path=explicit)
    except Exception as e:  # noqa: BLE001 — no capability manifest -> skip prompt, keep the rest
        cap = None
        print(f"  note: capability manifest for {te.target!r} unavailable ({type(e).__name__}: {e}); "
              f"STARTER_PROMPT.md not rendered (manifests + grant files still written).")
    written: list[Path] = []
    for variant in variants:
        for bundle_id, manifest in generate_bundles(te, variant=variant).items():
            bdir = dest / bundle_id
            bdir.mkdir(parents=True, exist_ok=True)
            out = bdir / "input_bundle_manifest.yaml"
            out.write_text(_dump_manifest(manifest))
            written.append(out)
            _materialize_prompt_and_grants(te, bdir, bundle_id, variant, manifest, cap, written)
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
