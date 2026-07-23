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
