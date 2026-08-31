"""The named catalog of ARM-GATED tools — the one place that says what each rung of the ladder adds.

WHY THIS EXISTS
---------------
The ladder's rungs were declared in two unrelated places that could not be varied together. The
file-read tools came from per-rung literal lists inside :mod:`merlin.targetgen.generate_bundles`, and
the interactive (brokered) tools were gated by a *substring test on the arm name* inside the experiment
driver. So "this arm, plus exactly one more tool" was not expressible: it needed a hand-authored bundle
manifest (which then bypasses the generator the readiness gate compares against) AND an edit to the
driver. An ablation you cannot express is an ablation nobody runs.

This module holds both mechanisms behind one name per tool, so a rung is a SET of names and an ablation
cell is that set plus or minus one element.

WHAT IS AND IS NOT IN HERE
--------------------------
Only the **treatment** — the tools an arm is given *because of which arm it is*. Deliberately absent:

* the COMMON SUBSTRATE every arm gets (the redacted self-check broker, the async sim-job broker, the
  task, the target's own ISA headers). Ablating those would not measure a Merlin capability, it would
  measure whether the benchmark is runnable at all.
* the HARNESS (preflight, grading, the anti-cheat audit). The agent never sees it.

TARGET-AGNOSTIC
---------------
Every ``bundle_paths`` entry is a literal merlin module path, identical for every target — that was
already true of the lists this replaces. The one target-varying grant (a target's extracted RTL facts)
is named INDIRECTLY, as the attribute of the descriptor that derives it, so no target name appears here
and a new target needs no edit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

_PY = "merlin/python/merlin/"  # the agnostic merlin package prefix (identical for every target)


@dataclass(frozen=True)
class BrokerSpec:
    """A driver-side broker + the in-sandbox shims that reach it.

    Brokered tools exist because the tool cannot run INSIDE the box: it needs either the oracle or the
    target model, both of which are masked there. The broker runs outside, the shim forwards over a
    channel dir in the (bind-mounted) workspace. See ``harness/isa_tools_broker.py`` for the pattern.
    """
    channel: str                                  # channel dir under the workspace, e.g. ".isa_channel"
    module: str                                   # the driver-side broker module
    log: str                                      # its log file, written into the channel dir
    shims: tuple[tuple[str, str], ...]            # (shim module, name it is staged as in the workspace)


@dataclass(frozen=True)
class ToolSpec:
    """One arm-gated tool: what it grants, how it is reached, and whether it may be ablated alone."""
    name: str
    blurb: str
    #: Repo-root-relative paths granted read access. Literal + target-agnostic.
    bundle_paths: tuple[str, ...] = ()
    #: ``TargetExperiment`` attribute names whose value is an additional granted path. Used for the one
    #: grant that legitimately varies per target (its extracted RTL facts) — named indirectly so this
    #: table stays free of target names.
    derived_paths: tuple[str, ...] = ()
    #: Set when the tool is reached through a driver-side broker rather than by reading files.
    broker: BrokerSpec | None = None
    #: The note recorded beside each granted path in the manifest (kept per-tool so regenerating a
    #: bundle reproduces the existing files exactly).
    note: str = ""
    #: The reason recorded when this tool is DENIED to an arm that does not carry it.
    deny_reason: str = ""
    #: False for a tool that other tools import. Dropping one alone does not measure its absence, it
    #: breaks every tool that depends on it — which has already happened once (see ``merlin_infra``).
    ablatable: bool = True


#: The ISA dev tools and the CCA contract tools are ORACLE-FREE by construction: the assembler encodes
#: the syntax the agent chose, the disassembler and linter inspect the agent's OWN emitted words, and
#: the CCA calls diff public schema against public routes. They are gated to the assisted arms not
#: because they leak anything, but because unaided raw-ISA authoring is what the baseline measures.
_ISA_BROKER = BrokerSpec(".isa_channel", "isa_tools_broker.py", "isa_tools_broker.log",
                         (("isa_tools_shim.py", "isa_tools.py"),))
_CCA_BROKER = BrokerSpec(".cca_channel", "cca_broker.py", "cca_broker.log",
                         (("cca_shim.py", "cca_contract.py"), ("cca_shim.py", "action_catalog.py")))


TOOLS: dict[str, ToolSpec] = {
    "cpp_oot_generators": ToolSpec(
        "cpp_oot_generators",
        "Generic C++ out-of-tree backend generators: MLIR scaffold, LLVM lowering plan, target repo.",
        bundle_paths=tuple(f"{_PY}targetgen/generate/{m}.py"
                           for m in ("mlir_scaffold", "llvm_plan", "target_repo")),
        note="ALLOWED tool: generic C++ OOT generator",
        deny_reason="denied tool (kept a strict subset of the xDSL arm)"),

    # Shared, answer-free INFRASTRUCTURE every granted merlin tool imports. `targetgen/rtl/facts.py`
    # opens with `from merlin.common.paths import artifacts_dir, targets_dir`, so without this grant the
    # RTL-facts generators die in the sandbox with ModuleNotFoundError: No module named 'merlin.common'.
    # Measured across three live runs (codex 6 hits, codex2 6, nemotron 5): every model tried the granted
    # generators, failed, and either worked around them or stopped reaching for them — so the treatment
    # was partly unavailable to all of them and the arm-4-vs-arm-3 contrast was understated. That is why
    # this one is NOT independently ablatable: dropping it does not remove a capability, it silently
    # removes several.
    "merlin_infra": ToolSpec(
        "merlin_infra",
        "Answer-free support modules imported transitively by the xDSL, CCA and RTL-profile authoring "
        "tools. Not an oracle, grader or answer surface — granting them widens no moat, and withholding "
        "one disables the advertised tool that imports it.",
        bundle_paths=(
            f"{_PY}common/",
            # ``targetgen.synthesize`` imports these at package import time.  Grant exact modules rather
            # than all of targetgen: that tree also contains the grader and callable oracle routes.
            f"{_PY}targetgen/families.py",
            f"{_PY}targetgen/compute_units.py",
            f"{_PY}targetgen/semantic_families.py",
            f"{_PY}targetgen/target_experiment.py",
            f"{_PY}targetgen/evidence/store.py",
            # ``interface_emit`` needs these only for a pooled COMMIT, which is why import-only smoke
            # missed them. They are pure command-buffer shape helpers; the oracle-bearing runtime
            # reference/simulator modules remain explicitly denied.
            f"{_PY}runtime/commandbuffer.py",
            f"{_PY}runtime/tensor.py",
            # ``rtl_backend.derived_levers`` consults endpoint roles.  Again keep the closure exact:
            # the complete kernels tree contains unrelated implementation and benchmark surfaces.
            f"{_PY}kernels/endpoints.py",
            f"{_PY}kernels/roles.py",
        ),
        note="ALLOWED tool: xDSL kit / CCA spine",
        ablatable=False),

    "xdsl_kit": ToolSpec(
        "xdsl_kit",
        "The xDSL authoring kit: dialect synthesis, the generators, the dialect definitions, the "
        "interface emitters and the out-of-tree starter kit.",
        bundle_paths=(f"{_PY}targetgen/synthesize/", f"{_PY}targetgen/generate/", f"{_PY}xdsl_dialects/",
                      f"{_PY}targetgen/contract/interface_emit.py",
                      f"{_PY}targetgen/contract/linalg_iface.py", f"{_PY}targetgen/oot_starterkit/"),
        note="ALLOWED tool: xDSL kit / CCA spine"),

    "cca_spine": ToolSpec(
        "cca_spine",
        "The Common-Compute-Abstraction spine — the where/how of modifying a compiler: extract a CCA, "
        "diff two, check the CCA<->action bijection, walk the escalation ladder, author a microkernel.",
        bundle_paths=tuple(f"{_PY}kernels/{m}.py" for m in
                           ("cca", "cca_compare", "cca_contract", "action_catalog", "microkernel"))
                     + (f"{_PY}targetgen/rtl_backend.py",),
        note="ALLOWED tool: xDSL kit / CCA spine"),

    "rtl_generators": ToolSpec(
        "rtl_generators",
        "The CIRCT RTL-fact generators: derive an ISA encoder module, a distilled RTL digest and a "
        "numeric-shape checker from the target's elaborated RTL rather than from its documentation.",
        bundle_paths=(f"{_PY}targetgen/rtl/",),
        note="ALLOWED (CIRCT arm): RTL-facts generators",
        deny_reason="CIRCT RTL generators (CIRCT arm only)"),

    "rtl_facts": ToolSpec(
        "rtl_facts",
        "The facts already extracted from THIS target's RTL — the generators' output, granted directly.",
        derived_paths=("rtl_facts_pin",),
        note="ALLOWED (CIRCT arm): RTL-extracted facts",
        deny_reason="RTL facts (CIRCT arm only)"),

    # The treatment under test is the SEAM itself — the agent registers its own implementation as an
    # alternative in an e-class and the extractor chooses — so the arm carrying it must differ from the
    # arm it is compared against in exactly this one declared way.
    "eqsat_seam": ToolSpec(
        "eqsat_seam",
        "The equivalence seam: an e-graph over real IR plus the persistent equivalence store.",
        bundle_paths=(f"{_PY}targetgen/contraction_egraph.py",
                      f"{_PY}targetgen/persistent_equivalence.py"),
        note="ALLOWED (eqsat arm): the equivalence seam",
        deny_reason="equivalence seam (eqsat arm only)"),

    "isa_tools": ToolSpec(
        "isa_tools",
        "Derived assembler, disassembler, static linter and lite debugger for the target's own ISA. "
        "Oracle-free: it encodes the syntax you chose and inspects the words you emitted.",
        broker=_ISA_BROKER),

    "cca_tools": ToolSpec(
        "cca_tools",
        "The two mandated CCA introspection calls — check_bijection and escalation_ladder — reachable "
        "as plain imports inside the sandbox. Oracle-free: public schema against public routes.",
        broker=_CCA_BROKER),
}


#: Which tools each rung of the ladder carries. The ladder is NESTED — every set is a superset of the
#: one before it — which is what makes a delta attributable to exactly one addition. Order matters: it
#: is the order grants are written into a manifest.
_ASSISTED = ("merlin_infra", "xdsl_kit", "cca_spine", "isa_tools", "cca_tools")
ARM_TOOLS: dict[str, tuple[str, ...]] = {
    "raw_baseline":     (),
    "cpp_merlininfra":  ("cpp_oot_generators",),
    "merlin_assisted":  _ASSISTED,
    "merlin_rtlchecks": _ASSISTED + ("rtl_generators", "rtl_facts"),
    # The eqsat arm shares the xDSL arm's denials on purpose: an arm that also gained the RTL facts
    # would differ in TWO ways and its result would not attribute to the seam.
    "merlin_eqsat":     _ASSISTED + ("eqsat_seam",),
}


class UnknownTool(KeyError):
    """Raised for a tool name no rung declares — fail closed rather than silently ablate nothing."""


def spec(name: str) -> ToolSpec:
    try:
        return TOOLS[name]
    except KeyError:
        raise UnknownTool(f"unknown tool {name!r}; known: {sorted(TOOLS)}") from None


def known_tools() -> tuple[str, ...]:
    return tuple(sorted(TOOLS))


def ablatable_tools() -> tuple[str, ...]:
    """The tools an ablation may add or drop on its own (excludes shared infrastructure)."""
    return tuple(sorted(n for n, t in TOOLS.items() if t.ablatable))


def arm_tools(arm: str, *, add: tuple[str, ...] = (), drop: tuple[str, ...] = ()) -> tuple[str, ...]:
    """The resolved tool set for ``arm``, plus ``add``, minus ``drop`` — the ABLATION CELL.

    Adding a tool the arm already has, or dropping one it does not have, is a no-op rather than an
    error: a sweep that walks every tool against every arm should not need to know the ladder's shape.
    Dropping a non-ablatable tool RAISES — it would disable the tools that import it, so the cell would
    not measure what its name says.
    """
    for n in (*add, *drop):
        spec(n)                                   # fail closed on a typo before anything is generated
    bad = [n for n in drop if not spec(n).ablatable]
    if bad:
        raise ValueError(
            f"cannot ablate {bad} alone: other granted tools import it, so the cell would measure their "
            f"absence too. Ablatable tools: {list(ablatable_tools())}")
    try:
        base = ARM_TOOLS[arm]
    except KeyError:
        raise KeyError(f"unknown arm {arm!r}; known: {sorted(ARM_TOOLS)}") from None
    dropped = set(drop)
    out = [n for n in base if n not in dropped]
    out += [n for n in add if n not in out]
    return tuple(out)


def brokers_for(tools: tuple[str, ...]) -> tuple[BrokerSpec, ...]:
    """The driver-side brokers the resolved tool set requires (empty for a file-read-only cell)."""
    return tuple(spec(n).broker for n in tools if spec(n).broker is not None)


def cell_suffix(add: tuple[str, ...] = (), drop: tuple[str, ...] = ()) -> str:
    """The bundle-id suffix naming an ablation cell, so a run directory says which cell produced it.

    Empty when nothing was varied — which is what keeps every default bundle id, and therefore every
    existing run path, unchanged.
    """
    return "".join([*(f"+{n}" for n in sorted(add)), *(f"-{n}" for n in sorted(drop))])
