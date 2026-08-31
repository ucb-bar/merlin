"""The arm-gated tool registry and the ablation cells it generates.

These pin the properties the ablation MEASUREMENT depends on. Each one corresponds to a way a cell can
look like it varied one thing while actually varying none, or several:

* the ladder must stay nested, or a delta is not attributable to one addition;
* a dropped tool must be DENIED rather than merely absent, or a shared grant re-exposes it and the cell
  silently measures nothing;
* the default (no add/drop) must reproduce the ladder exactly, or every historical bundle moves;
* a tool other tools import must not be droppable alone, or the cell measures their absence too.
"""
from __future__ import annotations

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen import tool_registry as TR
from merlin.targetgen.generate_bundles import _arm_manifest, generate_bundles
from merlin.targetgen.target_experiment import load_target_experiment

# One real descriptor is enough: every tool path in the registry is a literal shared by all targets, and
# the single target-varying grant is exercised through the descriptor attribute that derives it.
_DESCRIPTOR = ("merlin/experiments/capsule_bench/targets/gemmini/target_experiment.yaml")


@pytest.fixture(scope="module")
def te():
    return load_target_experiment(repo_root() / _DESCRIPTOR)


def _sets(te, arm, add=(), drop=()):
    m = _arm_manifest(te, arm, "bid", add_tools=add, drop_tools=drop)
    return ({e["path"] for e in m["allowed"]}, {e["path"] for e in m["denied"]}, m)


def test_every_registry_path_exists_on_disk():
    """A grant naming a path that moved silently grants nothing — the arm loses the tool and nobody sees
    it. This is not hypothetical: the arm-4 generators were unusable for three live runs because one
    import was missing from the grant set."""
    missing = [p for t in TR.TOOLS.values() for p in t.bundle_paths
               if not (repo_root() / p).exists()]
    assert not missing, f"registry names paths that do not exist: {missing}"


def test_every_tool_is_reachable_either_by_grant_or_by_broker():
    for name, t in TR.TOOLS.items():
        assert t.bundle_paths or t.derived_paths or t.broker, f"{name} grants nothing and has no broker"


def test_the_ladder_is_nested():
    """Each rung must be a superset of the one before it, or the arm-to-arm delta has two causes."""
    order = ["raw_baseline", "merlin_assisted", "merlin_rtlchecks"]
    for lo, hi in zip(order, order[1:]):
        assert set(TR.ARM_TOOLS[lo]) <= set(TR.ARM_TOOLS[hi]), f"{hi} is not a superset of {lo}"


def _granted(path: str, allow: set[str]) -> bool:
    """Is ``path`` reachable in ``allow`` — literally, or through a directory grant that contains it?

    Containment has to be directory-aware or the property is untestable where it matters most: arm-2
    grants three FILES under ``targetgen/generate/`` and arm-3 grants that whole DIRECTORY, so a literal
    subset test says arm-2 escapes the ladder when in fact it is contained.
    """
    return path in allow or any(g.endswith("/") and path.startswith(g) for g in allow)


def _tool_paths(te, name):
    """Every repo path one tool grants — literal plus the descriptor-derived one."""
    t = TR.spec(name)
    return [*t.bundle_paths, *(getattr(te, attr) for attr in t.derived_paths)]


def test_every_rung_is_contained_in_the_rung_above_it(te):
    """Path-level nesting across the WHOLE ladder, not just the three arms whose tool NAMES nest.

    ``test_the_ladder_is_nested`` compares tool-name sets, and by that measure arm-2 is not a rung at all:
    ``cpp_oot_generators`` is not in ``_ASSISTED``. But arm-2's three file grants sit inside the directory
    arm-3 grants, so the ladder IS nested where it counts -- in what the sandbox can actually read. Nothing
    asserted that, which left the real invariant resting on a coincidence of spelling: replace arm-3's
    ``targetgen/generate/`` directory grant with individual files and arm-2 silently stops being a subset,
    with no test to notice.
    """
    for lo, hi in (("raw_baseline", "cpp_merlininfra"),
                   ("cpp_merlininfra", "merlin_assisted"),
                   ("merlin_assisted", "merlin_rtlchecks")):
        a_lo, _, _ = _sets(te, lo)
        a_hi, _, _ = _sets(te, hi)
        escaped = [p for p in a_lo if not _granted(p, a_hi)]
        assert not escaped, f"{lo} grants what {hi} does not: {escaped}"


def _readable(path: str, allow: set[str], deny: set[str]) -> bool:
    """Can the sandbox actually read ``path`` under this manifest? Granted (directly or by a directory
    grant) and not masked by a denial (likewise directory-aware, since deny wins)."""
    return _granted(path, allow) and not _granted(path, deny)


def test_a_rung_cannot_read_any_tool_the_rungs_above_it_add(te):
    """The arm-to-arm contrast has to be symmetric: no rung may reach a tool that defines a higher rung.

    Stated as REACHABILITY rather than as "is explicitly denied", because the ladder legitimately uses two
    mechanisms and only the effect is the invariant. ``targetgen/rtl/`` is explicitly denied to arm-3
    (belt and braces: it would otherwise be one broadened grant away from exposure), while the eqsat seam
    modules sit at a path no arm-3 grant covers and are excluded by deny-by-default alone. Demanding an
    explicit denial for both would fail on a manifest that is perfectly correct, and demanding neither
    would miss the case this exists to catch -- a shared grant widening to re-expose a higher rung's tool
    while the arms still claim to differ by one thing.

    This covers the two grants most likely to drift and previously untested: ``rtl_facts``, whose path is
    DERIVED from the descriptor rather than spelled in the registry, and ``eqsat_seam``, whose entire claim
    to attribution is that the eqsat arm differs from arm-3 in exactly one way.
    """
    for lower, higher in (("merlin_assisted", "merlin_rtlchecks"),
                          ("merlin_assisted", "merlin_eqsat"),
                          ("cpp_merlininfra", "merlin_assisted")):
        allow, deny, _ = _sets(te, lower)
        added = set(TR.ARM_TOOLS[higher]) - set(TR.ARM_TOOLS[lower])
        assert added, f"precondition: {higher} adds nothing to {lower}"
        for tool in sorted(added):
            for path in _tool_paths(te, tool):
                assert not _readable(path, allow, deny), (
                    f"{lower} can read {path!r} (from tool {tool!r}, which is supposed to be what "
                    f"{higher} ADDS) -- the two arms no longer differ by that tool")


def test_dropping_the_rtl_tools_from_arm4_reproduces_arm3(te):
    """The headline identity: arm-4 minus its own additions IS arm-3. If this drifts, the ablation and
    the A/B are measuring different contrasts."""
    a3, _, _ = _sets(te, "merlin_assisted")
    a4m, _, _ = _sets(te, "merlin_rtlchecks", drop=("rtl_generators", "rtl_facts"))
    assert a4m == a3


def test_adding_arm4s_tools_to_the_baseline_grants_exactly_those_paths(te):
    """The other direction: the baseline plus arm-4's tool set grants precisely arm-4's tool paths and
    nothing else — no shared grant leaks in through the ablation path."""
    a1, _, _ = _sets(te, "raw_baseline")
    a4, _, _ = _sets(te, "merlin_rtlchecks")
    a1p, _, _ = _sets(te, "raw_baseline", add=TR.ARM_TOOLS["merlin_rtlchecks"])
    assert (a1p - a1) == (a4 - a1)


def test_a_dropped_tool_is_denied_not_merely_absent(te):
    """Deny-by-default is not enough: the shared contract/toolchain grants can re-expose a path that is
    simply missing from the allow list. A dropped tool must be actively masked."""
    a4, _, _ = _sets(te, "merlin_rtlchecks")
    a3, _, _ = _sets(te, "merlin_assisted")
    _, denied, man = _sets(te, "merlin_rtlchecks", drop=("rtl_generators", "rtl_facts"))
    for path in a4 - a3:
        assert path in denied, f"{path} was dropped but not denied"
    reasons = [e["reason"] for e in man["denied"] if e["path"] in (a4 - a3)]
    assert all("ablated" in r for r in reasons), reasons


def test_an_added_tool_loses_the_denial_its_rung_wrote(te):
    """Deny wins in the sandbox, so granting a path while its own denial still stands grants nothing."""
    _, denied, _ = _sets(te, "merlin_assisted")
    rtl_paths = set(TR.TOOLS["rtl_generators"].bundle_paths)
    assert rtl_paths & denied, "precondition: arm-3 denies the RTL generators"
    allowed2, denied2, _ = _sets(te, "merlin_assisted", add=("rtl_generators",))
    assert rtl_paths <= allowed2
    assert not (rtl_paths & denied2), "granted path is still denied — the grant is inert"


def test_default_generation_is_unchanged_by_the_ablation_machinery(te):
    """No add/drop must reproduce the ladder's bundle ids exactly, so existing runs and report paths
    keep resolving."""
    plain = generate_bundles(te, variant="hwbringup_v0")
    assert set(plain) == {
        "raw_baseline_hwbringup_v0", "cpp_merlininfra_hwbringup_v0", "merlin_assisted_hwbringup_v0",
        "merlin_assisted_rtlchecks_hwbringup_v0", "merlin_assisted_eqsat_hwbringup_v0"}
    for bid, man in plain.items():
        assert bid.endswith("hwbringup_v0"), "a default bundle must carry no ablation suffix"
        assert man["bundle_id"] == bid


def test_a_cell_names_itself_in_its_bundle_id(te):
    cells = generate_bundles(te, variant="hwbringup_v0", drop_tools=("rtl_generators",),
                             arms=("merlin_rtlchecks",))
    assert list(cells) == ["merlin_assisted_rtlchecks_hwbringup_v0-rtl_generators"]


def test_shared_infrastructure_cannot_be_ablated_alone(te):
    """merlin/common is imported by the granted tools themselves. Dropping it would not measure its
    absence, it would disable several tools at once — which already happened once, unnoticed."""
    assert not TR.TOOLS["merlin_infra"].ablatable
    with pytest.raises(ValueError, match="import"):
        _arm_manifest(te, "merlin_rtlchecks", "bid", drop_tools=("merlin_infra",))


def test_an_unknown_tool_name_fails_closed(te):
    """A typo must not silently produce a cell identical to its own rung."""
    with pytest.raises(TR.UnknownTool):
        _arm_manifest(te, "merlin_rtlchecks", "bid", drop_tools=("rtl_genrators",))
    with pytest.raises(TR.UnknownTool):
        TR.arm_tools("merlin_assisted", add=("no_such_tool",))


def test_adding_a_tool_the_arm_already_has_is_a_noop(te):
    """A sweep walks every tool against every arm; it should not need to know the ladder's shape."""
    base = TR.arm_tools("merlin_rtlchecks")
    assert TR.arm_tools("merlin_rtlchecks", add=("rtl_generators",)) == base
    assert TR.arm_tools("raw_baseline", drop=("rtl_generators",)) == TR.arm_tools("raw_baseline")


def test_brokered_tools_grant_no_paths_and_gate_the_brokers():
    """A brokered tool is staged into the workspace, never bound from the repo — so it must contribute
    no grant, and the driver must learn about it from the tool set rather than from the arm's name."""
    for name in ("isa_tools", "cca_tools"):
        t = TR.spec(name)
        assert not t.bundle_paths and not t.derived_paths
        assert t.broker is not None
    assert len(TR.brokers_for(TR.ARM_TOOLS["merlin_rtlchecks"])) == 2
    assert TR.brokers_for(TR.ARM_TOOLS["raw_baseline"]) == ()
    assert len(TR.brokers_for(TR.arm_tools("merlin_rtlchecks", drop=("isa_tools",)))) == 1


def test_the_registry_names_no_target():
    """The cardinal rule: a rung is the same set of literal module paths for every target. The one
    target-varying grant is named indirectly, as a descriptor attribute."""
    from merlin.targetgen import tool_registry
    src = (repo_root() / "merlin/python/merlin/targetgen/tool_registry.py").read_text()
    for token in ("gemmini", "atlas", "radiance", "saturn", "muon", "vortex"):
        assert token not in src.lower(), f"registry names the target {token!r}"
    assert TR.TOOLS["rtl_facts"].derived_paths == ("rtl_facts_pin",)
    assert hasattr(load_target_experiment(repo_root() / _DESCRIPTOR), "rtl_facts_pin")
    del tool_registry


def _workflow(te, arm, **kw):
    """The mandatory-workflow block a bundle's OWN grant set produces (cheap: no capability manifest)."""
    from merlin.targetgen.generate_prompt import _enforced_workflow
    m = _arm_manifest(te, arm, "bid", **kw)
    granted = {e["path"] for e in m["allowed"]
               if str(e.get("path", "")).startswith(("merlin/", "experiments/"))}
    return _enforced_workflow(arm, "inline_asm_insn", granted, te.target)


def test_the_prompt_stops_mandating_a_tool_the_cell_withheld(te):
    """A cell must never tell the agent to use a tool it cannot open.

    The mandatory-workflow block is derived from the bundle's grant set, so withholding a tool has to
    remove its step. Without this the agent is handed its full rung's checklist and burns rounds
    reaching for something that is not there -- which is how an arm once ran a whole campaign without
    executing a single one of its generators.
    """
    full = _workflow(te, "merlin_rtlchecks")
    assert "xDSL pass pipeline" in full
    assert "xDSL pass pipeline" not in _workflow(te, "merlin_rtlchecks", drop_tools=("xdsl_kit",))
    assert "check-bijection" in full
    assert "check-bijection" not in _workflow(te, "merlin_rtlchecks", drop_tools=("cca_spine",))


def test_the_rtl_mandate_needs_both_rtl_tools_dropped(te):
    """The prompt reads the generators and the extracted facts as ONE capability: it mandates deriving
    from the RTL if EITHER is granted. So dropping the generators alone is a real cell, but it is not a
    'no RTL' cell -- the agent still has the facts and is still told to use them. Anyone reading the
    ablation table needs that distinction, so pin it rather than leave it to be rediscovered.
    """
    assert "RTL-checks arm" in _workflow(te, "merlin_rtlchecks", drop_tools=("rtl_generators",))
    assert "RTL-checks arm" not in _workflow(te, "merlin_rtlchecks",
                                             drop_tools=("rtl_generators", "rtl_facts"))
