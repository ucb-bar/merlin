"""The optimization agent's feedback channel: what it is told about the machine, and by whom.

Three separate defects left the phase-2 agent close to blind, and all three were silent by
construction -- each one produced a well-formed document that simply omitted something, which is
indistinguishable from a document about a program that had nothing to report.

1. THE INSTRUCTION SET NEVER REACHED IT. `agent_guidance` has carried two complete findings --
   `declared_capability_unused` and `declared_capability_refused` -- since they were written, and
   nothing in production ever wrote the keys they read. Repo-wide the only writers were a test. The
   prompt of a live round (`phase2_arm4_frombest_20260909`, round 2) contains zero mentions of the
   instruction set: the agent's only view of the machine was a count of the instructions it had
   already emitted, which cannot name a capability it never reached for.

2. THE MEASURED PROBE WAS ADVERTISED AND REFUSED. `profile-reduced-global-witness` was registered,
   budgeted at one call a round, described at length in the prompt as the route to a calibrated
   cost -- and returned rc=125 in that same run, because its provider is installed only when the
   launcher is given a probe interface. The document said `probes_available: False` in a field
   nothing joined to the action list.

3. NO CYCLE FLOOR UNDER THE DISPATCH DELTA. The capsule grade attaches a `cost_plane` to every row
   it scores; none of it crossed into the optimization loop, which got a structural dispatch delta
   whose own licence reads "fewer dispatches do not prove fewer cycles", beside
   `cycle_selection: UNMEASURED`.

EVERY TEST HERE CARRIES A MUTATION THAT FAILS IT. A comment cannot detect silence: the assertions
are written so that removing the production wiring -- not the library function, the WIRING -- turns
them red. Where the check is structural it says so and says which edit it would catch.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys

import pytest
from merlin_experiments.phase2 import agent_view as AV
from merlin_experiments.phase2 import broker as PB
from merlin_experiments.phase2 import broker_policy as BP
from merlin_experiments.phase2 import emission_analysis as EA
from merlin_experiments.phase2 import emission_diagnostics as ED
from merlin_experiments.phase2 import portfolio_authoring as PAuthor
from merlin_experiments.phase2 import whole_model as WM

from merlin.common.paths import merlin_dir, repo_root
from merlin.common.provenance import load_artifacts
from merlin.perf.gate_phase import configured_phase

SCRIPTS = repo_root() / "merlin" / "experiments" / "gemmini_perf_bench" / "scripts"

#: A target whose facts no checkout derives. Not a name this repo owns -- the point is that nothing
#: can be derived for it, so every producer below must report UNKNOWN rather than a clean number.
UNDERIVABLE_TARGET = "a_target_whose_rtl_facts_are_not_derivable_here"


def _load(name: str):
    """Import one of the phase-2 stage scripts by path (they are scripts, not a package)."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def stage():
    from merlin_experiments.phase2 import authoring

    return authoring


@pytest.fixture(scope="module")
def runner():
    return _load("run_global_perf_experiment")


@pytest.fixture(scope="module")
def derivable_target() -> str:
    """A selected support provider with the complete instruction-facts inputs.

    Derived, never named: the roster comes from the repo's own target descriptors and the first one
    with names and ISA constants is used. Cached decoder metadata alone does not
    establish that its OOT backend and provider-owned RTL facts are available.
    """
    from merlin.perf.task_instruction_evidence import target_instruction_facts
    from merlin.targetgen.target_registry import list_targets as target_names

    unavailable = []
    for name in sorted(target_names()):
        try:
            facts = target_instruction_facts(name)
        except (KeyError, FileNotFoundError) as exc:
            unavailable.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if facts.get("instruction_names") and facts.get("isa", {}).get("CUSTOM_OPCODE") is not None:
            return name
        unavailable.append(f"{name}: incomplete instruction names or ISA constants")
    pytest.skip("no selected support has complete instruction facts; " + "; ".join(unavailable))


def test_derivable_fixture_requires_complete_selected_support(monkeypatch):
    from merlin.kernels.decode import rocc
    from merlin.perf import task_instruction_evidence as evidence
    from merlin.targetgen import target_registry

    monkeypatch.setattr(target_registry, "list_targets", lambda: ["a_metadata_only", "b_complete"])
    monkeypatch.setattr(rocc, "funct_table_for", lambda name: {"names": {0: "operation"}, "custom_opcode": 1})

    def facts(name):
        if name == "a_metadata_only":
            raise KeyError("selected backend unavailable")
        return {"instruction_names": {0: "operation"}, "isa": {"CUSTOM_OPCODE": 1}}

    monkeypatch.setattr(evidence, "target_instruction_facts", facts)
    assert derivable_target.__wrapped__() == "b_complete"


def test_derivable_fixture_reports_missing_support(monkeypatch):
    from merlin.kernels.decode import rocc
    from merlin.perf import task_instruction_evidence as evidence
    from merlin.targetgen import target_registry

    monkeypatch.setattr(target_registry, "list_targets", lambda: ["fixture"])
    monkeypatch.setattr(rocc, "funct_table_for", lambda name: {"names": {0: "operation"}, "custom_opcode": 1})

    def absent(name):
        raise FileNotFoundError("selected support has no RTL facts")

    monkeypatch.setattr(evidence, "target_instruction_facts", absent)
    with pytest.raises(pytest.skip.Exception, match="selected support has no RTL facts"):
        derivable_target.__wrapped__()


def test_derivable_fixture_does_not_hide_unexpected_errors(monkeypatch):
    from merlin.perf import task_instruction_evidence as evidence
    from merlin.targetgen import target_registry

    monkeypatch.setattr(target_registry, "list_targets", lambda: ["fixture"])

    def malformed(name):
        raise ValueError("malformed ISA evidence")

    monkeypatch.setattr(evidence, "target_instruction_facts", malformed)
    with pytest.raises(ValueError, match="malformed ISA evidence"):
        derivable_target.__wrapped__()


@pytest.fixture(scope="module")
def model_capsule():
    """A tracked model capsule with real interface MLIR, for the refusal census."""
    import yaml

    root = merlin_dir() / "contract" / "capsules" / "model"
    for descriptor_path in sorted(root.glob("*/capsule.yaml")):
        descriptor = yaml.safe_load(descriptor_path.read_text(encoding="utf-8")) or {}
        interface = descriptor_path.parent / str(descriptor.get("interface_mlir") or "capsule.interface.mlir")
        if interface.is_file():
            return descriptor_path.parent, descriptor
    pytest.skip("no tracked model capsule carries interface MLIR")


def _brief(analysis):
    from merlin.perf.agent_guidance import PackageOptimizationInventory, guidance_for_emission_analysis

    return guidance_for_emission_analysis(analysis, PackageOptimizationInventory(symbols=(), surfaces=()))


def _kinds(brief) -> set[str]:
    return {row["kind"] for row in brief["ranked_actions"]}


def _assignments_in(path, function: str) -> set[str]:
    """Every literal subscript key assigned into any name inside ``function``.

    Structural (``ast``) rather than a word search: the key names appear in this file's own prose,
    in the consumer's comments and in a test, and counting those would report a dead wiring as live
    -- which is the exact failure this file exists about.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            return {
                target.slice.value
                for inner in ast.walk(node)
                if isinstance(inner, ast.Assign)
                for target in inner.targets
                if isinstance(target, ast.Subscript)
                and isinstance(target.slice, ast.Constant)
                and isinstance(target.slice.value, str)
            }
    raise AssertionError(f"{path} has no function {function!r}")


def _returned_keys(path, function: str) -> set[str]:
    """Every literal key of every dict ``function`` returns. Structural, for the same reason."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            return {
                key.value
                for inner in ast.walk(node)
                if isinstance(inner, ast.Return) and isinstance(inner.value, ast.Dict)
                for key in inner.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
    raise AssertionError(f"{path} has no function {function!r}")


# --------------------------------------------------------------------------------------------
# DEFECT 1 -- the ISA reaches the agent, and the two dead consumers fire
# --------------------------------------------------------------------------------------------


def test_production_writes_the_two_keys_the_brief_reads() -> None:
    """The keys are set in the SAME production function that builds the brief.

    MUTATION: delete either `diagnostics["isa_capability_utilization"] = ...` or
    `diagnostics["capability_refusals"] = ...` from `analyze_whole_model_emission` and this fails.
    The functional tests below would still pass if the producers existed and nothing called them,
    which was the state for the whole life of both findings -- so the wiring gets its own assertion.
    """
    from pathlib import Path

    assigned = _assignments_in(Path(EA.__file__), "analyze_whole_model_emission")
    assert "isa_capability_utilization" in assigned, (
        "nothing in the whole-model analysis writes isa_capability_utilization; "
        "agent_guidance.declared_capability_unused reads it and would never fire"
    )
    assert "capability_refusals" in assigned, (
        "nothing in the whole-model analysis writes capability_refusals; "
        "agent_guidance.declared_capability_refused reads it and would never fire"
    )


def test_there_is_one_producer_of_declared_versus_emitted_use(stage) -> None:
    """The capsule-bench broker and the optimization stage measure with the SAME function.

    MUTATION: reimplement either caller's own declared table and this fails -- which matters because
    the half that drifted would be the half reporting a clean number.
    """
    from merlin.perf.isa_utilization import capability_utilization_for_target

    broker = _load_broker()
    assert broker._capability_utilization.__doc__, "the broker producer lost its statement of intent"
    text = ".insn r 0x0b, 0x3, 0x0, x0, $0, $1"
    assert broker._capability_utilization(text, UNDERIVABLE_TARGET) == capability_utilization_for_target(
        text, target=UNDERIVABLE_TARGET
    )
    assert ED.isa_capability_utilization(text, target=UNDERIVABLE_TARGET)["status"] == "UNKNOWN"


def _load_broker():
    from merlin_experiments.phase1.brokers import isa_tools

    return isa_tools


def test_an_underivable_instruction_table_is_unknown_and_not_full_coverage(stage) -> None:
    """The failure this producer exists to avoid: a clean answer produced by not looking.

    MUTATION: return `capability_utilization(text, declared_functs={}, custom_opcode=0)` instead of
    the UNKNOWN and this fails -- an empty declared table reports nothing unused, so the brief would
    say the compiler reaches for everything the machine offers.
    """
    out = ED.isa_capability_utilization(".insn r 0x0b, 0x3, 0x0, x0, $0, $1", target=UNDERIVABLE_TARGET)
    assert out["status"] == "UNKNOWN"
    assert out["declared_count"] is None
    assert out["unused"] == []
    assert "declared_capability_unused" not in _kinds(_brief({"isa_capability_utilization": out}))


def test_the_unused_capability_finding_fires_on_the_production_producer(stage, derivable_target) -> None:
    """The first dead consumer, driven by the production producer rather than by a test literal.

    MUTATION: have `isa_capability_utilization` return `{}` on an emitted artifact and this fails.
    """
    emitted = 'llvm.inline_asm has_side_effects ".insn r 0x0b, 0x3, 0x0, x0, $0, $1"'
    utilization = ED.isa_capability_utilization(emitted, target=derivable_target)
    if utilization.get("status") != "measured":
        pytest.skip(f"declared table not derivable here: {utilization.get('reason')}")
    brief = _brief({"isa_capability_utilization": utilization})
    assert "declared_capability_unused" in _kinds(brief), (
        "the target declares instructions this artifact never emits and the brief said nothing"
    )
    finding = next(row for row in brief["ranked_actions"] if row["kind"] == "declared_capability_unused")
    assert finding["evidence"]["declared_count"] == utilization["declared_count"]
    assert finding["evidence"]["unused"], "a finding about unused capabilities named none of them"


def test_the_refused_capability_finding_fires_on_the_production_producer(stage, model_capsule) -> None:
    """The second dead consumer. It says WHY a capability is absent, which the first cannot.

    MUTATION: return `[]` from `capability_refusals` and this fails; returning `[]` is also the
    shape the old code had, and it reads downstream as "nothing was refused".
    """
    from merlin.perf.model_placement import prepare_captured_source
    from merlin.targetgen.target_registry import list_targets as target_names

    capsule_dir, descriptor = model_capsule
    prepared = prepare_captured_source(capsule_dir / str(descriptor.get("interface_mlir")))
    datapath = ED.declared_capsule_datapath(descriptor)
    for target in sorted(target_names()):
        rows = ED.capability_refusals(prepared, target=target, datapath=datapath)
        assert rows, "the refusal census returned nothing at all, not even an UNKNOWN"
        if rows[0].get("status") != "measured" or not rows[0].get("clauses"):
            continue
        brief = _brief({"capability_refusals": rows})
        assert "declared_capability_refused" in _kinds(brief)
        finding = next(row for row in brief["ranked_actions"] if row["kind"] == "declared_capability_refused")
        assert finding["evidence"]["clauses"], "a refusal finding named no clause"
        assert finding["evidence"]["caveat"], "the short-circuit caveat did not travel with the census"
        return
    pytest.skip("no target in this checkout produced a measured placement census for this capsule")


def test_a_census_that_could_not_be_taken_says_so_rather_than_reporting_none(stage) -> None:
    """Fail-closed. An empty list and "nothing was refused" are the same bytes downstream.

    MUTATION: return `[]` on the un-parsed path and this fails.
    """
    rows = ED.capability_refusals(None, target=UNDERIVABLE_TARGET, datapath=None)
    assert len(rows) == 1
    assert rows[0]["status"] == "UNKNOWN"
    assert rows[0]["clauses"] == []
    assert "not a statement that none was refused" in rows[0]["caveat"]


def test_the_declared_instruction_set_survives_into_the_prompt(runner, derivable_target) -> None:
    """The prompt of the live round mentions the instruction set zero times; this is that repair.

    MUTATION: drop the `declared_instruction_prompt(...)` line from the round's prompt builder and
    the structural half of this fails; make the brief return its hash instead of its members and the
    functional half fails.
    """
    brief = runner.declared_instruction_set_brief(derivable_target)
    assert brief["status"] == "derived"
    assert brief["declared_count"] == len(brief["instructions"]) > 0
    prompt = AV.declared_instruction_prompt(brief)
    for entry in brief["instructions"]:
        assert entry["name"] in prompt, f"the prompt omits the declared instruction {entry['name']}"
    source = inspect.getsource(PAuthor)
    assert "text += AV.declared_instruction_prompt(" in source, (
        "the round's prompt no longer includes the declared instruction set"
    )


def test_an_underivable_instruction_set_is_stated_in_the_prompt_not_omitted(runner) -> None:
    """A prompt silent about the ISA and a prompt whose ISA is UNKNOWN must not read alike."""
    brief = runner.declared_instruction_set_brief(UNDERIVABLE_TARGET)
    assert brief["status"] == "UNKNOWN"
    assert "UNKNOWN" in AV.declared_instruction_prompt(brief)


def test_task_instruction_evidence_keeps_the_declared_set_not_only_its_hash(derivable_target) -> None:
    """`task_instruction_binding` reduced the facts to `target_facts_sha256` and the declared set,
    freshly derived on every analysis, was discarded.

    MUTATION: drop `declared_instruction_set` from the summary's result and this fails.
    """
    from merlin.perf.task_instruction_evidence import declared_instruction_set, target_instruction_facts

    facts = target_instruction_facts(derivable_target)
    declared = declared_instruction_set(facts)
    assert declared["status"] == "derived"
    assert declared["declared_count"] == len(facts["instruction_names"])
    assert declared_instruction_set({})["status"] == "UNKNOWN"
    assert declared_instruction_set({})["instructions"] == []
    # And the summary CARRIES it, rather than the helper merely existing beside it -- which is
    # exactly the state both dead consumers were in.
    returned = _returned_keys(
        merlin_dir() / "python" / "merlin" / "perf" / "task_instruction_evidence.py",
        "summarize_task_instructions",
    )
    assert {"declared_instruction_set", "declared_instruction_use"} <= returned, (
        "the summary still reduces the target's facts to their hash; the declared set it derives "
        "on every analysis is computed and thrown away"
    )


# --------------------------------------------------------------------------------------------
# DEFECT 2 -- a refused probe cannot report itself as available
# --------------------------------------------------------------------------------------------


def test_an_action_whose_provider_is_absent_is_never_advertised_available(stage) -> None:
    """The live shape: registered, budgeted, advertised, rc=125.

    MUTATION: stop threading `unavailable=` into `build_action_registry` (or drop `available` from
    `BrokerAction.advertised`) and this fails.
    """
    refusals = BP.unavailable_global_actions({})
    assert set(refusals) == {name for name, _attribute, _reason in BP.ACTION_PROVIDER_REQUIREMENTS}
    for name, reason in refusals.items():
        action = PB.BrokerAction(name, ("x",), (), "p", False, unavailable_reason=reason)
        row = action.advertised()
        assert row["available"] is False
        assert row["unavailable_reason"] == reason


def test_every_provider_is_installed_makes_every_action_available(stage) -> None:
    """The other direction, so the check is not satisfied by marking everything unavailable."""
    installed = {attribute: object() for _name, attribute, _reason in BP.ACTION_PROVIDER_REQUIREMENTS}
    assert BP.unavailable_global_actions(installed) == {}
    assert PB.BrokerAction("a", ("x",), (), "p", False).advertised()["available"] is True


def test_a_provider_guarded_handler_missing_from_the_table_is_caught(stage) -> None:
    """Anti-silence. A new action that guards on a provider and is not declared in the table would
    be advertised plainly and refuse at runtime -- exactly the defect, one action later.

    Structural: every `self.global_*_provider is None` guard in the broker's dispatch must name a
    provider the requirement table knows.

    MUTATION: add a handler guarding on a new `self.global_X_provider` without adding its row to
    `ACTION_PROVIDER_REQUIREMENTS` and this fails.
    """
    declared = {attribute for _name, attribute, _reason in BP.ACTION_PROVIDER_REQUIREMENTS}
    tree = ast.parse(inspect.getsource(WM))
    guarded = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Compare)
                and isinstance(inner.left, ast.Attribute)
                and isinstance(inner.left.value, ast.Name)
                and inner.left.value.id == "self"
                and inner.left.attr.startswith("global_")
                and inner.left.attr.endswith("_provider")
                and any(isinstance(op, ast.Is) for op in inner.ops)
            ):
                guarded.add(inner.left.attr)
    assert guarded, "no provider guard was found; this check would pass vacuously"
    assert guarded <= declared, (
        f"these broker providers guard an action that ACTION_PROVIDER_REQUIREMENTS does not "
        f"declare, so the action would be advertised as available and refuse: {sorted(guarded - declared)}"
    )


def test_the_round_names_a_provider_slot_for_every_declared_requirement(runner, stage) -> None:
    """The other end of the same wiring: a provider the table requires that the round never names
    would be reported unavailable while the broker holds it, and the two would disagree.

    MUTATION: add a row to `ACTION_PROVIDER_REQUIREMENTS` naming a provider `run_round`
    does not pass and this fails, instead of a live run quietly advertising the wrong availability.
    """
    tree = ast.parse(inspect.getsource(PAuthor))
    body = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "run_round")
    named = {
        key.value
        for node in ast.walk(body)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "round_providers" for t in node.targets)
        and isinstance(node.value, ast.Dict)
        for key in node.value.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert named, "the round no longer names the providers its availability map is built from"
    required = {attribute for _name, attribute, _reason in BP.ACTION_PROVIDER_REQUIREMENTS}
    assert required <= named, f"the round names no provider slot for {sorted(required - named)}"
    constructor = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    captured = {
        target.attr
        for node in ast.walk(constructor)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name)
        for target in node.targets
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
        and target.attr == node.value.id
    }
    assert required <= captured, "the concrete authoring owner must retain every declared provider"


def test_a_second_refusal_cannot_destroy_the_first_ones_record(stage, tmp_path) -> None:
    """The reason the live probe refused had to be recovered from a stderr HASH, because the record
    naming it was overwritten: the macro path carries no round index, so all three rounds wrote
    `round_None_call_4.txt` and round 2's edit-scope refusal replaced round 0's probe refusal.

    MUTATION: write to a fixed name again and this fails.
    """
    from types import SimpleNamespace

    private = tmp_path / "private"
    private.mkdir()
    holder = SimpleNamespace(feedback_evaluator=SimpleNamespace(work_root=private), global_experiment=None)
    for message in ("the probe provider is absent", "the edit exceeds frozen authority"):
        BP._record_host_refusal(holder, ValueError(message), round_index=None, call_index=4)
    written = sorted((private / "host_refusals").iterdir())
    assert len(written) == 2, f"a later refusal overwrote an earlier one: {[p.name for p in written]}"
    assert "the probe provider is absent" in "".join(p.read_text() for p in written)


def test_the_prompt_names_every_action_this_run_cannot_answer(runner, stage) -> None:
    """A prompt silent about availability reads the same whether every provider is installed or
    none is -- which is how the agent spent its only probe call on a refusal.

    MUTATION: drop `unavailable_action_notice(...)` from the round's prompt and the structural half
    fails; return `""` from it and the functional half fails.
    """
    refusals = BP.unavailable_global_actions({})
    notice = AV.unavailable_action_notice(refusals)
    for name, reason in refusals.items():
        assert name in notice, f"the prompt does not say {name} cannot be answered"
        assert reason in notice
    assert "no retry will make it available" in notice
    assert AV.unavailable_action_notice({}) != notice
    assert "installed" in AV.unavailable_action_notice({})
    source = inspect.getsource(PAuthor)
    assert "text += AV.unavailable_action_notice(" in source


def test_the_stage_context_cannot_advertise_an_action_its_own_flag_calls_absent(runner) -> None:
    """The exact contradiction the live document carried: `profile-reduced-global-witness` listed
    plainly in `broker_actions`, beside `probes_available: False` in the same file.

    Structural, over the round builder itself: the availability map that feeds the registry is the
    same one the context reports, so the two cannot disagree.

    MUTATION: build the registry without `unavailable=` and this fails.
    """
    source = inspect.getsource(PAuthor)
    tree = ast.parse(source)
    body = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "run_round")
    calls = [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "action_registry"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "BP"
    ]
    assert calls, "the round no longer builds an action registry"
    for call in calls:
        assert any(keyword.arg == "unavailable" for keyword in call.keywords), (
            "the round advertises its action registry without the availability map, so an action "
            "with no provider is listed exactly like one that can be answered"
        )
    assert '"unavailable_broker_actions"' in source, (
        "the stage context does not report which actions cannot be answered"
    )


# --------------------------------------------------------------------------------------------
# DEFECT 3 -- a cycle floor under the structural dispatch delta
# --------------------------------------------------------------------------------------------


def _timing_capsule(tier: str = "L3", ceiling: int | None = None) -> dict:
    capsule: dict = {"performance": {"acceptance": {"evidence": {"timing_tier": tier}}}}
    if ceiling is not None:
        capsule["performance"]["cost"] = {"projected_cycles": ceiling}
    return capsule


def test_the_iteration_carries_a_derived_cycle_floor(stage, derivable_target) -> None:
    """What phase 2 got instead was a dispatch delta whose own licence says it proves no cycles.

    MUTATION: drop the per-arm `floor` from `iteration_cost_plane` and this fails.
    """
    plane = ED.iteration_cost_plane(
        _timing_capsule(),
        target=derivable_target,
        arms={"baseline": {"macs": 2_000_000, "exact": True}, "candidate": {"macs": 1_000_000, "exact": True}},
        phase=configured_phase("cost_plane"),
        artifacts=load_artifacts(),
    )
    floors = {arm: plane["arms"][arm]["floor"] for arm in ("baseline", "candidate")}
    if any(floor["cycles"] is None or not isinstance(floor["cycles"], (int, float)) for floor in floors.values()):
        pytest.skip(f"this target's array geometry is not derivable here: {floors}")
    assert floors["candidate"]["cycles"] < floors["baseline"]["cycles"]
    assert plane["array"]["rows"] and plane["array"]["cols"], "the array geometry was not derived"
    assert plane["licence"], "the plane crossed without the statement of what a floor can decide"


def test_the_whole_plane_can_be_written_even_when_nothing_is_derivable(stage) -> None:
    """It travels as JSON in STAGE_CONTEXT, and an underivable floor is a sentinel, not a number.

    MUTATION: embed `derived_floor`'s raw result and this fails -- a document that cannot be written
    is a document nobody reads, and the case it fails in is exactly the unknown one.
    """
    import json

    plane = ED.iteration_cost_plane(
        {},
        target=UNDERIVABLE_TARGET,
        arms={"candidate": {"macs": None}, "baseline": {}},
        phase=configured_phase("cost_plane"),
        artifacts=load_artifacts(),
    )
    json.dumps(plane)  # must not raise
    for arm in ("baseline", "candidate"):
        floor = plane["arms"][arm]["floor"]
        assert floor["cycles"] is None
        assert floor["reason"], "an unknown floor crossed without saying why"


def test_an_unmeasured_iteration_is_incomplete_and_never_a_pass(stage, derivable_target) -> None:
    """`incomplete` is a STATUS orthogonal to phase, and the one this loop reports today.

    MUTATION: let the plane report `within` when no cycle count exists and this fails -- "not
    measured" reading as "measured fine" is the failure this repo keeps rediscovering.
    """
    from merlin.perf.gate_phase import STATUS_INCOMPLETE

    plane = ED.iteration_cost_plane(
        _timing_capsule(),
        target=derivable_target,
        arms={"candidate": {"macs": 1_000}},
        tiers=None,
        phase=configured_phase("cost_plane"),
        artifacts=load_artifacts(),
    )
    assert plane["status"] == STATUS_INCOMPLETE
    assert plane.get("admitted") is False
    assert plane.get("blocking") is False
    assert plane["measured_cycles"] is None


def test_a_measured_count_below_the_array_floor_is_refused_but_blocks_nothing(stage, derivable_target) -> None:
    """The plane's one decidable verdict with nothing invented, at the phase it ships in.

    MUTATION: land this gate at `fail` and the `blocking is False` assertion fails; drop the floor
    and the `below_floor` assertion fails.
    """
    from merlin.perf.gate_phase import PHASE_REPORT, configured_phase

    if configured_phase("cost_plane") != PHASE_REPORT:
        pytest.skip("the cost plane no longer ships at report; this assertion is about that rollout")
    capsule = _timing_capsule()
    design = {
        "substrate": "s",
        "hw_config": "h",
        "hwdb_config_artifact_sha256": "d",
        "engine": "e",
        "cycles": 1,
    }
    plane = ED.iteration_cost_plane(
        capsule,
        target=derivable_target,
        arms={"candidate": {"macs": 10_000_000, "exact": True}},
        tiers={"L3": design},
        phase=configured_phase("cost_plane"),
        artifacts=load_artifacts(),
    )
    if plane.get("floor_cycles") is None:
        pytest.skip(f"this target's array geometry is not derivable here: {plane.get('floor_reason')}")
    assert plane["status"] == "below_floor"
    assert plane["blocking"] is False, "a plane landing at report must block nothing"


def test_the_cost_plane_reaches_the_digest_the_prompt_starts_from(runner) -> None:
    """The prompt tells the agent to start with `portfolio_action_digest`; the plane has to be there.

    MUTATION: drop `"cost_plane"` from the digest row and this fails.
    """
    plane = {"schema": "cost_plane_verdict_v1", "status": "incomplete", "floor_cycles": 512.0}
    utilization = {"schema": "isa_capability_utilization_v1", "status": "measured", "declared_count": 26}
    record = {
        "candidate_sha256": "a" * 64,
        "portfolio": {"portfolio_sha256": "b" * 64, "members": [{"identity": {}, "readiness": {}}]},
        "analysis": {
            "optimization_brief": {},
            "diagnostics": {
                "cost_plane": plane,
                "isa_capability_utilization": utilization,
                "capability_refusals": [{"capability": "c", "clauses": [{"clause": "x", "sites": 1}]}],
            },
        },
    }
    digest = AV.portfolio_action_digest(record, complete_evidence="E.json", edit_contract=None)
    row = digest["members"][0]
    assert row["cost_plane"] == plane
    assert row["isa_capability_utilization"] == utilization
    assert row["capability_refusals"][0]["clauses"]


def test_a_secondary_member_keeps_the_same_three_signals(runner) -> None:
    """`agent_analysis_view` prunes secondary members to a fixed key list; the machine's own
    capabilities and the cycle floor must be in it.

    MUTATION: drop the three keys from `global_signals` and this fails.
    """
    plane = {"schema": "cost_plane_verdict_v1", "status": "incomplete"}
    record = {
        "analysis": {"optimization_brief": {}, "diagnostics": {}},
        "portfolio": {
            "members": [
                {
                    "analysis": {
                        "candidate_sha256": "c" * 64,
                        "optimization_brief": {},
                        "diagnostics": {
                            "cost_plane": plane,
                            "isa_capability_utilization": {"status": "measured"},
                            "capability_refusals": [{"capability": "c"}],
                        },
                    }
                }
            ]
        },
    }
    view = AV.agent_analysis_view(record, complete_evidence="E.json")
    signals = view["portfolio"]["members"][0]["analysis"]["global_signals"]
    assert signals["cost_plane"] == plane
    assert signals["isa_capability_utilization"]["status"] == "measured"
    assert signals["capability_refusals"][0]["capability"] == "c"
