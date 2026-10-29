"""The joint campaign's prohibited instruction roles: policy loading, the author's prompt, and silence.

The policy names roles of the closed vocabulary (``prohibited_instruction_roles``). It must load and
reject anything outside that vocabulary, state the rule in one plain prompt line derived from the policy,
and nothing the author is offered (skills, the optimization catalogue, the mechanism hints and the
library pass in ``optimization_next``) may suggest the hardware-loop mechanism the role names.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from merlin_experiments.phase1 import joint_control as JC
from merlin_experiments.phase1.feedback import mechanism_hints as MH
from merlin_experiments.phase1.feedback import performance_brief as PB

from merlin.common.paths import repo_root
from merlin.compile.analysis.instruction_roles import ROLES

ROLE = "loop_descriptor"
#: How a suggestion would name the mechanism the role prohibits.
HARDWARE_LOOP_PHRASES = (ROLE, "hardware loop", "hardware-loop", "loop unroller", "loop-unroller")
SKILLS = Path(JC.__file__).with_name("author_skills")
FIXTURE = json.loads(Path(__file__).with_name("library_pass_fresh4.json").read_text(encoding="utf-8"))


def _mentions_hardware_loop(text: str) -> list[str]:
    lowered = " ".join(str(text).lower().split())
    words = set("".join(c if c.isalnum() else " " for c in lowered).split())
    return [p for p in HARDWARE_LOOP_PHRASES if p in lowered] + (["fsm"] if "fsm" in words else [])


def _policy(tmp_path, **extra):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"version": 1, **extra}))
    return JC.policy(path)


def test_the_policy_loads_the_role_and_rejects_anything_else(tmp_path):
    assert _policy(tmp_path, prohibited_instruction_roles=[ROLE])["prohibited_instruction_roles"] == [ROLE]
    assert JC.prohibited_instruction_roles({"policy": _policy(tmp_path)}) == ()
    for bad in (["not_a_role"], [ROLE, ROLE], ROLE, [1]):
        with pytest.raises(ValueError, match="prohibited_instruction_roles"):
            _policy(tmp_path, prohibited_instruction_roles=bad)


def test_every_grader_reads_the_roles_from_the_stored_campaign_state(tmp_path, monkeypatch):
    monkeypatch.delenv(JC.ENV, raising=False)
    assert JC.prohibited_instruction_roles() == ()
    state = tmp_path / "joint_campaign.json"
    state.write_text(json.dumps({"policy": _policy(tmp_path, prohibited_instruction_roles=[ROLE])}))
    monkeypatch.setenv(JC.ENV, str(state))
    assert JC.prohibited_instruction_roles() == (ROLE,)


def test_the_prompt_states_the_rule_once_in_one_line_derived_from_the_policy(tmp_path, monkeypatch):
    policy = _policy(tmp_path, prohibited_instruction_roles=[ROLE])
    line = JC.instruction_policy_line(policy).strip()
    assert "\n" not in line and f"`{ROLE}`: {ROLES[ROLE]}" in line and "fails at L2" in line
    assert JC.instruction_policy_line(_policy(tmp_path)) == ""
    state = tmp_path / "joint_campaign.json"
    state.write_text(json.dumps({"policy": policy}))
    monkeypatch.setenv(JC.ENV, str(state))
    assert JC.prompt_block().count(line) == 1
    # The editable full-compiler campaign (the fresh runs) states it too.
    from merlin_experiments.phase1 import full_compiler_workspace

    external = _policy(tmp_path, prohibited_instruction_roles=[ROLE], compiler_execution_mode="snapshot_external_v1")
    state.write_text(json.dumps({"policy": external}))
    monkeypatch.setattr(full_compiler_workspace, "selected", lambda: {"selected": True})
    block = JC.prompt_block()
    assert "## Joint editable compiler development" in block and block.count(line) == 1
    state.write_text(json.dumps({"policy": _policy(tmp_path, compiler_execution_mode="snapshot_external_v1")}))
    assert "Prohibited instruction role" not in JC.prompt_block()


def test_no_author_skill_suggests_a_hardware_loop():
    found = {
        str(path.relative_to(SKILLS)): hits
        for path in sorted(SKILLS.rglob("*"))
        if path.is_file() and path.suffix in (".md", ".py", ".json", ".yaml")
        for hits in [_mentions_hardware_loop(path.read_text(encoding="utf-8", errors="replace"))]
        if hits
    }
    assert found == {}


def test_no_catalogue_mechanism_or_hint_suggests_a_hardware_loop():
    design = repo_root() / "docs" / "design"
    for name in ("joint_optimization_catalog.json", "tightly_coupled_optimization_extensions.json"):
        assert _mentions_hardware_loop((design / name).read_text(encoding="utf-8")) == [], name
    # Every staged mechanism made available, then every bottleneck class the hints address.
    selection = copy.deepcopy(FIXTURE["suggestions"])
    for row in selection["all_mechanisms"]:
        row["requirements"] = dict.fromkeys(row.get("requirements") or {"any": 1}, "supported")
        row["status"] = "supported"
    named = []
    for bound in ("requests", "compute", MH.FENCE_BOUND):
        named += MH.hints(selection, bound_by=bound, limit=64)
    for resource in ("accelerator.storage.accumulator.all", "memory.l1d.probe", "cpu.issue", "accelerator.execute"):
        named += MH.hints(selection, dominant={"name": resource, "modeled_activity": 1.0}, limit=64)
    assert named and _mentions_hardware_loop(json.dumps(named)) == []


@pytest.mark.parametrize("version", [20, 21, 22])
def test_no_exported_scheduling_capability_is_a_hardware_loop(version):
    from merlin.compile.scheduling.api_catalog import scheduling_capabilities

    assert _mentions_hardware_loop(json.dumps(scheduling_capabilities(version))) == []


def test_no_library_pass_in_optimization_next_suggests_a_hardware_loop(tmp_path):
    (tmp_path / "OPTIMIZATION_SUGGESTIONS.json").write_text(json.dumps(FIXTURE["suggestions"]))
    (tmp_path / "submission").mkdir()
    (tmp_path / ".agents" / "skills" / "ab-library-pass").mkdir(parents=True)
    written = []
    for key, rows in (("scalability", "cases"), ("core_gsim", "per_capsule")):
        document = copy.deepcopy(FIXTURE[key])
        PB.attach(document, document[rows], workspace=tmp_path)
        written += [row.get("optimization_next") for row in document[rows] if row.get("optimization_next")]
    assert written and _mentions_hardware_loop(json.dumps(written)) == []
