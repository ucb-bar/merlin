"""Protected policy parsing never substitutes for a live command authority."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2.component_instruction_policy import (
    POLICY_SCHEMA,
    IndependentInstructionPolicy,
    _exclusions,
    _resolve,
    issue_independent_instruction_policy,
)
from merlin_experiments.phase2.contracts import StageGateError


def policy(symbols):
    return {"schema": POLICY_SCHEMA, "target": "private_policy", "prohibited_source_symbols": symbols}


def declarations():
    return {"names": {"5": "CONTROL", "7": "PARAMS"}, "other_source_bindings": {"5": ["OTHER_NAMESPACE"]}}


def test_only_selected_function_symbol_identity_resolves():
    assert _resolve(policy(["CONTROL", "PARAMS"]), declarations(), target="private_policy") == (
        ("CONTROL", 5),
        ("PARAMS", 7),
    )
    with pytest.raises(StageGateError, match="absent or ambiguous"):
        _resolve(policy(["OTHER_NAMESPACE"]), declarations(), target="private_policy")


@pytest.mark.parametrize("change", ["numeric_override", "foreign_target", "vacuous", "duplicate", "unknown"])
def test_policy_cannot_smuggle_values_or_weaken_its_identity(change):
    selected = policy(["CONTROL"])
    if change == "numeric_override":
        selected["selector"] = 7
    elif change == "foreign_target":
        selected["target"] = "other"
    elif change == "vacuous":
        selected["prohibited_source_symbols"] = []
    elif change == "duplicate":
        selected["prohibited_source_symbols"] = ["CONTROL", "CONTROL"]
    else:
        selected["prohibited_source_symbols"] = ["UNKNOWN"]
    with pytest.raises(StageGateError):
        _resolve(selected, declarations(), target="private_policy")


def test_selected_span_itself_must_resolve_each_symbol_unambiguously():
    declared = declarations()
    declared["names"]["11"] = "CONTROL"
    with pytest.raises(StageGateError, match="ambiguous"):
        _resolve(policy(["CONTROL"]), declared, target="private_policy")


def test_metadata_and_constructor_cannot_issue_instruction_policy(tmp_path):
    command = SimpleNamespace(sha256="caller_hash", hardware=SimpleNamespace(target="private_policy"))
    selected = tmp_path / "policy.json"
    selected.write_text('{"status":"derived"}')
    with pytest.raises(StageGateError, match="actual independently issued"):
        issue_independent_instruction_policy(
            command_intake=command,
            routing_intake=None,
            policy_file=selected,
            forbidden_roots=(tmp_path,),
            output=tmp_path / "receipt.json",
        )
    authority = IndependentInstructionPolicy(
        command, None, selected, "caller_hash", (("CONTROL", 5),), (), tmp_path / "receipt.json", "caller_hash"
    )
    with pytest.raises(StageGateError, match="live independent"):
        authority.verify()
    with pytest.raises(StageGateError, match="live independent"):
        replace(authority, selectors=(("CONTROL", 7),)).verify()


def test_absent_protected_prefix_is_retained_without_creation(tmp_path):
    present = tmp_path / "present"
    present.mkdir()
    absent = tmp_path / "absent" / "private_answers"
    assert _exclusions((present, absent)) == (present, absent)
    assert not absent.parent.exists()
    # Later materialization remains inside the same excluded lexical prefix.
    assert (absent / "policy.json").is_relative_to(_exclusions((absent,))[0])
    absent.mkdir(parents=True)
    assert _exclusions((absent,)) == (absent,)


@pytest.mark.parametrize("kind", ["alias", "dangling_alias", "ancestor_alias", "file", "parent_escape"])
def test_protected_prefix_refuses_indirection_and_non_directory(tmp_path, kind):
    directory = tmp_path / "ordinary"
    directory.mkdir()
    selected = tmp_path / "protected"
    if kind == "alias":
        selected.symlink_to(directory, target_is_directory=True)
    elif kind == "dangling_alias":
        selected.symlink_to(tmp_path / "never_created", target_is_directory=True)
    elif kind == "ancestor_alias":
        selected.symlink_to(directory, target_is_directory=True)
        selected = selected / "absent_child"
    elif kind == "file":
        selected.write_text("ordinary file, not a protected directory prefix")
    else:
        selected = directory / ".." / "protected"
    with pytest.raises(StageGateError, match="canonical ordinary or absent"):
        _exclusions((selected,))


@pytest.mark.parametrize("roots", [(), [], None])
def test_protected_prefix_roster_is_explicit_and_nonempty(roots):
    with pytest.raises(StageGateError, match="explicit protected"):
        _exclusions(roots)
