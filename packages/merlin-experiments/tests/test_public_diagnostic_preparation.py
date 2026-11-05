"""Original preparation/edit custody; neutral fixtures confer no target qualification."""

import shutil
from types import SimpleNamespace

import pytest
import test_compiler_pass_preparation as PREP
from merlin_experiments.phase2 import compiler_pass_preparation as P
from merlin_experiments.phase2 import public_diagnostic_launch as L
from merlin_experiments.phase2.contracts import StageGateError
from merlin_experiments.phase2.edit_authority import FrozenEditAuthority
from merlin_experiments.phase2.retained_revisions import RetainedRevisionStore

from merlin.perf.phase2_edit_contract import seal
from merlin.targetgen.compiler_pass_selection import CompilerPassSelection, OriginalPassFactory


def _restore_not_executed(inputs, *, work_root):
    raise AssertionError("this source-only fixture must never run admission or a target")


@pytest.fixture
def prepared_inputs(tmp_path):
    generator = PREP.setup.__wrapped__(tmp_path)
    selection, _, original, _, owner = generator
    factory = OriginalPassFactory(_restore_not_executed, {"neutral_original_selection": True})
    selection = CompilerPassSelection(selection.registry, selection.binding, mechanisms=("composite",), factory=factory)
    preparation = P.CompilerPassPreparation(
        "composite", baseline_policy={"enabled": False}, candidate_policy={"enabled": False}
    )
    library = SimpleNamespace(contract=SimpleNamespace(sha256=selection.binding.library))
    return dict(
        selection=selection,
        preparation=preparation,
        library=library,
        factory=factory,
        plan={"compiler": str(original), "compiler_inventory": L._inventory(original)},
        owner=owner,
    )


def _prepare(values, tmp_path):
    output = tmp_path / "launch"
    output.mkdir()
    return L._prepare_diagnostic_sources(output=output, **{k: v for k, v in values.items() if k != "owner"})


def test_preparation_preserves_raw_parent_and_identical_immutable_comparison_seed(prepared_inputs, tmp_path):
    values = prepared_inputs
    original = values["plan"]["compiler"]
    before = L._inventory(original)
    candidate, baseline, record = _prepare(values, tmp_path)
    assert L._inventory(original) == before == values["plan"]["compiler_inventory"]
    assert record["original"]["sha256"] != record["baseline"]["sha256"]
    assert record["baseline"]["sha256"] == record["candidate_seed"]["sha256"]
    assert L._inventory(candidate) == L._inventory(baseline)
    assert values["owner"].sources[0][0] == values["owner"].sources[1][0]
    assert not (baseline / "candidate_policy.py").stat().st_mode & 0o222
    assert (candidate / "candidate_policy.py").stat().st_mode & 0o200
    P.verify_pair(record, values["selection"])


def test_retained_policy_and_new_pass_edit_require_explicit_original_authority(prepared_inputs, tmp_path):
    candidate, baseline, _ = _prepare(prepared_inputs, tmp_path)
    contract = seal(
        dict(
            schema="compiler_edit_contract_v2",
            existing_symbols=[dict(path="candidate_policy.py", symbol="choose", surface_id="policy")],
            helper_extensions=[],
            new_python_components=[dict(directory="passes", surface_ids=["policy"], reason="candidate-owned rewrites")],
        )
    )
    output = tmp_path / "authority"
    output.mkdir()
    authority = FrozenEditAuthority(output)
    authority.freeze(candidate, contract, has_iterations=False)
    store = RetainedRevisionStore(
        tmp_path / "revisions", edit_authority=authority, max_bytes=1 << 20, max_files=32, max_total_bytes=4 << 20
    )
    initial = store.freeze(candidate, options={"hypothesis": "same prepared seed", "edit_family": "control"})
    changed = tmp_path / "changed"
    shutil.copytree(candidate, changed)
    (changed / "passes").mkdir()
    (changed / "passes/new_rule.py").write_text("def rule():\n    return True\n")
    (changed / "candidate_policy.py").write_text(
        "def choose():\n    from passes.new_rule import rule\n    return rule()\n"
    )
    revision, receipt = L._retain_candidate_source(store, initial, changed, L._inventory(changed))
    assert revision.parent_revision == initial.revision_id and receipt["distinct_source"]
    assert (baseline / "candidate_policy.py").read_text().endswith("return False\n")
    assert not (baseline / "passes").exists()
    (changed / "compiler.py").write_text("def compile(source): return 'foreign'\n")
    with pytest.raises(ValueError, match="exceeds host-frozen authority"):
        L._retain_candidate_source(store, initial, changed, L._inventory(changed))


@pytest.mark.parametrize("change", ["saved_selection", "saved_choices", "unequal", "no_factory", "callback"])
def test_preparation_cannot_acquire_authority_from_json_or_rebound_callbacks(prepared_inputs, change):
    values = prepared_inputs
    selection, preparation = values["selection"], values["preparation"]
    if change == "saved_selection":
        selection = selection.record()
    elif change == "saved_choices":
        preparation = preparation.record()
    elif change == "unequal":
        preparation = P.CompilerPassPreparation(
            "composite", baseline_policy={"enabled": False}, candidate_policy={"enabled": True}
        )
    elif change == "no_factory":
        selection = CompilerPassSelection(selection.registry, selection.binding, mechanisms=selection.mechanisms)
    else:
        values["owner"].prepare_compiler = lambda *a, **kw: {"status": "success"}
    with pytest.raises(StageGateError):
        L._validate_preparation(selection, preparation)


@pytest.mark.parametrize("change", ["library", "factory"])
def test_foreign_library_or_factory_refuses_before_preparation_copy(prepared_inputs, tmp_path, change):
    values = prepared_inputs.copy()
    if change == "library":
        values["library"] = SimpleNamespace(contract=SimpleNamespace(sha256="foreign-release"))
    else:
        values["factory"] = OriginalPassFactory(_restore_not_executed, {"different": True})
    output = tmp_path / "launch"
    output.mkdir()
    with pytest.raises(StageGateError, match="original factory or selected library"):
        L._prepare_diagnostic_sources(output=output, **{k: v for k, v in values.items() if k != "owner"})
    assert list(output.iterdir()) == []
    assert prepared_inputs["owner"].sources == []


def test_v1_plan_does_not_select_preparation_or_expand_edit_authority():
    assert "compiler_pass_preparation" not in L.FIELDS
    assert "compiler_pass_selection" not in L.FIELDS
    assert L._validate_preparation(None, None) is None
