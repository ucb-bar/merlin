"""Checked result custody retains its original facts, software and numeric policy.

These tests use real source discovery/scheduling and the ordinary preparation
transaction. They do not compile native providers or claim target qualification.
"""

import hashlib
import json
from dataclasses import replace

import pytest
import yaml
from test_pointwise_source import case, source_program
from test_selected_dense_contracts import inputs as dense_inputs
from test_selected_dense_contracts import schedule
from test_source_pointwise_lowering import fixture, run

from merlin.compile.scheduling.pointwise_source import compile_source_pointwise
from merlin.compile.scheduling.selected_contracts import read_scheduling_contracts
from merlin.llvmlower.prepared_model_transform import RECEIPT

MUTATIONS = ("rtl", "numerical_policy", "software")


def other_selection(original, directory, mutation):
    """Create a separately valid selection without changing the original files."""
    directory.mkdir()
    facts = json.loads(original.facts_bytes)
    software = yaml.safe_load(original.software_bytes)
    if mutation == "rtl":
        # A new independently selected RTL identity, with unchanged geometry.
        facts["facts"]["independent_revision"] = "different-neutral-rtl-revision"
    elif mutation == "numerical_policy":
        software["scheduling"]["numerical_policy"] += "-different-policy"
        if "row_reduction" in software["scheduling"]:
            from merlin.compile.scheduling.row_softmax import RowSoftmaxNumerics

            # A different but internally consistent complete row policy is
            # independently admissible; its old result must still reject it.
            algorithm = software["scheduling"]["row_reduction"]["algorithm"]
            algorithm["source_policy"] = software["scheduling"]["numerical_policy"]

            def tree(value):
                return tuple(map(tree, value)) if isinstance(value, list) else value

            numerics = RowSoftmaxNumerics(**{**algorithm, "tree": tree(algorithm["tree"])})
            for instruction in software["isa"]["instructions"]:
                instruction["semantics"]["parameters"]["numerical_identity"] = numerics.identity
    else:
        # Different software/ISA identity with compatible encoded instructions.
        software["isa"]["namespace"] += "_different_software"
    fact_path, software_path = directory / "facts.json", directory / "software.yaml"
    fact_path.write_text(json.dumps(facts))
    software["isa"]["facts_sha256"] = hashlib.sha256(fact_path.read_bytes()).hexdigest()
    software_path.write_text(yaml.safe_dump(software))
    selected = read_scheduling_contracts(fact_path, software_path, target=original.target)
    selected.revalidate()
    original.revalidate()
    if mutation == "rtl":
        assert selected.hardware_identity != original.hardware_identity
    elif mutation == "numerical_policy":
        assert selected.numerical_policy != original.numerical_policy
    else:
        assert selected.isa.identity() != original.isa.identity()
    return selected


@pytest.mark.parametrize("family", ("pointwise", "dense"))
@pytest.mark.parametrize("mutation", MUTATIONS)
def test_checked_result_rejects_single_field_valid_contract_substitution(tmp_path, family, mutation):
    original = tmp_path / "original"
    original.mkdir()
    if family == "pointwise":
        options = case(original)
        module, operation = source_program()
        result = compile_source_pointwise(module, operation, **options)
    else:
        result = schedule(dense_inputs(original, shape=(3, 7, 5)))
    assert result is not None
    result.revalidate()
    selected = other_selection(result.selected_contracts, tmp_path / "other", mutation)
    changed = replace(result, selected_contracts=selected)
    # This is a single-field substitution; the original checked payload stays
    # intact, and the new selection is valid on its own original input files.
    assert changed.selected_contracts is selected
    for name in result.__dataclass_fields__:
        if name != "selected_contracts":
            assert getattr(changed, name) is getattr(result, name)
    with pytest.raises(ValueError, match="contract|selection|changed|replan"):
        changed.revalidate()
    with pytest.raises(ValueError, match="contract|selection|changed|replan"):
        if family == "pointwise":
            changed.images()
        else:
            changed.emit_c(None, symbol="must_not_emit")
    result.revalidate()


@pytest.mark.parametrize("mutation", MUTATIONS)
def test_original_source_transaction_rejects_prepare_callback_contract_substitution(tmp_path, mutation):
    seen = []

    def prepare(bindings, work):
        assert bindings
        for index, binding in enumerate(bindings):
            original = binding.compilation
            original.revalidate()
            selected = other_selection(original.selected_contracts, work / ("different_" + str(index)), mutation)
            changed = replace(original, selected_contracts=selected)
            # Model a callback replacing exactly its retained compilation's
            # contract, rather than changing original selection/configuration.
            object.__setattr__(binding, "compilation", changed)
            seen.append((original, changed))

    selected, mapped, prepared = fixture(tmp_path, prepare=prepare)
    original_files = (selected.selected_contracts.facts_bytes, selected.selected_contracts.software_bytes)
    module, _ = source_program(m=3, n=5)
    # The existing fixture's provider build fails if reached. The real source
    # transaction must instead refuse immediately after its prepare callback.
    with pytest.raises(ValueError, match="contract|selection|changed|replan"):
        run(tmp_path, selected, module)
    assert len(mapped) == len(prepared) == len(seen) == 1
    assert not (tmp_path / "preparation" / RECEIPT).exists()
    assert original_files == (selected.selected_contracts.facts_bytes, selected.selected_contracts.software_bytes)
    selected.selected_contracts.revalidate()
    seen[0][0].revalidate()


@pytest.mark.parametrize("mutation", ("remove", "insert"))
def test_compact_dense_retains_original_optional_selection_sentinel(tmp_path, mutation):
    from test_compact_dense import fixture as compact_fixture
    from test_compact_dense import schedule as compact_schedule

    model = dense_inputs(tmp_path, shape=(3, 7, 5))
    if mutation == "remove":
        result = schedule(model)
        substituted = None
    else:
        result = compact_schedule(*compact_fixture(shape=(3, 7, 5)))
        substituted = model[0]
    assert result is not None
    result.revalidate()
    changed = replace(result, selected_contracts=substituted)
    with pytest.raises(ValueError, match="contract|selection|changed|replan"):
        changed.revalidate()
    with pytest.raises(ValueError, match="contract|selection|changed|replan"):
        changed.emit_c(None, symbol="must_not_emit")
    result.revalidate()


@pytest.mark.parametrize("mutation", MUTATIONS)
def test_dense_source_wrapper_rejects_nested_compiled_contract_substitution(tmp_path, mutation):
    from test_dense_source import compile_source, source

    original = tmp_path / "original"
    original.mkdir()
    model = dense_inputs(original, shape=(3, 7, 5))
    module, operation = source((3, 7, 5), model[0].dense.dtypes)
    result = compile_source(model, module, operation)
    assert result is not None
    result.revalidate()
    selected = other_selection(model[0], tmp_path / "other", mutation)
    changed = replace(result, compiled=replace(result.compiled, selected_contracts=selected))
    assert changed.selected_contracts is result.selected_contracts
    assert changed.compiled.kernel is result.compiled.kernel
    with pytest.raises(ValueError, match="contract|selection|changed|replan"):
        changed.revalidate()
    with pytest.raises(ValueError, match="contract|selection|changed|replan"):
        changed.emit_c(None, symbol="must_not_emit")
    result.revalidate()


def checked_result(directory, family):
    directory.mkdir()
    if family == "pointwise":
        options = case(directory)
        module, operation = source_program()
        return compile_source_pointwise(module, operation, **options)
    if family == "dense":
        return schedule(dense_inputs(directory, shape=(3, 7, 5)))
    from test_selected_row_contracts import inputs as row_inputs
    from test_selected_row_contracts import schedule as row_schedule

    return row_schedule(row_inputs(directory, rows=2, columns=19))


def replace_companion(result, selected, family):
    changed = replace(result, selected_contracts=selected)
    if selected is None:
        digest = None
    else:
        record = selected.record()
        if family == "pointwise":
            payload = json.dumps(record, separators=(",", ":"))
        elif family == "dense":
            payload = json.dumps(record, sort_keys=True, separators=(",", ":"))
        else:
            payload = json.dumps(record, sort_keys=True)
        digest = hashlib.sha256(payload.encode()).hexdigest()
    # Pre-fix pointwise/dense results have no companion field. Attribute
    # injection keeps the red test focused on admission, not constructor errors.
    # The original pass stamps, checked kernel and byte images stay unchanged.
    object.__setattr__(changed, "checked_selection", digest)
    for name in result.__dataclass_fields__:
        if name not in ("selected_contracts", "checked_selection"):
            assert getattr(changed, name) is getattr(result, name)
    return changed


def refuse_result(result, family):
    with pytest.raises(ValueError, match="contract|selection|changed|replan"):
        result.revalidate()
    with pytest.raises(ValueError, match="contract|selection|changed|replan"):
        if family == "pointwise":
            result.images()
        else:
            result.emit_c(None, symbol="must_not_emit")


@pytest.mark.parametrize("family", ("pointwise", "dense", "row"))
@pytest.mark.parametrize("mutation", MUTATIONS)
def test_recomputed_companion_cannot_rebind_original_verification_stamp(tmp_path, family, mutation):
    result = checked_result(tmp_path / "original", family)
    assert result is not None
    result.revalidate()
    selected = other_selection(result.selected_contracts, tmp_path / "other", mutation)
    refuse_result(replace_companion(result, selected, family), family)
    result.revalidate()


@pytest.mark.parametrize("mutation", MUTATIONS)
def test_row_result_rejects_valid_contract_substitution(tmp_path, mutation):
    result = checked_result(tmp_path / "original", "row")
    selected = other_selection(result.selected_contracts, tmp_path / "other", mutation)
    refuse_result(replace(result, selected_contracts=selected), "row")
    result.revalidate()


@pytest.mark.parametrize("mutation", ("remove", "insert"))
@pytest.mark.parametrize("companion", (False, True))
def test_row_original_optional_selection_cannot_be_added_or_removed(tmp_path, mutation, companion):
    from test_row_softmax import fixture as row_fixture
    from test_selected_row_contracts import inputs as row_inputs

    from merlin.compile.scheduling.row_softmax import compile_row_softmax

    if mutation == "remove":
        result = checked_result(tmp_path / "original", "row")
        selected = None
    else:
        result = compile_row_softmax(**row_fixture(rows=2, columns=19))
        (tmp_path / "selected").mkdir()
        selected = row_inputs(tmp_path / "selected", rows=2, columns=19)[0]
    result.revalidate()
    changed = replace_companion(result, selected, "row") if companion else replace(result, selected_contracts=selected)
    refuse_result(changed, "row")
    result.revalidate()
