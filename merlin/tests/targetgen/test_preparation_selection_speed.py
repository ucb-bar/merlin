from pathlib import Path

import pytest
from merlin_experiments.corpus.preparation import _bundle_selection
from merlin_experiments.spec import SpecError

from merlin.targetgen.tool_profile_toolchain import _SelectedMembers


def test_selected_inventory_exact_prefix_and_duplicate_refusal():
    document = {"files": [
        {"path": "/sdk/lib/a", "sha256": "a"},
        {"path": "/sdk/library/b", "sha256": "b"},
        {"path": "/sdk/lib/sub/c", "sha256": "c"},
    ]}
    selected = _SelectedMembers(document, Path('/manifest.json'))
    assert selected.under(Path('/sdk/lib')) == {
        Path('/sdk/lib/a'): 'a', Path('/sdk/lib/sub/c'): 'c'}
    assert selected.under(Path('/absent')) == {}
    document['files'].append(document['files'][1])
    with pytest.raises(ValueError, match='repeats a member'):
        _SelectedMembers(document, Path('/manifest.json'))


def test_one_treatment_selection_and_historical_default():
    arms, variants = _bundle_selection((), None, None)
    assert len(arms) == 5 and len(variants) == 3
    assert _bundle_selection((), ('EL4',), ('hwbringup_v0',)) == (
        ('merlin_rtlchecks',), ('hwbringup_v0',))


@pytest.mark.parametrize('levels,variants', [
    ((), None), (('EL5',), None), (('EL4', 'EL4'), None),
    (None, ()), (None, ('unknown',)),
])
def test_selection_refuses_empty_duplicate_and_unenabled(levels, variants):
    with pytest.raises(SpecError):
        _bundle_selection((), levels, variants)
