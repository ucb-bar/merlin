"""Original pass-interface construction, fallback and exact selected scheduler transport."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from merlin.transforms.pipeline_policy import execute_policy


def _recipe():
    path = (Path(__file__).resolve().parents[1] / 'src/merlin_experiments/phase1/author_skills'
            '/heavy-pass-recipes/scripts/shared_rhs_epoch_recipe.py')
    spec = importlib.util.spec_from_file_location('shared_rhs_recipe_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _owner(tmp_path, eligible):
    package = tmp_path / 'package'
    (package / 'tools').mkdir(parents=True)
    (package / 'tools/shared_rhs_owner.py').write_text(f'''
from merlin.transforms.pipeline_state import PipelineContext

def context(package):
    return PipelineContext(hardware="selected-facts", resources="selected-stores",
                           numerical_policy="source-integer-readout", execution="original-epoch-law")

def eligible(source, *, package):
    return {eligible!r}

def compile(source, *, package, scheduler):
    name = "incumbent" if scheduler is None else scheduler.__module__ + ":" + scheduler.__name__
    return {{"emitted": source + "\\n" + name + "\\n", "source_was_retained": source}}
''')
    source = tmp_path / 'source.mlir'
    source.write_text('original complete group')
    return package, source


def test_describe_is_inert_and_owner_is_required(tmp_path, capsys):
    recipe = _recipe()
    assert recipe.main(['--describe']) == 0
    assert json.loads(capsys.readouterr().out)['timing'] == 'UNKNOWN'
    with pytest.raises(ValueError, match='construction required'):
        recipe.prepare(tmp_path / 'absent.mlir', package=tmp_path, mechanism=None)
    with pytest.raises(ValueError, match='unsupported'):
        recipe.prepare(tmp_path / 'absent.mlir', package=tmp_path, mechanism='invented')


@pytest.mark.parametrize('eligibility,status', [(None, 'unknown'), (False, 'unsupported')])
def test_missing_or_refused_law_keeps_original_emission(tmp_path, eligibility, status):
    recipe = _recipe()
    package, source = _owner(tmp_path, eligibility)
    baseline = execute_policy(**recipe.prepare(source, package=package, mechanism=None))
    selected = execute_policy(**recipe.prepare(source, package=package, mechanism=recipe.MECHANISM))
    assert baseline.artifact == selected.artifact == source.read_text() + '\nincumbent\n'
    assert selected.receipt['selected_variant'] == 'incumbent'
    assert selected.receipt['decisions'][0]['status'] == status
    assert selected.module == source.read_text()


def test_selected_route_receives_actual_checked_library_callable(tmp_path):
    recipe = _recipe()
    package, source = _owner(tmp_path, True)
    binding = recipe.prepare(source, package=package, mechanism=recipe.MECHANISM)
    original = recipe.prepare(source, package=package, mechanism=None)
    assert original['facts'] == binding['facts']
    assert original['session'].stamp() == binding['session'].stamp()
    selected = execute_policy(**binding)
    assert selected.receipt['selected_variant'] == recipe.MECHANISM
    assert 'merlin.compile.scheduling.dense_shared_rhs:schedule_dense_shared_rhs' in selected.artifact
    assert binding['session'].analysis('compilation')['source_was_retained'] == source.read_text()
    assert selected.receipt['before']['source_sha256'] == selected.receipt['after']['source_sha256']


def test_linked_owner_cannot_replace_the_authored_package_boundary(tmp_path):
    recipe = _recipe()
    package, source = _owner(tmp_path, True)
    owner = package / 'tools/shared_rhs_owner.py'
    saved = tmp_path / 'foreign.py'
    saved.write_bytes(owner.read_bytes())
    owner.unlink()
    owner.symlink_to(saved)
    with pytest.raises(ValueError, match='inside the package'):
        recipe.prepare(source, package=package, mechanism=None)
