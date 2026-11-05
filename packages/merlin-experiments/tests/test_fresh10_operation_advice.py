"""Catalog withholding applies to staged JSON as well as the visible TASK section."""

from pathlib import Path

import pytest

from merlin_experiments.phase1 import author_profiles as A
from merlin_experiments.phase1 import fresh10_profiles as P
from merlin_experiments.phase1 import operation_optimizations as O

REPO = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize('name', P.NAMES[:2])
@pytest.mark.parametrize('owner', [O.stage, O.stage_selected])
def test_withheld_catalog_is_neither_staged_nor_read(name, owner, tmp_path, monkeypatch):
    monkeypatch.setenv(A.ENV, name)
    ws, run = tmp_path / 'workspace', tmp_path / 'run'
    ws.mkdir(); run.mkdir()
    task = 'The complete common compiler contract.\n'
    for root in (ws, run):
        (root / 'TASK.md').write_text(task)
    monkeypatch.setattr(O, 'load', lambda _repo: pytest.fail('withheld catalog was read'))
    result = owner(ws, run, declarations=[], repo=REPO)
    assert result is None
    for root in (ws, run):
        assert (root / 'TASK.md').read_text() == task
        assert not (root / O.FILENAME).exists()


@pytest.mark.parametrize('name', [P.NAMES[2], A.SCRATCH_NAMES[3], ''])
def test_full_library_and_historical_profiles_keep_their_original_operation_advice(name, tmp_path, monkeypatch):
    monkeypatch.setenv(A.ENV, name)
    ws, run = tmp_path / 'workspace', tmp_path / 'run'
    ws.mkdir(); run.mkdir()
    (ws / 'TASK.md').write_text('The complete common compiler contract.\n')
    result = O.stage_selected(ws, run, declarations=[{'operation':{'op':'matmul'}}], repo=REPO)
    assert result is not None
    assert (ws / O.FILENAME).is_file()
    assert (ws / O.FILENAME).read_bytes() == (run / O.FILENAME).read_bytes()
    assert O.HEADING in (ws / 'TASK.md').read_text()
