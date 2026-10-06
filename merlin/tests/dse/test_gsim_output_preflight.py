"""Bad output destinations must fail before backend/emulator work."""
import os
from pathlib import Path

import pytest
from merlin.perf.layer_bench.run import run_on_gsim
from merlin.runtime.backends import base


@pytest.mark.parametrize('kind', ['directory', 'missing_parent', 'elf', 'hardlink'])
def test_output_preflight(tmp_path, monkeypatch, kind):
    elf = tmp_path / 'input.elf'
    elf.write_bytes(b'unchanged executable')
    def forbidden(*args, **kwargs):
        pytest.fail('backend must not be invoked for invalid output destination')
    monkeypatch.setattr(base, 'get_backend', forbidden)
    output = tmp_path
    if kind == 'missing_parent':
        output = tmp_path / 'absent' / 'stdout.txt'
    elif kind == 'elf':
        output = elf
    elif kind == 'hardlink':
        output = tmp_path / 'alias'
        output.hardlink_to(elf)
    with pytest.raises((OSError, ValueError)):
        run_on_gsim(elf, target='test_target', max_cycles=1, timeout_s=1,
                    stdout_path=output)
    assert elf.read_bytes() == b'unchanged executable'


def test_existing_output_not_truncated_by_preflight(tmp_path, monkeypatch):
    elf = tmp_path / 'input.elf'
    elf.write_bytes(b'input')
    output = tmp_path / 'stdout.txt'
    output.write_text('retained')
    def stop(*args, **kwargs):
        raise RuntimeError('after preflight')
    monkeypatch.setattr(base, 'get_backend', stop)
    with pytest.raises(RuntimeError, match='after preflight'):
        run_on_gsim(elf, target='test_target', max_cycles=1, timeout_s=1,
                    stdout_path=output)
    assert output.read_text() == 'retained'


@pytest.mark.parametrize("through_symlink", [False, True])
def test_fifo_refused_without_open_or_backend(tmp_path, monkeypatch, through_symlink):
    elf = tmp_path / "input.elf"
    elf.write_bytes(b"input")
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    output = fifo
    if through_symlink:
        output = tmp_path / "pipe_alias"
        output.symlink_to(fifo)

    def forbidden(*args, **kwargs):
        pytest.fail("nonregular destination must be refused before open/backend")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(base, "get_backend", forbidden)
    with pytest.raises(ValueError, match="regular file"):
        run_on_gsim(elf, target="test_target", max_cycles=1, timeout_s=1,
                    stdout_path=output)
