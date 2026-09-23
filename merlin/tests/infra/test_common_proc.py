"""merlin.common.proc.run_checked: one subprocess-failure contract, with each caller's error and timeout rule."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from merlin.common.proc import run_checked


class _Boom(RuntimeError):
    pass


def test_success_returns_the_completed_process_and_stringifies_argv(tmp_path):
    proc = run_checked([sys.executable, "-c", "import sys; print(sys.argv[1])", Path(tmp_path)])
    assert proc.returncode == 0 and proc.stdout.strip() == str(tmp_path)


def test_failure_raises_the_callers_error_with_status_and_output_tails():
    with pytest.raises(_Boom) as exc:
        run_checked([sys.executable, "-c", "import sys; print('o'*50); sys.stderr.write('e'*50); sys.exit(3)"],
                    error=_Boom, tail=10)
    msg = str(exc.value)
    assert "rc 3" in msg and "STDOUT:" + "o" * 9 in msg and "STDERR:" + "e" * 10 in msg and "o" * 11 not in msg


def test_timeout_becomes_the_callers_error_when_wrapped():
    with pytest.raises(_Boom, match="timed out after 0.2s \\(slow\\)"):
        run_checked([sys.executable, "-c", "import time; time.sleep(5)"], error=_Boom, timeout=0.2,
                    timeout_hint=" (slow)")


def test_timeout_propagates_unchanged_when_not_wrapped():
    with pytest.raises(subprocess.TimeoutExpired):
        run_checked([sys.executable, "-c", "import time; time.sleep(5)"], error=_Boom, timeout=0.2,
                    wrap_timeout=False)
