"""P3: the MERLIN_TARGET_EXPERIMENT override + scaffolding guard in the capsule-bench drivers' _common.

The drivers live in one experiment dir but must be able to target ANOTHER target's descriptor without
copies. _common honors MERLIN_TARGET_EXPERIMENT (a path to a target_experiment.yaml): unset ⇒ the
gemmini dir it lives in; set ⇒ that descriptor's dir. A target that ships only a descriptor (no task/ +
input_bundles/) must fail loudly via require_scaffolding(), not deep inside a run.

_common self-bootstraps (git root + sys.path) at import, so these run it in a fresh subprocess with the
env set, which is exactly how the drivers import it.
"""
from __future__ import annotations

import os
import subprocess
import sys

from merlin.common.paths import repo_root

_SCRIPTS = repo_root() / "merlin/experiments/gemmini_capsule_bench_v0/scripts"
_ATLAS = repo_root() / "merlin/experiments/atlas_capsule_bench_v0/target_experiment.yaml"

_PROBE = f"""
import sys
sys.path.insert(0, r"{_SCRIPTS}")
import _common as C
print("TARGET", C.TARGET)
print("EXP", C.EXP.name)
try:
    C.require_scaffolding()
    print("SCAFFOLD", "ok")
except SystemExit:
    print("SCAFFOLD", "missing")
"""


def _run(target_experiment: str) -> dict:
    env = dict(os.environ)
    env["MERLIN_TARGET_EXPERIMENT"] = target_experiment
    r = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True,
                       env=env, cwd=str(repo_root()))
    assert r.returncode == 0, r.stderr
    return dict(line.split(" ", 1) for line in r.stdout.strip().splitlines())


def test_default_targets_gemmini_and_has_scaffolding():
    out = _run("")                                        # empty ⇒ treated as unset
    assert out["TARGET"] == "gemmini"
    assert out["EXP"] == "gemmini_capsule_bench_v0"
    assert out["SCAFFOLD"] == "ok"                        # gemmini ships task/ + input_bundles/


def test_override_switches_target_to_another_descriptor():
    out = _run(str(_ATLAS))
    assert out["TARGET"] == "atlas"                       # drivers now target atlas's descriptor
    assert out["EXP"] == "atlas_capsule_bench_v0"


def test_guard_trips_on_a_descriptor_without_scaffolding(tmp_path):
    """The guard must fire for a descriptor-only target.

    This used to be asserted against atlas, on the premise that atlas "ships only a descriptor".
    Atlas has since been scaffolded (task/ + input_bundles/), so the assertion started failing --
    the guard was fine, the fixture had simply outgrown it. Synthesize a descriptor-only dir
    instead, so the test keeps testing the guard rather than tracking whichever target happens to
    be unscaffolded this week.
    """
    desc = tmp_path / "probe_capsule_bench_v0"
    desc.mkdir()
    (desc / "target_experiment.yaml").write_text("target: probe\n", encoding="utf-8")
    out = _run(str(desc / "target_experiment.yaml"))
    assert out["TARGET"] == "probe"
    assert out["SCAFFOLD"] == "missing"                   # no task/ + input_bundles/ ⇒ loud guard
