"""Board-LOCAL MetaSchedule measurement + tuning for the K1 (no tvm_rpc / tracker).

The tracker<->server RPC handshake has always deadlocked. Instead, measure BOARD-LOCALLY: LocalBuilder
compiles each candidate .so on the host for rv64gcv; this runner scp's the .so + an args manifest to
the board and runs ``board_kernel_timer`` over plain ssh, returning the measured per-call latency to
the search. Same board-local transport that fixed execution (board_runner), now for measurement.

Usage: python board_tune.py <model> <int8|fp32> <max_trials> <wall_cap_sec>
Env: MERLIN_K1_HOST, MERLIN_K1_SSH_KEY (board). Prints tuned-vs-untuned RVV%/latency.
"""
from __future__ import annotations
import os, sys, time, subprocess, tempfile
from pathlib import Path

REPO = Path("/scratch/agustin/projects/oscar-merlin")
RT_DIR = REPO / "build/baselines/tvm-rv64"
TIMER = RT_DIR / "board_runner" / "board_kernel_timer"
RT_SO = RT_DIR / "libtvm_runtime.so"
BDIR = "/root/tvm_tune"

_DT = {"float32": (2, 32), "float64": (2, 64), "int64": (0, 64), "int32": (0, 32),
       "int8": (0, 8), "uint8": (1, 8), "bool": (1, 1), "float16": (2, 16)}


def _ssh(host, key):
    return ["ssh", "-i", key, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10", host]


def _scp(key):
    return ["scp", "-i", key, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no"]


def make_board_runner(host, key):
    """Build a picklable-enough PyRunner that measures each candidate board-locally."""
    from tvm.meta_schedule.runner import PyRunner, RunnerFuture, RunnerResult
    from tvm.meta_schedule.runner.runner import PyRunnerFuture
    from tvm.meta_schedule.utils import derived_object

    ssh = _ssh(host, key)
    scp = _scp(key)
    # deploy the timer + runtime once
    subprocess.run(ssh + [f"mkdir -p {BDIR}"], capture_output=True, timeout=60)
    for src, dst in [(RT_SO, "libtvm_runtime.so"), (TIMER, "board_kernel_timer")]:
        subprocess.run(scp + [str(src), f"{host}:{BDIR}/{dst}"], capture_output=True, timeout=300)
    subprocess.run(ssh + [f"chmod +x {BDIR}/board_kernel_timer"], capture_output=True, timeout=30)

    @derived_object
    class _Fut(PyRunnerFuture):
        def __init__(self, res, err):
            super().__init__(); self.res = res; self.err = err
        def done(self):  # noqa: D401
            return True
        def result(self):
            return RunnerResult(self.res, self.err)

    @derived_object
    class BoardLocalRunner(PyRunner):
        def run(self, runner_inputs):
            futures = []
            for ri in runner_inputs:
                try:
                    so = ri.artifact_path
                    # manifest from args_info (TensorInfo -> code bits ndim shape)
                    lines = []
                    for ai in ri.args_info:
                        j = ai.as_json()  # ["TENSOR", dtype, [shape...]]
                        dtype = str(j[1]); shape = list(j[2])
                        code, bits = _DT.get(dtype, (2, 32))
                        lines.append(f"{code} {bits} {len(shape)} " + " ".join(str(int(s)) for s in shape))
                    man = so + ".manifest"
                    with open(man, "w") as f:
                        f.write("\n".join(lines) + "\n")
                    base = os.path.basename(so)
                    subprocess.run(scp + [so, f"{host}:{BDIR}/{base}"], capture_output=True, timeout=300)
                    subprocess.run(scp + [man, f"{host}:{BDIR}/{base}.manifest"], capture_output=True, timeout=120)
                    cmd = (f"cd {BDIR} && LD_LIBRARY_PATH={BDIR} ./board_kernel_timer {base} "
                           f"{base}.manifest main 10 3")
                    r = subprocess.run(ssh + [cmd], capture_output=True, timeout=120, text=True)
                    subprocess.run(ssh + [f"rm -f {BDIR}/{base} {BDIR}/{base}.manifest"], capture_output=True, timeout=30)
                    lat = None
                    for ln in (r.stdout or "").splitlines():
                        if ln.startswith("LAT_SEC"):
                            lat = float(ln.split()[1])
                    if lat is not None and lat > 0:
                        futures.append(_Fut([lat], None))
                    else:
                        futures.append(_Fut(None, "no LAT_SEC: " + (r.stderr or "")[-120:]))
                except Exception as e:  # noqa: BLE001
                    futures.append(_Fut(None, str(e)[:150]))
            return futures

    return BoardLocalRunner()
