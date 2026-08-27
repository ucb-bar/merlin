"""Hardware PERFORMANCE-COUNTER measurement on the K1 board (cycles / instructions / IPC).

Wall time alone cannot distinguish the two ways a kernel can be slow: *executing too many
instructions* (a codegen-quality problem the schedule can fix) from *stalling on each instruction*
(a memory/dependency problem the schedule cannot fix). Ranking a beam on wall time alone therefore
hides WHICH of the two a fork improved. This module supplies the missing axis.

The board runs Bianbu with a working PMU (``/sys/bus/event_source/devices/cpu``) but ships **no
``perf(1)`` binary and no native compiler**, so we cross-compile a ~60-line ``perf_event_open``
wrapper (``pmustat``) with the same SpacemiT clang used for every other board artifact and push it
once. ``pmustat`` execs an arbitrary ELF and reports counters for that child only, so ANY binary the
harness already builds can be measured without recompiling it.

Fail-closed: if the toolchain, the board, or the PMU is unavailable, :func:`measure` returns ``None``
and callers keep their wall-time number — a missing counter is never reported as a zero.
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import k1

#: Where the cross-built wrapper lives on the board (pushed once, reused by every measurement).
REMOTE_PMUSTAT = "/tmp/merlin_pmustat"

_PMUSTAT_C = r"""
/* pmustat -- minimal perf_event_open wrapper: run a command, report cycles/instructions.
 * The K1 (Bianbu) exposes a PMU but has no perf(1) binary, so we count via the raw syscall.
 * Counters are opened against the CHILD pid with inherit=1, so they cover the measured ELF (and
 * its threads) and NOT this wrapper or the shell -- the child is held on a pipe until the counters
 * are armed, so no instruction of the payload escapes the window. */
#define _GNU_SOURCE
#include <linux/perf_event.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

static int pe_open(int pid, unsigned type, unsigned long long cfg, int group) {
  struct perf_event_attr a; memset(&a, 0, sizeof a);
  a.type = type; a.size = sizeof a; a.config = cfg;
  a.disabled = (group == -1); a.inherit = 1;
  a.exclude_kernel = 1; a.exclude_hv = 1;
  return syscall(__NR_perf_event_open, &a, pid, -1, group, 0);
}
static long long pe_read(int fd) {
  long long v = 0; if (read(fd, &v, sizeof v) != sizeof v) return -1; return v;
}

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: pmustat <cmd> [args...]\n"); return 2; }
  int pipefd[2]; if (pipe(pipefd)) { perror("pipe"); return 1; }
  pid_t pid = fork();
  if (pid == 0) {                       /* child: block until the counters are armed, then exec */
    close(pipefd[1]); char c; read(pipefd[0], &c, 1); close(pipefd[0]);
    execvp(argv[1], &argv[1]); perror("execvp"); _exit(127);
  }
  close(pipefd[0]);
  int cyc = pe_open(pid, PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, -1);
  int ins = pe_open(pid, PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, cyc);
  if (cyc >= 0) ioctl(cyc, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
  write(pipefd[1], "g", 1); close(pipefd[1]);   /* release the child */
  int st = 0; waitpid(pid, &st, 0);
  if (cyc >= 0) ioctl(cyc, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
  long long c = cyc >= 0 ? pe_read(cyc) : -1, i = ins >= 0 ? pe_read(ins) : -1;
  fprintf(stderr, "MERLIN_PMU cycles=%lld instructions=%lld\n", c, i);
  return WIFEXITED(st) ? WEXITSTATUS(st) : 1;
}
"""


@dataclass(frozen=True)
class PmuCounts:
    """Hardware counters for one measured run. ``ipc`` is the diagnostic: a LOW ipc means the core
    is stalling (memory/dependency bound), a HIGH ipc with a big instruction count means the codegen
    is emitting too much work. The two call for different compiler fixes."""

    cycles: int
    instructions: int

    @property
    def ipc(self) -> float | None:
        return round(self.instructions / self.cycles, 4) if self.cycles > 0 else None

    def as_dict(self) -> dict:
        return {"pmu_cycles": self.cycles, "pmu_instructions": self.instructions, "pmu_ipc": self.ipc}


def parse(stderr_text: str) -> PmuCounts | None:
    """Parse the wrapper's counter line. Returns None if absent or if the kernel refused a counter
    (``-1``) — a refused counter must never be reported as zero work."""
    for line in (stderr_text or "").splitlines():
        if not line.startswith("MERLIN_PMU "):
            continue
        fields = {}
        for tok in line.split()[1:]:
            key, _, val = tok.partition("=")
            try:
                fields[key] = int(val)
            except ValueError:
                return None
        cyc, ins = fields.get("cycles", -1), fields.get("instructions", -1)
        return PmuCounts(cycles=cyc, instructions=ins) if cyc > 0 and ins >= 0 else None
    return None


def ensure_deployed(*, timeout: int = 180) -> bool:
    """Cross-build ``pmustat`` and push it to the board (idempotent). False if unavailable."""
    if not k1.available():
        return False
    probe = k1._ssh(f"test -x {REMOTE_PMUSTAT} && echo yes", timeout=30)
    if "yes" in (probe.stdout or ""):
        return True
    cc = k1.toolchain_cc()
    if cc is None:
        return False
    with tempfile.TemporaryDirectory(prefix="merlin_pmu_") as tmp:
        src, binp = Path(tmp) / "pmustat.c", Path(tmp) / "pmustat"
        src.write_text(_PMUSTAT_C, encoding="utf-8")
        build = subprocess.run(
            [str(cc), "--target=riscv64-unknown-linux-gnu", f"-march={k1.K1_MARCH}",
             f"-mabi={k1.K1_MABI}", "-O2", "-static", "-o", str(binp), str(src)],
            capture_output=True, text=True, timeout=timeout)
        if build.returncode != 0 or not binp.is_file():
            return False
        push = subprocess.run(
            ["scp", "-i", k1.K1_SSH_KEY, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
             str(binp), f"{k1.K1_HOST}:{REMOTE_PMUSTAT}"],
            capture_output=True, text=True, timeout=timeout)
        if push.returncode != 0:
            return False
    k1._ssh(f"chmod +x {REMOTE_PMUSTAT}", timeout=30)
    return "MERLIN_PMU" in (k1._ssh(f"{REMOTE_PMUSTAT} /bin/true", timeout=60).stderr or "")


def wrap(remote_cmd: str) -> str:
    """Wrap a board command so its counters are reported. Caller must have ensure_deployed()."""
    return f"{REMOTE_PMUSTAT} {remote_cmd}"
