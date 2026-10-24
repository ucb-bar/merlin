"""The FASED latency-pipe memory model: its cycle timing against FASED's own Chisel, and the path that runs it.

The claim under test is narrow and exact: at its AXI boundary, ``mm_latency_pipe_t`` presents the same
outputs, cycle for cycle, as FireSim's FASED latency-bandwidth pipe. The reference is a transcription of
that model's Chisel (firesim ``sim/midas/src/main/scala/midas/models/dram`` at b084672c) register for
register -- its counters, its flow queues and its one-entry releaser -- written as the Chisel is written,
not as the C++ is, so a shared misreading has to survive two different shapes. A random, legal AXI4 master
drives the C++ model and records every cycle; the reference replays the master's half and must agree on
the model's half everywhere. Mutants of the model are built and shown to fail that comparison, because a
comparison that cannot fail certifies nothing.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import textwrap
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import pytest

from merlin.targetgen import mem_perturb as MP

# --------------------------------------------------------------------------------------------------
# a random, legal AXI4 master, recording every cycle
# --------------------------------------------------------------------------------------------------

MASTER = r"""
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <vector>
#include <sys/mman.h>
#include "mm_perturb.h"

// Each request the master presents is held, unchanged, until it fires (AXI4 forbids withdrawing one). W
// beats are paced independently of AW, so a burst's data may lead, meet or trail its address. One line per
// cycle: what the master drove, '|', what the model presented. Write data is checked at the end.
static uint64_t st;
static uint64_t rnd() {
  uint64_t z = (st += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
static bool coin(uint64_t pct) { return rnd() % 100 < pct; }
static uint64_t opt(int argc, char **argv, const char *key, uint64_t dflt) {
  size_t n = strlen(key);
  for (int i = 1; i < argc; i++)
    if (!strncmp(argv[i], key, n)) return strtoull(argv[i] + n, nullptr, 0);
  return dflt;
}
static uint64_t tag(uint64_t addr) { return addr * 0x9e3779b97f4a7c15ULL; }
struct req_t { uint64_t addr, id, len; };

int main(int argc, char **argv) {
  st = opt(argc, argv, "stim_seed=", 1);
  const uint64_t nreads = opt(argc, argv, "reads=", 200), nwrites = opt(argc, argv, "writes=", 120);
  const uint64_t pace = opt(argc, argv, "pace=", 60), ready = opt(argc, argv, "ready=", 75);
  const uint64_t max_cycles = opt(argc, argv, "max_cycles=", 200000);
  const uint64_t base = 0x80000000, sz = 1 << 20, word = 8, line = 64, half = sz / 2;
  uint8_t *d = (uint8_t *)mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  for (uint64_t a = 0; a < sz; a += word) *(uint64_t *)(d + a) = tag(base + a);
  backing_data_t dat{d, sz};
  mm_t *m = mm_perturb_make(mm_perturb_parse(argc, (const char *const *)argv), base, sz, word, line, dat);

  static const uint64_t lens[] = {0, 1, 3, 7};
  std::deque<req_t> ar, aw, w;  // reads to issue; AWs to issue; bursts whose W beats are still to send
  std::vector<uint64_t> expect(half / word);  // the write half as the writes must leave it
  for (uint64_t i = 0; i < half / word; i++) expect[i] = tag(base + half + i * word);
  for (uint64_t i = 0; i < nreads; i++)
    ar.push_back({base + (rnd() % (half / line)) * line, rnd() % 4, lens[rnd() % 4]});
  for (uint64_t i = 0; i < nwrites; i++) {  // each write owns a fresh line, so the final image is unambiguous
    req_t r{base + half + i * line, rnd() % 4, lens[rnd() % 4]};
    aw.push_back(r);
    w.push_back(r);
  }
  bool arv = false, awv = false, wv = false;
  uint64_t wbeat = 0, wdata = 0, wstrb = 0, reads_done = 0, writes_done = 0, idle = 0;
  for (uint64_t cyc = 0; cyc < max_cycles && idle < 64; cyc++) {
    bool arr = m->ar_ready(), awr = m->aw_ready(), wr = m->w_ready(), rv = m->r_valid(), bv = m->b_valid();
    uint64_t rid = m->r_id(), rlast = m->r_last(), rdata = *(uint64_t *)m->r_data(), bid = m->b_id();
    if (!arv && !ar.empty() && coin(pace)) arv = true;
    if (!awv && !aw.empty() && coin(pace)) awv = true;
    if (!wv && !w.empty() && coin(pace)) {
      wv = true;
      wdata = tag(w.front().addr + wbeat * word) ^ 0xa5a5a5a5a5a5a5a5ULL;
      wstrb = coin(80) ? 0xff : (rnd() & 0xff);
    }
    bool wlast = wv && wbeat == w.front().len;
    bool rr = coin(ready), br = coin(ready);
    req_t a = arv ? ar.front() : req_t{0, 0, 0}, b = awv ? aw.front() : req_t{0, 0, 0};
    printf("C %d %llx %llu %llu %d %llu %llu %d %d %d %d | %d %d %d %d %llu %llu %llx %d %llu\n", arv,
           (unsigned long long)a.addr, (unsigned long long)a.id, (unsigned long long)a.len, awv,
           (unsigned long long)b.id, (unsigned long long)b.len, wv, wlast, rr, br, arr, awr, wr, rv,
           (unsigned long long)rid, (unsigned long long)rlast, (unsigned long long)rdata, bv,
           (unsigned long long)bid);
    m->tick(false, arv, a.addr, a.id, 3, a.len, awv, b.addr, b.id, 3, b.len, wv, wstrb, &wdata, wlast, rr, br);
    if (arv && arr) { ar.pop_front(); arv = false; }
    if (awv && awr) { aw.pop_front(); awv = false; }
    if (wv && wr) {
      uint64_t i = (w.front().addr + wbeat * word - base - half) / word;
      for (int k = 0; k < 8; k++)
        if (wstrb >> k & 1) ((uint8_t *)&expect[i])[k] = ((uint8_t *)&wdata)[k];
      if (wlast) { w.pop_front(); wbeat = 0; } else wbeat++;
      wv = false;
    }
    if (rv && rr && rlast) reads_done++;
    if (bv && br) writes_done++;
    bool drained = ar.empty() && aw.empty() && w.empty() && reads_done == nreads && writes_done == nwrites;
    idle = drained ? idle + 1 : 0;
  }
  printf("END reads=%llu writes=%llu\n", (unsigned long long)reads_done, (unsigned long long)writes_done);
  for (uint64_t i = 0; i < half / word; i++)
    if (*(uint64_t *)(d + half + i * word) != expect[i]) {
      printf("WRITES_BAD %llx\n", (unsigned long long)(base + half + i * word));
      return 3;
    }
  printf("WRITES_OK\n");
  return reads_done == nreads && writes_done == nwrites ? 0 : 4;
}
"""


@dataclass
class Cycle:
    arv: int
    araddr: int
    arid: int
    arlen: int
    awv: int
    awid: int
    awlen: int
    wv: int
    wlast: int
    rr: int
    br: int
    out: dict


def _parse(stdout: str) -> list[Cycle]:
    cycles = []
    for ln in stdout.splitlines():
        if not ln.startswith("C "):
            continue
        drove, _, shown = ln[2:].partition(" | ")
        d, s = drove.split(), shown.split()
        ins = [int(d[0]), int(d[1], 16), *map(int, d[2:])]
        out = dict(zip(("arr", "awr", "wr", "rv", "rid", "rlast"), map(int, s[:6]), strict=True))
        out.update(rdata=int(s[6], 16), bv=int(s[7]), bid=int(s[8]))
        cycles.append(Cycle(*ins, out=out))
    return cycles


# --------------------------------------------------------------------------------------------------
# the reference: FASED's latency pipe, transcribed from its Chisel
# --------------------------------------------------------------------------------------------------


def _tag(addr: int) -> int:
    return (addr * 0x9E3779B97F4A7C15) & (2**64 - 1)


def _sat(value: int, inc: bool, dec: bool, n: int) -> int:
    """midas.widgets.SatUpDownCounter: full = value >= max; inc and dec together cancel."""
    if inc and not dec and value < n:
        return value + 1
    if dec and not inc and value > 0:
        return value - 1
    return value


class FasedPipe:
    """LatencyPipe + SplitTransactionModel + AXI4Releaser, one call of :meth:`step` per target cycle.

    Field for field: ``pending_*`` are TimingModel's three SatUpDownCounters; ``aw_queue``, ``read_pipe``
    and ``write_pipe`` are the flow=true Queues (an entry enqueued this cycle may leave this cycle);
    ``current_read``/``current_write`` are AXI4Releaser's Queue(1, pipe=true), whose occupant IS what the R
    and B channels present. releaseCycle = latency + tCycle - egressUnitDelay, with egressUnitDelay = 1.
    """

    def __init__(self, read_latency: int, write_latency: int, max_reads: int, max_writes: int):
        self.rl, self.wl, self.mr, self.mw = read_latency, write_latency, max_reads, max_writes
        self.t = 0
        self.pending_reads = self.pending_aw = self.pending_w = 0
        self.aw_queue: deque[int] = deque()
        self.read_pipe: deque[list[int]] = deque()  # [id, addr, beats, releaseCycle]
        self.write_pipe: deque[list[int]] = deque()  # [id, releaseCycle]
        self.current_read: list[int] | None = None  # [id, addr, beats, beat]
        self.current_write: int | None = None

    def outputs(self) -> dict:
        cr = self.current_read
        return {
            "arr": int(self.pending_reads < self.mr),  # nastiReq.ar.ready := ~pendingReads.full
            "awr": int(self.pending_aw < self.mw),
            "wr": int(self.pending_w < self.mw),
            "rv": int(cr is not None),
            "rid": cr[0] if cr else None,
            "rlast": int(cr[3] == cr[2] - 1) if cr else None,
            "rdata": _tag(cr[1] + 8 * cr[3]) if cr else None,
            "bv": int(self.current_write is not None),
            "bid": self.current_write,
        }

    def step(self, c: Cycle) -> None:
        o = self.outputs()
        ar_fire, aw_fire, w_fire = bool(c.arv and o["arr"]), bool(c.awv and o["awr"]), bool(c.wv and o["wr"])
        r_fire, b_fire = bool(o["rv"] and c.rr), bool(o["bv"] and c.br)
        w_last_fire, r_last_fire = w_fire and bool(c.wlast), r_fire and bool(o["rlast"])
        # SplitTransactionModel.newWReq (no LLC), on the counters' register values
        pw, paw = self.pending_w, self.pending_aw
        new_w = (pw > paw and aw_fire) or (pw < paw and w_last_fire) or (w_last_fire and aw_fire)
        if aw_fire:
            self.aw_queue.append(c.awid)
        if new_w:
            self.write_pipe.append([self.aw_queue.popleft(), self.wl + self.t - 1])
        if ar_fire:
            self.read_pipe.append([c.arid, c.araddr, c.arlen + 1, self.rl + self.t - 1])
        # AXI4Releaser: currentRead.ready := r.fire && r.last; currentWrite.ready := b.fire
        take_r = bool(self.read_pipe) and self.read_pipe[0][3] <= self.t and (self.current_read is None or r_last_fire)
        take_b = bool(self.write_pipe) and self.write_pipe[0][1] <= self.t and (self.current_write is None or b_fire)
        if r_fire:
            if r_last_fire:
                self.current_read = None
            else:
                self.current_read[3] += 1
        if take_r:
            rid, addr, beats, _ = self.read_pipe.popleft()
            self.current_read = [rid, addr, beats, 0]
        if b_fire:
            self.current_write = None
        if take_b:
            self.current_write = self.write_pipe.popleft()[0]
        self.pending_reads = _sat(self.pending_reads, ar_fire, r_last_fire, self.mr)
        self.pending_aw = _sat(self.pending_aw, aw_fire, b_fire, self.mw)
        self.pending_w = _sat(self.pending_w, w_last_fire, b_fire, self.mw)
        self.t += 1


def _first_disagreement(cycles: list[Cycle], cfg: tuple[int, int, int, int]) -> str | None:
    ref = FasedPipe(*cfg)
    for n, c in enumerate(cycles):
        want = ref.outputs()
        got = c.out
        for k in ("arr", "awr", "wr", "rv", "bv"):
            if got[k] != want[k]:
                return f"cycle {n}: {k} model={got[k]} fased={want[k]}"
        if want["rv"] and (got["rid"], got["rlast"], got["rdata"]) != (want["rid"], want["rlast"], want["rdata"]):
            return f"cycle {n}: R beat model={got} fased={want}"
        if want["bv"] and got["bid"] != want["bid"]:
            return f"cycle {n}: B id model={got['bid']} fased={want['bid']}"
        ref.step(c)
    return None


# --------------------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def master(mm_build, tmp_path_factory):
    return mm_build(MASTER, tmp_path_factory.mktemp("lbp"))


def _pipe_args(cfg) -> list[str]:
    rl, wl, mr, mw = cfg
    return [
        "+mem_latency_pipe=1", f"+mem_latency_pipe_read_latency={rl}", f"+mem_latency_pipe_write_latency={wl}",
        f"+mem_latency_pipe_max_reads={mr}", f"+mem_latency_pipe_max_writes={mw}",
    ]  # fmt: skip


def _drive(exe: Path, *args: str, env: dict | None = None, check: bool = True) -> subprocess.CompletedProcess:
    run_env = {k: v for k, v in (env if env is not None else os.environ).items()}
    p = subprocess.run([str(exe), *args], capture_output=True, text=True, env=run_env, check=False)
    if check:
        assert p.returncode == 0, f"master failed ({p.returncode}): {p.stdout[-400:]} {p.stderr[-400:]}"
    return p


# (read_latency, write_latency, max_reads, max_writes), master pace %, ready %
CONFIGS = {
    "fased_stock": ((30, 30, 16, 16), 60, 75),
    "fased_stock_saturated": ((30, 30, 16, 16), 100, 100),
    "unit_latency": ((1, 1, 16, 16), 60, 75),
    "tight_bounds": ((7, 3, 2, 1), 80, 60),
    "one_read": ((2, 9, 1, 4), 70, 90),
    "backpressured": ((5, 5, 4, 4), 100, 20),
}


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_agrees_with_fased_cycle_for_cycle(master, name, seed):
    cfg, pace, ready = CONFIGS[name]
    p = _drive(master, *_pipe_args(cfg), f"stim_seed={seed}", f"pace={pace}", f"ready={ready}")
    assert "WRITES_OK" in p.stdout, "every write must land, whether its data led or trailed its address"
    cycles = _parse(p.stdout)
    assert len(cycles) > 500
    assert _first_disagreement(cycles, cfg) is None, _first_disagreement(cycles, cfg)


def test_the_master_reaches_the_corners_the_comparison_needs(master):
    """The comparison is only as good as the traffic: the master must make the read bound bite, send W data
    ahead of its AW, and stall responses -- or agreement on those paths would be agreement on nothing."""
    cfg, pace, ready = CONFIGS["tight_bounds"]
    cycles = _parse(_drive(master, *_pipe_args(cfg), "stim_seed=1", f"pace={pace}", f"ready={ready}").stdout)
    assert any(c.arv and not c.out["arr"] for c in cycles), "the read bound never backpressured"
    assert any(c.awv and not c.out["awr"] for c in cycles), "the write bound never backpressured"
    assert any(c.out["rv"] and not c.rr for c in cycles), "the R channel was never stalled"
    aws = wlasts = 0
    led = False
    for c in cycles:
        wlasts += bool(c.wv and c.wlast and c.out["wr"])
        led |= wlasts > aws
        aws += bool(c.awv and c.out["awr"])
    assert led, "no W burst completed ahead of its AW"


def _fires(cycles, valid, ready):
    return [n for n, c in enumerate(cycles) if getattr(c, valid) and c.out[ready]]


@pytest.mark.parametrize("latency", [1, 2, 30])
def test_a_read_is_answered_exactly_read_latency_after_acceptance(master, latency):
    cycles = _parse(
        _drive(master, *_pipe_args((latency, 30, 16, 16)), "reads=1", "writes=0", "pace=100", "ready=100").stdout
    )
    accepted = _fires(cycles, "arv", "arr")[0]
    first = next(n for n, c in enumerate(cycles) if c.out["rv"])
    assert first - accepted == latency


def test_a_write_is_answered_exactly_write_latency_after_address_and_data_are_both_in(master):
    cycles = _parse(_drive(master, *_pipe_args((30, 17, 16, 16)), "reads=0", "writes=1", "stim_seed=4").stdout)
    aw = _fires(cycles, "awv", "awr")[0]
    wlast = next(n for n, c in enumerate(cycles) if c.wv and c.wlast and c.out["wr"])
    first_b = next(n for n, c in enumerate(cycles) if c.out["bv"])
    assert first_b - max(aw, wlast) == 17


def test_reads_beyond_max_reads_wait(master):
    """With nothing ever drained, exactly max_reads reads are accepted and the next waits forever."""
    cycles = _parse(
        _drive(
            master, *_pipe_args((3, 3, 5, 5)), "reads=40", "writes=0", "ready=0", "max_cycles=400", check=False
        ).stdout
    )
    assert len(_fires(cycles, "arv", "arr")) == 5


def test_off_is_the_stock_model(master):
    """Absent or 0, the pipe is not built: the factory answers in order, one cycle after acceptance."""
    for args in ([], ["+mem_latency_pipe=0"]):
        p = _drive(master, *args, "reads=1", "writes=0", "pace=100", "ready=100")
        assert MP.reported_models(p.stderr) == ("in_order",)
        cycles = _parse(p.stdout)
        assert next(n for n, c in enumerate(cycles) if c.out["rv"]) - _fires(cycles, "arv", "arr")[0] == 1


def test_the_factory_names_the_model_and_its_settings(master):
    p = _drive(master, *_pipe_args((30, 29, 16, 15)), "reads=1", "writes=0")
    assert MP.reported_models(p.stderr) == ("latency_pipe",)
    assert "read_latency=30 write_latency=29 max_reads=16 max_writes=15" in p.stderr


def test_the_environment_spelling_is_the_same_model(master):
    env = {
        **os.environ, "MERLIN_MEM_LATENCY_PIPE": "1", "MERLIN_MEM_LATENCY_PIPE_READ_LATENCY": "7",
        "MERLIN_MEM_LATENCY_PIPE_WRITE_LATENCY": "3", "MERLIN_MEM_LATENCY_PIPE_MAX_READS": "2",
        "MERLIN_MEM_LATENCY_PIPE_MAX_WRITES": "1",
    }  # fmt: skip
    by_env = _drive(master, "stim_seed=5", env=env).stdout
    by_plusarg = _drive(master, *_pipe_args((7, 3, 2, 1)), "stim_seed=5").stdout
    assert by_env == by_plusarg


@pytest.mark.parametrize(
    "args",
    [
        ["+mem_latency_pipe=1", "+mem_latency_pipe_read_latency=abc"],
        ["+mem_latency_pipe=1", "+mem_latency_pipe_read_latency=0"],
        ["+mem_latency_pipe=1", "+mem_latency_pipe_max_writes=0"],
        ["+mem_latency_pipe=2"],
    ],
)
def test_a_bad_knob_is_refused_not_ignored(master, args):
    p = _drive(master, *args, "reads=1", "writes=0", check=False)
    assert p.returncode != 0 and "[mem-latency-pipe]" in p.stderr


def test_both_models_at_once_is_refused(master):
    p = _drive(master, "+mem_latency_pipe=1", "+mem_perturb_seed=3", "reads=1", "writes=0", check=False)
    assert p.returncode != 0 and "give one" in p.stderr


# Each mutant is one plausible mistake in reading FASED; each must make the comparison fail somewhere.
MUTANTS = {
    "answers a cycle early (the releaser register forgotten)": (
        "rburst_t b{ar_id, cycle + cfg.read_latency, {}};",
        "rburst_t b{ar_id, cycle + cfg.read_latency - 1, {}};",
    ),
    "acknowledges writes a cycle late": (
        "bq.push_back(bresp_t{a.id, cycle + cfg.write_latency});",
        "bq.push_back(bresp_t{a.id, cycle + cfg.write_latency + 1});",
    ),
    "never bounds outstanding reads": ("    pending_reads++;\n", "\n"),
    "bounds W by beats instead of complete bursts": ("pending_w += w_last;", "pending_w += 1;"),
}


@pytest.mark.parametrize("mistake", sorted(MUTANTS))
def test_a_mutant_model_disagrees_with_fased(mm_build, tmp_path, mistake):
    anchor, replacement = MUTANTS[mistake]
    src = (MP.model_sources_dir() / MP.LATENCY_PIPE_SOURCE).read_text()
    assert src.count(anchor) == 1, f"mutation anchor drifted: {anchor!r}"
    mutant = tmp_path / MP.LATENCY_PIPE_SOURCE
    mutant.write_text(src.replace(anchor, replacement))
    exe = mm_build(MASTER, tmp_path, replace={MP.LATENCY_PIPE_SOURCE: mutant})
    caught = []
    for name, (cfg, pace, ready) in CONFIGS.items():
        p = _drive(exe, *_pipe_args(cfg), f"pace={pace}", f"ready={ready}", "max_cycles=20000", check=False)
        caught.append(p.returncode != 0 or _first_disagreement(_parse(p.stdout), cfg) is not None)
    assert any(caught), f"no configuration tells the model that {mistake} from FASED"


# --------------------------------------------------------------------------------------------------
# the run path: knobs, the pinned model, and the console's word for what ran
# --------------------------------------------------------------------------------------------------

STOCK = {"read_latency": 30, "write_latency": 30, "max_reads": 16, "max_writes": 16}


def _echo(tmp_path: Path, model_line: str | None) -> Path:
    """A stand-in engine: prints its argv and the latency-pipe environment, and the factory line if given."""
    p = tmp_path / "engine"
    line = f"sys.stderr.write({model_line!r} + '\\n')" if model_line else "pass"
    p.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import os, sys
            print(' '.join(sys.argv[1:]))
            print(' '.join(f'{{k}}={{v}}' for k, v in sorted(os.environ.items()) if k.startswith('MERLIN_MEM_')))
            {line}
            """
        )
    )
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def test_the_pipe_knobs_reach_the_engine_in_either_spelling(tmp_path):
    eng = _echo(tmp_path, None)
    rc, text, _ = MP.run_once(eng, "p.elf", seed=None, latency_pipe=STOCK, plusarg_wrap=("+on", "+off"))
    assert text.splitlines()[0].split() == [
        "+on", "+mem_latency_pipe=1", "+mem_latency_pipe_read_latency=30", "+mem_latency_pipe_write_latency=30",
        "+mem_latency_pipe_max_reads=16", "+mem_latency_pipe_max_writes=16", "+off", "p.elf",
    ]  # fmt: skip
    rc, text, _ = MP.run_once(eng, "p.elf", seed=None, latency_pipe=STOCK, via="env")
    assert text.splitlines()[0].split() == ["p.elf"], "the environment form leaves the command line alone"
    assert text.splitlines()[1].split() == [
        "MERLIN_MEM_LATENCY_PIPE=1", "MERLIN_MEM_LATENCY_PIPE_MAX_READS=16", "MERLIN_MEM_LATENCY_PIPE_MAX_WRITES=16",
        "MERLIN_MEM_LATENCY_PIPE_READ_LATENCY=30", "MERLIN_MEM_LATENCY_PIPE_WRITE_LATENCY=30",
    ]  # fmt: skip


def test_an_inherited_knob_never_leaks_into_a_run(tmp_path, monkeypatch):
    """A stray MERLIN_MEM_LATENCY_PIPE in the caller's environment must not turn an in-order run into a timed
    one, nor a timed run's knobs into another's."""
    monkeypatch.setenv("MERLIN_MEM_LATENCY_PIPE", "1")
    monkeypatch.setenv("MERLIN_MEM_PERTURB_SEED", "9")
    rc, text, _ = MP.run_once(_echo(tmp_path, None), "p.elf", seed=None, via="env")
    assert text.splitlines()[1].strip() == ""


def test_a_run_is_under_one_model(tmp_path):
    with pytest.raises(ValueError, match="ONE memory model"):
        MP.run_once(_echo(tmp_path, None), "p.elf", seed=3, latency_pipe=STOCK)


@pytest.mark.parametrize(
    "declared",
    [
        {"kind": "fased_fr_fcfs", **STOCK},
        {"kind": MP.FASED_LATENCY_PIPE, "read_latency": 30, "write_latency": 30, "max_reads": 16},
        {**STOCK, "kind": MP.FASED_LATENCY_PIPE, "read_latency": 0},
        {**STOCK, "kind": MP.FASED_LATENCY_PIPE, "max_writes": "16"},
        {**STOCK, "kind": MP.FASED_LATENCY_PIPE, "max_reads": True},
    ],
)
def test_a_memory_model_the_pipe_cannot_reproduce_is_refused(declared):
    with pytest.raises(MP.MemoryModelError):
        MP.latency_pipe_knobs(declared)


def _registry(tmp_path: Path, memory_model: dict | None) -> Path:
    art = {"path": "firesim.tar.gz", "digest": "a" * 64, "role": "firesim_bitstream"}
    if memory_model is not None:
        art["memory_model"] = memory_model
    reg = tmp_path / "pins.yaml"
    reg.write_text(json.dumps({"artifacts": {"board": art}}))
    return reg


def test_the_pinned_memory_model_is_what_a_run_reproduces(tmp_path):
    reg = _registry(tmp_path, {"kind": MP.FASED_LATENCY_PIPE, **STOCK, "evidence": "read from source"})
    assert MP.pinned_latency_pipe("board", registry=reg) == STOCK
    with pytest.raises(MP.MemoryModelError, match="declares no memory_model"):
        MP.pinned_latency_pipe("board", registry=_registry(tmp_path, None))
    with pytest.raises(MP.MemoryModelError, match="no artifact"):
        MP.pinned_latency_pipe("other", registry=reg)


def test_every_declared_memory_model_in_the_registry_is_one_the_engine_reproduces():
    from merlin.common import provenance

    declared = {n: a.memory_model_dict() for n, a in provenance.load_artifacts().items() if a.memory_model}
    assert declared, "the registry declares no memory model, so no board's timing can be reproduced"
    for name, model in declared.items():
        assert MP.latency_pipe_knobs(model), name
        assert model.get("evidence"), f"{name}: a memory_model must say where its values were read"


@pytest.mark.parametrize(
    "bad",
    [
        {"read_latency": 30},
        {"kind": "", "read_latency": 30},
        {"kind": "x", "read_latency": 30.0},
        {"kind": "x", "a": [1]},
    ],
)
def test_a_malformed_memory_model_is_refused_at_load(tmp_path, bad):
    from merlin.common import provenance

    with pytest.raises(provenance.PinsError, match="memory_model"):
        provenance.load_artifacts(_registry(tmp_path, bad))


def test_the_console_names_the_model_each_channel_ran(tmp_path):
    con = "x\n[mem-perturb] model=latency_pipe read_latency=30\n[mem-perturb] model=in_order (mm_magic_t)\n"
    assert MP.reported_models(con) == ("latency_pipe", "in_order")
    assert MP.reported_models("no factory here\n") == ()


@pytest.mark.parametrize(
    ("line", "confirmed"),
    [
        ("[mem-perturb] model=latency_pipe read_latency=30", True),
        ("[mem-perturb] model=in_order (mm_magic_t)", False),  # an engine that ignored the knobs
        (None, False),  # an engine with no factory at all
    ],
)
def test_a_timed_run_is_believed_only_when_the_console_confirms_it(tmp_path, line, confirmed):
    elf = tmp_path / "p.elf"
    elf.write_bytes(b"x")
    rec = MP.run_under(_echo(tmp_path, line), elf, latency_pipe=STOCK)
    assert rec["model_requested"] == "latency_pipe" and rec["model_confirmed"] is confirmed


def test_the_operator_command_takes_the_timing_from_the_pin_and_refuses_an_unconfirmed_run(
    tmp_path, monkeypatch, capsys
):
    from merlin.common import provenance as P
    from merlin.targetgen import tool_cli

    monkeypatch.setattr(P, "pins_path", lambda: _registry(tmp_path, {"kind": MP.FASED_LATENCY_PIPE, **STOCK}))
    elf = tmp_path / "p.elf"
    elf.write_bytes(b"x")
    good = _echo(tmp_path, "[mem-perturb] model=latency_pipe read_latency=30")
    assert (
        tool_cli.main(["mem-model-run", "--emulator", str(good), "--elf", str(elf), "--firesim-artifact", "board"]) == 0
    )
    rec = json.loads(capsys.readouterr().out)
    assert rec["timing_of"]["artifact"] == "board" and rec["timing_of"]["digest"] == "a" * 64
    assert rec["latency_pipe"] == STOCK and rec["model_confirmed"]
    stale = tmp_path / "stale"
    stale.mkdir()
    old = _echo(stale, "[mem-perturb] model=in_order (mm_magic_t)")
    assert (
        tool_cli.main(["mem-model-run", "--emulator", str(old), "--elf", str(elf), "--firesim-artifact", "board"]) == 2
    )
    capsys.readouterr()
    assert tool_cli.main(["mem-model-run", "--emulator", str(old), "--elf", str(elf), "--in-order"]) == 0
    assert json.loads(capsys.readouterr().out)["model_requested"] == "in_order"


# --------------------------------------------------------------------------------------------------
# the variant build: every model compiled in, from a relative output directory
# --------------------------------------------------------------------------------------------------

FAKE_HARNESS = r"""
#include <cstdio>
#include <sys/mman.h>
#include "mm.h"
// A harness of the shape the rewrite targets: it constructs the stock model, then drives one read and
// prints how many cycles the answer took.
int main(int argc, char **argv) {
  const size_t sz = 1 << 16;
  uint8_t *d = (uint8_t *)mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  backing_data_t dat{d, sz};
  mm_t *mm = (mm_t *) (new mm_magic_t(0x80000000, sz, 8, 64, dat));
  uint64_t w = 0;
  int accepted = -1;
  for (int c = 0; c < 1000; c++) {
    if (accepted >= 0 && mm->r_valid()) { printf("ANSWERED %d\n", c - accepted); return 0; }
    bool arv = accepted < 0;
    if (arv && mm->ar_ready()) accepted = c;
    mm->tick(false, arv, 0x80000000, 0, 3, 0, false, 0, 0, 3, 0, false, 0, &w, false, true, true);
  }
  return 1;
}
"""


def test_the_variant_links_every_model_and_builds_from_a_relative_out_dir(mm_build, tmp_path, monkeypatch):
    """A miniature of the Verilator build the variant is cut from -- a makefile whose dry run names each
    compile and the link -- built for real, then rebuilt as a variant into a RELATIVE directory (which used
    to compile nothing: the patched source was named relative to a cwd that was the output directory)."""
    gen, obj = tmp_path / "gen", tmp_path / "obj"
    gen.mkdir()
    obj.mkdir()
    for n in ("mm.h", "mm.cc"):
        (gen / n).write_text((mm_build.mm_dir / n).read_text())
    (gen / "SimDRAM.cc").write_text(FAKE_HARNESS)
    flags = f"-std=c++17 -I{gen} -I{mm_build.fesvr}"
    (obj / "V.mk").write_text(
        f"VM_USER_CLASSES = \\\n\tSimDRAM \\\n\tmm \\\n\nVM_USER_DIR = \\\n\t{gen} \\\n\n"
        f"sim: SimDRAM.o mm.o\n\t{mm_build.cxx} SimDRAM.o mm.o -o sim\n"
        f"SimDRAM.o: {gen}/SimDRAM.cc\n\t{mm_build.cxx} {flags} -c {gen}/SimDRAM.cc -o SimDRAM.o\n"
        f"mm.o: {gen}/mm.cc\n\t{mm_build.cxx} {flags} -c {gen}/mm.cc -o mm.o\n"
    )
    subprocess.run(["make", "-C", str(obj), "-f", "V.mk"], check=True, capture_output=True)
    monkeypatch.chdir(tmp_path)
    receipt = MP.build_verilator_variant(obj, "variant", makefile="V.mk", argc_expr="argc", argv_expr="argv")
    assert set(receipt["model_sources"]) == set(MP.MODEL_FILES)
    assert receipt["models"]["latency_pipe"].startswith("+mem_latency_pipe=1")
    emu = Path(receipt["binary"]["path"])
    assert emu.is_absolute() and emu.is_file()

    def answered(*args: str) -> tuple[int, tuple[str, ...]]:
        p = subprocess.run([str(emu), *args], capture_output=True, text=True, check=True)
        return int(p.stdout.split()[-1]), MP.reported_models(p.stderr)

    assert answered() == (1, ("in_order",))
    assert answered(*_pipe_args((9, 9, 16, 16))) == (9, ("latency_pipe",))
    assert answered("+mem_perturb_seed=5", "+mem_perturb_max_latency=0")[1] == ("perturbed",)
