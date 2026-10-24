"""The memory-order-perturbing simulator variant: the harness rewrite, the C++ model, and the sweep verdict.

Each property is held by a test that a specific broken version would fail, and the broken version is
BUILT here and shown to fail it -- a race detector whose own tests cannot fail is how a clean sweep gets
read as a certificate.
"""

from __future__ import annotations

import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from merlin.targetgen import mem_perturb as MP

# --------------------------------------------------------------------------------------------------
# the harness rewrite
# --------------------------------------------------------------------------------------------------

# Two harness shapes the rewrite must handle without knowing either: a DPI SimDRAM (multi-line args
# carrying a nested subscript) and a GSIM-style blackbox (a single line against file-scope constants).
DPI_SHAPE = textwrap.dedent(
    """\
    #include "mm_dramsim2.h"
    void *memory_init(int chip_id) {
        if (use_dramsim)
          mm = (mm_t *) (new mm_dramsim2_t(mem_base, mem_size, word_size, line_size, ini));
        else
          mm = (mm_t *) (new mm_magic_t(mem_base, mem_size, word_size, line_size,
                                        backing_mem_data[chip_id][mem_base]));
        return mm;
    }
    """
)
GSIM_SHAPE = "static mm_t* mm = nullptr;\nvoid SimDRAM() { if (!mm) { mm = new mm_magic_t(MEM_BASE, MEM_SIZE, 8, 64, g_dram); } }\n"


def test_rewrite_carries_the_constructor_arguments_verbatim():
    out = MP.patch_harness_source(DPI_SHAPE, "info.argc", "info.argv")
    assert out.startswith('#include "mm_perturb.h"\n')
    assert "new mm_magic_t(" not in out
    assert (
        "mm_perturb_make(mm_perturb_parse(info.argc, (const char *const *)(info.argv)), mem_base, mem_size, "
        "word_size, line_size,\n                                    backing_mem_data[chip_id][mem_base])" in out
    )
    # the other memory model is untouched: only the stock one is replaced
    assert "new mm_dramsim2_t(mem_base, mem_size, word_size, line_size, ini)" in out


def test_rewrite_is_harness_agnostic():
    out = MP.patch_harness_source(GSIM_SHAPE, "g_argc", "g_argv")
    assert (
        "mm_perturb_make(mm_perturb_parse(g_argc, (const char *const *)(g_argv)), MEM_BASE, MEM_SIZE, 8, 64, g_dram)"
        in out
    )


@pytest.mark.parametrize("text", ["int x;\n", GSIM_SHAPE + GSIM_SHAPE])
def test_rewrite_refuses_anything_but_one_constructor(text):
    with pytest.raises(MP.PerturbBuildError):
        MP.patch_harness_source(text, "argc", "argv")


def test_makefile_variables_are_read_across_continuations(tmp_path):
    gen = tmp_path / "gen"
    gen.mkdir()
    for n in ("SimDRAM.cc", "mm.cc", "uart.cc"):
        (gen / n).write_text("//\n")
    mk = tmp_path / "V.mk"
    mk.write_text(f"VM_USER_CLASSES = \\\n\tSimDRAM \\\n\tmm \\\n\tuart \\\n\nVM_USER_DIR = \\\n\t{gen} \\\n\n")
    assert [p.name for p in MP._user_sources(mk)] == ["SimDRAM.cc", "mm.cc", "uart.cc"]


def test_commands_are_relocated_so_the_source_build_is_never_written(tmp_path):
    obj, out = tmp_path / "obj", tmp_path / "out"
    obj.mkdir()
    (obj / "a.o").write_text("")
    argv = ["c++", "-I.", "a.o", "b.o", "-o", "/elsewhere/sim"]
    got = MP._relocate(argv, obj, out, {"b.o": out / "b.o"})
    assert got == ["c++", f"-I{obj}", str(obj / "a.o"), str(out / "b.o"), "-o", str(out / "sim")]


# --------------------------------------------------------------------------------------------------
# the C++ model, compiled against the real testchipip mm.h/mm.cc
# --------------------------------------------------------------------------------------------------

DRIVER = r"""
#include <cstdio>
#include <sys/mman.h>
#include "mm_perturb.h"
int main(int argc, char **argv) {
  const size_t sz = 1 << 16, N = 96, line = 64;
  uint8_t *d = (uint8_t *)mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  for (size_t i = 0; i < sz / line; i++) *(uint64_t *)(d + i * line) = i;
  backing_data_t dat{d, sz};
  mm_t *m = mm_perturb_make(mm_perturb_parse(argc, (const char *const *)argv), 0x80000000, sz, 8, 64, dat);
  uint64_t wbuf = 0; size_t issued = 0, done = 0;
  for (int cyc = 0; done < N && cyc < 100000; cyc++) {
    if (m->r_valid()) { printf("R %llu %llu\n", (unsigned long long)m->r_id(), (unsigned long long)*(uint64_t *)m->r_data()); done++; }
    bool arv = issued < N;
    m->tick(false, arv, 0x80000000 + issued * line, issued % 4, 3, 0, false, 0, 0, 3, 0, false, 0, &wbuf, false, true, true);
    if (arv) issued++;
  }
  return done == N ? 0 : 3;
}
"""


def _order(exe: Path, *plusargs: str, env: dict | None = None) -> list[tuple[int, int]]:
    p = subprocess.run([str(exe), *plusargs], capture_output=True, text=True, check=True, env=env)
    return [(int(a), int(b)) for _, a, b in (ln.split() for ln in p.stdout.splitlines() if ln.startswith("R "))]


def _same_id_order_kept(seq) -> bool:
    last: dict[int, int] = {}
    for rid, idx in seq:
        if idx <= last.get(rid, -1):
            return False
        last[rid] = idx
    return True


@pytest.fixture(scope="module")
def model_exe(mm_build, tmp_path_factory):
    return mm_build(DRIVER, tmp_path_factory.mktemp("mm"))


def test_without_a_seed_the_stock_in_order_model_answers(model_exe):
    seq = _order(model_exe)
    assert [idx for _, idx in seq] == list(range(96))


def test_a_seed_reorders_across_ids_and_never_within_one(model_exe):
    seq = _order(model_exe, "+mem_perturb_seed=7", "+mem_perturb_max_latency=32")
    idx = [i for _, i in seq]
    assert sorted(idx) == list(range(96)), "every request answered exactly once, with its own data"
    assert all(rid == i % 4 for rid, i in seq), "each response carries the data of the request with its id"
    assert idx != sorted(idx), "a seeded run must actually reorder, or it tests nothing"
    assert _same_id_order_kept(seq), "AXI4 keeps same-id responses in acceptance order"


def test_a_schedule_is_a_function_of_its_seed(model_exe):
    a = _order(model_exe, "+mem_perturb_seed=11")
    assert a == _order(model_exe, "+mem_perturb_seed=11")
    assert a != _order(model_exe, "+mem_perturb_seed=12")


def test_the_same_id_check_catches_a_model_that_breaks_axi_order(mm_build, tmp_path):
    """MUTATION: a model that lets a response overtake an older one with the SAME id violates AXI4 and
    would manufacture failures real hardware cannot produce. Build that model and show the check fails."""
    src = (MP.model_sources_dir() / MP.MODEL_SOURCE).read_text()
    anchor = "blocked = q[j].id == q[i].id;"
    assert src.count(anchor) == 1
    mutant = tmp_path / "mm_perturb_mutant.cc"
    mutant.write_text(src.replace(anchor, "blocked = false;"))
    exe = mm_build(DRIVER, tmp_path, replace={MP.MODEL_SOURCE: mutant})
    broken = [
        _same_id_order_kept(_order(exe, f"+mem_perturb_seed={s}", "+mem_perturb_max_latency=32")) for s in range(1, 9)
    ]
    assert not all(broken), "the same-id check must fail on a model that reorders within an id"


def test_a_malformed_knob_is_refused_not_ignored(model_exe):
    p = subprocess.run([str(model_exe), "+mem_perturb_seed=abc"], capture_output=True, text=True)
    assert p.returncode != 0 and "malformed" in p.stderr


# --------------------------------------------------------------------------------------------------
# the sweep verdict, against a scripted emulator
# --------------------------------------------------------------------------------------------------

FAKE = """#!{py}
import sys
seed = None
for a in sys.argv[1:]:
    if a.startswith("+mem_perturb_seed="):
        seed = int(a.split("=", 1)[1])
racy = {racy}
print("OUT Y 1 2", 3, 4 if (seed is None or not racy or seed % 2) else 1)
print("DONE")
"""


def _fake(tmp_path: Path, racy: bool) -> Path:
    p = tmp_path / ("racy" if racy else "ordered")
    p.write_text(FAKE.format(py=sys.executable, racy=racy))
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def _judge(rc: int, text: str) -> str:
    if rc != 0 or "DONE" not in text:
        return "unjudged"
    return "pass" if "OUT Y 1 2 3 4" in text else "fail"


def test_sweep_flags_a_program_whose_answer_depends_on_the_schedule(tmp_path):
    elf = tmp_path / "prog.elf"
    elf.write_bytes(b"x")
    res = MP.sweep(_fake(tmp_path, True), elf, [1, 2, 3, 4], _judge)
    assert res["verdict"] == "order_sensitive"
    assert res["in_order_outcome"] == "pass", "the in-order run is what every cheap tier already saw"
    assert res["diverging_seeds"] == [2, 4]


def test_sweep_does_not_flag_an_ordered_program(tmp_path):
    elf = tmp_path / "prog.elf"
    elf.write_bytes(b"x")
    res = MP.sweep(_fake(tmp_path, False), elf, [1, 2, 3, 4], _judge, workers=3)
    assert res["verdict"] == "order_stable" and res["diverging_seeds"] == []
    assert [r["seed"] for r in res["runs"]] == [None, 1, 2, 3, 4], "parallel runs keep the in-order run first"


def test_sweep_that_cannot_judge_the_in_order_run_says_so(tmp_path):
    bad = tmp_path / "bad"
    bad.write_text(f"#!{sys.executable}\nraise SystemExit(4)\n")
    bad.chmod(bad.stat().st_mode | stat.S_IEXEC)
    elf = tmp_path / "prog.elf"
    elf.write_bytes(b"x")
    assert MP.sweep(bad, elf, [1], _judge)["verdict"] == "unjudged"


def test_the_seed_knobs_reach_the_simulator_inside_the_wrap(tmp_path):
    echo = tmp_path / "echo"
    echo.write_text(f"#!{sys.executable}\nimport sys\nprint(' '.join(sys.argv[1:]))\n")
    echo.chmod(echo.stat().st_mode | stat.S_IEXEC)
    rc, text, _ = MP.run_once(echo, "p.elf", seed=5, max_latency=9, extra_args=["+x=1"], plusarg_wrap=("+on", "+off"))
    assert rc == 0
    assert text.split() == ["+on", "+x=1", "+mem_perturb_seed=5", "+mem_perturb_max_latency=9",
                            "+mem_perturb_tail_permille=0", "+off", "p.elf"]  # fmt: skip
    rc, text, _ = MP.run_once(echo, "p.elf", seed=None)
    assert text.split() == ["p.elf"], "the in-order run passes no perturbation knob at all"


def test_the_environment_spelling_is_the_same_model(model_exe):
    """A caller that cannot add plusargs to a simulator's command line sets the environment instead."""
    import os

    env = {**os.environ, "MERLIN_MEM_PERTURB_SEED": "7", "MERLIN_MEM_PERTURB_MAX_LATENCY": "32"}
    assert _order(model_exe, env=env) == _order(model_exe, "+mem_perturb_seed=7", "+mem_perturb_max_latency=32")


def test_region_latency_is_a_property_of_the_address(model_exe):
    """With a region class, every request to one region is slow or fast together, so whole runs of
    consecutive addresses overtake whole other runs -- the shape a cache hit/miss split produces."""
    seq = _order(model_exe, "+mem_perturb_seed=3", "+mem_perturb_max_latency=256", "+mem_perturb_region_log2=10")
    idx = [i for _, i in seq]
    assert sorted(idx) == list(range(96)) and idx != sorted(idx) and _same_id_order_kept(seq)
    # 16 lines per 1 KiB region: a fast region's requests are delivered as a group, ahead of slow ones
    first_region = {i // 16 for i in idx[:16]}
    assert len(first_region) <= 3, f"the earliest deliveries should come from few regions, got {first_region}"


def test_the_invocation_is_derived_from_the_builds_own_content(tmp_path):
    """The preload plusarg is used only when the memory harness parses it; the wrap only when a library
    the link line names carries both spellings. Remove either piece of evidence and it is not used."""
    src = tmp_path / "SimMem.cc"
    lib = tmp_path / "libhost.so"
    link = ["c++", f"-L{tmp_path}", "-lhost", "-o", "sim"]
    plan = MP.LinkPlan(tmp_path, tmp_path / "V.mk", {}, link, tmp_path / "sim")
    src.write_text('if (arg.find("+loadmem=") == 0) load(arg);')
    lib.write_bytes(b"...+permissive ... +permissive-off...")
    assert MP.derive_invocation(plan, src)["args"] == ["+permissive", "+loadmem={elf}", "+permissive-off"]
    lib.write_bytes(b"nothing here")
    assert MP.derive_invocation(plan, src)["args"] == ["+loadmem={elf}"]
    src.write_text("// no preload support")
    assert MP.derive_invocation(plan, src)["args"] == []


def test_a_missing_receipt_means_no_invocation(tmp_path):
    assert MP.invocation_for(tmp_path / "emulator") == []


def test_an_unreadable_receipt_is_refused_not_read_as_no_invocation(tmp_path):
    """A receipt that exists but cannot be read must not become "run without the preload": the engine
    would then leave the operands cached, perturb nothing and report a clean sweep."""
    (tmp_path / MP.RECEIPT_NAME).write_text("{ truncated")
    with pytest.raises(ValueError):
        MP.invocation_for(tmp_path / "emulator")
    (tmp_path / MP.RECEIPT_NAME).write_text('{"invocation": {"args": ["+loadmem={elf}"]}}')
    assert MP.invocation_for(tmp_path / "emulator") == ["+loadmem={elf}"]


def test_absent_model_sources_refuse_the_build(tmp_path, monkeypatch):
    """An installed wheel does not ship the C++ model; building from it must say so, not half-build."""
    monkeypatch.setattr(MP, "model_sources_dir", lambda: tmp_path / "absent")
    with pytest.raises(MP.PerturbBuildError, match="absent"):
        MP.build_verilator_variant(tmp_path / "obj", tmp_path / "out", makefile="V.mk")


def test_the_operator_command_matches_the_control_against_the_pinned_digest(tmp_path, monkeypatch, capsys):
    """`merlin-target-tools mem-perturb-variant` reads the stock simulator's digest from the pin
    registry, builds into a fresh directory only, and exits non-zero when the control relink is not
    the pinned simulator -- the receipt's one claim is that the memory model is the only difference."""
    import json

    from merlin.common import provenance as P
    from merlin.targetgen import tool_cli

    registry = tmp_path / "pins.yaml"
    registry.write_text(json.dumps({"artifacts": {"stock_sim": {"path": "sim", "digest": "d" * 64}}}))
    monkeypatch.setattr(P, "pins_path", lambda: registry)
    seen = {}

    def build(obj_dir, out_dir, **kw):
        seen.update(kw)
        return {"control_relink": {"performed": True, "matches_declared_base": kw["base_digest"] == "d" * 64}}

    monkeypatch.setattr(MP, "build_verilator_variant", build)
    argv = ["mem-perturb-variant", "--obj-dir", str(tmp_path / "obj"), "--makefile", "V.mk"]
    assert tool_cli.main([*argv, "--out", str(tmp_path / "v1"), "--base-artifact", "stock_sim"]) == 0
    assert seen == {"makefile": "V.mk", "control": True, "base_digest": "d" * 64}
    registry.write_text(json.dumps({"artifacts": {"stock_sim": {"path": "sim", "digest": "e" * 64}}}))
    assert tool_cli.main([*argv, "--out", str(tmp_path / "v2"), "--base-artifact", "stock_sim"]) == 1
    full = tmp_path / "v3"
    full.mkdir()
    (full / "emulator").write_bytes(b"pinned engine")
    with pytest.raises(SystemExit, match="not empty"):
        tool_cli.main([*argv, "--out", str(full)])
    capsys.readouterr()
