"""Explicit CPU maps and scalar controllers grant no tile/context authority."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.runtime.backends import zephyr_model as zm
from merlin.runtime.backends.zephyr_placement import controller_entry
from merlin.runtime.worker_placement import plan_placement, worker_placement


def board():
    return SimpleNamespace(harts=3, matrix_harts=(0, 1), hart_ids_for=lambda backend: (0, 1, 2))


def test_two_compute_workers_and_distinct_scalar_controller():
    p = plan_placement(board(), cpus=3, workers=2, worker_harts=(0, 1), controller_hart=2, matrix=True)
    assert p.worker_cpus == p.worker_harts == (0, 1)
    assert p.controller_cpu == p.controller_hart == 2
    assert p.to_dict()["os_cpu_count"] == 3 and p.to_dict()["worker_count"] == 2
    assert p.to_dict()["tile_context_safety_established"] is False
    assert p.to_dict()["concurrent_call_reentrancy_established"] is False


def test_nonidentity_explicit_map_translates_hart_ids_to_pin_indices():
    p = plan_placement(
        board(), cpus=3, workers=2, worker_harts=(1, 0), controller_hart=2, cpu_hart_ids=(2, 0, 1), matrix=True
    )
    assert p.worker_cpus == (2, 1) and p.controller_cpu == 0
    assert p.mapping_basis == "caller_declared"


def test_existing_tuple_api_and_noncontiguous_capabilities():
    brd = SimpleNamespace(harts=4, matrix_harts=(1, 3), hart_ids_for=lambda _: (0, 1, 3))
    assert worker_placement(brd, cpus=4, workers=2, worker_harts=(3, 1), matrix=True) == (3, 1)


@pytest.mark.parametrize(
    "kw",
    [
        {"cpus": 0},
        {"cpus": True},
        {"cpus": 4},
        {"workers": 0},
        {"workers": 4},
        {"worker_harts": (0, 0)},
        {"worker_harts": (0,)},
        {"worker_harts": [0, 1]},
        {"worker_harts": (0, True)},
        {"worker_harts": (0, 2)},
        {"controller_hart": 1},
        {"controller_hart": 3},
        {"controller_hart": True},
        {"cpu_hart_ids": (0, 0, 2)},
        {"cpu_hart_ids": (0, 1)},
        {"cpu_hart_ids": (0, 1, 4)},
        {"cpu_hart_ids": [0, 1, 2]},
    ],
)
def test_invalid_or_incapable_placement_refuses(kw):
    args = dict(cpus=3, workers=2, worker_harts=(0, 1), controller_hart=2, matrix=True)
    args.update(kw)
    with pytest.raises(ValueError):
        plan_placement(board(), **args)


def test_default_harness_has_no_extra_controller():
    text = zm._main_c(0)
    assert "merlin_controller_thread" not in text
    assert "k_thread_cpu_pin(t, MERLIN_RVV_HART)" in text


def test_opt_in_controller_preserves_pool_worker_and_completion_boundaries():
    text = zm._main_c(0, n_harts=2, controller_cpu=2)
    assert "k_thread_cpu_pin(t, 2)" in text
    assert "k_thread_cpu_pin(t, 0)" in text
    assert text.index("merlin_omp_init(2)") < text.index("uint64_t c0 = rd_mcycle()")
    assert "k_sem_take(&merlin_done, K_FOREVER)" in text
    assert "k_sem_take(&merlin_controller_done, K_FOREVER)" in text


def test_explicit_worker_list_overrides_discovery_without_rewriting_default():
    paths = [Path("archive"), Path("runtime"), Path("abi"), Path("generated")]
    default = zm._cmakelists(*paths, omp=True, n_harts=2, hart_ids=(0, 1))
    selected = zm._cmakelists(*paths, omp=True, n_harts=2, hart_ids=(0, 1), explicit_placement=True)
    assert "MERLIN_OMP_VECTOR_POOL=1" in default
    assert "MERLIN_OMP_VECTOR_POOL=0" in selected
    assert "MERLIN_OMP_HART_IDS={0, 1}" in selected


@pytest.mark.parametrize("controller", [-1, 0, True])
def test_controller_fragment_rejects_invalid_os_cpu(controller):
    with pytest.raises(ValueError):
        controller_entry(0, controller, "return")


def test_controller_fragment_is_valid_c_before_worker_definition(tmp_path):
    import shutil
    import subprocess

    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("host C compiler unavailable")
    declarations, main = controller_entry(0, 2, "stop()")
    stub = r"""
#include <stddef.h>
struct k_thread { int x; }; struct k_sem { int x; }; struct cpu {int id;};
typedef void *k_tid_t;
#define K_THREAD_STACK_DEFINE(name,n) static char name[n]
#define K_PRIO_COOP(x) (x)
#define K_FOREVER (-1)
#define MERLIN_WORKER_STACK 65536
static char merlin_worker_stack[MERLIN_WORKER_STACK];
static struct k_thread merlin_worker_thread;
static struct k_sem merlin_done;
extern void k_sem_init(struct k_sem*,int,int);
extern void k_sem_take(struct k_sem*,int);
extern void k_sem_give(struct k_sem*);
extern k_tid_t k_thread_create(struct k_thread*,char*,size_t,void(*)(void*,void*,void*),void*,void*,void*,int,int,int);
extern int k_thread_cpu_pin(k_tid_t,int);
extern void k_thread_start(k_tid_t);
extern struct cpu *arch_curr_cpu(void);
extern int printk(const char*,...);
extern void stop(void);
"""
    path = tmp_path / "controller.c"
    path.write_text(stub + declarations + "\nstatic void merlin_worker(void*a,void*b,void*c){}\n" + main)
    subprocess.run(
        [compiler, "-std=c11", "-Werror=implicit-function-declaration", "-fsyntax-only", str(path)],
        check=True,
        capture_output=True,
    )


@pytest.mark.parametrize(
    "selection, message",
    [
        ({"worker_harts": (0, 1), "controller_hart": 1}, "outside the compute"),
        ({"worker_harts": (0, 1), "controller_hart": 2, "cpus": 2}, "online physical"),
        ({"worker_harts": (0, 2), "controller_hart": 1}, "incompatible"),
        ({"worker_harts": (0, 1), "controller_hart": 2, "cpu_hart_ids": (2, 0, 1)}, "identity"),
    ],
)
def test_model_builder_refuses_bad_mapping_before_toolchain_or_filesystem(monkeypatch, tmp_path, selection, message):
    from merlin.runtime import boards

    brd = SimpleNamespace(
        harts=3,
        flow="zephyr",
        zephyr_default_ram_bytes=1 << 20,
        zephyr_link_limit_bytes=1 << 20,
        matrix_harts=(0, 1),
        hart_ids_for=lambda _: (0, 1),
    )
    monkeypatch.setattr(boards, "board", lambda *args, **kwargs: brd)

    def forbidden():
        raise AssertionError("placement must refuse before toolchain discovery")

    monkeypatch.setattr(zm, "build_available", forbidden)
    work = tmp_path / "never-created"
    kwargs = {"board": "fixture", "cpus": 3, "n_harts": 2, **selection}
    with pytest.raises(zm.ZephyrModelError, match=message):
        zm.build_app(tmp_path / "not-read", work, **kwargs)
    assert not work.exists()
