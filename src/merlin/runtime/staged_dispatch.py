"""Explicit prepare/controller and compute-only roles for one synchronous stage.

This renders a separate opt-in harness. It neither segments a monolithic model
nor establishes target context, coherence, vector completion or ISA safety.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class StagedRoles:
    cpu_hart_ids: tuple[int, ...]
    model_init_hart: int
    pack_hart: int
    dispatch_hart: int
    finish_hart: int
    compute_harts: tuple[int, ...]
    profile_harts: tuple[int, ...]

    def __post_init__(self):
        mapping = self.cpu_hart_ids
        if (
            not isinstance(mapping, tuple)
            or not mapping
            or any(type(h) is not int or h < 0 for h in mapping)
            or len(set(mapping)) != len(mapping)
        ):
            raise ValueError("explicit distinct CPU-to-hart mapping required")
        controller = self.dispatch_hart
        if any(
            type(h) is not int or h not in mapping or h != controller
            for h in (self.model_init_hart, self.pack_hart, self.dispatch_hart, self.finish_hart)
        ):
            raise ValueError("initialization, packing, dispatch and finish require one online controller")
        if (
            not isinstance(self.compute_harts, tuple)
            or not 1 <= len(self.compute_harts) <= 8
            or len(set(self.compute_harts)) != len(self.compute_harts)
            or any(type(h) is not int or h not in mapping or h == controller for h in self.compute_harts)
        ):
            raise ValueError("one to eight distinct online compute harts must exclude the controller")
        if self.profile_harts != self.compute_harts:
            raise ValueError("only the declared compute harts may be profiled")

    def validate_compute_capabilities(self, allowed_harts: tuple[int, ...]) -> None:
        """Compare with caller-selected board/provider capabilities, never guess them."""
        if (
            not isinstance(allowed_harts, tuple)
            or any(type(h) is not int or h < 0 for h in allowed_harts)
            or any(h not in allowed_harts for h in self.compute_harts)
        ):
            raise ValueError("compute hart lacks the caller-declared capability")

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "worker_cpus": [self.cpu_hart_ids.index(h) for h in self.compute_harts],
            "controller_cpu": self.cpu_hart_ids.index(self.dispatch_hart),
            "mapping_basis": "caller_declared_requires_target_bootstrap_verification",
        }


@dataclass(frozen=True)
class StagedExecutionContract:
    shared_memory_coherence: str
    synchronous_compute_completion: str
    pinned_worker_context: str
    preserved_interrupt_context: str
    single_owner_storage_lifetime: str

    def __post_init__(self):
        if any(not isinstance(value, str) or not value.strip() for value in asdict(self).values()):
            raise ValueError("all target execution obligations require explicit caller evidence descriptions")

    def to_dict(self) -> dict:
        return {"status": "caller_declared_not_independently_proved", **asdict(self)}


@dataclass(frozen=True)
class StagedBindings:
    context: str
    model_init: str
    prepare: str
    compute: str
    finish: str
    current_hart: str
    global_ticks: str
    worker_cycles: str
    worker_instret: str
    completed: str
    relax: str | None = None

    def __post_init__(self):
        for name, value in asdict(self).items():
            if value is None and name == "relax":
                continue
            if (
                not isinstance(value, str)
                or not value
                or not (value[0].isalpha() or value[0] == "_")
                or not all(c.isascii() and (c.isalnum() or c == "_") for c in value)
            ):
                raise ValueError("stage bindings must be C identifiers")


def render_zephyr_staged(
    roles: StagedRoles,
    contract: StagedExecutionContract,
    bindings: StagedBindings,
    *,
    compute_capable_harts: tuple[int, ...],
    sequence: int,
    max_wait_polls: int,
    stack_bytes: int = 65536,
) -> tuple[str, dict]:
    """Render pinned startup around the portable C runtime; return explicit metadata.

    Caller supplies target hooks/callbacks, compilation ISA separation, terminal
    console policy and numerical gate. `completed` runs on the controller and
    may publish worker observations; the controller reads no profiling counters.
    No ordinary model build uses this API unless explicitly selected.
    """
    roles.validate_compute_capabilities(compute_capable_harts)
    if (
        type(sequence) is not int
        or not 0 < sequence < 1 << 64
        or type(max_wait_polls) is not int
        or not 0 < max_wait_polls < 1 << 64
        or type(stack_bytes) is not int
        or stack_bytes < 1024
    ):
        raise ValueError("positive bounded sequence, poll budget and worker stack required")
    b = bindings
    workers = len(roles.compute_harts)
    worker_cpus = [roles.cpu_hart_ids.index(h) for h in roles.compute_harts]
    controller_cpu = roles.cpu_hart_ids.index(roles.dispatch_hart)
    relax_decl = f"extern void {b.relax}(void *);" if b.relax else ""
    source = f"""/* Generated explicit staged dispatch; callbacks contain the selected ABI. */
#include <zephyr/kernel.h>
#include <stdint.h>
#include "merlin_staged.h"
extern void *{b.context};
extern int {b.model_init}(void *);
extern int {b.prepare}(void *);
extern int {b.compute}(void *, unsigned);
extern int {b.finish}(void *);
extern unsigned {b.current_hart}(void *);
extern uint64_t {b.global_ticks}(void *);
extern uint64_t {b.worker_cycles}(void *);
extern uint64_t {b.worker_instret}(void *);
extern void {b.completed}(const merlin_staged_report *);
{relax_decl}
static merlin_staged_team stage_team;
static struct k_thread stage_workers[{workers}], stage_controller_thread;
K_THREAD_STACK_ARRAY_DEFINE(stage_worker_stacks, {workers}, {stack_bytes});
K_THREAD_STACK_DEFINE(stage_controller_stack, {stack_bytes});
static struct k_sem stage_worker_exits, stage_controller_exit;
static int stage_status;
static merlin_staged_report stage_published_report;
static void stage_worker(void *slot, void *unused_b, void *unused_c) {{
  (void)unused_b; (void)unused_c;
  (void)merlin_staged_worker(&stage_team, (unsigned)(uintptr_t)slot);
  k_sem_give(&stage_worker_exits);
}}
static void stage_controller(void *a, void *b, void *c) {{
  (void)a; (void)b; (void)c;
  const merlin_staged_config config = {{
    .controller_hart = {roles.dispatch_hart}, .worker_count = {workers},
    .worker_harts = {{{", ".join(map(str, roles.compute_harts))}}},
    .sequence = UINT64_C({sequence}), .max_wait_polls = UINT64_C({max_wait_polls})
  }};
  const merlin_staged_hooks hooks = {{{b.current_hart}, {b.global_ticks},
    {b.worker_cycles}, {b.worker_instret}, {b.relax or "NULL"}}};
  const merlin_staged_callbacks callbacks = {{{b.model_init}, {b.prepare},
    {b.compute}, {b.finish}}};
  stage_status = merlin_staged_team_init(&stage_team, &config, &hooks, &callbacks, {b.context});
  if (stage_status) goto complete;
  k_sem_init(&stage_worker_exits, 0, {workers});
  const int cpus[{workers}] = {{{", ".join(map(str, worker_cpus))}}};
  for (unsigned i = 0; i < {workers}; ++i) {{
    k_tid_t thread = k_thread_create(&stage_workers[i], stage_worker_stacks[i],
      K_THREAD_STACK_SIZEOF(stage_worker_stacks[i]), stage_worker,
      (void *)(uintptr_t)i, NULL, NULL, K_PRIO_COOP(0), 0, K_FOREVER);
    stage_status = k_thread_cpu_pin(thread, cpus[i]);
    if (stage_status) goto complete;
  }}
  /* Start only after every pin succeeds; failed pins never dispatch callbacks. */
  for (unsigned i = 0; i < {workers}; ++i) k_thread_start(&stage_workers[i]);
  stage_status = merlin_staged_controller(&stage_team);
  if (stage_team.report.quiescent)
    for (unsigned i = 0; i < {workers}; ++i) k_sem_take(&stage_worker_exits, K_FOREVER);
complete:
  if (stage_status && !stage_team.report.status) stage_team.report.status = stage_status;
  /* Never copy observations while an unacknowledged worker may write them. */
  if (stage_team.report.quiescent) stage_published_report = stage_team.report;
  else {{
    stage_published_report.status = stage_status;
    stage_published_report.sequence = UINT64_C({sequence});
    stage_published_report.global_clock_hart = {roles.compute_harts[0]};
  }}
  {b.completed}(&stage_published_report);
  k_sem_give(&stage_controller_exit);
}}
int main(void) {{
  k_sem_init(&stage_controller_exit, 0, 1);
  k_tid_t thread = k_thread_create(&stage_controller_thread, stage_controller_stack,
    K_THREAD_STACK_SIZEOF(stage_controller_stack), stage_controller,
    NULL, NULL, NULL, K_PRIO_COOP(0), 0, K_FOREVER);
  stage_status = k_thread_cpu_pin(thread, {controller_cpu});
  if (stage_status) return stage_status;
  k_thread_start(thread);
  k_sem_take(&stage_controller_exit, K_FOREVER);
  return stage_status;
}}
"""
    return source, {
        "schema": "merlin.staged_dispatch.v1",
        "roles": roles.to_dict(),
        "contract": contract.to_dict(),
        "bindings": asdict(bindings),
        "sequence": sequence,
        "max_wait_polls": max_wait_polls,
        "stack_bytes_each": stack_bytes,
        "global_interval": "compute-leader global clock before controller init/prepare through final-finish ACK",
        "worker_interval": "local cycle/retirement reads around synchronous compute callback only",
        "controller_counter_reads": False,
        "single_inflight_stage": True,
        "reentrant_team": False,
        "full_model_segmentation_proved": False,
        "hardware_qualified": False,
        "physical_busy_qualified": False,
        "requires": ["merlin_staged.h", "merlin_staged.c", "caller-owned target hooks and callbacks"],
    }
