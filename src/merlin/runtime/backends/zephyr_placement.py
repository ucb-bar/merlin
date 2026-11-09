"""Render the existing pinned cooperative controller/worker startup protocol."""

from __future__ import annotations


def controller_entry(worker_cpu: int, controller_cpu: int | None, halt: str) -> tuple[str, str]:
    """Return declarations and main; no vector/tile use occurs in the controller.

    The worker/pool remains one nonreentrant team. Callback and termination code
    are authored by the surrounding harness, never selected from target names.
    """
    start = f"""  k_sem_init(&merlin_done, 0, 1);
  k_tid_t t = k_thread_create(&merlin_worker_thread, merlin_worker_stack,
                              MERLIN_WORKER_STACK, merlin_worker, NULL, NULL, NULL,
                              K_PRIO_COOP(0), 0, K_FOREVER);
  int rc = k_thread_cpu_pin(t, {worker_cpu});
  if (rc != 0) {{
    printk("FAIL k_thread_cpu_pin hart %d rc=%d\\n", {worker_cpu}, rc);
    {halt};
    return 1;
  }}
  k_thread_start(t);
  k_sem_take(&merlin_done, K_FOREVER);
"""
    # Keep the legacy spelling as well as behavior in the unselected template.
    if controller_cpu is None:
        start = start.replace(f"k_thread_cpu_pin(t, {worker_cpu})", "k_thread_cpu_pin(t, MERLIN_RVV_HART)")
        start = start.replace(f'\\n", {worker_cpu}, rc)', '\\n", MERLIN_RVV_HART, rc)')
        return "", "int main(void) {\n" + start + f"  {halt};\n  return 0;\n}}\n"
    if (
        type(worker_cpu) is not int
        or type(controller_cpu) is not int
        or min(worker_cpu, controller_cpu) < 0
        or worker_cpu == controller_cpu
    ):
        raise ValueError("distinct nonnegative worker/controller OS CPUs required")
    declarations = f"""static void merlin_worker(void *, void *, void *);
K_THREAD_STACK_DEFINE(merlin_controller_stack, 65536);
static struct k_thread merlin_controller_thread;
static struct k_sem merlin_controller_done;
static int merlin_controller_error;
static int merlin_start_worker(void) {{
{start}  return 0;
}}
static void merlin_controller(void *a, void *b, void *c) {{
  (void)a; (void)b; (void)c;
  printk("METRIC controller_cpu %d\\n", arch_curr_cpu()->id);
  merlin_controller_error = merlin_start_worker();
  k_sem_give(&merlin_controller_done);
}}
"""
    main = f"""int main(void) {{
  k_sem_init(&merlin_controller_done, 0, 1);
  k_tid_t t = k_thread_create(&merlin_controller_thread, merlin_controller_stack,
                            65536, merlin_controller, NULL, NULL, NULL,
                            K_PRIO_COOP(0), 0, K_FOREVER);
  int rc = k_thread_cpu_pin(t, {controller_cpu});
  if (rc) {{ printk("FAIL controller pin cpu %d rc=%d\\n", {controller_cpu}, rc); {halt}; return 1; }}
  k_thread_start(t);
  k_sem_take(&merlin_controller_done, K_FOREVER);
  if (merlin_controller_error) {{ {halt}; return 1; }}
  {halt};
  return 0;
}}
"""
    return declarations, main
