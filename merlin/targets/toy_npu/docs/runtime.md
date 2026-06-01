# ToyNPU Runtime (reference target)

Documentation only.

- Persistent handles for resident tensors.
- A single command queue with explicit submit/wait.
- Profiling regions around resident matmul.

Maps onto the `merlin.runtime` dialect concepts (command buffers, dispatches, queues,
persistent handles, waits, profiling regions).
