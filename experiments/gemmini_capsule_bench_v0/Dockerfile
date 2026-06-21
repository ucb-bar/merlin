# Optional container image for the agent arm of gemmini_capsule_bench_v0.
# The PRIMARY isolation path is bwrap (scripts/run_agent_experiment.py --sandbox bwrap), which needs
# no image. This Dockerfile is provided for environments that prefer container isolation; the
# RISC-V toolchain + spike/verilator live on the host under /scratch2/chipyard and are bind-mounted
# read-only at run time (grading happens on the host, outside the agent container).
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash git ca-certificates curl python3 python3-venv build-essential ninja-build cmake \
    && rm -rf /var/lib/apt/lists/*

# The claude CLI is mounted from the host (~/.local/bin/claude) at run time, along with ~/.claude.
# LLVM/MLIR 23 (third_party/llvm-install) and the RISC-V toolchain are bind-mounted read-only.

WORKDIR /workspace
# Agent sees only the assembled workspace (allowed bundle paths) mounted here; denied paths are not
# mounted. The entrypoint runs the headless agent against TASK.md.
ENTRYPOINT ["/bin/bash", "-lc", "claude --print --model ${MODEL:-opus} --effort ${EFFORT:-high} \
  --permission-mode bypassPermissions --output-format stream-json --verbose < TASK.md"]
