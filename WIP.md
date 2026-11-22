# WIP: Merlin integration snapshot (wip/atlas-nov22)

A work-in-progress snapshot (e563d58ce) of the Merlin side of the Atlas integration stack: the declared SoC and memory budget, host f32 math and source families, placement and staging, residency with the counted region loop, and dispatch. It sits on `refactor/repository-layout` 2451b8134 and already includes the fixes from https://github.com/ucb-bar/merlin/pull/66. A cleaned-up history with nearly the same final tree will replace it on https://github.com/reed-nicolas/merlin/tree/feat/quantized-model-deployment (https://github.com/ucb-bar/merlin/pull/65), and the final push rebases onto PR 66 itself.

atlas-mlir `wip/atlas-nov22` pins this exact commit by archive URL in its pyproject.toml. Merlin-side ideas from the Atlas WIP.md (such as cost-based plan selection) land here, and each one needs a matching Atlas pin update: push the Merlin commit to your own branch first, then point the Atlas pin at that commit.

Never rewrite or force-push this branch. Put new work in commits on your own branch off `wip/atlas-nov22`, so it moves over later with `git rebase --onto feat/quantized-model-deployment wip/atlas-nov22 <branch>`.

## Setup and tests (from CONTRIBUTING.md)

```sh
git clone -b wip/atlas-nov22 https://github.com/reed-nicolas/merlin.git && cd merlin
uv venv --python 3.12
uv pip install -e '.[dev,verify,semantic-search,targetgen]'
.venv/bin/python build_tools/hooks/install_git_hooks.py
.venv/bin/python build_tools/checks/check_structure.py
.venv/bin/python build_tools/checks/check_artifact_layout.py
.venv/bin/python build_tools/checks/check_docs.py
.venv/bin/python build_tools/checks/check_test_budget.py --output out/build/python/ci/test-budget.json
.venv/bin/python -m pytest tests -q -ra
```

`semantic-search` pins z3-solver 4.16.0.0; keep it, because placement counts Z3 work and 5.x changes results. The suite is capped at 150 collected cases. Some workflows need private inputs, compilers, or hardware: report the passing subset and the missing prerequisites, and never make fixtures or weaken checks to get a green run. The native bridge needs Cargo.

Focused files for the integration work: `tests/runtime/test_dispatch_c.py`, `test_dispatch_program_regions.py`, `test_application_shared_loop.py`, `test_compact_parameters.py`, `test_dispatch_loop_c.py`, `test_region_loop.py`, `tests/infra/test_memory_budget.py`, `test_selected_control.py`, `tests/targetgen/test_application_residency.py`, and `test_residency.py`. Changes here must also pass the Atlas checks in its WIP.md with the updated pin.

## Rules

Models reach Atlas only through atlas-torchao FX to model2MLIR to Merlin to atlas-mlir, with no hand-written Atlas code. Runtime and cross-region loops belong in Merlin; in-layer loops belong in Atlas's `compile_region`. Correctness means bit-exact against npu_model; never loosen a check or gate. Follow AGENTS.md: no target names, model names, regex, or hardcoded hardware constants in library code; derive target facts from the target. Subject-only conventional commits, each small and reviewable.
