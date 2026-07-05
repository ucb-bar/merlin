# AGENT.md — third_party

## Purpose

Pinned hard build/test dependencies that merlin itself cannot build or test without (e.g. a vendored LLVM/MLIR or lit, if and when needed).

## What belongs here

- True build-time dependencies, pinned for reproducibility.
- Small header-only libs or test utilities required by the build.

## What does not belong here

- External analysis repos (XNNPACK, Autocomp, Exo, Triton, ...) — those are integrations.
- Anything merlin can run without. Prefer a `pyproject.toml` dependency or env-var path.

## Carve-out: `baselines/`

`third_party/baselines/` is a **deliberate, documented exception** to the "hard build deps only /
integrations go via env-var" rule above. It holds external **baseline compilers/runtimes** (TVM,
ExecuTorch, Buddy, EXO, llama.cpp/ggml) that we compare merlin against by running the *same* models
end-to-end on the *same* SpacemiT K1 board with RVV. They are pinned as **submodules** (not env-var
integrations) specifically so the cross-framework comparison is reproducible — the exact framework
commit is part of the measurement. See `third_party/baselines/AGENT.md`. Build trees stay in
`build/`; results in `artifacts/compare/` + `artifacts/measurements/`.

## Interfaces

Consumed by the build system (`CMakeLists.txt`, `build_tools/`).

## Invariants

- Only hard build/test dependencies belong here.
- Do not add external analysis repos here by default.
- Prefer 'bring your own LLVM build' over vendoring at this stage.

## Testing expectations

Build must remain green; document any pin and its rationale.

## Notes for future agents

xDSL is a `pyproject.toml` optional dependency by default; only vendor it here if a pinned local copy becomes necessary for CI or local patches.
