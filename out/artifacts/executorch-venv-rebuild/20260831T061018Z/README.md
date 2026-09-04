# ExecuTorch exporter identity rebuild

Status: **promoted; exact top-level exporter/runtime identity gate passes**.

This run rebuilt the paper exporter environment without exporting a model or contacting the K1.
The prior environment remains recoverable.

## Result

- Runtime source: `/scratch/agustin/projects/oscar-merlin/third_party/baselines/executorch`
- Runtime source commit: `7fc34bf6f53d2098e3e16c1fa71c23222f607330`
- Active exporter venv: `/scratch/agustin/projects/oscar-merlin/out/build/baselines/executorch/et-venv`
- Preserved old venv: `/scratch/agustin/projects/oscar-merlin/out/build/baselines/executorch/et-venv.backup-20260831T061018Z-4af91c3`
- Old exporter identity: `1.4.0a0+4af91c3`, `4af91c3d6a4c16c5b0e5745620d7b3d208ba6928`
- New exporter identity: `1.4.0a0+7fc34bf`, `7fc34bf6f53d2098e3e16c1fa71c23222f607330`
- `require_matching_executorch`: pass, `matches=true`
- `et_identity_error()`: empty
- `et_venv_available()`: true
- Candidate focused tests: 35 passed in 0.34 s
- Active-path focused tests: 35 passed in 0.29 s

## Package identity

- Python: 3.12.13
- PyTorch: `2.12.0+cpu` (the version pinned by this ExecuTorch source revision)
- torchvision: `0.27.0+cpu`
- torchao: `0.17.0+git02105d46c`; source submodule `02105d46c61dc80a8c9d39d5836e827ba3af8439`
- transformers: `5.0.0rc1`
- lerobot: `0.6.0`
- NumPy: `2.2.6`
- fsspec: `2025.3.0`
- built wheel SHA-256 reported by pip: `7e6e91b134b954f50433331e73b4996c0b608127f59acf920779768bf5fc52e0`
- installed `executorch/version.py` SHA-256: `3a5f95b7db06ff58acdaf18df23d89fed6d293cd2924119411bfad00a912bd32`
- installed `_portable_lib` SHA-256: `69b1818c09f529e94bfee172b416bb0bf2a520402ce13e942402688f358df9cf`
- installed ExecuTorch `RECORD` SHA-256: `f56641fe5d27cba689e1b35a007102b7e81f525bc55d4df447e409117d564fb7`

`15_candidate_freeze.log` is the full package freeze. `16_candidate_packages.log` records the
important package metadata. `17_candidate_file_hashes.log` records installed file hashes.

## Build and timing

The successful source build used a fresh `pip-out`, CPU PyTorch, and explicitly disabled unrelated
CUDA, QNN, Vulkan, and OpenVINO wheel components. XNNPACK remained enabled and its partitioner import
was checked after installation.

Successful source install command:

```text
env CMAKE_BUILD_PARALLEL_LEVEL=16 \
  CMAKE_ARGS='-DEXECUTORCH_BUILD_CUDA=OFF -DEXECUTORCH_BUILD_QNN=OFF -DEXECUTORCH_BUILD_VULKAN=OFF -DEXECUTORCH_BUILD_OPENVINO=OFF' \
  <candidate>/bin/python -m pip install --force-reinstall --no-deps --no-build-isolation -v \
  ./third_party/baselines/executorch
```

- Full wheel build/install: 6m27.05s wall, 1,454,528 KiB peak RSS
- CPU PyTorch/torchvision install: 52.12s wall
- loader-dependency restoration: 4.10s wall
- full venv copy: 29.27s wall
- promotion: same-filesystem renames; effectively instantaneous

Every attempt's exact command, stdout/stderr, exit status, and `/usr/bin/time -v` output is retained
in numbered logs. The failed reflink clone and failed CUDA builds are retained too.

## Failures and remediation

1. Reflink cloning is unsupported on this filesystem; a normal recoverable copy was used.
2. The old environment contained a CUDA 13.0 PyTorch wheel while the host toolkit is CUDA 12.6.
   CMake therefore could not configure the pybind build. The source revision's pinned
   `torch==2.12.0` CPU wheel and matching `torchvision==0.27.0` CPU wheel were installed in the
   candidate. The active environment was not touched until validation passed.
3. `torchaudio==2.12.0+cpu` is not published in the configured test CPU index; the existing
   `torchaudio==2.11.0+cpu` remains. No paper model in the active set uses audio.

## Known compatibility warning

`lerobot==0.6.0` declares `torch<2.12.0` and `torchvision<0.27.0`, while this ExecuTorch checkout
pins `torch==2.12.0` and `torchvision==0.27.0`. Imports of lerobot, transformers, torchao, the
XNNPACK partitioner, and ExecuTorch all pass, but `pip check` correctly reports those two metadata
conflicts. This rebuild did not weaken the ExecuTorch identity gate and did not run a held-out
SmolVLA export; the conflict must remain visible until that separate package/export qualification.

## Recursive source note

The top-level ExecuTorch source is at the exact required commit, but the shared checkout already had
two submodule pointer differences:

- `backends/vulkan/third-party/Vulkan-Headers`: expected `8864cdc8...`, present `10739e8e...`
- `third-party/pybind11`: expected `d03662f0...`, present `a2e59f0e...`

Vulkan was disabled for this exporter build. The full state is recorded in `14_source_identity.log`.
The existing identity gate intentionally checks the full top-level commit only, so paper provenance
should also retain this run artifact if recursive-source reproduction is required.

## Scope

No `.pte`/`.bpte` package was produced, no held-out model was loaded or exported, and no K1 command
or measurement was run.
