# Gemmini target setup

[`descriptor.yaml`](descriptor.yaml) declares experiment inputs and resource ownership.
[`contracts/target_contract.yaml`](contracts/target_contract.yaml) and
[`contracts/residual.yaml`](contracts/residual.yaml) are the authored prototype
capability inputs; [`evidence_concepts.yaml`](evidence_concepts.yaml) supplies the
target's evidence vocabulary. These are public source inputs, not generated
capsules, extracted RTL facts or a certified compiler. The checkout registry can
inspect this example as reference metadata; it does not load a backend from it.

[`probe_oracles.py`](probe_oracles.py) checks the configured Chipyard tools and prebuilt
simulator paths without building or running anything by default:

Runtime implementation lives in the OOT `gemmini-mlir` support package, not
Merlin's retained reference metadata. Select the provider explicitly:

```sh
export MERLIN_TARGET_PATH=/path/to/gemmini-mlir/merlin-support
```

Use the companion revision recorded in `build_tools/upstreams/target_support.json`.
Those local companion commits have not been pushed; an arbitrary upstream checkout
may not contain this support package yet. Its `backend/` and `build_support/` are
the single implementation owners. Its `tools/` owns reference-program corroboration,
and `cost_model/` owns the calibration driver, inputs, vocabulary and coefficients.
Shared Merlin resolves these through the selected provider; there are no in-tree copies.
Calibration executes configured compiler/simulator tools unless explicitly using the
driver's offline refit mode. Historical coefficients are screening estimates, not new
hardware measurements or certification.

The provider currently lacks a reviewed default RTL-facts pin and its contract
lacks `legality`; backend import or pure tests do not certify it. A provisioned
Chipyard elaboration and mlc/CIRCT source tree can produce an **explicit,
generated** facts artifact for the dynamic kernel-ABI harness check:

```sh
# Run from Merlin's checkout after selecting the OOT support provider above.
facts_dir="${MERLIN_OUT_ROOT:-$PWD/out}/artifacts/audits/rtl-facts/gemmini"
mkdir -p "$facts_dir"
.venv/bin/python -m merlin.targetgen.rtl.circt_introspect \
  --target gemmini --out "$facts_dir/facts.json" --validate
MERLIN_RTL_FACTS="$facts_dir/facts.json" \
  .venv/bin/python build_tools/scripts/check_kernel_abi_arg_order.py --verbose
```

The extractor records source, contract and implementation hashes in `facts.json`;
inspect and verify those commitments against the selected RTL revision before
using the result. The ABI gate compares the emitted harness's pointer order with
the shared command-shape contract. Its success does not review or install the
generated facts as the provider's default pin, repair `legality`, or certify
compiler correctness or hardware behavior. Target-policy tests likewise require
explicit support selection.

For the status-only tool check:

```sh
python examples/gemmini/target/probe_oracles.py
```

Set `MERLIN_CHIPYARD` to your own checkout. Optional overrides are
`MERLIN_GEMMINI_SPIKE`, `MERLIN_GEMMINI_VERILATOR`, `MERLIN_RISCV_GCC`, and
`MERLIN_GEMMINI_HARNESS_DIR`. The script's default `/path/to/chipyard` is a placeholder.

**`--run` executes a simulator** against an existing prebuilt test binary; it is not
a status-only check. Run only on explicitly provisioned, owned resources:

```sh
python examples/gemmini/target/probe_oracles.py --run spike --timeout 300
python examples/gemmini/target/probe_oracles.py --run verilator --timeout 300
```

Status reports file availability, not executable validity, revision provenance,
compiler correctness, or certification. Status-only mode returns zero even when tools
are missing. An explicitly requested unavailable oracle is not executed and contributes
exit status 2. An optional probe run reports the prebuilt binary's exit status, not a certificate for a generated
compiler. Preserve the normal provenance and numerical grading gates for experiments.
