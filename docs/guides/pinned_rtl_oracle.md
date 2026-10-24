---
title: Pin a Chipyard RTL oracle
kind: guide
status: current
owner: targetgen
last_verified: 2026-10-14
related: [target_resolution]
code_refs:
  - src/merlin/targetgen/rtl_simulator_pin.py
  - src/merlin/targetgen/rtl_simulator_execution.py
  - src/merlin/targetgen/rtl_simulator_slot.py
  - src/merlin/targetgen/oracle_policy.py
  - merlin/contract/hardware_pins.yaml
---

# Pin a Chipyard RTL oracle

Select an explicit support provider with `MERLIN_TARGET_PATH` and its exact
fact bundle with `MERLIN_RTL_FACTS`. Set `MERLIN_EXT_CHIPYARD` to the checkout
that contains the generated RTL and simulator. A support backend may separately
resolve its checkout through `MERLIN_CHIPYARD`; configure both consistently.

Declare a simulator pin in `merlin/contract/hardware_pins.yaml`. Use full source
commits, source paths verified by content, and `build_products` hashes for both
the executable and its FIRRTL. Cover separately versioned generator and ABI
header repositories with their own pins. Retain a build reproduction receipt
connecting the generated RTL to the executable. Pinning observed source files
alone does not prove that those sources produced an existing elaboration.

Set `MERLIN_REQUIRED_RTL_ENGINE=verilator` and `MERLIN_RTL_SIM_PIN` to the pin's
name. Engine selection then verifies the backend's resolved executable against
that pin. The selected facts must bind a verified FIRRTL-to-HW source bundle,
match the runtime configuration, and read the pin's exact FIRRTL product.
Missing identities and changed products refuse selection.

Use the ordinary `oracle_adapters(target, sim_via="chipyard")` inventory when
constructing a grade. Explicitly supplying only an L2 adapter omits L3 even if
a verified RTL simulator is available. Capsule requirements remain authoritative:
a missing required adapter makes the result incomplete.

The direct Verilator adapter holds one of four per-user advisory slots under
`/tmp/merlin_verilator_slots_<uid>` and includes slot wait in its wall budget.
All participating grades share these slots. Unrelated native simulator launches
do not participate. Do not nest reservations or launch concurrent children in
one reservation. Runtime checks reopen the pin before and after execution and
retain its provenance in the oracle result.

Calibrate timeouts with an observed successful execution of the selected binary.
The target timing reader checks the target, runtime configuration and executable
SHA-256; copy the resulting record into the selected experiment resources as
`.oracle_timing.<target>.json` only when preparing that experiment. Preserve the
command, ELF hash, wall time, host load and slot concurrency with the observation.
A small-program timing is an extrapolation basis; layer and model workloads need
their own measurements before claiming a complete-suite runtime.
