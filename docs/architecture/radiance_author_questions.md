# Radiance/Gluon Maintainer Questions (HAL Bring-up)

This document tracks open questions needed to move the Merlin Radiance HAL from
scaffold to end-to-end execution.

Please answer inline (preferred) or link to the source-of-truth file/spec/PR.

## 1. Launch Packet ABI

1. What is the canonical launch payload layout expected by Gluon today?
2. Which fields are required vs optional (`_start`, kernel entry, grid/block,
   registers/thread, shared-memory bytes, TLS/stack, printf buffer)?
3. What are the field units and limits (bytes vs words, element counts, max
   values)?
4. Are there alignment requirements for payload buffers and packet headers?

## 2. Command Submission + Completion Contract

1. What is the stable packet header format for kernel launch, copy, fill, wait,
   and sync?
2. Are command IDs globally monotonic, per stream, or per queue?
3. What ordering is guaranteed across streams? What does a fence/sync drain?
4. What completion/error payload schema should HAL parse and surface?

## 3. Muon ELF / Code Object Contract

1. Which ELF relocation types must runtime support immediately?
2. Which sections/symbols are required for execution (`_start`, kernel symbol,
   constants, metadata)?
3. Where should runtime source entry metadata (regs/thread, shmem/block,
   occupancy hint)?
4. Is there a versioned executable format string we should advertise through
   `hal.executable.format`?

## 4. Occupancy and Launch Validation

1. What is the exact occupancy policy currently implemented in simulator/runtime
   for register pressure and shared-memory pressure?
2. Should HAL enforce occupancy limits strictly at submit-time, or defer to
   backend errors?
3. Are there kernel metadata overrides that must bypass default occupancy rules?

## 5. Memory Model and Addressing

1. Confirm current device pointer semantics: raw VRAM physical addresses (no
   GPU VA)?
2. What host-visible address ranges are valid/invalid for runtime operations?
3. Which address spaces must HAL distinguish in v1 (`global`, `shared`,
   others)?
4. Are there required cache/fence operations for host-visible copies?

## 6. RPC-Compat Bring-up Contract

1. Which `rad_rpc` operations are mandatory for HAL bring-up?
2. Which handshake/version checks should be enforced at connect time?
3. Is the RPC protocol stable enough to pin, or should HAL negotiate a protocol
   version?

## 7. Test Corpus and Golden Outputs

1. Which kernels should be considered required for first-pass bring-up CI?
   Suggested: `launch`, `address_space`, `vecaddx`.
2. Do you provide canonical stdout/printf output and/or memory-result goldens?
3. What simulator flags/environment are required for deterministic results?

## 8. Near-term Neutrino/Gemmini Scope

1. Should HAL v1 ignore Neutrino payloads entirely, or reserve payload fields
   now?
2. Is there a minimal accelerator invocation path that must be supported in the
   first release?

## Where HAL Needs Confirmed Answers

The most immediate code paths waiting for these answers are:

- `runtime/src/iree/hal/drivers/radiance/dispatch_builder.c`
- `runtime/src/iree/hal/drivers/radiance/executable.c`
- `runtime/src/iree/hal/drivers/radiance/submission.c`
- `runtime/src/iree/hal/drivers/radiance/transport/rpc_compat.c`
- `runtime/src/iree/hal/drivers/radiance/transport/direct_submit.c`
