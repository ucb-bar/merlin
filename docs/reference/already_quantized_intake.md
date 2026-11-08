---
title: Already-quantized source intake
kind: reference
status: current
owner: core
last_verified: 2026-11-07
code_refs: [src/merlin/semantic_compiler/registered_source.py, src/merlin/targetgen/_pt2_capture_worker.py, src/merlin/targetgen/_m2m_capture_worker.py]
---

# Already-quantized source intake

`translate_linalg_text` accepts an explicit `source_operation_translators` registry from the selected provider. Registered callbacks must preserve source operands, result types, and order. Unregistered source operations are refused. Numeric semantics, immutable contraction intake, target storage encodings and legality, and arithmetic qualification belong to the provider. Reusable packing algorithms remain generic compiler infrastructure.

The exported-program capture worker accepts `--already-quantized` and repeatable `--register-module` options. Registration runs before deserialization. This path converts the saved export once, retains lifted identities and guards, and writes the same conversion's frontend trace and bundle. Read-only buffers are declared from the export's mutation signature and become immutable compiler parameters; mutable buffers retain their state role. A dtype label on the loader worker cannot cast an already-quantized graph, and numerical rewrite options are refused.

The loader worker accepts an external quantization adapter with exact contract and policy paths only with `--already-quantized`. It passes the preapplied external configuration to generic capture and invokes the loader's normalization hook before bundle publication. Application compilation and replay require the source-bound precision policy to agree with the selected provider. Intake byte identity does not establish model quality or device agreement.
