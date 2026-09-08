# Pure build support

Target-owned C caller formatting and storage-word representation only. No candidate
imports, backend discovery, golden/reference execution, simulator, tool discovery or
ambient hardware policy. Generic build services load this package by its exact pinned
path. Legacy backend wrappers inject their existing layout/measurement callbacks.

Do not duplicate harness semantics here: the legacy caller delegates to this one
renderer, with generated-byte parity tests. Explicit storage preserves host prepack
authorization checks and does not authorize arithmetic or change tensor values.
