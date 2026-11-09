---
title: "Numerical provider evidence identity"
kind: reference
status: current
owner: core
last_verified: 2026-11-09
related: [ordered_fma_certificates, quantized_host_optimizations]
code_refs:
  - src/merlin/runtime/numeric_provider_identity.py
---

# Numerical provider evidence identity

`validate_numeric_provider_identity` validates the current authoritative native
numerical witness before a source-bound provider is installed. It does not
prove numerical equivalence and does not execute untrusted libraries.

The assembler supplies the actual selected library's queried workspace bytes
and minimum alignment, the current witness, and its compile manifest. Declared
alignment may be a stronger power of two. The validator checks library and
manifest hashes, matching compile-command representations, selected output,
compiler executable, compiler dependency file output and complete dependency
set, local/transitive pin agreement, and witness coverage of every dependency.
A stale baseline command, workspace size, library identity or dependency map
is refused. Historical recipes belong in a separately named provenance record.

Opaque witness hashes in generic dispatch contracts bind identity, not the
schema or internal consistency of arbitrary external proof formats. Assemblers
using this native-provider manifest format must invoke this validator before
constructing those contracts. A successful numerical run remains an observed
result even if its accompanying evidence seal fails; the seal must be repaired
in a fresh record without rewriting historical evidence.

The checker lives in `merlin.runtime.numeric_provider_identity`, alongside
runtime artifact admission. It consumes supplied query results and exact files;
loading the provider, establishing its numerical theorem, and collecting actual
execution evidence remain separate operations.
