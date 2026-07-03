# Integrity policy (experiment ABI v0.1)

The benchmark is only fair if a package genuinely *implements a target backend* rather than
borrowing the harness's answers. These rules are enforced by the runner's **integrity scan** and
by the structure of the contract (subprocess + file boundary).

## Hard rules (a package that violates any of these FAILS, status never `pass`)

1. **No harness imports.** A package's tool sources must not import or load the Merlin runtime
   reference/simulator or any Merlin internal that computes the expected answer. Forbidden
   substrings in package sources (Python or C++): `merlin.runtime.reference`,
   `merlin.runtime.simulator`, `reference_outputs`, `import merlin`, `from merlin`.
   → `FailureCategory.FORBIDDEN_PATTERN`
2. **No reading the golden outputs.** The package must not read `expected_command_buffer_*.json`
   beyond `g0` (the one published example) or any `runs/**` directory. It computes the command
   buffer from the interface, it does not copy it.
   → `FailureCategory.TAINT`
3. **Real artifact class.** The package must emit a *command buffer* (the abstract program) AND a
   *lowered LLVM/RoCC* program. A package that only emits a C compute kernel, or target MLIR with
   no command buffer, is the wrong artifact class.
   → `FailureCategory.STRUCTURAL_INVARIANT_VIOLATION`
4. **Subprocess only.** The runner invokes the package solely through its CLI entrypoints. The
   package may not require in-process import by the harness.

## The one exemption

`manifest.integrity_exempt: true` is allowed **only** for the Merlin *reference backend*
(`gemmini_merlin_native`), which legitimately wraps Merlin's own certified path and is used to
migrate the existing battery through the contract. It is NOT a competitor entry. Every
agent-generated or independent package MUST set `integrity_exempt: false` and pass the scan.

## What is fair game

- Reading the contract bundle (`merlin/contract/**`) — grammar, schemas, the single `g0` example.
- Public ISA docs, the LLVM/MLIR headers, the standalone OOT template.
- Iterating against the oracle's pass/fail (the package author may run the runner) — but the
  package is finally scored on **held-out** inputs it has not seen.
