<!-- Versioned RVV kernel-mining prompt (v1). The agent reads a kernel DOSSIER (pre-extracted
facts + framework contract + code), NOT raw files cold. It does only the high-value judgment the
deterministic tooling can't: algorithm intent, whether the kernel is exemplary, caveats, and any
framework-contract refinement or compiler-lever implication. The prompt itself is a tuned artifact
— new versions are evaluated as candidates (see kernels/agent_mine.compare_modes). -->
You are mining an expert RISC-V Vector (RVV) kernel to improve a COMPILER's RVV codegen (not to
copy the kernel). You are given a structured dossier; the mechanical facts are already extracted —
focus on judgment.

## Dossier
- source / op / dtype: {source} / {op} / {dtype}
- RVV decisions (measured): {decisions}
- structure (AST): {struct}
- classified motifs: {motifs}
- framework contract (caller-side assumptions): {framework_contract}
- code:
```c
{code}
```

## Answer ONLY with a JSON object, these keys:
- "algorithm": one-sentence description of what the kernel computes and HOW (blocking/dataflow).
- "is_exemplary": true|false — is this a kernel whose codegen decisions our compiler should emulate
  for this (op, dtype)? 
- "why": one line justifying is_exemplary.
- "contract_refinements": list of corrections/additions to the framework contract you can infer
  from the code (e.g. a prepack/transpose/layout detail the descriptor missed); [] if none.
- "compiler_levers": list of concrete RVV codegen levers this kernel implies our compiler should
  adopt (e.g. "fuse mul+add into vfmacc.vf", "use e32m4 LMUL", "register-block MR=4",
  "vsetvl-loop tail"); map to schedule knobs / lowering patterns where possible.
- "caveats": list of reasons this kernel might NOT generalize (framework-specific assumptions,
  shape constraints); [] if none.
Output JSON only, no prose.
