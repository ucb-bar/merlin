# Next tools — proposals grounded in P19 source evidence (Phase 5)

Propose-only (no code this pass). Each tool is justified by a concrete source/MLIR finding from the audit,
not by a wish. "Already exists" notes prevent rebuilding. No perf claims; structural/requirements only.

## Tool A — mapspace_seed_extractor (Timeloop-style)  ·  difficulty: medium  ·  level: MLIR
**Evidence:** every recovered op carries M/N/K (+ batch fold) with operand identities (`prov.fqn` →
activation vs weight; attention = both-activation). The 4-column audits show loop dims + operand roles are
fully present in the flat MLIR. **Build:** per op emit `{equation, loops:[M,N,K(,B)], operands:{A,B,C
identities}, legal_spatial/temporal axes, bypass + dataflow(WS/OS/IS) candidates}`. **Captures beyond P16:**
turns the shape table into Timeloop problem-shapes + a dataflow candidate space. **Caveat:** K-reduction
spatial legality needs the partial-sum note already in the boundary catalog. (This is the Phase-3 seed —
see mapspace_seed_candidates.yaml; the *tool* would generate it in-pipeline.)

## Tool B — operand_locality_analyzer (CADOSys-style)  ·  difficulty: medium  ·  level: MLIR+sidecar
**Evidence:** per-operand bytes are recovered (lhs/rhs/output); reuse scope is recoverable for within-op and
across-ops, and *across-K/decode/replan* is derivable from the (assumed) K. rdt2 showed loop-carried latent
+ KV-as-inputs as the real residency target. **Build:** per (region, op, operand) → bytes + reuse scope
{within_op/across_ops/across_K/across_decode/across_replan} + cache-vs-scratchpad candidate. **Beyond P16:**
makes residency a per-operand locality map, not just a per-region resident-weight number. **Caveat:**
across-K scopes inherit the configured-K caveat.

## Tool C — capture_fidelity_probe  ·  difficulty: low  ·  ALREADY EXISTS (extend)
**Evidence:** P18 `capture_fidelity` + P19 `capture_erasure_evidence` + `dump_exported_fx.py` already produce
the source→export→MLIR ladder and the erasure proof. **Do NOT rebuild.** Extend only to: (1) ingest the
`dump_exported_fx` ATen histogram automatically (so the export column is pipeline-generated, not manual),
(2) emit the per-feature absent-vs-unparsed verdict as a committed matrix.

## Tool D — attention_kv_accounting  ·  difficulty: low  ·  MOSTLY EXISTS (fixed in P19)
**Evidence:** P18 recovery + the P19 attention-classifier fix already recover attention MACs (incl. the
SDPA-fused `prov.op=sdpa` form and the batched-MLP-bmm separation). **Remaining gap to build:** KV-cache
*state* sizing across the decode loop (seq-growth) — needs a loop/decode-preserving capture or a sidecar
with seq/heads/head_dim. Until then KV capacity stays `blocked` (honestly). So this tool is ~80% done; the
KV-state part is capture-blocked, not analyzer-blocked.

## Tool E — quant_metadata_capture  ·  difficulty: medium  ·  level: capture (m2m)
**Evidence:** bitvla source HAS real int2 BitLinear packing + absmean scales, but the captured branch is
fake-quant→f32; the **qdq-level recapture already restores `quant_ext.dequantize`** (P18-B capture-level
ablation). **Build:** a parser that, on the qdq/high-level recaptures, extracts storage dtype / compute
dtype / accumulator / scale granularity / packed layout / dequant placement into a committed
quant_metadata_visibility table. **Beyond P16:** unblocks the 7 low-bit abstractions for the workloads that
have a qdq capture. **Caveat:** only as faithful as the qdq recapture (torchao int8, not the model's native
ternary — note the gap).

## Tool F — requirements_envelope  ·  difficulty: low  ·  ALREADY EXISTS (P17)
P17 `timing_requirement_envelope` already crosses base facts with deadline/K/H/control-rate grids
(requirements, not measured). **Do NOT rebuild.** Extend only to consume Tool B's per-operand locality and
Tool D's attention MACs so the envelope covers attention + locality, not just linear-GEMM + residency.

## Priority
1. **Tool A (mapspace seeds)** — the biggest new DSE value (turns facts into a Timeloop search space); source
   evidence is fully present. 2. **Tool E (quant metadata)** — unblocks low-bit, qdq capture already exists.
3. **Tool B (operand locality)** — sharpens residency. C/D/F are extensions of existing P17/P18 tools, not
   new builds. The capture-side frontier (loop/KV-preserving, native-ternary) is the limiting factor for
   D-KV and E-native — a m2m/frontend project, not an analysis tool.
