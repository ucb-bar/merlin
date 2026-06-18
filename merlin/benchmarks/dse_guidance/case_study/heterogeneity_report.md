# Heterogeneity report — one bigger / many identical / specialized

> Evidence comparing three resource-multiplicity search spaces. **Evidence and implications only — no architecture is selected, no speedup claimed.**

## one_bigger_unit

**Evidence for:**
- low inter-op parallelism (avg 1.291×, 9/11 workloads near-sequential) — a single large unit is not starved by inter-op concurrency
- some workloads are dominated by one op

**Evidence against:**
- 5 distinct geometry classes coexist (gemv_like, projection_like, squareish_gemm, unknown, wide_skinny)
- GEMV/decode shapes (skinny family) are a poor match for a square matrix unit
- phases run at different cadences (backbone vs head vs control) — pipelineable

## multiple_identical_units

**Evidence for:**
- 2 workload(s) expose some inter-op parallelism
- 1797 (op,axis) M/N shards split with no tail — reduction-free replication

**Blocked by:**
- reduction/partial-sum cost for K-sharding is unknown (not measured)
- memory bandwidth is unknown — replicas may contend for weight reload
- data dependencies serialize work (avg parallelism only 1.291×)

## multiple_specialized_units

**Evidence for:**
- distinct operator families coexist: dense GEMM 35% of MACs vs skinny/GEMV 64%
- epilogue/requant appears on 667 ops
- DMA/memory can overlap compute (resident loop-invariant weights)
- backbone and head run at different rates (multi-rate contract)
- the control loop decouples from replan inference

**Candidate units:** `matrix_engine`, `vector_gemv_engine`, `epilogue_requant_unit`, `dma_engine`, `loop_controller`, `scalar_control_unit`, `kv_cache_unit`

## Search-space implication

the evidence structurally suggests a HETEROGENEOUS (specialized) resource search space: distinct operator families (dense GEMM + skinny/GEMV), a frequent epilogue, resident-weight DMA, and multi-rate phases all coexist, while low inter-op parallelism argues against many identical units kept busy by concurrency. A future DSE should explore specialized units; this is an evidence-based search-space implication, NOT an architecture selection.

**Caveat (structural, not realized):** this is an evidence-based search-space implication. **No speedup**, throughput, cycle, or area is claimed, and no design is chosen; the missing measurements that block a quantitative decision (reduction cost, memory bandwidth, per-unit throughput, timing) are named in the option evidence.
