# Why golden wins on model shapes and generated wins on a single tile — a utilization mechanism analysis

*Mechanism half of task #131. Data: `runs/perf_full_0001/perf_results.json`, `runs/perf_full_0001/firesim_arm_results.json`. Companion: `reports/perf_comparison.md`.*

## Thesis

The cross-approach result is a **utilization** story, not an instruction-count story. On the cycle-accurate RTL (verilator L3 + FireSim L5), the hand-tuned golden C library (`tiled_matmul_auto`) runs the 16×16 systolic array at ~52–56% utilization on model/attention shapes, while the generated MLIR backends (baseline-gen v0, merlin-gen v1, and the native RoCC emitter — all of which emit the same lean per-tile RoCC sequence) sit at ~14–15% on the same shapes. They issue *correct* RoCC, but their tiling and data-movement schedule does not keep the array fed, so it stalls. The single 16×16×16 tile (G00) inverts this: there is nothing to tile, so golden's fixed setup/loop overhead becomes pure dead weight (559 cyc) while the generated lean sequence wins (308 cyc). The crossover sits between "one tile, overhead dominates" and "many tiles, scheduling dominates." Static instruction counts cannot see this — they do not capture loop trip-counts or the stall cycles that a cycle-accurate model exposes.

Two further framing points the data forces:
- **Performance is RTL-only.** spike is functional; it does not model the systolic array, so its cycle counts are not timing (see §5). Every number in the crossover argument below is verilator (L3) or FireSim (L5).
- **Capability is a separate axis.** Only merlin-gen (v1) compiles conv2d + movement among the generated backends. That is a coverage differentiator, not part of the perf mechanism, and is kept distinct (§6).

## 2. The crossover, in cycles and utilization (cycle-accurate RTL only)

Utilization = MACs / (cycles × 256), since the Gemmini array is 16×16 = 256 MACs/cycle at peak. All cells below are verilator (`L3`) or FireSim (`L5`) — the two simulate the same RTL and are directly comparable; verilator covers ≤~32K-MAC kernels, FireSim covers the 64³+/model/attention shapes. The "generated" column is merlin-gen (v1); on the verilator shapes baseline-gen and merlin-native are bit-identical to it (same emitted RoCC), and golden is the hand-tuned reference.

| kernel | shape | MACs | sim | golden cyc (util) | generated cyc (util) | winner | golden/gen cyc ratio |
|---|---|---|---|---|---|---|---|
| **G00_single_tile** | 16×16×16 | 4,096 | L3 | 559 (2.86%) | **308 (5.19%)** | **generated** | 1.82× (gen faster) |
| G02_rect | 32×64×16 | 32,768 | L3 | 1137 (11.26%) | **919 (13.93%)** | **generated** | 1.24× (gen faster) |
| G03_kaccum | 16×128×16 | 32,768 | L3 | 1342 (9.54%) | **983 (13.02%)** | **generated** | 1.37× (gen faster) |
| G04_wideN | 16×16×128 | 32,768 | L3 | 1599 (8.01%) | **1269 (10.09%)** | **generated** | 1.26× (gen faster) |
| G05_tallM | 128×16×16 | 32,768 | L3 | 1159 (11.04%) | **1102 (11.62%)** | **generated** | 1.05× (gen faster) |
| G01_multitile_sq | 64×64×64 | 262,144 | L5 | **4843 (21.14%)** | 7439 (13.77%) | **golden** | 0.65× (golden faster) |
| G06_acc_scale_i8 | 64×64×64 | 262,144 | L5 | **3091 (33.13%)** | 7082 (14.46%) | **golden** | 0.44× |
| G07_relu_i8 | 64×64×64 | 262,144 | L5 | **3091 (33.13%)** | 7082 (14.46%) | **golden** | 0.44× |
| K_attn_qk | 64×64×64 | 262,144 | L5 | **4847 (21.13%)** | 7437 (13.77%) | **golden** | 0.65× |
| K_attn_pv | 64×64×64 | 262,144 | L5 | **4847 (21.13%)** | 7437 (13.77%) | **golden** | 0.65× |
| M00_smolvla | 16×32×960 | 491,520 | L5 | **5851 (32.81%)** | 18176 (10.56%) | **golden** | 0.32× |
| M03_openvla | 32×256×128 | 1,048,576 | L5 | **7582 (54.02%)** | 28061 (14.60%) | **golden** | 0.27× |
| M04_openvla | 32×128×256 | 1,048,576 | L5 | **7826 (52.34%)** | 27894 (14.68%) | **golden** | 0.28× |
| M01_smolvla | 64×720×32 | 1,474,560 | L5 | **10218 (56.37%)** | 41081 (14.02%) | **golden** | 0.25× |

The transition is sharp and monotone in problem size:

- **1 tile (G00, 4K MACs):** generated is 1.82× faster than golden.
- **Single-tile-in-one-dimension rectangles (G02–G05, 32K MACs):** generated still wins, but the margin collapses to 1.05–1.37× — golden's relative overhead is already shrinking as there is more real work to amortize it against.
- **Multi-tile and model shapes (64³ and up, ≥262K MACs):** golden wins decisively, 1.5×–4×, and its utilization advantage widens with size — from ~21% vs 14% at 64³ up to **56% vs 14% at the largest model shape (M01)**.

The single number that captures the mechanism: on the four largest model shapes (M00, M01, M03, M04) golden holds **52–56% util** (M00's 33% is the outlier, a thin 16×32×960 shape where even golden can't fill the array on the short M=16 edge), while generated is pinned at **10.6–14.7%** regardless of shape. Generated utilization is essentially flat across all the big shapes; golden's rises with size. That flatness is the fingerprint of a fixed per-tile stall, not a shape-dependent one.

## 3. The mechanism — why utilization differs, and why the single tile flips

**Why golden keeps the array busy.** The hand-tuned C library schedules tiling and data movement so that the systolic array rarely waits on operands: it overlaps mvin of the next tile's operands with the compute of the current tile, sizes tiles to the scratchpad/accumulator, and reuses stationary weights across the moving dimension so a loaded weight tile feeds many output rows before it is evicted. The array stays fed; utilization climbs toward the structural ceiling as the problem grows and the fixed pipeline fill/drain is amortized (G00→M01: 2.86% → 56.37%).

**Why the generated backends stall.** The generated lowering emits a correct per-tile RoCC sequence — preload, compute, mvin/mvout — but with no overlap and poor reuse: each tile's compute waits on its own mvin, weights are re-moved rather than held stationary across the moving dimension, and there is no software pipelining across tiles. The array therefore idles for a roughly fixed fraction of every tile, which is exactly why generated utilization is pinned near 14% independent of shape (13.77% at 64³, 14.02–14.68% at the million-MAC model shapes). More tiles just means more copies of the same per-tile stall — the inefficiency is constant per tile, so it never amortizes away. The util ceiling is set by the schedule, not the shape.

**Why the single tile flips it (G00).** With exactly one 16×16×16 tile there is no tiling to schedule, no second tile to overlap with, and no reuse to exploit — every advantage golden has on big shapes is inapplicable. What remains is overhead: golden carries the fixed setup/loop-nest machinery of a general tiled routine (`tiled_matmul_auto`'s configuration, address generation, and loop preamble) regardless of how small the problem is, costing 559 cycles for 4096 MACs (2.86% util). The generated path emits a lean, direct RoCC sequence with none of that generality, completing the same tile in 308 cycles (5.19% util). On one tile, leanness beats a tiling strategy that has nothing to tile. As soon as there is more than one tile to coordinate (G01 onward), the tiling strategy is what matters and golden pulls ahead.

So the crossover is the meeting point of two opposite-signed effects: golden's **fixed overhead** (hurts most when work is tiny) versus generated's **per-tile stall** (hurts proportionally more as work grows). Below ~32K MACs the overhead term dominates and generated wins; above it the stall term dominates and golden wins.

## 4. What this implies for the generated compiler

The gap is **not** closed by emitting fewer or cheaper instructions. The generated path already issues a correct, lean RoCC stream and even *beats* golden on a single tile — instruction-level economy is not the deficit. The deficit is **schedule quality**, specifically:

1. **Tile scheduling / software pipelining.** Overlap the next tile's mvin with the current tile's compute so the array is not idle during operand load. This is the dominant lever: it is what turns a ~14% per-tile-serial schedule into golden's ~50%+.
2. **Data reuse / operand stationarity.** Hold weight (or activation) tiles stationary across the moving dimension instead of re-moving them per output tile, cutting the data-movement traffic that the array stalls behind.
3. **Tile-size/loop-order selection** matched to scratchpad and accumulator capacity, so reuse and overlap are actually achievable rather than spilled.

These are loop-nest/data-movement transforms, not new RoCC opcodes. The right capability to add to the generated backend is a **movement-aware tiling/scheduling pass** that reasons about array occupancy across tiles — precisely the thing the hand-written C library encodes by hand and the generated lowering currently omits. The flat ~14% utilization is the budget this pass would recover; closing to golden's ~50% on model shapes is a ~3.5–4× speedup ceiling on the very shapes (M00–M04, attention) that dominate real inference.

## 5. Methodology note — RTL, not spike

Every performance claim here is from cycle-accurate RTL: verilator (L3) for ≤~32K-MAC kernels, FireSim (L5, Alveo U250) for the 64³+/model/attention shapes verilator cannot reach in its time budget. Both simulate the same Gemmini RTL, so L3 and L5 cells are directly comparable.

**spike is functional-only and must never be cited as timing.** spike models the RoCC matmul as retiring in ~0 cycles — it does not simulate the systolic array — so its cycle counts reflect only scalar issue overhead and plateau (~118–121 cycles) from 4K MACs all the way to 2M MACs. Run through the util formula this yields impossible numbers: golden's spike line on M01 is 121 cycles → **4760% "utilization"**, and on G08 (128³) 619 cycles → **1323%**. Those are artifacts of a model that does not count compute cycles, not performance. The spike column exists only for the correctness/coverage axis (§6). The iree_dialect arm (the deprecated IREE merlin at `/path/to/merlin-iree`, distinct from this merlin project) currently runs spike-only and therefore has **no** cycle-accurate column — its spike PASS confirms functional correctness of the hand-written dialect, but contributes nothing to this perf comparison.

## 6. Separate axis: capability (coverage)

Distinct from the perf mechanism above, the spike correctness table records a coverage difference: among the generated backends, **only merlin-gen (v1) compiles conv2d and movement**. Baseline-gen (v0) and the native RoCC emitter fail to lower those ops (`merlin_iface.conv2d`/`movement` unknown, or `expected RES_PACK + matmuls==commits` codegen errors); golden's conv2d C template is not wired. All four matmul-capable arms handle matmul and attention correctly. This is a real differentiator for merlin-gen, but it is an op-coverage property, orthogonal to the utilization gap that governs matmul/attention performance — a backend can compile an op and still schedule it at 14% util, which is exactly merlin-gen's situation on the matmul shapes.

---

### One-line takeaway

Golden beats generated on model shapes because its hand-tuned tiling + data-movement schedule holds the 16×16 array at ~52–56% utilization while the generated backends' un-overlapped, low-reuse per-tile schedule pins them at a shape-independent ~14%; generated beats golden on a single tile only because there is no tiling to do and golden's fixed overhead has nothing to amortize against — so the fix is a movement-aware tiling/pipelining pass, not fewer instructions, and the evidence is cycle-accurate RTL utilization, which static instruction counts and functional spike both miss.
