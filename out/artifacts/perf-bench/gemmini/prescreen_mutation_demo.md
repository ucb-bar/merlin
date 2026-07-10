# Pre-screen iteration-saving — demonstrated by mutation vs real verilator (#142)

Closes the honest gap from the 383-run corroboration: the checks were proven *safe* (0 false-rejects) but
that corpus had no real RTL-tier codegen failures, so the "catches bugs before the RTL sim" value was
asserted, not shown. Here we inject realistic codegen-bug classes into a known-good kernel
(A2 single-tile 16×16×16 matmul) and run **both** the real RTL sim (verilator) and the RTL pre-screen on
each. Both derive from one artifact (`lowered.llvm.mlir`), so a single mutation propagates consistently to
the ELF (verilator) and the decoded trace (pre-screen). Mutants are **verified to actually fail verilator**,
so each "RTL failure" is real, not assumed. Script: `scripts/demo_prescreen_mutation.py`.

| mutant | verilator | sim time | pre-screen | pre-screen time | caught before RTL |
|---|---|---|---|---|---|
| original (unmutated) | **pass** (Y0==golden, 308 cyc) | 178.2 s | **ok** | 2.3 ms | — (0 false positive) |
| illegal_funct (4→99) | fail (verilator exit 255) | 110.1 s | reject | 3.2 ms | **YES** |
| drop_compute | fail (verilator exit 255) | 112.0 s | reject | 3.2 ms | **YES** |
| drop_mvout | fail (Y0≠golden, 256 mism.) | 165.5 s | reject | 3.7 ms | **YES** |
| swap_compute_operands (numerical) | fail (Y0≠golden, 256 mism.) | 198.1 s | **ok** | 2.4 ms | **MISS — by design** |

## Result

- **3 genuine RTL failures caught statically in ~3 ms each**, vs 110–165 s per verilator run — a ~50,000×
  faster verdict on the structural/encoding bug classes. Estimated **≈ 388 verilator-seconds saved** across
  the three (sum of their measured `sim_active_s`).
- **0 false-positives** on the unmutated original: verilator passes it (Y0==golden, 308 cyc) and the
  pre-screen says `ok`. Safe to act on a `reject` (skip the oracle).
- Caught by overlapping, RTL-grounded checks: `decode_funct_legal` (illegal funct), `movement_compute_balance`
  (drop_compute), `tile_coverage` (drop_mvout) + the TRACE FileCheck — not a single brittle signal.
- **The numerical mutation (`swap_compute_operands`) is honestly missed**: verilator fails it (256
  mismatches) but the static pre-screen says `ok`, because it is structurally legal — only the *values* are
  wrong. This is the documented limit (static checks don't verify numerics; that needs the oracle or the
  arcilator middle-tier). It corroborates the 9 real `functional_mismatch` FNs seen in the corpus study.

## Honesty notes
- Mutants are injected, not agent-produced — but the bug *classes* are real (observed in intermediate
  agentic drafts), and the RTL failure of each is *measured*, not assumed. This is standard mutation testing.
- verilator here runs the full-SoC bare-metal flow (boot dominates the 110–198 s; the matmul is ~308 cyc) —
  which is exactly why a millisecond static pre-screen on the structural bug classes is valuable, and why a
  faster RTL-faithful middle-tier (arcilator, #143) is the path for the numerical class.
