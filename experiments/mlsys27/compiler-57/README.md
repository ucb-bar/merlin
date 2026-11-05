# Gemmini compiler: 57/57 device ops proven

`submission/` is an authored Gemmini OOT compiler package. Under the release grader and the frozen 193-overload harness
(`../h193/`), it proves all 57 device core-ATen ops.

**Verdict** (`evidence/m27b-S1div/summary.md`, stage m27b, evaluated 2026-11-05):

    device 57/57 proven (weak lowering 57, full definition 57, corpus-clean 57); host 0/136 proven
    regrade: core 318/321; scalability 111/111

"Proven" for a device op means both of the following:
- the 193-overload harness case passes the FULL definition: it lowers on the device with RoCC instructions retired and
  exact output, and passes the numerical, host_compute (grader census and audit) and trace checks;
- every graded corpus capsule credited to the op passes the release grader.

The corpus regrade is native Spike L0-L2, run by `tooling/eval/eval_candidate.sh` in lorenshung/aws. Models, L3 RTL and
FireSim were not run for this table. The 3 non-passing core capsules are credited to no device op:
- 2 expected-refusal capsules (MUST_REFUSE_VIOLATED);
- 1 whole-model capsule that was not run.

The host-136 column is outside this claim.

## Provenance

The package comes from one unseeded Codex Phase 1 run followed by seeded continuations, plus one targeted port.

1. clean3 was an unseeded Codex run on fin6w3c. It reached 23/57.
2. seed5, seedB and cx-B2 were seeded continuations on fin6w3c, each starting from the previous final package.
3. S1 was a seeded Codex Phase 1 run on stage m27, branch tip 69aa427c5:
   - run p12-g-m27-codex-seedcx-B2-20261105T000424Z;
   - gpt-6.1-sol, effort high, 6 subagents, 5 h wall clock;
   - the general author brief was passed as operator errata.

   Its final submission scores 56/57 (`evidence/m27-S1/`). The only failing op is div.Scalar.
4. ff6f921 ports the exact quantized device lowering of div.Scalar onto S1. That lowering was developed by a Codex fixer
   on the sibling fin-B2 lineage. The port adds `div_scalar.py` and touches the hooks in `backend.py` and `frontend.py`:
   253 insertions, nothing else changed (`evidence/div-port-REPORT.md`). After the port, div.Scalar is bit exact,
   offloaded, with 0 host ops per element.

The final evaluation ran on stage m27b. m27b is a fresh, unsealed, diagnostic release prepare from branch commit
e70dea82e, which embeds the author brief. Its graded corpus is byte-identical to m27's: 321 public core capsules and 111
public scalability capsules.

`submission.MANIFEST.sha256` pins the package bytes. The candidate package-id is 8df75aad7284f0ba.

## Reproduce

Point `tooling/eval/eval_candidate.sh` (lorenshung/aws) at `submission/`, with the fin6w2-m27b corpus template and
MODE=full. Step 2 of that script runs `../h193/run_193.sh` with FULL=1.

## Open

An unseeded confirmation run on m27b was started on 2026-11-05: run p12-g-m27b-codex-clean-*, with
`--fresh-backend`, a 10 h wall clock and the brief as errata. It tests whether the result reproduces from an empty
start. Its outcome is not part of this claim.
