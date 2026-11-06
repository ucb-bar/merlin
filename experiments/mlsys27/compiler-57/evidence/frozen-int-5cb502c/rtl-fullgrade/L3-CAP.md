# Two capsules that exceed the grader's 10M-cycle L3 cap

The release-grader-equivalent grade (`per-capsule.csv`) ran all 320 non-model core capsules, including the mandatory
L3 Verilator RTL tier. 318 pass outright. The two others pass L0-L2 and fail only at L3, with the simulator stopped by
the grader's fixed hang bound:

    *** FAILED *** (timeout) after 10000001 simulation cycles      (+max-cycles=10000000)

| capsule | credited to (tier 1) | grader L3 at 10M cycles | uncapped RTL run of the same ELF | grader harness with an EMPTY kernel at 10M |
|---|---|---|---|---|
| SY_conv_operand_capacity_boundary | convolution | timeout, 0 of 229 output chunks printed | completes; all 231 `OUT_B64` lines identical to Spike L2 | timeout, 0 output chunks printed |
| SY_multi_block_beyond | mm | timeout, 32 of 55 output chunks printed | completes; all 57 `OUT_B64` lines identical to Spike L2 | timeout, 13 output chunks printed |

The empty-kernel ELF links the grader's own generated harness, CRT, HTIF and libc objects with a `kernel_entry` that
does nothing. It runs under the grader's simulator binary and arguments (`+permissive +dramsim ... +max-cycles=10000000`).
It cannot finish either capsule in 10M cycles. The cycles go to loading the ELF's inputs and printing the output
tensor one character at a time over HTIF, so no compiler can pass these two at L3 under that cap. No earlier Phase 1
run passed them at L3 either: every run's self-check stopped them on the wall-clock budget.

Decision (Loren Hung, 2026-11-06): count both as L3-verified on this evidence. The uncapped RTL output is bit-identical
to the Spike output that the grader passed at L2.

Files in `l3-cap/`:
- `<capsule>.capsule_result.json` and `.grader-L3-10M.console-tail.txt`: the grader's L3 result and console.
- `<capsule>.spike-L2.console.txt`: the passing Spike L2 console.
- `<capsule>.uncapped-rtl.console.txt`: the same ELF on the same simulator binary with `+max-cycles=300000000`.
- `<capsule>.empty-kernel-10M.console.txt`: the empty-kernel harness at the grader's 10M cap.
