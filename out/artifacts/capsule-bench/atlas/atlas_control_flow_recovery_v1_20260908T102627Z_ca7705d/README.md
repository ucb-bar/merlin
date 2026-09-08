# Atlas control-flow recovery package

This is the runnable source snapshot of the 49/61 fresh Atlas backend after repairing its
large-program control flow. It is a recovery candidate, **not a certified champion**.

## What failed

The backend resolved every relative branch as `(target_word - source_word) * 4`, the conventional
RISC-V byte displacement. Atlas does not use a byte-indexed scalar PC here. The target schedule
contract declares two decoded immediate units per emitted instruction: the RISC-V-shaped decoder
divides the immediate by two and the scalar PC is word-indexed.

On the diagnosed squareish GEMM, the JAL at word 3470 was intended to reach word 2797. The four-unit
encoding sent the core to word 2124 instead, exactly doubling the backward edge and producing an
infinite loop. The package now uses the contract's two-unit displacement for BNE, BEQ, and JAL. It
also enables the compact loop form at 16 tile products and removes the NOP-padding workaround that
had only kept conservative static checks in range.

## Evidence and limit

With the fix, `SY_geometry_squareish_gemm` emits 8,721 words and GSIM halts after 13,228,115 cycles.
That resolves the codegen nontermination defect. It does **not** make the capsule pass: the independent
golden still reports 24,919/50,176 mismatches (maximum absolute error 128). The output is materially
closer to a tile-32 BF16 accumulation model than to the capsule's per-MAC BF16 model, so the remaining
work is a datapath/accumulation-contract mismatch. Full measurements and hardware identities are in
`evidence/gsim_result.json`.

## Reproduce the regression guard

From this directory:

```bash
python test_control_flow.py
python -m pytest -q test_control_flow.py
```

The test imports this exact packaged backend, decodes its emitted BNE/BEQ/JAL immediates, and compares
them with the copied target schedule contract. Restoring the old `* 4` calculation makes both tests
fail; the squareish test additionally pins the observed 3470-to-2797 back edge.

The compiler entry point remains `submission/mlir_oot/atlas-opt`; its original package manifest is
preserved at `submission/manifest.yaml`.
