# WP-4a handoff: scalar arith/math, linalg.generic/fill/index, func.call, IEEE instance

Status: **not started in code.** Stopped on request (token budget) after reading the design doc,
every G0-frozen file and the Arith spec. `Sem/Ops/{Arith,Linalg,Func}.lean` are still the G0 stubs
(every op `stuck`). No helper modules or tests exist yet. Nothing is proved.

## Environment facts (verified)
- Build: `ln -s /home/jeremy/Projects/merlin/out/build/lean/packages <worktree>/out/build/lean/packages`,
  then `lake build` in `merlin/lean` (about 3 s, no Mathlib download). `globs = MerlinVerify.+`, so
  `MerlinVerify/Tests/*.lean` with `#guard` run during `lake build`.
- A fresh agent worktree may be cut from `main`. Run `git reset --hard origin/feat/pass-verification` first.

## Spec facts for the implementer (llvm-project `a47bddcce`, `ArithBase.td` / `ArithOps.td`)
- Exporter (WP-2) is not written, so attribute encodings follow MLIR generic form:
  - `predicate` (cmpi/cmpf) is an `I64EnumAttr`, so `Attr.int n _`.
    cmpi: eq0 ne1 slt2 sle3 sgt4 sge5 ult6 ule7 ugt8 uge9.
    cmpf: false0 oeq1 ogt2 oge3 olt4 ole5 one6 ord7 ueq8 ugt9 uge10 ult11 ule12 une13 uno14 true15.
  - `roundingmode` is an `I32EnumAttr` (`Attr.int`): to_nearest_even0 downward1 upward2
    toward_zero3 to_nearest_away4. It is optional on addf/subf/mulf/divf/truncf/convertf.
  - `fastmath` is `Attr.enum "arith" "fastmath" s`, with flags separated by `","`. The values are
    none, reassoc, nnan, ninf, nsz, arcp, contract, afn, plus the group `fast` (all seven). Default
    is none. Canonicalise to a bitmask before building `FloatOpKey.modifiers`.
  - `overflowFlags` is `Attr.enum "arith" "overflow" s`, separated by `", "`, with values none, nsw,
    nuw. Used by addi/subi/muli/shli/trunci.
  - `isExact` is a unit attr on divui/divsi/shrui/shrsi. `nonNeg` is a unit attr on extui (and
    on uitofp and index_castui; check them).
- UB versus poison, per the spec text:
  - **Immediate UB:** divui/remui/ceildivui on division by zero. divsi/ceildivsi/floordivsi on
    division by zero or `INT_MIN / -1`. remsi only on division by zero.
  - **Poison:** `exact` on a non-multiple or nonzero shifted-out bits. A shift amount ≥ width.
    shli nuw/nsw (LLVM shl rules). trunci nuw (dropped bits nonzero) and nsw (dropped bits differ
    from the result's top bit). extui `nneg` on a negative input. A poison select condition.
  - **Ambiguity 1:** `floordivsi` is declared `Pure`, but its description says UB. Follow the
    description.
  - **Ambiguity 2:** for remsi `INT_MIN % -1` the spec says nothing, so it is defined (0). LLVM
    `srem` makes it UB. Record this in the ledger.
- Float semantics:
  - maximumf/minimumf propagate NaN and order -0 < +0.
  - maxnumf/minnumf return the non-NaN operand; on ±0 they may return either.
  - remf takes the sign of the dividend.
- Ops not in the original brief:
  - flush_denormals, convertf, scaling_extf/truncf: leave `stuck`.
  - addui_extended, subui_extended, mulsi_extended, mului_extended: these have two results.
  - ceildivui.

## Exact next steps
1. `Sem/Support.lean`:
   - attribute decoders (above);
   - `Ty` to scalar width (`index` uses `ctx.indexBits`);
   - dense decoding, little-endian, storage width `alignTo(w, 8)`, i1 one byte each, `index` 8
     bytes, splat and non-splat, byte count must match or `stuck`;
   - elementwise lifting over equal static shapes.

   Push it first; WP-4b depends on it.
2. `Sem/Ops/Arith.lean`:
   - integer ops on `BitVec`;
   - float ops via `ctx.M` with key `(op, "fastmath=<mask>;rm=<n>;pred=<n>")`;
   - `math.*` as uninterpreted `M.op`;
   - `supported?`.
3. `Linalg.genericSem` (keep the frozen signature and `IterKind`):
   - split ins/outs by `operandSegmentSizes`;
   - each loop bound comes from an operand dim whose map result is `dim d`, otherwise `stuck`;
   - lexicographic iteration with `loopIndex` set;
   - also `linalg.fill` and `linalg.index`.
4. `Func.sem`: `func.call` takes `callee` (`symbolRef`) and calls `call`.
5. `Sem/IEEE.lean`:
   - executable `FloatModel` (Float32, Float, bf16/f16 on bits with RNE);
   - its own `roundeven`;
   - fptosi/fptoui return `none` for NaN or out of range;
   - `exactOps`.
6. `Tests/{Arith,Linalg,Func,IEEE}.lean` `#guard` tests; then progress lemmas; then `evalArr` (stretch).

## Interface change requests
None identified yet.
