# Actual two-epoch reuse → nested segmentation

The full modules and their exact hashes are in ../memory-scf-mode-v1/EXAMPLE.json
and README.md. The following excerpt is explanatory; it is not substituted for
those unabridged checked buffers. Two independently seeded2×3i32 initializers,
private loads/updates, caller-touch calls and untouched functions are present.

```mlir
// Original: independent logical allocations, each fully initialized.
%a = memref.alloc() : memref<2x3xi32>
scf.for %r = %zero to %two step %one {
  scf.for %c = %zero to %three step %one {
    memref.store %seed0, %a[%r, %c] : memref<2x3xi32>
  }
}
// loads, update, dimension, and caller-touch observation
memref.dealloc %a : memref<2x3xi32>
%b = memref.alloc() : memref<2x3xi32>
// Same rectangular initializer with seed1 and b, then observations/free.
```

Reuse removes the first free/second allocation and maps the entire second epoch
to `%a`. The actual next pass changes BOTH initializers. Each becomes:

```mlir
%stride = arith.constant 2 : index
scf.for %base = %zero to %two step %stride {
  %sum = arith.addi %base, %stride : index
  %end = arith.minui %sum, %two : index
  scf.for %r = %base to %end step %one {
    scf.for %c = %zero to %three step %one {
      memref.store %seed1, %a[%r, %c] : memref<2x3xi32>
    }
  }
}
```

The registry records original scalar/IV SSA IDs and per-operation keys; the old
reuse protocol records two logical epochs sharing one slot. Both changed loop
boundaries supply generated stride/add/min/inner/yield IDs. Lean checks their
control equalities, bounds, lexical non-shadowing, full rectangular store coverage,
live initialization/free protocol and whole-list composition. The full Python
matcher binds every unchanged operation/function and each source body clone to
that registry. That matcher/parser remains trusted, not formally verified.

Retained checker output is in CHECKER_OUTPUT.json. The positive records
returncode0 and lean_complete_composition_checked=true; independent review and
pipeline gating remain false. A compiler-valid negative changes ONLY the second
initializer's min upper operand to one. It is structurally decoded, then Lean
rejects its generated min binding: returncode1 and checked=false. Its exact full
mutant, exported trees, model, query and first diagnostics are retained.

The actual full instruction sequence also has a checked nonzero abstract witness
with public loads17,34,29,58. Separate finite xDSL tests bind real returned values,
caller effects and body traces. Neither establishes native primitive fidelity.
The panel gSIM figure is execution evidence separate from this theorem.
