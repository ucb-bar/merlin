# Independent rotary reference v3

`whole_reference.py` computes rotary preparation in its original-ATen
interpreter. `rotary_source.py` reads the dependency slice of a complete captured
source FX trace, scalar arguments, tensor metadata, and unchanged lifted
inverse-frequency words. The K=1 product uses exact integer binary32 RNE;
transpose and concatenation preserve words. `rotary_exact.py` computes sine and
cosine without libm or framework trigonometric kernels, preserves the source
binary32 scale of one, and applies integer BF16 RNE.

The proof uses exact `Fraction` alternating-arctangent enclosures and Machin's
identity for pi, outward-rounded 160-bit dyadic intervals, and 28 Taylor terms
after the seed. Range reduction propagates the pi interval, a derivative bound
of one encloses input uncertainty, and a factorial Lagrange remainder encloses
the truncated series. Both interval endpoints must round to the same binary32
word; ambiguous rounding refuses rather than selecting a precision heuristic.

Qualification covers positions 0..2047 and all 32 captured inverse frequencies:
65,536 angles and 262,144 duplicated sine/cosine BF16 words. The independent
reference agrees exactly with the selected CORE-MATH archive at binary32 and
BF16 and with torch at BF16. Binary32 framework equivalence is not claimed;
positions beyond 2047 are unqualified. This is host source arithmetic, with no
Atlas execution claim. Chipyard newlib was rejected because its sine changes
one BF16 word at position 1897.

## Reference entry points

The package includes the original-ATen whole-model interpreter, bounded rotary
source replay, independent arithmetic, and retained head reference history.
Use `validate_rotary_v3.py` with an explicit complete source trace and a passing
archive gate. The validator writes authenticated intermediate and output words.
`validate_rotary_sequence_v3.py` rejects rotary fixture substitution.
Set `ROTARY_SOURCE_TRACE` when running source-slice tests; they skip without it.
A reference check does not establish whole-model compilation or execution.
