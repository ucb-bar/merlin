# Actual historical per-channel i32 fix replay

Passed on 2026-10-23 UTC. The complete actual passes_xdsl.py from the parent of
3d281b41453bd1ffc02629c5b7bc453bb748a964 and from that fixed commit were separately
loaded and executed. No scalar body was substituted or reconstructed. Each transformed
one identical signed-i8/i32-zero-point/f32 per-channel source operation.

The authoritative MLIR exporter accepted both outputs; source and both targets pass
source-support checking. The existing C1 whole-function AND complete-module checkers
refuse the historical output and accept the fixed output. Thus refusal is distinguished
from an unsupported-source case. The extracted four proof roots, including DequantSite,
C1, module and support soundness, use only standard axioms; exact statements are retained.
The ordinary checker's conditional verdict is not a new human contract approval.

Native execution of the actual historical output returns cb7fffff (-16777215).
Native execution of the actual fixed output returns cb800000 (-16777216), agreeing
with pinned PyTorch2.10.0+cpu/Numpy2.4.6 for byte1, zero point16777217 and scale1.
The oracle additionally returns the input byte identity; that auxiliary result is
explicit in the full retained response, separate from the one-result native fixture.
Native compiler/pipeline/bindings/source identities, exact inputs, commands, loaded
Python source references and all original/fixed source bytes are retained.

This uses the CURRENT recorded xDSL/MLIR dependency environment with historical pass
source; it does not reconstruct the dependency environment at the historical commit.
The new frontend candidate's frozen public helper and axiom auditor were read-only.
Original production sources, helper sources and checker/auditor identities are unchanged.
The first v1 attempt is retained as inconclusive because of a collector output-arity
assertion. An auxiliary statement-parser error in v2 is also explicitly retained;
its successful auditor stdout was parsed correctly without repeating that extraction.

This case is the already-fixed i32/F6 case. It does not resolve the still-open i64/F12
counterexample or qualify the newer per-tensor common-model API, which rejects per-channel
inputs. This is one independently replayed semantic fix in a registered file-level
candidate population, not an overall historical detection rate or full-pass claim.
Source-contract review, final capture, native generalization and combined qualification
remain separate. Sessions53745 and80117 are terminal/exit0; no worker capsule was rebuilt.
