# Configuration graph verification merged into Graham's branch

Target: `feat/pass-verification-dce-cse`, starting at e0484f488.
Imported primary: `feat/pass-verification` at 89ac040ae.
Merge commit: ccde4e252. Compatibility fix: beff617f2.
Original commit ancestry, proof sources and historical qualification remain intact.
The three implementation workers remain stopped; no new task was assigned.

## What changed

- Added the complete-entry Lean checker bridge for **redundant configuration
  writes**. It begins with unknown configuration, removes only already-redundant
  total pure writes, handles unknown/unordered effects and clobbers, and preserves
  modeled success/error/terminal-state observations for the supported modules.
- Added the partial-loop proof and actual Kernel checker for **loop-invariant
  configuration hoisting**: one constant pure setter moves before a positive
  static loop with an ordered body that preserves its fields. The proof includes
  failing body and surrounding calls. Zero-trip, nested, multi-prefix, overlapping
  writes and completion-token/asynchronous variants remain outside this subset.
- Added source/IR binding, fresh private proof builds, full root/type/axiom
  audits, live exact-byte consumer authority, replay and adversarial controls.
  Hoisting pins the e048 producer's 2,009 Python source blobs, checks used origins
  and complete typed Kernel roundtrips, and anchors proof paths to the installed
  verification package instead of caller environment settings.
- Included worker source/evidence archives, actual worked examples, counterexample
  and failure records, usage guides and stopped-state portable handoffs.
- Bound the newer `common/owned_children.py` process helper in elision receipts
  and added a regression showing that a changed helper invalidates the request.
  Hoist receipts already collect that helper by its actual imported origin.

The only merge conflicts were the generated docs index and onboarding text.
The index was regenerated; onboarding preserves both histories and the current
stop instruction. Frozen evaluator/specification/lock files were not changed.

## Combined-tree validation

250 focused tests passed. Scoped lint passed. Both families passed a fresh Lean
baseline check with exact-byte delivery and a separate-process fresh replay:

| Family | Roots per check | Transitive theorem audit | Violations |
| --- | --- | --- | --- |
| Configuration elision | 17 | 247 | 0 |
| Scoped loop hoisting | 15 | 315 | 0 |

Only the standard three axioms were present. Counts overlap and must not be
summed. These repeat one retained baseline per family; the historical full
campaigns (13 elision pairs and seven hoist pairs) keep their original sources,
environments and identities. No new full-campaign/new-host qualification is claimed.
See [the exact merged-tree result and reports](../../verification/merlin-passes/configuration-graph-integration-20261028/RESULT.json).

The structure guard reports 16 inherited failures: two optional-package import
violations and 14 module-size violations. Every offending source is byte-identical
to the target before this merge; evidence is in STRUCTURE-BASELINE.json beside
the result. No guard or expected source pin was weakened.

## Claim boundary and entry points

The existing opt-in compiler selector remains **DCE/CSE only**. The new
configuration APIs are direct developer validation paths. Their correctness
statements assume the documented common instruction/state-effect interpretation
and trusted IR projection. Native/hardware fidelity, independent human contract
approval, complete model coverage and adjacent-pass composition remain open.
Neither the complete hoisting algorithm nor all graph transformations are claimed.

- [Elision API guide](../../../../docs/guides/lean_configuration_elision_gate.md)
- [Hoist API guide](../../../../docs/guides/lean_configuration_hoist_gate.md)
- [Elision qualification](../../verification/merlin-passes/loop-scope-20261028/NEXT-CONFIGURATION/README.md)
- [Hoist result and three stopped handoffs](../../verification/merlin-passes/loop-scope-20261028/NEXT-HOIST/README.md)

The old statements that these results were not yet ported describe their original
primary checkpoints. This integration supersedes that transport status, without
relabeling old qualification. The detached e048 producer worktree remains read-only;
the merge was performed in a separate checkout.
