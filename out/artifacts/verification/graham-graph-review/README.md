# Independent graph verification review

The latest execution results are in
[the a26 reproduction report](execution-a26-20261026-v1/README.md).
Three workers reviewed the changed admission sources, selected a fresh official
Lean 4.35.0 SDK, and attempted five fixed proof profiles on 2026-10-26 Pacific
(2026-10-27 UTC). All workers finished with no live jobs.

The tested compiler/source baseline is **c62f81a8a11347ff53b60101fa75280e69d002a6**
on `feat/pass-verification`. This review branch retains its own history from
**5a7f872db33d378d6b0c4a294e4099e80efd4956**; it does not merge the later compiler
source commits. The original reviews below concern verification source
**58dbc7c00acdb6770d440b773cc2954328c78472** and producer **4499eebb9**.

| Work | Current result |
| --- | --- |
| Changed source/admission review | Ten changed Python files reviewed; 32/32 independent controls pass on Python 3.12/xDSL 0.68.0. The original 81-test replay is unavailable. |
| Fresh SDK | Official archive digest and members verified; eight diagnostics pass. Old runtime holds remain. |
| Q schedule | Seven modules and all three exact queries pass; two changed cases and one identity. |
| Q inline | Twenty-three modules pass; the empty query passes; four nonempty queries and all five witness files fail. |
| Q tail | Stops at the last support module's arity proof; four dependent queries are unexecuted. |
| W caller-context and entries | Both stop at the shared renaming proof; their 26 dependent queries are unexecuted. |

Four of 38 selected case queries compiled and passed exact root audits; four
failed and 30 were unexecuted. The 76 requested final/support root-audit
occurrences use only `propext`, `Classical.choice`, and `Quot.sound`. These are
conditional source/query results, not four verified passes. Fresh semantic
qualification, human approval, native fidelity and ordinary compiler enforcement
remain incomplete.

## Deliverable updates

- [Coverage and claim update](deliverable-1/UPDATE-20261026.md), supplementing
  [the original coverage review](deliverable-1/coverage-review.md) and
  [gap list](deliverable-1/gaps.md).
- [IR correspondence update](deliverable-2/UPDATE-20261026.md), supplementing
  [the original mapping](deliverable-2/ir-map.md) and
  [omissions/bugs](deliverable-2/omissions-bugs.md).
- [Assumptions and proof-strength update](deliverable-3/UPDATE-20261026.md),
  supplementing [the original assumption review](deliverable-3/assumption-review.md).
- [Fresh reproduction and exact blockers](execution-a26-20261026-v1/README.md),
  [case roster](execution-a26-20261026-v1/agent-3/CASE_ROSTER.csv), and
  [machine-readable combined findings](execution-a26-20261026-v1/SUMMARY.json).

The original manifest-bound review files remain unchanged. Updates have their
own identity and do not transfer historical qualification to later candidates.
See [the publication boundary](execution-a26-20261026-v1/PUBLICATION.md) before
attempting relocation or reproduction. SDKs and compiled build products are not
part of this publication.
