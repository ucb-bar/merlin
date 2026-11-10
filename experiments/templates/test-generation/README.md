# Shared Phase 0 performance template

[`performance.yaml`](performance.yaml) is the authored, target-independent definition
of performance families and their comparison policies. Each target's experiment
definition selects it explicitly as `config.performance_template`; the target recipe
supplies hardware-specific inputs without copying the shared policy.

The shared declarations name oracle *tiers*, not simulator binaries. Phase 0
uses concrete `L2` and `L3` engines declared by the selected capability
contract; when that contract names only a fidelity, the target's recipe must
select a concrete engine through `performance_oracles`. Materialization freezes the
selected names, the `rtl_<L3 engine>` oracle kind, the `<L3 engine>_L3_cycles`
metric, and the placeholders from which they were resolved. Missing or abstract
selections fail; Phase 0 does not probe whichever simulator happens to be
installed on the build host. A target with no admitted performance family does
not need to invent an oracle selection.

The target-neutral PK and PR acceptance contracts use new analyzer versions;
their claim thresholds and cohort membership are unchanged. The analyzers also
retain exact interpretation of historical frozen contracts. New runs freeze
their declared template bytes and selected recipe; existing frozen runs retain
their own original inputs. Regenerate in a new run rather than editing a prior
receipt. A target-neutral declaration does not by itself qualify a target's
Phase 2 measurement controller or simulator.

These declarations are not generated capsules or performance evidence. See the
[experiment workflow](../../README.md) for derivation, review and phase handoff.
