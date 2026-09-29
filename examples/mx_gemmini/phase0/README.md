# MX Gemmini Phase 0

The [`mx-gemmini-functional`](../experiment.yaml) experiment derives capsules from the
selected RTL facts, [software specification](../target/software-spec.yaml), and
observed application captures. The [recipe](recipe.yaml) uses `derived_only` membership.
The former hand-authored recipe is retained verbatim as
[`recipe-legacy.yaml`](recipe-legacy.yaml) for reproduction; it is not a current
qualification claim.

The software specification selects `GemminiMxFPConfigs.standaloneMxFPConfig` at
`f0167390b56fb315deea90ac1fc3983772e92d82`. It declares separate MXFP8,
MXFP6, and MXFP4 contraction contracts. Source-bound elaboration and a local
Verilator simulator now pass the pinned format tests and bounded compiler
payload diagnostics at 32³, 64³, and one two-wave 32×32×64 split. A selected
Spike extension also matched the RTL goldens for 18 exact ELFs, including
zero-block, two-wave, and one element-subnormal vector per format. The spec remains **unreviewed**: full numerical
edge cases, a complete toolchain-closure receipt, Phase 0 L0–L3 capsules,
and whole-model/host semantics still need admission.
The selected independent MX numerical model is loaded through `MERLIN_MLC_DIR`; its frozen
source identity and the selected RTL identity must agree. A TorchAO fake-quant
capture is only an operand-conversion and coverage diagnostic, not an L2/L3
oracle. The source-level config audit is recorded in the guide; the retained
Phase 1 `hwbringup_mx_v0` inputs still describe an older default/GPU-local
mapping and do not qualify the selected standalone config.

The selected `mx_gemmini.synth.yaml` is a retained, unverified legacy sidecar.
Phase 0 preflight refuses it; its older BF16/int8 entries are not an MX corpus.
Fresh synthesis currently also needs an explicit same-target backend capability
contract and derived tile geometry bound to the selected RTL. The retained
conformance requirement also admits BF16/int8 accelerator cells that conflict
with this three-format software spec; synthesis rejects those cells. Select
verified MX application captures, derive a new requirement, and review a new
digest-bound sidecar under the artifact root before running Phase 0. Preserve
the historical inputs.

[The MX Phase 0 guide](../../../docs/guides/mx_gemmini_phase0.md) records the
format rules, supported operation boundary, capture policy, and evidence gates.

Provision the selected OOT support and numerical model, replace the legacy
sidecar, then inspect and run with fresh paths under the configured output root:

```sh
merlin experiment inspect mx-gemmini-functional --phase 0
merlin experiment preflight mx-gemmini-functional --phase 0
merlin experiment run mx-gemmini-functional --phase 0 --run-dir /configured/out/runs/mx_gemmini/phase0/example-1
merlin experiment corpus prepare /configured/out/runs/mx_gemmini/phase0/example-1 --output /configured/out/artifacts/protocols/mx_gemmini-review-1
merlin experiment corpus inspect /configured/out/artifacts/protocols/mx_gemmini-review-1
```

A prepared corpus needs operator review before sealing. A seal acknowledges
reviewed inputs; it does not certify numerical behavior, source closure, or a
functional compiler. Keep generated runs, holdouts, weights, and goldens outside
source examples.
