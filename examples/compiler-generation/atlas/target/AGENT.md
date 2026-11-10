# Authored target descriptor

`descriptor.yaml` is the single source for this example's target setup and experiment
policy. Its explicit `resources_root` retains the current authored task/harness
location during migration; do not copy private bundles or generated capsules here.
Generated releases remain under the configured artifact root. Historical receipts
retain their original paths; the legacy descriptor path is only a compatibility link.

`contracts/target_contract.yaml` is loadable reference metadata, and
`contracts/residual.yaml` carries the matching intent for fact-backed generation.
Both are prototype inputs. Neither supplies executable OOT support or qualified
RTL facts; keep geometry, instruction codes, numerical review and model-operand
scale choices out until selected evidence establishes them.
