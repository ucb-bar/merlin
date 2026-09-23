# Authored target descriptor

`descriptor.yaml` is the single source for this example's target setup and experiment
policy. Its explicit `resources_root` retains the current authored task/harness
location during migration; do not copy private bundles or generated capsules here.
Generated releases remain under the configured artifact root. Historical receipts
retain their original paths; the legacy descriptor path is only a compatibility link.
