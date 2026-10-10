# Gemmini Phase 1 contract resources (intentionally empty)

The [target descriptor](../../target/descriptor.yaml) selects this directory as
`contracts_root`. Gemmini grants no curated runtime harness, ISA C headers or
hardware bring-up links: `hardware_spec` is empty, and authors derive the
hardware from the selected RTL facts and target contract instead.

Keeping the declared root present and empty is deliberate. Without it,
contract lookups (for example the ISA/RTL cross-check's bring-up set) would fall
back to the retained sibling tree under
`merlin/experiments/capsule_bench/targets/gemmini/contracts/`, whose legacy
headers and RTL links are not reviewed inputs for a fresh experiment.

Release preparation removes `contracts_root` from the prepared descriptor and
stages only declared resources; nothing here is copied into a release.
