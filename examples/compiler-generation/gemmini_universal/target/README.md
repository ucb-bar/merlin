# Authored Universal target inputs

[`descriptor.yaml`](descriptor.yaml) selects the Universal configuration and
its explicitly retained backend generation resource root. Its
[`target_contract.yaml`](contracts/target_contract.yaml) and
[`abi/gemmini_params.h`](contracts/abi/gemmini_params.h) are public,
configuration-specific inputs. The ABI and its [provenance record](contracts/abi/abi.yaml)
must be read together; the header is not interchangeable with Gemmini's.

For this target's L2 Spike model, set `MERLIN_GEMMINI_UNIVERSAL_SPIKE_EXTLIB`
to the absolute path of the operator-provided `libgemmini.so`. The resolver checks
that file against the SHA-256 in `target_contract.yaml` and refuses missing or
different bytes; it never substitutes the ordinary Gemmini extension.

Compatibility links remain at the old descriptor and contract paths so historical
code and receipts can be inspected. The retained resource root contains old
generated input bundles and an absolute RTL symlink into another user's checkout.
Those are neither portable nor newly frozen grants. Prepare an operator-owned,
ordinary input tree and a new reviewed release before verified execution.
