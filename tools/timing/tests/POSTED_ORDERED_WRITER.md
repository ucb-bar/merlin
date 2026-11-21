# Ordered writer handoff contract

`bind_ordered_deferred_values` permits a caller-owned ordered pipeline to pass
one matching predecessor writer at its actual internal-consumption token. The
caller supplies original value ID, predecessor final-publication token and
predecessor consumption token. It must establish FIFO publication order and
safe internal forwarding or a write that does not consume the old payload.

The generic kernel supplies no pipeline or latency law. An exact original
writer identity is required; a stale identity preserves the publication wait.
Independent reader obligations remain. New external readers still depend on
final publication, including all earlier ordered writes. The ordinary
`bind_deferred_values` entrypoint retains its original semantics. No state is
recycled and the existing visibility/token limits remain.
