# AGENT.md — packages/merlin-experiments/src/merlin_experiments/execution

The provisioned `native_supervisor` is the direct parent/reaper of every guardian.
`native_guardian` owns only its descendant tree. `chia_native` adapts public Chia
hooks; no worker-side Popen or private upstream tracker mutation is permitted.
`_protocol` owns bounded local packets, descriptor transfer and peer identity.

Require a private authenticated same-host/PID-namespace endpoint, explicit sessions,
worker and driver pidfds, and independent native-tree and guardian-reaping evidence.
Unknown cleanup is never success. Preserve native stdio/environment without exposing
session capabilities to candidates or receipts. Do not launch a background daemon on
import or automatically change an existing cluster. Service/host failure is not
qualified by worker-loss tests. Historical task receipts retain their old meaning.

Frozen services and guardians use the existing frozen Python command builder and
snapshot seal. Their source reference comes from verified process-local bootstrap
state, never ambient environment or rehashed live checkout files. Session admission
and guardian READY must match that reference; frozen callers cannot silently downgrade.
The reference attributes service/guardian implementation, not an ordinary worker's
loaded imports or the candidate command. Production launchers remain separately wired.

`chia_group` owns driver reservations, task-reference association and independent
native lifecycle publication. It never changes the existing Chia task receipt's
meaning or upstream accounting. Its current locality policy is explicitly single-node:
only the driver's node may run managed work, with resource feasibility checked first.
Returned task results require matching native-start/exit evidence; bypass is not success.
Attempt every owned receipt despite an earlier failure, preserving primary exceptions.
