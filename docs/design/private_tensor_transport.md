---
title: Private tensor transport
kind: design
status: current
owner: core
last_verified: 2026-10-18
related: [architecture, shared_tensor_preparation]
code_refs: [src/merlin/xdsl_dialects/lowering/private_transport.py, merlin/tests/ir/test_private_transport.py]
---

# Private tensor transport

An external producer may widen its native readout solely for a consumer that
immediately narrows it again. `rewrite_private_transports` can preserve an
explicit alternative representation along that private edge. This removes
unnecessary host materialization without asserting that data remains resident
on a device or that device commands can be fused.

Providers supply `PrivateTensorTransport` contracts: equivalent producer and
consumer entries and the intermediate scalar type. Both endpoints must be
complete synchronous operations whose composition preserves the original
results, intermediate rounding and exceptional behavior. A consumer that
previously refused NaNs must retain that refusal even when a narrower input
would otherwise skip conversion. Symbol identity alone proves none of these
numerical properties; qualification belongs to the provider.

The generic pass checks destination-only external calls with explicit read
inputs and a fresh private write destination. The producer result and every
intervening expand/collapse view must have a sole use. Shapes are static and
unencoded, and views preserve flat contents. Dynamic views, transposes, casts,
fanout, observed intermediates and unknown effects retain original execution.
The provider selector cannot override these checks.

Rewriting allocates the producer's new destination, rebuilds its private views
and changes only the selected consumer input. It preserves call positions,
public tensor types and arithmetic bodies. Declarations retain read/write
contracts; inconsistent signatures and existing-symbol collisions refuse.
The original module is unchanged and the returned clone must verify.

The caller owns cost selection. Measure transfers, conversion work, allocation,
guards, target workspace and any padded readout crop under the existing timing
boundary. A legal representation change is not automatically profitable.
Separate device scratch residency, direct readout into a host tensor and
cropping through a temporary readout in resource and performance claims.
