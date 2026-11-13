---
title: Application buffer views
kind: reference
status: current
owner: core
last_verified: 2026-11-12
related: [architecture, core_dialects]
code_refs: [src/merlin/compile/application_views.py, src/merlin/compile/semantic_regions.py, src/merlin/compile/semantic_application.py, src/merlin/semantic_compiler/codegen.py]
---

# Application buffer views

`native-compile --application --application-buffer-views` can replace eligible
internal reshapes and copies with storage views. Use the same flag with
`native-verify`. Python callers select
`CodegenOptions(application_buffer_views=True)` inside `codegen_options(...)`.
The option defaults off and participates in code-generation identity when set.

Merlin proves the transformation, outlines a view node, and gives its alias to
the bounded arena allocator. A view and its root retain the allocation until
every reader completes. The generated C view checks that its input and output
pointers are identical and performs no copy or device dispatch. The logical
operation and its source receipt remain present; there is still a C callback.

Admission requires a single selected pure `reshape` or `copy`, unchanged dtype,
numerical policy and positive element count, no attributes or indexing maps,
and identical dense, plain boundary representations. Both values must be
internal intermediates. Caller-owned inputs, constants and public outputs keep
their existing materializations. Allocated layouts, permutations, slices and
representation conversions do not qualify.

The pass also checks all selected region boundaries. It declines an alias if
that alias, including through a chain of views, would make two distinct input
ports of any consumer overlap. Existing target wrappers may forbid such
overlap even when both operands are read-only. Supporting those cases requires
an additional provider contract, rather than weakening their checks.

Compilation and independent publication replay use the same shared source
proof and C generation. The outliner rechecks the proof before granting view
semantics. The normal arena checker validates alias bounds and lifetimes, and
the C dispatcher grants overlapping input/output permission only to the view
callback. Other regions retain their disjoint-buffer checks.

This is a shared graph, buffer-ownership and dispatch optimization. Target
resource limits and folding actual layout transformations into accelerator
loads or stores belong to the target provider. A view does not grant a new
device layout or arithmetic capability.
