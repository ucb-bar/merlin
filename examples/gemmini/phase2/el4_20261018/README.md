# Historical bounded Phase 2 controllers

This directory contains earlier manual exploration and diagnostic adapters over
Merlin's source, compiler, linking, simulator and retained-revision APIs. It is
not the current joint-campaign launcher or a self-contained reproduction preset.
Target schedules and implementations belong in the selected out-of-tree backend.

Use [artifact setup](../../../../docs/guides/artifact_evaluation.md) for a fresh
joint campaign. For the separately selected retained-compiler workflow, see
[capsule source development](../../../../docs/guides/phase2_capsule_source_development.md)
and [shared-pass integration](../../../../docs/guides/shared_pass_phase2.md).

These adapters require their explicitly selected inputs and preserve original
runtime, source, numerical and timing admission. Local evidence and deployment
selections are not distributed here. Inspect each entrypoint's arguments before
using it; source availability does not establish a working target compiler,
qualified timing model or completed optimization campaign.
