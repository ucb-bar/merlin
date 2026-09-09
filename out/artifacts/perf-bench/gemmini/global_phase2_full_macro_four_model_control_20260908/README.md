# Four-model Phase 2 launch control

This is the control surface for the next full Phase 2 authoring run. It does not contain a claimed performance result and it does not rerun Phase 1.

The run is armed against the integrated generalized compiler at tree SHA-256 `f8992bc35be76e805dc8dce9bfe5864167d50f0276da685306a2fb7d9dddfa36`. It trains simultaneously on complete ResNet-50, TinyLLaMA, LSTMNetViT, and SmolVLA graphs. Each revision is judged per model with whole-graph compilation and static work, movement, representation, residency, synchronization, dispatch, and coverage evidence. Unlike model cycles are not added into a synthetic score.

Search is host-only. It does not run a complete model or layer in GSIM, L3, or FireSim. The analytical provider is wired into the launcher but fails closed to exact-only evaluation until a target-bound calibration and four held-out host-quality corpora are complete. FireSim remains a post-freeze measurement step.

The authoring order is macro-first: delete whole-program work and host/device boundaries, improve global dataflow/layout/residency, improve issue/overlap/synchronization, and only then tune operators or tiles. One coherent generalized compiler mechanism is attempted per round.

Calibration preparation is now complete and explicitly `incomplete`, not silently accepted. The trusted evidence establishes 10 GSIM counter windows, partial-overlap efficiency of approximately 0.9199, and a controlled movement fit of 16.119 bytes/cycle with a 128.502-cycle intercept. It does not establish compute-unit, physical-movement-unit, or encoding-transition coefficients. The canonical receipt is `../phase2_host_analytical_calibration_20260908/preparation.json`, with receipt SHA-256 `79abe4eb3faae1080c906de9a4afb5fdeaeff5a46e5e452d5ba600d2d8a799b0`; therefore this run intentionally uses exact-only evaluation until those fields have defensible evidence.

The feature-calibration preparer now verifies paired controls, target/source/evidence digests, typed JSON pointers and physical-interface units, then recomputes coefficients. Current canonical compute and physical-movement preparations are under `../phase2_feature_calibration_preparation_20260908/`; both report `incomplete` because the frozen corpus contains zero valid controlled pairs. Compute needs two points, and the two-coefficient physical-movement fit needs four. Separately, the emitted-artifact encoding seam verifies one retained reduced materialization as one transition and 72 physical bytes, but does not generalize that single point into a model-wide coefficient.

The existing-evidence audits are sealed under `../phase2_compute_feature_existing_evidence_audit_20260908/` and `../movement_pair_contract_audit_20260908/`. They prove that no hidden reusable calibration exists: the compute runs lack bound controls and warm-predecessor identity, while the movement series has only one distinct issued-count arm, no executed-command or physical-interface byte measurement, no controls binding, and no warm-predecessor proof. Logical command-buffer byte volumes were not relabeled as physical traffic.

## Current gate

At the recorded preflight, approximately 8 GiB of swap was occupied. The launch ceiling is 2 GiB, with one analysis worker and 48 GiB minimum available RAM. The run is therefore waiting rather than overriding the safeguard that protects the workstation.

Use `STATUS.json` for the machine-readable state. Once the resource gate is healthy, `launch_exact_only_when_safe.sh` starts the fresh run directory. The calibrated launcher flags can replace exact-only fallback only after the calibration preparation emits an admissible, content-addressed bundle.

Run `./current_status.sh` for a live, read-only resource/watcher check. The traveling checkpoint includes `DISARMED`, so both launcher paths refuse to start work after the machine move. On the destination machine, restore the ignored compiler/objective artifacts referenced below, verify every recorded digest, reclaim swap below 2 GiB, deliberately remove `DISARMED`, and then explicitly start `watch_and_launch_when_safe.sh`. It invokes the hardened launcher only after admission and refuses to create a second run when the planned output already exists.

The earlier `global_phase2_full_macro_four_model_v1_20260908` path contains only a failed worker's `host_resource_telemetry.json`; it contains no authoring round or performance result. The checkpoint therefore reserves fresh `global_phase2_full_macro_four_model_v2_20260908` for the next explicit launch.
