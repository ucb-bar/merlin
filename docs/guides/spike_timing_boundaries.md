# Keep setup and execution evidence separate

The original counter review selects the execution window. Packing, layout,
transfers, dispatch, accelerator work and host arithmetic performed inside that
window remain execution cost. A kernel-only review cannot establish complete
model latency or exclude its feeding costs by implication. Cold/warm state needs
its own selection; unknown state stays unknown.

Compilation, input staging outside the window, simulator startup, output
publication and grading are setup/verification costs. Record their operator
walltime and CPU costs separately. Simulator walltime never substitutes for
target cycles. Do not add overlapping operator walltimes.

`spike-timing native-diagnostics` joins one original public record to its exact
selected native model artifact:

```sh
spike-timing native-diagnostics --dataset original-dataset.json \
  --record-id ORIGINAL_RECORD_ID --model-artifact model.json \
  --output native-feedback.json
```

The output retains the ELF, input, timing window, device, report and model
identities. Execution-window counts and modeled cycles are separate from the
observer's `cycle_breakdown.scope=whole_trace` resource context. The latter
includes setup and teardown, can contain unknown events even when the selected
window is complete, and is nonadditive. Do not subtract or normalize these totals
to invent execution-window utilization, stalls, overlap or a physical floor.
Those fields remain UNKNOWN until independently checked scoped observations
exist. Compilation/staging costs also remain UNKNOWN in this report; obtain
them from the original operator receipts.

This diagnostic grants no source, numerical, statistical or hardware
qualification. Use the original complete-output admission and unchanged native
held gates separately. Public feedback refuses historical private datasets.
