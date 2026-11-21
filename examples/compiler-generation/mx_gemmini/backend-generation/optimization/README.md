# Optimize a MX Gemmini backend

Correctness and performance are developed together by the native backend
generation engine. Start with the target's [authored workflow](../../experiment.yaml)
and the [backend-generation instructions](../README.md).

Select a reviewed test-generation release, its matching input bundle, the
campaign policy and timing evidence explicitly. The [experiment guide](../../../../../docs/guides/12-experiments.md)
shows the named workflow and its required inputs. A target name or a successful
process exit does not establish numerical correctness or measured performance.

Keep generated candidates, compiler snapshots and measurements under ignored
`out/`. Historical specialist configurations require their original source
revision, frozen compiler and operator inputs; they are not catalog defaults.
Do not substitute another target's compiler, hardware profile or measurements.
