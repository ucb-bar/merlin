# Experiment reference inputs

Status: reference

This directory retains explicitly selected experiment inputs and source-library
manifests. It is not an experiment controller or a source of implicit defaults.
Preserve declared source identities when changing or relocating a manifest;
existing sealed runs retain their original inputs. New results and operator
records belong under the configured output root, not alongside reference data.
