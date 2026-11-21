# tools

`console/` is the Go terminal frontend for Merlin experiment records. Its Python
data process remains in `merlin.experiments.tracking.console`.

`timing/` is the independent timing observer distribution and portable C++ engine.
It installs separately, without importing the compiler or target providers.
Its source, calibration and measurement authority remain separate from compiler
candidates. Moving the project does not change its public package or CLI names.

Keep build products beneath the configured `out/build/` root. Preserve portable
source export identities and frozen study inputs; do not add compiler code here.
