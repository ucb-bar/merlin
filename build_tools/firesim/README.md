# FireSim build support

Merlin builds binaries and can prepare the files used to run them. FireSim
provisioning, bitstream builds, infrastructure launch and job scheduling remain
external. See [running programs](../../docs/guides/08-running.md).

- `preflight.py` reads the selected environment and reports missing prerequisites.
  It does not submit a job, start a daemon or access the FPGA.
- `make_chipyard_overlay.py` creates a private build overlay for an explicitly
  selected Chipyard checkout. It does not modify that shared checkout.
- `firesim_baremetal.ld` and `htif-nano.specs` are existing bare-metal link inputs.
  Zephyr uses its own linker script; do not apply these inputs to a Zephyr image.

```sh
python build_tools/firesim/preflight.py --json
python build_tools/firesim/make_chipyard_overlay.py --help
```

These source files and a successful preflight do not establish that a binary
ran on hardware. Retain the actual image, selected configuration and execution
evidence for each run.
