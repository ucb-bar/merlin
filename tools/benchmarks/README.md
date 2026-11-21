# Board measurement tools

These two operators use Merlin's existing compilation and independent comparison
implementations. They require the selected board, toolchain and reviewed inputs.

- `k1_int8_fair_compare.py`: measure an interleaved Merlin/ExecuTorch int8 comparison.
- `linux_board_op_profile.py`: collect a model's operator timings and whole-model wall time on a
  `flow: linux` board selected with `--board`.

Use each command's `--help` for its inputs. Generated measurements belong under
ignored `out/`. A software-only check does not establish a board measurement.
