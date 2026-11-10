# Calibration program sources

The small `*_ablation.c` programs are authored paired calibration diagnostics.
They exercise operand retention, accumulator commit, dispatch batching and vector
tails when an explicitly selected target/compiler can execute them.

Keep generated binaries, consoles, measurements and fitted coefficients under
ignored `out/`. A program source is not a measured cost model or evidence that a
target implements the operation. Record exact input and tool identities when
running a calibration investigation. Historical study drivers require their
original source revision; there is no default study-tree launcher here.
