# Broadcast masked softmax

The source takes `scores [2,3,35]` and an additive `mask [3,35]`. It returns
softmax over the last axis. The mask is an ordinary runtime input. It is
broadcast across the first axis and can change without recompilation.

The example includes a column tail and rows with different mask patterns.
Target compilation needs an explicitly selected numerical policy for the mask,
exponential, reduction and division. The source itself does not select one.
For the selected bounded BF16 implementation, see the captured-application guide
in the target support repository. Generated captures, compiler packages and
execution results belong in the invocation's artifact directory.
