# Functional model state

The model takes `[2,3,35]` history and input tensors. It returns ReLU of their
sum followed by the updated history. The caller supplies initial history and
passes the returned history to the next successful invocation. The input state
is preserved; input and output buffers must be separate.

Capture `loader.py` using the existing model2MLIR capture worker. With the staged
program hash, declare `history` as an `updated` state input, `output_index: 1`,
and `initialization: "caller"`. Pass that declaration with
`native-compile --application --application-state state.json`, and supply the
same declaration to `native-verify`. A selected numerical policy determines
addition and conversion semantics. This example does not establish arbitrary
mutable Python state, KV-cache support or original-model numerical quality.

`readonly_loader.py` has a registered `[1,3,1]` scale buffer and a
`[2,3,35]` input. Declare the scale argument as `read_only`, with
`output_index: null` and `initialization: "captured"`. The C/H package contains
an initializer for the captured scale values. The caller may retain or change
that input between invocations; no recompilation or embedded constant
substitution is needed. See the [state interface guide](../../../docs/reference/host_device_compilation.md#explicit-model-state).
