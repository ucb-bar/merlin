# AGENT.md ? src/merlin/compiler/quantization

Keep numerical intent separate from calibration and application policy. Existing
v1 methodology derivation is an explicit compatibility mechanism. New profile
selection binds source bytes and exposes changed numerical intent as an exact
alternative before returning an executable recipe. Reuse existing recipe and
capture adapters; profile selection grants no framework or target qualification
and changes no independent quality/reference gates.

`admission.py` joins saved numerical declarations, hardware candidates and actual
operation observations. It does not resolve a backend, place regions or import
framework engines. Its lazy capture-recipe export preserves the historical public
binding; the projection algorithm lives in `torchao/recipes.py` and retains explicit
preparation-profile and unknown-semantics refusals. Authored validators remain in
`target/spec/quantization.py`; diagnostic candidates supply no execution authority.

`recipe.py` owns serialized numerical records, vocabulary and content identity.
`capabilities.py` derives readout candidates and preserves the historical public
recipe namespace by exact aliases to actual definitions. `static.py` and
`dynamic.py` construct explicit numerical choices; they do not apply TorchAO,
place regions or qualify application quality. `cli.py` owns the existing JSON
command. Keep normal module dependencies: no global-patch propagation or callback
registry. The legacy capture `for_target` seam shares the actual capabilities
module; private patches belong at their real defining owner.
