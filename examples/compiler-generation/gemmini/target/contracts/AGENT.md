# Public capability inputs

The contract and residual are human-authored prototype inputs, not generated or
certified results. `residual.yaml` feeds the generic contract deriver;
`target_contract.yaml` is the selected reference read by discovery, the example
experiment, and direct probes. Their parsed YAML currently matches; the OOT
generation test pins this relationship. Do not treat them as independent evidence
or let edits drift. Keep the selected contract as a regular file: backend generation source
admission rejects a symlinked provider contract. A future derived contract
belongs in a content-identified output artifact, not beside these authored
sources. Executable plugin declarations belong to the selected OOT support
provider, not these example inputs.
Qualified RTL facts remain explicit external artifacts; never invent missing facts
or copy private support here to satisfy tests.
