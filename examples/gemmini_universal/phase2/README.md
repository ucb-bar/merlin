# Phase 2: optimize a frozen Universal compiler

No Universal Phase 2 run is declared yet. A new run must consume the exact
frozen Universal Phase 1 compiler bytes, its functional identity, the same
configuration descriptor, qualifying certificates and a managed worker. Do not
use Gemmini's compiler, performance data or hardware profile as a surrogate.

The shared [measured-claims](../../../experiments/definitions/measured-claims-template.yaml)
and [model-portfolio](../../../experiments/definitions/model-portfolio-template.yaml)
templates define the installed routes. Copy one outside this example, fill in
the Universal-specific operator inputs and budgets, then inspect/preflight the
new definition. Keep it a template until every required input is real. Store
candidates, checkpoints and measurements under the configured output root;
preserve the published compiler payload and its records separately.
