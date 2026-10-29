# Prepared reference-free EL1/EL2/EL3 replacements

All three replacement configurations exclude every expert kernel, including Exo sources,
archives, ELFs, disassemblies, reference maps and learned artifacts. The existing Spike
model and feedback are preserved. No replacement author has launched.

The original task and branch are unchanged: `cc/el4-grader-throughput-20261024`, HEAD
`388ee5ff2ed57b06b5f229e1abadfba78d7fbac4`, with the preexisting pending merge preserved.
The detached local source snapshot is `02d14a7d523f1ff72eaf4fb6ba6c56d2bbd47b33`.
It contains the original matched r2 source plus the explicitly reviewed no-expert policy,
preparation and common output-comparison repairs. No branch commit, push or merge occurred.

## Completed

- Fresh profiles and run IDs, empty backend and new provider contexts.
- Four hours authoring, 900 seconds drain, 15 CPUs and at most eight helpers per arm.
- Native input preparation, corpus inspection/sealing, ordering preparation and launch plans.
- Independent staged-input audit: 28 common grants per arm, no expert grants/bank, exact
  0/12/34 supplied source inventories, matching datasets and Spike settings.
- Preflight scripts rendered. `preparation/preflight.json` deliberately records
  `rendered_only=true` and `all_passed=false`; it is not native qualification.
- Frozen-source tests: 213 passed, 1 skipped, 5 preexisting fake-target fixture failures
  reproduced on baseline. All 25 new common-output tests pass. See test-qualification.json.
- Prepared boundary-canary controls in profiles/; 12 static/refusal tests pass. No boundary
  control, model or workload simulator was run by this preparation.
- Configurations, receives inventories, native plans and receipts archived in the repo's
  examples/gemmini/phase1/studies/kernel_abstractions/configurations/20261028/el123-no-expert/.

## Required before author launch

The existing tooling campaigns occupy the future CPU groups. Their scheduled author end is
22:31:57–22:32:12 UTC, with final drain through 22:46:57–22:47:12 UTC. Use live process and
allocation evidence, not those scheduled times alone, to decide whether the groups are free.
Preserve external CPUs 4–37 and 48–55. Preparation/control probes use CPUs 0–3.

1. After those CPUs are actually released, run each frozen native launcher's `preflight`
   command with its exact spec and export ID plus `--execute`. Retain the snapshots (`drop`
   lists are empty). Capture the outer launcher output, including `PREFLIGHT OK` and the
   tooling-readiness line, in a wave-local log. This performs native smoke checks without
   launching a paid author.
2. Emit preparation/native-preflight-executed.json only from successful executed checks.
   The schema and required identities, exact 15-CPU groups, environment/log hashes and
   markers are specified in profiles/README.md and profiles/canary_policy.py. Do not
   manufacture this receipt from the rendered script or an old run's readiness.
3. Run profiles/run-native-canary.py with `--execute` for each arm on CPUs 0–3. It requires
   the executed native receipt and probes actual author/helper/candidate boundaries,
   including all retained expert/Exo paths and reference aliases. Its generic C++ control
   is a permission check, not a workload or performance qualification.
4. Audit the actual assembled initial and resumed prompts and empty backend. Require the
   common public ABI/schema inputs to remain readable and the expert material unavailable.
5. Recheck CPU/campaign admission and launch at most three replacement authors, with no more
   than five full 15-CPU campaigns machine-wide. The small two-core studies are exempt as
   directed by the user. Preserve all checkpoints, final frozen sources, complete roster
   admission and unknown/deferred verdicts separately from pass/fail.

## Live-run scope clarification

The live tooling A/B configurations intentionally received raw expert/Exo references when
launched; C did not. The user has now reinforced no expert Exo kernels. Root disclosed this
and asked whether that means the new EL1/EL2/EL3 only or replacing live A/B too. No answer
has arrived as of this handoff. Do not silently relabel exposed runs as reference-free.
The existing health-monitor subagent continues watching all three without changing them.

The earlier reference-assisted EL1/EL2/EL3 results and immutable CSV are preserved, with
protocol-deviation metadata marking them ineligible as primary reference-free ablations.
