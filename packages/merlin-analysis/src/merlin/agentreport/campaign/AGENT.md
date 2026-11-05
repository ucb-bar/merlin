# AGENT.md — packages/merlin-analysis/src/merlin/agentreport/campaign

Draws one agentic compiler-construction campaign (a chain of continuations, each possibly restarted)
as a single-column figure of ONE stitched run: agent trajectory (activity share, token rates, agents
active, milestones), test cases passed over time (L2 / L3), measured RTL performance (speedup, PE-array
utilization), estimated compute on the reserved cores, and final status by cohort.

```
python -m merlin.agentreport.campaign --manifest <campaign.yaml> [--out-dir DIR] [--formats pdf,svg,png]
python -m merlin.agentreport.campaign --run-dir RUN [--workspace WS] [--agent-home HOME] [--launch-log LOG]
                                      [--accounting-log LOG] [--target T] [--live]
```

Without `--out-dir` the output is a versioned product (`new_product("campaign-figure", ...)`, declared
under `agentic-report/campaign-figures` in `merlin/contract/storage.yaml`, target axis from the manifest).
Re-run any time during or after a run; edit the MANIFEST (not code) when an attempt is archived or a
new one starts. Keep a campaign's manifest with its outputs.

## Modules
- `manifest.py` — schema `merlin.agentreport.campaign_manifest.v1`: attempts (run dir, workspace, agent
  home, launch log, accounting log + block index, live flag), reserved cores, RTL engine identity,
  simulator display names, cohort rules (display order + `priority`). No campaign facts live in code.
- `records.py` — read-only readers -> one dataset (`records.json`). Record fields named after the corpus'
  vocabulary are read as written; every key it returns speaks of test cases.
- `figure.py` — stitching, series and drawing; `draw` returns the notes (every number on the figure).
- `cli.py` — product folder: `campaign_figure.{pdf,svg,png}`, `notes.json`, `records.json`,
  `campaign_manifest.yaml`, `README.md`.

## Rules encoded here
- **One stitched run.** Active minutes = each attempt's window from its lead agent's start to its last
  record, cut where a finished attempt's background grader recorded nothing for 45 min. Pass status is
  carried across attempts and never reset; a later failing grade drops (L2) or demotes (L3) it.
- **Performance binding** (after the console's `perf.py` / `capsule_metric`): an RTL row counts only with
  the manifest's engine identity, positive cycles, a 64-hex compiler source digest and a test-case digest
  equal to its structural-comparison binding; the roofline only when floor <= cycles. Model estimates
  count only when PROVISIONAL/QUALIFIED with complete event accounting. Exclusions are counted, never filled.
- **Compute** is an estimate: recorded job intervals scaled per launch (one factor <= 1, capped at the
  reservation) to the CPU time systemd measured; unattributable CPU is "other". Say so wherever it is shown.
- User-facing text says "test case(s)", never the corpus' internal word for them.
- Paired tests: `merlin/tests/infra/test_agent_report_campaign.py`.
