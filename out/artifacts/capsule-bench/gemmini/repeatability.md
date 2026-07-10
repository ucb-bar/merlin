# Repeatability — raw_baseline (claude-opus-4-8), rate-limit-aware

> **Why this matters.** Each full pilot run costs ~$18 / ~50 min and exhausts the org's five-hour session budget. Runs that the budget rejected (zero real work) are **blocked**, not failed — counting them as failures would falsely imply the baseline is unreliable.

- **Valid full-pass rate (public 4/4 AND hidden 3/3, blocked runs excluded): 1/1**
- runs: 3 total = 1 valid + 2 blocked-by-rate-limit
- integrity clean (valid): True; numeric exact (valid): True; answer-access leak (any run): False
- rounds-to-converge (passing runs): {'min': 2, 'max': 2, 'mean': 2, 'median': 2, 'stdev': 0.0}
- wall(s) valid: {'min': 2977.692, 'max': 2977.692, 'mean': 2977.69, 'median': 2977.692, 'stdev': 0.0}
- cost$ valid: {'min': 18.1395, 'max': 18.1395, 'mean': 18.14, 'median': 18.1395, 'stdev': 0.0}
- raw full-pass (NOT rate-limit-aware, for audit only): 1/3

| run_id | outcome | public | hidden | tier | rounds | rej/work | wall(s) | cost$ | answer-access |
|---|---|---|---|---|---|---|---|---|---|
| rb_pilot_rep_01 | **pass** | 4/4 | 3/3 | L3 | 2 | 0/2 | 2977.692 | 18.1395 | attempted_blocked |
| rb_pilot_rep_02 | **blocked_rate_limit** | 0/0 | 0/0 | None | 8 | 7/1 | 334.287 | 3.1187 | attempted_blocked |
| rb_pilot_rep_03 | **blocked_rate_limit** | 0/0 | 0/0 | None | 8 | 8/0 | 11.416 | 0.0 | clean |

**Legend.** outcome: pass / fail (ran for real, did not converge) / blocked_rate_limit (five-hour budget rejected the agent; excluded from pass-rate). rej/work: rounds rejected by the rate-limit / rounds that did real tool work. answer-access: clean / attempted_blocked (tried to read a masked-absent answer file => got nothing) / leak (real answer read). Only `leak` is disqualifying; masking-defeated attempts confirm the isolation held.
