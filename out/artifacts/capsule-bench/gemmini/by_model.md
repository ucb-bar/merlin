# capsule-bench by model - target `gemmini`

| model | runs | best public | best hidden | rounds | tool calls | tokens | cost |
|---|---|---|---|---|---|---|---|
| `glm5` | 1 | 0/20 | 0/5 | 6 | 599 | 46,441,832 | $28.19 metered |
| `gpt-5.6-sol` | 3 | 20/20 | 5/5 | 1 | 190 | 26,648,234 | $21.08 notional (subscription, not billed) |
| `nemotron` | 2 | 0/20 | 0/5 | 12 | 786 | 25,387,853 | $3.89 metered |

## per run (a best-of row hides a run that diverged)

| run | model | public | hidden | tier | rounds | tokens | cost |
|---|---|---|---|---|---|---|---|
| `merlincirct_gemarm4_glm5` | `glm5` | 0/20 | 0/5 | — | 6 | 46,441,832 | $28.19 |
| `merlincirct_gemarm4_codex` | `gpt-5.6-sol` | 20/20 | 5/5 | L3 | 1 | 6,591,224 | $5.63 notional |
| `merlincirct_gemarm4_codex2` | `gpt-5.6-sol` | 20/20 | 5/5 | L3 | 1 | 10,125,955 | $8.19 notional |
| `merlincirct_gemarm4_codex3` | `gpt-5.6-sol` | 20/20 | 1/5 | L2 | 1 | 9,931,055 | $7.26 notional |
| `merlincirct_gemarm4_nemotron` | `nemotron` | 0/20 | 0/5 | — | 12 | 18,506,132 | $2.83 |
| `merlincirct_gemarm4_nemotron3` | `nemotron` | 0/20 | 0/5 | — | 12 | 6,881,721 | $1.06 |

## how far capsules got (highest tier reached, summed over runs)

- `merlincirct_gemarm4_glm5` (glm5): L1=17, none=3
- `merlincirct_gemarm4_codex` (gpt-5.6-sol): L3=20
- `merlincirct_gemarm4_codex2` (gpt-5.6-sol): L3=20
- `merlincirct_gemarm4_codex3` (gpt-5.6-sol): L2=19, L3=1
- `merlincirct_gemarm4_nemotron` (nemotron): none=20
- `merlincirct_gemarm4_nemotron3` (nemotron): none=20

## arm conformance (did the run use the RTL-derived tooling it was given?)

| run | model | rounds conformant | isa_tools | cca | no-regex | full self-check |
|---|---|---|---|---|---|---|
| `merlincirct_gemarm4_glm5` | `glm5` | 0 | no | no | no | yes |
| `merlincirct_gemarm4_codex` | `gpt-5.6-sol` | 1 | yes | yes | yes | yes |
| `merlincirct_gemarm4_codex2` | `gpt-5.6-sol` | 1 | yes | yes | yes | yes |
| `merlincirct_gemarm4_codex3` | `gpt-5.6-sol` | 0 | no | yes | yes | yes |
| `merlincirct_gemarm4_nemotron` | `nemotron` | 0 | no | no | yes | yes |
| `merlincirct_gemarm4_nemotron3` | `nemotron` | 0 | no | no | yes | yes |

## how the run was spent

| run | model | actions | recon before 1st write | writes | invalid calls | distinct tools |
|---|---|---|---|---|---|---|
| `merlincirct_gemarm4_glm5` | `glm5` | 599 | 53 (9%) | 114 | 3 | 9 |
| `merlincirct_gemarm4_codex` | `gpt-5.6-sol` | 45 | 33 (73%) | 4 | 0 | 2 |
| `merlincirct_gemarm4_codex2` | `gpt-5.6-sol` | 78 | 28 (36%) | 13 | 0 | 2 |
| `merlincirct_gemarm4_codex3` | `gpt-5.6-sol` | 67 | 21 (31%) | 17 | 0 | 2 |
| `merlincirct_gemarm4_nemotron` | `nemotron` | 507 | 46 (9%) | 87 | 2 | 7 |
| `merlincirct_gemarm4_nemotron3` | `nemotron` | 279 | 24 (9%) | 21 | 16 | 6 |

## where capsules die (first failure plane, summed over runs)

- `glm5`: command_buffer_schema=3, spike=17
- `gpt-5.6-sol`: -
- `nemotron`: numeric_golden=20, parse=20

## telemetry reconciliation

- every run wrote a sink and its token totals agree with the run's own accounting

## spend

- metered (counts against the budget ceiling): **$32.07**
- notional (subscription; tokens real, dollars not billed per-token): $21.08
- billing mode unrecorded (older runs; NOT added to the metered total): $0.00

## caveats

- `notional (subscription)` = a ChatGPT/Claude-subscription run: tokens are real, dollars are notional.
- `unpriced` means the price table has no rate for that model - it is NOT zero spend.
- `(lower bound)` = at least one round ended without the provider reporting usage.
