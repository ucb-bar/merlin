"""Phase 2 ``whole_model_measured`` mode: a candidate compiler measured as a WHOLE MODEL on hardware.

The third Phase 2 mode, beside ``measured_claims`` and ``model_portfolio``.  An authoring agent edits
a candidate package forked from Phase 1's frozen compiler; every candidate is built as one whole-model
program and measured asynchronously, keyed by the digest of its exact bytes:

* a SCREEN machine (a board, paired with a functional model that grades every group locally) measures
  every candidate -- board time only for correct programs that obey the experiment's instruction
  policy over the whole linked ELF, several candidates batched per board job with the vendor
  reference as the in-batch control;
* a CERTIFIER (the elaborated-RTL emulator) re-measures each new best, and a best it finds wrong is
  retracted; a best must also hold up SOLO, beyond the store's own measured repeat spread;
* CELL mode measures a candidate on form-perf capsules' own group programs (the inner loop), and a
  cell win is transplanted to the whole model for adjudication.

FireSim adjudicates, GSIM certifies, the functional model and cells only rank.  Nothing in this
package names a target: machines, builders, capsules and policies are declared data.

Owners: :mod:`.identity` (digests, builder closure, store key), the core's
:mod:`merlin.perf.whole_model_verdict` (reading a program's log) and :mod:`merlin.perf.whole_model_builder`
(the service builder), :mod:`.machines` / :mod:`.registry` (devices and how to run them), :mod:`.gates`
(pre-machine refusals), :mod:`.worker` / :mod:`.batch` (one job, one board batch), :mod:`.service` (the
store), :mod:`.feedback` / :mod:`.transfer` (what the agent reads), :mod:`.objective` / :mod:`.config`
(the loop's objective), :mod:`.cells` (cell mode), :mod:`.sessions` (plateau, quota, model
verification, circuit breaker), :mod:`.snapshot` (commit snapshots and preflight), :mod:`.runs` (run
layout, prepare/resume) and :mod:`.watchdog`, :mod:`.launch` (a detached launch and its record),
:mod:`.store_admin` (operator surgery), :mod:`.profiles` (launch profiles), :mod:`.progress` (a run's
status and its changes), :mod:`.round_audit` (a recorded round judged again) and :mod:`.cell_runs` (cell
runs prepared, launched, read and confirmed on the board).
"""

MODE = "whole_model_measured"
