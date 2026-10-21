# Explicit fast native numerical and count diagnostics

The original host can select the fast loop for both compiler arms:

```python
from merlin.targetgen.checked_source_model_diagnostic import (
    NUMERICAL_COUNTS_PROTOCOL,
    select_checked_source_diagnostic,
)

checked = select_checked_source_diagnostic(
    passes, target=target, rtl_facts=facts, board_config=config,
    verification_protocol=NUMERICAL_COUNTS_PROTOCOL,
)
```

This is a distinct live selector and sealed-child record. The default selector
continues to require complete target source-effect observation. The existing
explicit native pointer ABI can be selected independently.

The fast consumer preserves the original frozen source eligibility, source/IR
and ABI/control-flow checks, exact ELF instruction/role checks, and complete
same-ELF native output comparison. It reopens L0/L1/L2 twice and requires unchanged
native receipt bytes. It then returns before creating a source-effect owner or
running a complete commit trace. Original deeper tier requirements remain pending.

The result uses schema
`merlin.checked_source_model_numerical_counts_result.v1` and status
`numerical_counts_L0_L1_L2_diagnostic_pass_source_effects_pending` in
`checked_source_status`. Its overall status remains `incomplete`. Source effects
are explicitly `PENDING`; formal admission, incumbent selection, promotion and
roofline convergence have no authority. The result supplies no timing-model fit.
Counts and frozen approximate estimates are separate observations.

A private queued numerical receipt binds the exact original native JSON. Before
using count feedback, its consumer reopens actual source, IR, ELF and full-output
witnesses, recomputes source eligibility, and reruns the matching original native
audits under the live selector. Missing or changed witnesses refuse. A saved fast
status cannot satisfy the original full-effect selection.

Fast diagnostics support iteration only. Complete source-effect checking and
the existing independent compiler acceptance and later hardware gates remain
mandatory before accepting or promoting a compiler.
