# v4_rtl_tools — Method Prompt

## Role

You have access to lightweight RTL analysis tools (grep, regex, signal tracing) applied to
`selected_rtl/`. Use them to extract hardware-grounded facts before schema planning.

## Process

1. Use Bash tools to grep for RoCC command encodings, state machine transitions, and
   memory address calculations in the RTL.
2. Emit extracted facts to `<run_dir>/contracts/rtl_facts.jsonl`.
3. Use the RTL facts as primary evidence for `dialect_plan.yaml` op semantics.
4. Derive contracts as in v3.

## What this measures

Whether automated RTL analysis (vs. human-curated docs) produces better-grounded
dialect designs. Comparison to v3 isolates the value of RTL tooling vs. doc-reading.
