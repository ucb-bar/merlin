# Inconclusive first historical replay attempt

The actual historical parent pass executed, its exported output was refused by C1,
and native execution returned the expected cb7fffff float word. The collector then
failed because it incorrectly expected an extra returned byte identity from the
three-argument, one-result fixture. That extra return belongs to the separate old
quant-ext qualification fixture/oracle. Native output here has exactly one word.
No fixed-pass execution or complete historical detection result is credited to v1.
Exact original collector, generated files and FAILURE.json are preserved. A new
version corrects only the collector's expected output arity and adds explicit source
support/module checks. No production/checker/contract/artifact is modified.
