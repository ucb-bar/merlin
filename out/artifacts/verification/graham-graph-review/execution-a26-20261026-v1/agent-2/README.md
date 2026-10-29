# Fresh official Lean CLI admission

This agent selected a new official Lean **4.35.0** Linux environment for source
reproduction. The tag resolves to **eb6cafb7290e8bd5f2937522d8d804f58f02bafc**.
The 925899528-byte archive matches the official publisher SHA256
`c8531faaa4a93d5a4dae0c742cbc1bd42a2993594d02f93a9d629ebf299fb2c8`.

The intake inspected 18571 ZIP entries and hashed all 17938 extracted regular
members. Declared expansion is 3365301905 bytes. All eight finite diagnostic
commands passed: version/prefix, Init/List/Omega compilation, root/axiom
inspection and diagnostic evaluation. [READY.json](READY.json) records the
executable, standard library, loader/runtime observations, environment selection
and exact evidence identities. [FINDINGS.json](FINDINGS.json) summarizes the run.

A preserved static review error came from passing the official `libc++.so` text
linker script to `readelf`. The successor inspection classified ELF magic
explicitly. No archive layout, member or digest policy was relaxed, and no native
diagnostic failed.

The selected scope is independent source compilation. It does not approve the
production Context/provider, activate a public compiler gate, establish native
fidelity or explain the earlier held runtime incident. Old runtime holds remain.
No owned jobs remain.

The SDK archive and extracted binaries are excluded from Git. Their member hashes
and raw setup/diagnostic records are retained here. The six exact plain Lean
diagnostic/audit sources are in [diagnostic-sources](DIAGNOSTIC_SOURCE_INDEX.json).
[OUTPUT_MANIFEST.json](OUTPUT_MANIFEST.json) binds the 132 original report files;
this README and source index are publication additions.
