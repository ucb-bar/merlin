# Reviewed original-copy verification — coordinator only

The user resumed deadline work. Installed SDK and all failed native profiles stay
held; this does not explain or repair the original libleanshared failure.

METADATA-REVIEW-v2.json selects explicit W files/prior runtime, F selected refs
(not excluded inventories) and Q bindings. All4527 SDK references agree across
the original contexts and the PRE-EXISTING independent-copy sha256.json; there
are zero missing/different/conflicting original digests. The old copy's recorded
Lean executable and libleanshared image hashes are exactly the original pins.
Its earlier verification source reports fresh inodes. That source was READ only;
it must not be executed because it opens the held installed SDK.

The coordinator now authorizes DATA-ONLY verification/copying of these exact
pre-existing archived files into a NEW private runtime-candidate directory.
Never open the installed SDK, libclang or Std.Time.Format.Basic. Refuse symlink/
nonregular source files, changed descriptor stats, wrong size or ORIGINAL hash.
Use new exclusive regular destination files, not hardlinks; verify source bytes
before writing and preserve exact provenance. On a first mismatch retain actual
hash/stats and STOP. No expected digest is learned from a new file.

This is an independent original-image delivery check, not old-profile retry or
qualification. Native invocation is NOT authorized by a successful data copy.
A later explicit context review must bind runtime/import/project closure, system
loader/libraries, commands and observations before any worker uses it. Workers
remain under the installed/native hold. Preserve every old capsule and diagnosis.

## New basic-kernel context review

The coordinator has inspected `probe_original_copy.py`, the frozen observational
trace/axiom parser it copies, the eleven original runtime-image references, and
the successful independent delivery of all 4,527 selected SDK images. The new
private files have the ORIGINAL digests; this does not learn replacement pins.
The new probe uses only `Init`, has no project imports, and has no path that
opens the installed SDK. The six original system images are checked before and
after use. The loader command, closed environment, finite diagnostic policy,
exact source and exact theorem root are retained before launch.

This review authorizes ONE coordinator-only `basic-kernel-probe-v1` invocation
against those private original images. New unknown diagnostics, mismatched
images, timeout, nonzero exit, or observer/audit rejection stop this profile and
retain the first failure. No unchanged retry or widened acceptance policy.
The probe is a new context; it neither retries nor repairs an old failed profile.
It is not graph qualification, a worker-context authorization, a proof of native
fidelity, or a proof of OS isolation. Kernel/OS, loader reporting and host
association remain explicit trust. A later graph context needs its own complete
source/import/object binding and review even if this basic probe succeeds.
