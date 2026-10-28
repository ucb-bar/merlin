# TVM implementation cleanup verification

TVM cleanup successor: `2c0857ab88b18d4a090f2dd6e401e06266af1877`.
The default entry point no longer imports old Relax graph substitution; explicit
legacy imports remain compatible. The historical verifier now binds the moved
source file in its compiler identity. No schedules, intrinsic definitions or
runtime arithmetic were changed. Thirteen existing semantic definitions compare
identically as ASTs after excluding only the new compatibility imports.

Backend checks: **41 tests + 64 subtests pass**. Nineteen ONNX tests could not run
because the environment lacks ONNX; these are not passes. The initial refactor
check caught a missing DataTypeCode import in shared host inlining; that was fixed
before the passing run. Both logs are retained. The structure check retains the
same nine inherited size failures. Earlier Spike ELFs/results still bind bb4ad9f8d.

The branch inventory is read-only. Deletion of the two duplicate unused local
aliases `gemmini/bringup` and `gemmini/legacy-template` awaits user approval under
the supplied machine rules. Both equal retained `gemmini/byoc`. Other branches,
all worktrees (including dirty historical ones), remote refs and frozen artifacts
are preserved. See summary.json for the recorded cleanup state.
