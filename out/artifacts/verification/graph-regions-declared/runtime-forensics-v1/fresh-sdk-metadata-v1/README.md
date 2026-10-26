# Fresh official Lean intake alternative — metadata only

This alternative can establish a **new environment identity** without claiming
to diagnose or repair the old hash incidents. Root authorized read-only official
metadata/source lookup. No release archive was downloaded, no SDK/runtime image
was read, and nothing was installed or executed. All old holds remain active.

## Primary-source identity

The official release is stable `v4.35.0`, published 2026-10-12. Its tag resolves
directly to commit `eb6cafb7290e8bd5f2937522d8d804f58f02bafc`, source tree
`ac07366d159bbc30b396b654019fffcc14c98c02`. The commit API reports `unsigned`;
no signed-commit or publisher-signature claim is made. The release's mutable
`target_commitish: master` is not used as source identity.
Sources: [official release API](https://api.github.com/repos/leanprover/lean4/releases/tags/v4.35.0),
[exact tag API](https://api.github.com/repos/leanprover/lean4/git/ref/tags/v4.35.0),
[commit API](https://api.github.com/repos/leanprover/lean4/git/commits/eb6cafb7290e8bd5f2937522d8d804f58f02bafc).

The proposed asset is `lean-4.35.0-linux.zip`, asset ID 634574461, **925,899,528
bytes**, SHA256 `c8531faaa4a93d5a4dae0c742cbc1bd42a2993594d02f93a9d629ebf299fb2c8`.
These are published GitHub release-asset metadata, not a digest learned by
accepting suspect local bytes. The smaller tar.zst alternative is retained in
the raw metadata but is not selected; ZIP permits a future bounded stdlib
extractor without invoking an additional decompressor executable.
[Official asset metadata](https://api.github.com/repos/leanprover/lean4/releases/tags/v4.35.0).

Linux x86_64 is the platform interpretation supported by the pinned CI release
matrix's amd64 runner and x86_64 Linux LLVM input; the same pinned build template
creates the release ZIP. This is source-based platform evidence, not inspection
or execution of the asset. See retained source copies and the exact upstream
[CI matrix](https://github.com/leanprover/lean4/blob/eb6cafb7290e8bd5f2937522d8d804f58f02bafc/.github/workflows/ci.yml)
and [packaging template](https://github.com/leanprover/lean4/blob/eb6cafb7290e8bd5f2937522d8d804f58f02bafc/.github/workflows/build-template.yml).

The retained API bytes, URLs, response dates/ETags and hashes live in
`official-metadata-v1` and `official-source-metadata-v1`. The build source is
in `official-build-source-v1`. All network requests were bounded metadata/text
lookups. An initial web-tool API access failure and guessed `release.yml` 404
are recorded separately; the subsequent actual API directory listing identified
the existing CI/template paths. No asset request was made.

## Concrete proposed recovery stages

`NEW-IDENTITY.json` and `INTAKE-PLAN.json` record exact identity, roots and bounds.
They remain **unapproved, not launchable**. No downloader/extractor/native runner
was implemented or invoked by this metadata study.

1. Root reviews a new bounded fetch implementation and exact source cut. Proposed
   intake root is `out/build/graph-paper-deadline/fresh-official-lean435-v1`.
   One HTTPS GET, no resume/retry, at most three redirects restricted to GitHub
   and its release-assets host, 30-second connection timeout and 1,800-second
   total deadline. Retain partial bytes and the first error. Read at most the
   published 925,899,528 bytes plus one oversize sentinel. Before extraction,
   require the independently recorded publisher-platform SHA256 and exact size.
2. Under separate review, parse the verified ZIP as data and extract exclusively
   into a new scratch tree. Bound members to 50,000, each to 1 GiB, total expanded
   bytes to 16 GiB, and require 20 GiB free disk. Reject traversal, absolute/NUL
   paths, duplicate/colliding members, links/nonregular entries, encryption,
   unsupported compression, CRC/size failures and exceeded caps. These are
   conservative limits, not claims about uninspected archive contents. No scripts,
   executable files, installer, elan command or setup hook runs during intake.
3. Derive a NEW member manifest under the authenticated archive identity, keeping
   parent archive provenance for every member. These would be new-environment
   pins, not replacements for old pins. Independently review module/IR/runtime
   membership and layout before selecting any execution context. Do not reuse old
   objects or assume the old 4,527/11 closure describes this asset. External host
   loader/system-library dependencies also need a concrete fresh context review;
   none is authorized or inspected here.
4. Only after those stages and new explicit native permission, prepare an exact
   closed diagnostic source/runtime cut. Start with the three retained Nat/List/
   Omega microcase sources in a NEW context, at most 18 native commands and
   whole-batch stop on first error. No numerical work or graph acceptance. A
   passing diagnostic is prerequisite evidence, not full qualification. Actual
   graph roots, audits and fresh replay then require their own new evidence.

Changing the environment intentionally breaks the old qualification identity.
The original source/proof packets stay unchanged; any future claim must cite
the new environment and fresh checks. The old runtime failures remain UNKNOWN
even if the alternative succeeds. A complete fresh SDK may address a missing-IR
prerequisite, but this has not been established; hash incidents and Repr/codegen
dependencies remain distinct questions.

## Trust and tests

HTTPS/GitHub official release metadata supplies the proposed archive expectation;
it is not independent cryptographic publisher-signature verification or a proof
of reproducible builds. Extraction code, decompression, host IO, source transport,
later runtime/loader reporting and Lean remain explicit trust boundaries.
The archive's actual link layout, full expanded size, runtime closure, module
contents and suitability are unknown until separately authorized inspection.

Six offline metadata tests pass in `offline-tests-v1`: exact retained identity,
duplicate/missing or wrong-platform asset, tag/commit mismatch, missing/invalid
digest or unofficial URL, and size/type bounds. They do not fetch an archive,
install Lean, prove a theorem or qualify a graph. All old packets remain intact;
coordinator alone publishes or authorizes the next concrete action.
