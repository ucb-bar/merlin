# Original graph evidence packets for independent review

All 8,522 indexed repository artifacts from five historical packets were reopened
as stable regular files and matched to their ORIGINAL byte counts and SHA-256
pins. Five new transport archives were written and independently read back
member by member. No original was changed, no SDK was read and no qualification
or native command was rerun. `DELIVERY.json` records original index identities,
exact archive identities and limits.

The archives are LOCAL, under:

`out/build/graph-paper-deadline/original-review-packets-v1/`

| Archive | Original indexed files | Compressed bytes |
| --- | --- | --- |
| shape-originals.tar.gz | 1,001 | 10,762,591 |
| integer-originals.tar.gz | 2,250 | 29,782,540 |
| pipeline-originals.tar.gz | 4,001 | 82,556,434 |
| memory-originals.tar.gz | 1,061 | 12,836,842 |
| declared-originals.tar.gz | 209 | 1,331,955 |

Each archive also includes its original index/manifest if not already indexed.
Members keep repository-relative names. New archive digests certify this
transport, while member digests remain the old original pins. The archived
packets contain their original source snapshots, queries, objects, raw test/
checker/audit output and replay evidence to the extent covered by each index.
This is not an assertion that all transitive external dependencies or every
paper-cited receipt outside the indexes were included.

To hand off: copy the selected archives and this `DELIVERY.json` to Graham or
the laptop, verify the listed archive digests, and inspect/extract into a NEW
empty review directory. Do not extract over the shared development checkout.
Check each packet's original manifest and inspect its raw observations using
the human review guide. Archive contents must be treated as data until the
native context has been independently selected and reviewed.

These large generated archives are intentionally separate from source commits;
publication of this small delivery record does not transmit the archives.
`independent_reviewer_received` remains false. This prepares the missing local
files for D1-G03 review; it does not close independent delivery/reproduction.
Absolute Linux paths inside old receipts are preserved, not rewritten. The
archives do not provide a relocated replay, parser/SDK installation, archive
authenticity proof, native conformance result, or expanded correctness claim.

Producer baseline remains4499eebb9; qualified source versions are the versions
inside the packets. The current gate/admission/actual-after fixes are NEW and
are not part of these historical qualifications.
