# Unqualified HTIF transport experiment

`htif1120.patch` preserves the complete source and host controls from helper
commit `11678f79561ddf0b417ba815906075bbd78228ef`. It is deliberately unapplied.
The selected runtime's `htif.c/h` stay byte-identical.

The patch batches complete output lines into 1120-byte buffers. Twenty-two host
controls passed and the explicit new-runtime diagnostic compared all 6400
integer outputs exactly. Its complete trace still exceeded the unchanged 64 MiB
ceiling, so no new source-owner or corpus-writer admission followed.

The original runtime contract binds source hashes, including these two files.
Applying the patch, even with a 256-byte default, changes that identity and makes
fresh setup under the old contract refuse. A future runtime release must select
its own implementation, recipe and contract and obtain fresh qualification.
The old baseline's numerical and timing qualifications do not transfer.

This retained patch supplies an experiment input, not an enabled runtime option
or a latency improvement. `experiment.json` pins its source and scope.
