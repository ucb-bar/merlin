# Independent components and retained research tests

`merlin-timing/` remains an independently owned observer distribution and uses the shared
extension setup hook. Its source and measurement authority are separate from the compiler.
The four former research distributions retain tests and supporting study inputs here;
their active Python implementations now live under `src/` and ship only in `merlin`.
Do not recreate code wheels or shared namespace copies for those workflows.

Preserve existing test, access-role and frozen source identities. Build products belong under
`out/build/python/`. Validate actual wheel/sdist bytes and installed origins outside checkout.
