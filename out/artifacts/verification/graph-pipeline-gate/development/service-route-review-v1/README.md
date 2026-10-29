# Independent service-route review v1 — immutable source cut

Source-snapshot holds EXACT12 owned source files from checkpoint-v9: CommonTransformService,
service binding/delivery module, gate/provider/matcher/emitter, tests and minimal Lean
adapter. MANIFEST records hashes/sizes. Actual transitive project source dependencies
are copied separately; external Python parser sources are references only. No SDK or
runtime image was opened/copied. All native permissions are suspended.

Independent source/API fixture: source-snapshot/primary/merlin/tests/ir/
test_lean_graph_pipeline_gate.py, class ServiceBoundary. Its setUp constructs actual
service with explicit mocked original-bundle/profile custody and typed mocked frozen
Context methods. It forbids legacy _toolchain, Context construction/runtime guards and
child processes. Actual binding, service.apply, gate adapter, pipeline and compiler_
eligible run. Select ServiceBoundary directly; do not rerun unrelated matrices.

Example includes actual input/before/after/query, marked MOCKED output and exact
source closure. Binding recipe and limits in example/README.md. Verify missing/saved/
copied injection, wrong workspace/options, public/context holds, second-stage refusal,
source/output joins and live eligibility. Profile/source digest matching is host
association, not native/human proof approval. Unchanged/no-change result does not
mint transformed-boundary authority. Cross-stage semantic bridge remains unproved.

Review does not authorize production activation or native execution. Full original
release snapshot/profile/import selection must still be reviewed by coordinator;
these were MOCKED for source-only unit execution. All old failure/capsule bytes remain
unchanged. This packet is immutable; later source/test changes need a new cut.
