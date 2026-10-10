# `packages/merlin-experiments/src/merlin_experiments/phase1/feedback`

Host-only grading, redaction, certificate promotion and simulator dispatch live here.
Inputs are explicit invocation context and corpus/contract paths, never native imports.
Keep actual adapter construction, observation order, missing-oracle behavior and
enqueue-time certificate attribution unchanged. Public clients remain separate.
Dispatch owns only the closed async simulator policy shared by broker and promotion.
Source moves invalidate new implementation identities, never rewrite old evidence.

`private_facts.py` selects the private roster's exact pinned RTL extraction for
its model builds, after checking that the public effective view names the same
verified FIRRTL target/config. It restores the public environment afterward;
distinct input roles do not imply equivalent host/SDK facts or compiled semantics.

`private_capture_roster.py` binds every single- or multi-program capture stage to
the ordinary root contract and receipts. It refuses missing, extra, opaque or
indirect stage inputs, and reports only the attested loader's declared input
provenance without private paths. It neither selects a validation subset nor certifies code.

`caller_layout.py` gives an answer-free, layout-only projection from an explicitly
selected harness provider for a submission-owned command buffer. It never grades or
imports candidate code; absent provider inspection support refuses.
`native_output_readback.py` privately reuses that projection to bind output strides,
physical words and writable ELF symbols before decoding an exact coherent terminal
dump or a Spike HTIF signature whose output symbols exactly tile the alias-bounded
physical window in ELF-address order. The separate fixed
ET_EXEC and signature-bound preflight APIs must run before a native launch; both
decoders independently recheck their preflight records. It does not select a
transport, run a simulator, check normal exit/DONE, compare a golden or grade;
the eventual consumer must bind its run/receipt and keep full-value checks mandatory.
`native_memory_readback.py` supplies the one-use trusted hook for an explicitly
selected backend transport. It stages a fresh command-buffer copy and run-owned
artifact paths, performs all ELF/alias preflight before launch, then rechecks
source, request and output bytes; core retains exit/DONE, receipt and grade authority.
Each invocation uses a fresh retained attempt directory even when two tiers share
the same capsule build directory. `native_packet_readback.py` independently binds
the optional packed arena/publication symbols and complete typed logical frames
to those same source/ELF pins. It never treats packet DONE as native completion
or claims physical-padding, numerical equivalence or a certificate.

`private_prebuilt_receipt.py` admits only diagnostic inspection of an existing
whole-model build. The shared post-build verifier still checks linked bytes and
source obligations, but an old receipt without exact producer/toolchain closure
cannot become a full-roster gate result or a build-cache hit.

The automatic private build caller accepts an explicitly supplied live
`LinkedElfAdmissionService`. `private_linked_elf_selection.py` freezes and
reopens its target, callback and source selection, and requires the exact same
service on the planned `DeviceRouting` before and after the ordinary saved-model
build. An active route without that selection refuses. This supplies no default
policy, native decoder or instruction/effect/runtime authority. The formal
coordinator currently supplies no issued instruction service: independently
selected public command/predicate/accessor intakes, protected source-symbol
prohibitions and an explicit decoder selection are still required. Diagnostic
prebuilt inspection cannot establish actual final-image policy evaluation.

The private complete-model build gate consumes the producer-bound completed
compilation recipe and independently rehashes its explicit compiler/link inputs
and final ELF. Missing historical recipes remain diagnostic-only, never newly
certified. The v11 gate additionally requires an empty-or-complete
`private_linkage_support.py` source roster. A reviewed f32 sine/cosine declaration
with a closed linkage contract records only a pending source requirement; the
consumer reselects host package, GCC, ISA flags and archive bytes independently
before rechecking the actual link's defining-supplier trace against the same
source/candidate/capture/ELF identity. It never infers source-call routing,
transitive toolchain closure, frontend/executable numerical equivalence, or a
host numerical grant from the trace or ignored metadata. Opt-in closed static
composite math forms additionally retain exact per-ordinal literal-bit and
intrinsic-obligation fields (including explicit null fields for older forms).
Pow/tanh source forms require an independently selected `powf`/`tanhf` archive
supplier in the same linked image; this is not source-call or numerical proof.
Historical v10 gate records lack the mandatory fields and cannot be upgraded.
Composite forms additionally reparse their exact raw and normalized captured
stage at link and final claim, verify the selected capture tree and receipt
bytes, and compare each source ordinal with the stored literals and intrinsic
roster. A missing or changed source is a refusal, not a numerical waiver.

`private_source_freeze.py` owns host-only run copies of authored software and
host-capability inputs selected by a verified Phase 0 derivation export. Bind
original path, semantic role, digest and archive owner; distinct archives may
share one identical byte copy, but conflicting byte pins refuse. Fresh formal
completion requires the run-owned record, workspace snapshot and private masks.
Historical direct-path reads remain diagnostic only. The copy proves selected
source identity, not host-operation correctness or whole-model execution.

`private_control_support.py` proves typed source intervals for a narrow
mask-count assertion chain, its single-consumer comparison predicates, exact
bounded mask-compaction/scatter cursors, and a separately closed internal
Boolean compact-then-cast chain. The latter binds count, allocation, ordered
cursor writes, shape-preserving cast and worst-case byte span without admitting
the separate reduction/cast arithmetic or dynamic external output ABI. The v9
gate requires its empty-or-complete internal roster; historical v8 results do
not acquire this proof. Linked-image identity does not prove compiled semantic
equivalence or numerical correctness. Its index width is a caller-supplied
premise, not selected compiler evidence. The full-model consumer may discharge
only exact proven source ordinals after binding the same producer-owned compiler
observation to the linked build. All other index arithmetic and host math retain
their separate admissions. `private_source_support_join.py` requires all
source-only proofs to cite the same exact linked program; it does not infer
numerical equivalence or preserve arbitrary assertion abort paths.

`private_pointwise_support.py` records exact normalized source ordinals only
when a reviewed host declaration opts into a typed static pointwise body.
It rechecks each parsed operation and joins the same linked whole-program build;
this source witness does not establish numerical equivalence.
`private_linalg_support.py` is the current versioned, mandatory empty-or-
populated witness for exact pointwise, Boolean and unary f32 sine/cosine source-body declarations.
Its opt-in projected-pointwise schema independently rechecks every static input
shape and singleton-projection map, with selected signed-index per-tensor byte
bounds; the older identity-only schema stays strict. Neither schema creates a
host admission or proves executable numerical behavior.
Its separate dynamic Boolean cast schema requires the same ordinal in the
closed internal-compaction proof and bounds its allocation by the selected
signed index width. Every body is rechecked against parsed source and joined
to the same linked candidate, capture and ELF. This is per-tensor source/build
evidence, not a global arena proof; it grants no host rule, external dynamic
output ABI, compiled semantics or PyTorch numerical equivalence.
`private_device_audit.py` also owns exact static board/DTS input checks for the
linked image; that check remains build-only and does not imply board execution.

`private_literal_arange.py` is a grant-none witness for fixed prepared i64
range literals and their closed typed source lowering. It records original
graph ancestry without claiming original-to-prepared equivalence; index width
is an explicit premise until separately bound to the linked compiler build.
`private_literal_arange_admission.py` requires that witness for each reviewed
host-admitted prepared range, binds the selected compiler index observation,
and joins the exact candidate, capture and ELF bytes after whole-program build.
It never creates a host declaration or proves numerical equivalence.

`private_integer_reduction_support.py` rechecks the exact source bodies of
reviewed integer sum, prefix-sum and paired minimum admissions. Its selected
index-width premise must match the producer-owned linked build observation.
An opt-in closed integer-reduction `source_body` admission is routed here, not
to the pointwise Linalg witness; its declaration/profile/context and every
parsed ordinal pattern are rechecked before the mandatory linked-build join.
`private_f32_maximum_support.py` similarly requires a separate reviewed host
decision before accepting a closed prepared maximumf reduction. It reparses
every selected source ordinal and the capture tree after build, and binds the
selected index observation plus candidate/capture/ELF bytes. It neither grants
host placement nor equates signed-zero or reduction-order behavior with PyTorch.
`private_index_source.py` is a grant-none diagnostic for two independent
prepared tensor-index forms: an i1 mask's closed i64 count, and an indexed
extract whose literal range/offset extrema are in bounds under a caller-supplied
index-width premise. It verifies receipt, trace, parsed SSA and static maps,
but does not associate distinct prepared index nodes, admit their host lowering,
or prove compiler execution or original-frontend numerical equivalence.
`private_index_host_support.py` consumes that versioned proof for exact
reviewed compute-root admissions. It redoes the source/trace/receipt proof,
requires every root to have its own matching typed host declaration, and
joins the selected compiler index observation and candidate/capture/ELF.
`private_host_source_dispatch.py` routes only recognized source-body schemas
to their mandatory owning witnesses; unknown bodies keep the old refusal.
`private_ordered_scan_support.py` independently checks a closed prepared f32
prefix scan with f64 carry, ordered loop/lane reset and exact source ancestry.
The v7 gate requires its empty-or-populated reviewed semantic tensor-root
admission roster, independently selected hardware exclusion, and exact proved
body-component ordinals before the selected-width/linked-image join; v6 records
cannot be upgraded. This does not prove original-to-prepared equivalence or
numerical correctness of the compiled loop.
`private_bucketize_support.py` joins the source/trace roster, proves literal
boundary ordering and requires an existing reviewed host placement for every
occurrence. An opted-in bucketize host source-body proof must exactly agree
with that independent right/literal/index witness; link and completion re-read
the selected capture tree, model, trace and receipt. Dynamic boundary ordering
remains unproved. The v13 private gate requires the empty-or-populated bucketize
record and binds the exact
candidate, capture and ELF. Neither proves frontend or executable numerical
equivalence, creates a host rule, or upgrades historical v3 gate records.

`rtlchecks.py` owns strict authored treatment selection, redaction and advisory
round/checkpoint RTL feedback. Its factory captures the explicit invocation context;
callbacks receive the actual graded public roots, including frozen promoted roots at
checkpoint. Never rediscover a live canonical corpus or select Chipyard at import.
Structural failures/errors do not change numerical pass/fail. The feedback subtree
already carries grader-only access identity; core RTL policy owners are explicit
members of the existing implementation source inventory.

`loop_grading.py` owns the authoring snapshot, language gate, numerical-grade call,
stage ledger, shape gate, promotion and verdict publication in their existing order.
GradingInputs keeps invocation roots separate; only the native diagnostic edge may
defer public-root materialization until after the submission gate. Certification
policy similarly resolves public roots only after finding additional cert tiers.
Keep archive-before-shape and publish-after-promotion ordering unchanged.

`codegen_scalability.py` observes public contractions at geometric multiples of
the selected fact-derived tile. It reads no validation capture or answer; the
ordinary round feedback records emitted text size, not executed work or a
correctness verdict. Size ratios never prescribe unrolling or reject smaller
looped code. Its separately selected pure build-only capability may compile the
exact emitted LLVM artifact within a bounded diagnostic budget and record direct
tool/source pins, object bytes and status. Default round feedback stays emit-only;
neither compiled objects nor their size/time grant execution, numerical or
certification evidence. Full-model and native numerical gates remain separate.

`brief.py` constructs and publishes the round brief from already-redacted round
verdicts, the candidate's notes and explicit operator errata. Preserve prompt bytes,
notes-staleness stamps, recognized resume prefixes and pre-launch refresh timing.
Refresh must not advance the notes stamp. This owner does not grade, inspect hidden
corpora or independently sanitize its trusted redacted inputs. Only its generated
brief is candidate-visible; the implementation remains in the private feedback tree.

`lifecycle.py` owns client staging, broker child lifetimes, the background-grading
cadence/single-flight handoff and durable channel-health accounting. BrokerConfig
supplies invocation context, resolved treatment tools, explicit timing storage and
distinct public/policy/schema roots; GradeCadence supplies only scheduler settings.
Grade callbacks are mandatory and contain the caller's existing numerical policy.
Background grading surrounds the agent launch; brokers live inside that launch.
Stop brokers first, then join the grader without a timeout before the authoritative
grade. Partial startup unwinds already-created children and closes parent log
handles. Shutdown attempts all owned processes despite signal/wait/kill failures;
after the 15-second graceful wait, forced stops use a bounded reap. Cleanup errors
are aggregated with unreaped PIDs. A propagating provider or startup exception
remains primary, with cleanup notes; otherwise cleanup failure raises. This owns
direct broker processes, not arbitrary detached descendants, and does not qualify
the full installed engine or unrelated orchestration cancellation.

`formal.py` owns public grade, iteration evidence, freeze/re-hash, hidden grade and
formal completion. `freeze.py` owns the existing freeze.json record using the shared
tree-hash and explicit repository identity. Preserve mandatory L3 simulator selection
even for no-oracle diagnostics. Only synthetic external execution is substituted in
installed lifecycle tests; those tier records are not hardware qualification.
