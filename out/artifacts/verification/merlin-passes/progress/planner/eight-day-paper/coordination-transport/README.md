# Current instructions must survive compaction

Observed2026-10-23 about22:26UTC. User explicitly requested continuous worker coordination,
then specifically corrected TorchAO obeying its historical stop instead of current tasks.

The delegation tool appends a standalone function-call output. It does not replace the
latest direct user message. Inspection of frontend's persisted message roles found its
last substantive direct user request was the Oct22 stop. After compaction at22:13UTC,
the worker repeatedly acknowledged that old request despite coordinator continuations.
Its22:18checkpoint commits5eb197305/89f4411ca preserve compiled scalar/tensor work;
the stopped final is historical, not an authorization to abandon the current assignment.

The existing app-server's normal `turn/start` input accepts a user instruction. Under the
user's explicit delegation, root relayed the resume as such, identifying it honestly as
coordinator-relayed authorization and preserving parent/root turn attribution. Existing
thread, model, permissions, server and checkout were retained. No new agent was created.
Official [protocol documentation](https://learn.chatgpt.com/docs/app-server) distinguishes
normal input from toolOutput and specifies WebSocket transport over the local Unix socket.

Successful frontend resume: [record](frontend-resume-v1.json), turn
01a16ade-ad9f-7681-b8ed-dd77215a6234. Root subsequently observed actual commands reading the
current assignment and source execution modules in the intended checkout. This is actual
resumption, not merely an active flag or an acknowledgement. The [helper](resume_frontend.py)
refuses an already-written record and refuses to start an unexpectedly active thread.

W8A8 also drifted into an old planner-handoff final. Root applied the same authorized fix:
[record](w8a8-resume-v1.json), turn01a16adf-2d85-70b1-950b-a8c75f38f40f. Its next work remains
denominator/quotient bounds and actual attention SSA, not another planner handoff. Quant-ext
was already executing qualification and was not restarted or sent this unrelated issue.

Quant-ext subsequently requested its next task after sealing. Root delivered that own-task
continuation with NORMAL turn/steer, guarded by the exact active turn ID:
[record](quant-ext-next-v1.json), [collector](queue_quant_ext.py). Its running29-replay batch
is preserved; initialized-helper-program work activates only after the immutable seal.
Thus all three conversations now have current continuation recorded as actual user input.

Read-only diagnostic failures before the fix: both installed0.163and managed0.164 proxy
attempts timed out at initialize with zero response bytes; no turn-start mutation was
attempted in those connections. Raw newline JSON on the Unix socket closed without a
response. Correct WebSocket Upgrade/framing then passed a metadata-only read before the
single successful frontend start. No server restart, authentication change, credential
read or global configuration change was needed. These transport observations say nothing
about the separate retained verification runtime failures or their causes.

After future compaction, inspect actual latest user instructions and live worker progress.
A NEW user stop wins. Do not loop on ineffective tool-output resume messages or blindly
repeat a turn-start after a timeout: inspect the actual thread/record first. No model or
reasoning overrides. This fixes instruction delivery; it does not qualify any proof.
