# FireSim queue cross-user deploy paths

This thin patch selects the reviewed queue source whose SHA-256 is
`af721197a5f6a5ffd0a2a46da2555c52026c2f2f97eafc84c5dcd976406d256a`.
Its external repository revision is unavailable. The patch SHA-256 is
`bc32f2c27e4e9b520ad5fa4e84cdbeb8d6576c1095a814227d1c5b5c880505bf`;
the resulting source SHA-256 is
`2caf88d819c5c58bd84f2c8dcc3cd8d722f1c9bb5b28a60bb7326c469cdc6c42`.
Apply only after explicit operator review, with `patch --batch --fuzz=0 -p1`.
No installation, daemon restart or hardware operation is part of this recipe.

The public FireSim manager at revision
`b084672c2f8cf32e55d78f73a001074c23f8a2b8` sets its working directory from
the invoked script path. Its platform list, platform script transfers and
unchanged `check_script()` comparisons also resolve `../platforms` relative
to that deployment directory. In the queue's cross-user path, a bare PATH
lookup selected the original CLI and bypassed private writers. Selecting the
overlay CLI exposed a second failure: its platform sibling was missing.

The correction invokes the explicit overlay `firesim` path when that overlay
is selected, and links its platform sibling to the original resolved source
directory. A missing source, foreign sibling link or pre-existing directory
refuses. The source, job and slot memberships remain the original selections.
The ordinary constructor also writes `generated-topology-diagrams`. That
directory joins logs, results and workloads as a private writer; an existing
source diagram directory is never aliased as a writable product.
Same-user dispatch retains its existing bare CLI selection. The lifecycle
lock, private inputs, consumption checks and mandatory teardown are unchanged.
This is path correspondence, not isolation or immutable source authority.

## Controls

```sh
MERLIN_TEST_QUEUE_DEPLOY_SOURCE=/explicit/reviewed/queue.py \
MERLIN_TEST_QUEUE_PUBLIC_MANAGER_DEPLOY=/explicit/public/manager/deploy \
MERLIN_TEST_FIRESIM_MANAGER_PYTHON=/explicit/manager/python \
python -m pytest -q build_tools/upstreams/firesim_queue/test_deployment_overlay.py
```

The controls reject changed source, test exact sibling membership and private
writers, reject unsupported siblings and execute real disposable fake-manager
phases under the existing flock. A distinct refused PATH launcher detects
fallback; every child must execute the overlay CLI in its private directory.
An explicitly selected actual public manager runs only `--help` to check its
ordinary import/path behavior. Its ordinary `RuntimeConfig` constructor also
executes its native Graphviz writer in the private directory and checks that
the original source diagram bytes stay unchanged. Missing selections skip
honestly. Replay the
existing committed-input and original neutral HWDB lifecycle controls too.
The committed-input fixture applies both reviewed patches and checks the final
source digest, so its original assertions exercise the deployed queue version.

No permission helper, privileged utility, remote connection or hardware is
executed. Dependency/source builds, device loading, clock/reset, observer and
effect integrity, descendant cleanup and physical timers remain unqualified.
