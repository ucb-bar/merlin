# Optional local public manager transport

Source: FireSim public commit `b084672c2f8cf32e55d78f73a001074c23f8a2b8`.
`source-pins.json` pins every existing file changed by the patch. The patch is
explicit operator infrastructure; its presence is not a compiler input, tool
grant, runtime issuer or installation default.

## Selection and scope

Apply with `patch --batch --fuzz=0 -p1` in a fresh selection of the exact public
revision after checking the source pins. An externally provisioned host spec
may explicitly select:

```yaml
command_transport: local
local_command_environment:
  PATH: /usr/bin:/bin
  HOME: /absolute/operator-owned/empty-directory
  LANG: C
local_command_timeout_seconds: 600
```

The selected host must be literal `localhost`. Bind the transport through the
normal `RuntimeConfig` constructor and its actual run-farm/host membership.
Absent selection delegates to the original Fabric SSH functions and does not
enumerate additional run-farm state. This preserves cloud and remote defaults.
No global monkeypatch, SSH agent, user key or implicit inherited process
environment is part of the local route. Commands use Bash without profile or
rc files. Explicit shell contexts still apply; this is trusted operator code,
not an execution sandbox or arbitrary compiler permission.

The supported slice is noninteractive commands, ordinary single-file get/put
and stock local rsync, including direction and trailing-slash behavior. PTYs,
stdin/stream callbacks, privileged file-copy options, remote addresses and
unsupported rsync options refuse. Shell output is UTF-8. Direct-process
timeouts do not qualify escaped descendant cleanup, OS hard deadlines or a
resource lease. Native rsync, interpreter/library origins and privileged
utilities require their own explicit selection and checks.

Stock `check_script()` remains unchanged and compares original source bytes.
Host, slot, job and renamed-ELF membership is unchanged. No hardware utility,
SSH session, FPGA programming or workload execution is exercised by these
controls; ordinary public instance liveness is tested through unprivileged
local `uname` and shell commands.

## Replay

Select the public checkout, native Python with the original manager's public
dependencies, and an owned independently source-bound manager configuration
fixture. The fixture supplies `config_runtime.yaml`, `selected-local-recipe.yaml`,
`config_hwdb.yaml`, `config_build_recipes.yaml` and `workloads/`. It is not an
author seed or a generalization/performance workload.

```sh
MERLIN_TEST_PUBLIC_FIRESIM_CHECKOUT=/explicit/public/source \
MERLIN_TEST_FIRESIM_MANAGER_PYTHON=/explicit/native/python \
MERLIN_TEST_MANAGER_CONFIG_DEPLOY=/explicit/owned/config/deploy \
python -m pytest -q build_tools/upstreams/firesim_manager/test_local_transport.py
```

Tests derive exact public Git blobs, reject changed/preexisting patch inputs,
and run 30 actual small command/file/manager assertions. They cover the genuine
stock source mismatch, refused remote/alias hosts, mutated live host selection,
unselected environment/terminal/transfer semantics and timeout. The default
SSH dispatch assertion intercepts that API to avoid authentication; local
commands and transfers execute normally. Missing explicit selections skip
honestly and cannot establish native readiness. Store receipts under `out/`.

Clock/reset/loading/observer/runtime-effect and physical-timer authority remain
outside this transport's scope.
