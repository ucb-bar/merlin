"""Actual native private-file and network-denial launch observation.

This fixed preflight retains the original reachable-listener and exact marker
checks. Its return is observation data only; the admitted launch owner still
controls runtime/readiness qualification. No saved result can mint authority.
"""

import shlex
import socket
import subprocess

from merlin.common.digest import sha256_bytes
from merlin_experiments.phase1.providers import codex_agent as CA

from . import contracts as C


def probe_private_denial(inputs, home, private_canary, *, timeout_s):
    from .component_launch import _control_command, _runtime_binds

    # A live reachable host TCP listener distinguishes actual network
    # denial from a harmless connection-refused on an absent service.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        controls = [
            str(private_canary),
            "/perf-control/private-golden-canary.json",
            "/perf-control/unadmitted-sibling.json",
            str(home / "auth.json"),
        ]
        script = (
            "import pathlib,socket\n"
            f"for path in {controls!r}:\n"
            " try: pathlib.Path(path).read_bytes()\n"
            " except (OSError,PermissionError): pass\n"
            " else: raise SystemExit('private sibling or control credential readable')\n"
            "client=socket.socket(socket.AF_INET,socket.SOCK_STREAM);client.settimeout(2)\n"
            f"try: client.connect(('127.0.0.1',{port}))\n"
            "except OSError: pass\n"
            "else: raise SystemExit('candidate TCP was not denied')\n"
            "print('COMPONENT_PRIVATE_FILES_AND_NETWORK_DENIED')\n"
        )
        native = shlex.join(
            (
                str(inputs.codex_binary),
                "sandbox",
                "--permission-profile",
                CA._CANDIDATE_PERMISSION_PROFILE,
                "-C",
                str(inputs.candidate),
                "--",
                "/usr/bin/python3",
                "-c",
                script,
            )
        )
        payload = _control_command(inputs, native, inputs.candidate, {}, extra_binds=_runtime_binds(inputs, home))
        denied = subprocess.run(["/bin/sh", "-c", payload], capture_output=True, timeout=timeout_s, check=False)
        if denied.returncode != 0 or denied.stdout.strip() != b"COMPONENT_PRIVATE_FILES_AND_NETWORK_DENIED":
            raise C.StageGateError("component actual private sibling/credential/network denial probe failed")
        return {
            "capability": "private_files_and_network_denial",
            "returncode": denied.returncode,
            "stdout_sha256": sha256_bytes(denied.stdout),
            "stderr_sha256": sha256_bytes(denied.stderr),
        }
