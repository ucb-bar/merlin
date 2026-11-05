"""Explicit optional container commands with an immutable native PID-1 owner.

Uses an existing authorized service. Never starts/configures a daemon, pulls an
image, grants its socket, loads credentials or accepts PASS JSON as authority.
This diagnostic transport does not qualify fresh authoring, independent runtime
roles or performance. Service implementation/failure and OS hard deadlines stay
unknown; the native owner's successful ECHILD observation scopes normal cleanup.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from merlin.common import invocation_record
from merlin_experiments.phase2.contracts import StageGateError

from .container_image import image_root_masks, verify_image_export
from .container_policy import ContainerMount as ContainerMount
from .container_policy import _verify_created_container, selected_environment, verify_owned_container
from .container_policy import mount_from_source as mount_from_source
from .container_policy import mounts_from_strict_policy as mounts_from_strict_policy

_PREPARED = {}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class PreparedContainerTransport:
    client: Path
    client_sha256: str
    endpoint: str
    configuration: Path
    configuration_sha256: str
    image_archive: Path
    image_archive_sha256: str
    guardian: Path
    guardian_sha256: str
    guardian_build: Path
    _issuer: object = field(repr=False, compare=False)

    @property
    def source_pins(self):
        """Reopened command owner, production record and explicit data/tool closure."""
        self.verify()
        build = invocation_record.verify(self.guardian_build)
        files = {
            self.client,
            self.configuration,
            self.image_archive,
            self.guardian,
            self.guardian_build,
            Path(__file__),
            Path(__file__).with_name("container_image.py"),
            Path(__file__).with_name("container_policy.py"),
            Path(invocation_record.__file__),
        }
        files.update(Path(row["path"]) for row in [build["executable"], *build["inputs"], *build["dependencies"]])
        return tuple((path.resolve(), _sha(path)) for path in sorted(files))

    @property
    def sha256(self):
        """Selection identity only; never an isolation/runtime qualification receipt."""
        binding = {
            "scope": "explicit compiler command transport; no author/runtime/physical authority",
            "endpoint": self.endpoint,
            "client": str(self.client),
            "guardian": str(self.guardian),
            "configuration": str(self.configuration),
            "archive": str(self.image_archive),
            "source_pins": [(str(path), digest) for path, digest in self.source_pins],
        }
        return hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def verify(self):
        if _PREPARED.get(self._issuer) is not self:
            raise StageGateError(
                "container transport was not prepared around actual original tools and native owner build"
            )
        if (
            not self.client.is_absolute()
            or not os.access(self.client, os.X_OK)
            or _sha(self.client) != self.client_sha256
            or _sha(self.guardian) != self.guardian_sha256
            or not self.endpoint.startswith("unix:///")
            or any(char in self.endpoint for char in "\n\r\x00")
        ):
            raise StageGateError("explicit container service/client/guardian selection changed")
        original_source = Path(__file__).with_name("container_guardian.c").resolve()
        record = invocation_record.verify(self.guardian_build)
        if (
            record.get("status") != "completed"
            or record.get("kind") != "subprocess"
            or {row["path"] for row in record["inputs"]} != {str(original_source)}
            or not any(
                row["path"] == str(self.guardian.resolve()) and row["sha256"] == self.guardian_sha256
                for row in record.get("outputs", [])
            )
            or "-static" not in record["argv"]
        ):
            raise StageGateError("container guardian has no reopened original native build")
        closure = verify_image_export(
            configuration=self.configuration, archive=self.image_archive, config_sha256=self.configuration_sha256
        )
        if closure["archive_sha256"] != self.image_archive_sha256:
            raise StageGateError("selected container filesystem export changed")
        return closure

    def execute(
        self,
        *,
        mounts: tuple[ContainerMount, ...],
        command: tuple[str, ...],
        cwd: str,
        evidence_root: Path,
        timeout_s: float = 30,
        environment: tuple[tuple[str, str], ...] | None = None,
        namespace_masks: tuple[str, ...] = (),
    ) -> subprocess.CompletedProcess:
        """Run a disposable command; original admission remains caller-owned.

        This helper cannot grant compiler/runtime roles. Unknown native cleanup
        is a refusal even when the delegated service reports a stopped container.
        No candidate rollback, native-service lease release or OS hard deadline.
        """
        from merlin_experiments.phase2.external_resources import active_resource_boundary

        if active_resource_boundary() is not None:
            raise StageGateError("hard resource boundary cannot admit an unbound external container executor")
        closure = self.verify()
        effective_environment = None
        if environment is not None:
            original = json.loads(self.configuration.read_bytes()).get("config", {}).get("Env", [])
            if type(original) is not list or any(type(value) is not str or "=" not in value for value in original):
                raise StageGateError("original selected image environment is unsupported")
            inherited = selected_environment(tuple(tuple(value.split("=", 1)) for value in original))
            effective_environment = tuple((inherited | selected_environment(environment)).items())
        masks = image_root_masks(self.image_archive)
        if (
            type(namespace_masks) is not tuple
            or len(namespace_masks) > 256
            or len(set(namespace_masks)) != len(namespace_masks)
            or any(
                type(value) is not str
                or not PurePosixPath(value).is_absolute()
                or str(PurePosixPath(value)) != value
                or ".." in PurePosixPath(value).parts
                or value in {"/", "/dev", "/proc", "/sys", "/tmp"}
                or any(c in value for c in ",\x00\n\r")
                for value in namespace_masks
            )
        ):
            raise StageGateError("container namespace masks require bounded canonical original destinations")
        masks = tuple(sorted(set((*masks, *namespace_masks))))
        if (
            not isinstance(mounts, tuple)
            or any(type(row) is not ContainerMount for row in mounts)
            or not command
            or any(type(value) is not str or "\x00" in value for value in command)
            or not math.isfinite(timeout_s)
            or not 0 < timeout_s <= 3600
        ):
            raise StageGateError("container command needs closed mounts/argv and a bounded budget")
        final_mounts = (*mounts, mount_from_source(self.guardian, "/usr/bin/merlin-command-owner", read_only=True))
        for row in final_mounts:
            row.verify()
        destinations = [row.destination for row in final_mounts]
        if set(namespace_masks) & set(destinations):
            raise StageGateError("original container mask conflicts with an explicit bind destination")
        if len(set(destinations)) != len(destinations) or cwd not in destinations:
            raise StageGateError("container destinations conflict or lack the original working directory")
        for writable in (row for row in final_mounts if not row.read_only):
            for other in final_mounts:
                if writable is not other and (
                    writable.source == other.source
                    or writable.source.is_relative_to(other.source)
                    or other.source.is_relative_to(writable.source)
                    or PurePosixPath(writable.destination).is_relative_to(other.destination)
                    or PurePosixPath(other.destination).is_relative_to(writable.destination)
                ):
                    raise StageGateError("writable container grant overlaps another original grant")
        evidence_root.mkdir(parents=True)
        config = evidence_root / "empty-client-config"
        config.mkdir()
        cid_file = evidence_root / "owned.cid"
        plan = evidence_root / "command-plan.json"
        plan.write_text(
            json.dumps(
                {
                    "image": closure,
                    "mounts": [row.__dict__ | {"source": str(row.source)} for row in final_mounts],
                    "command": command,
                    "cwd": cwd,
                    **({"selected_environment": environment} if environment is not None else {}),
                    **({"namespace_masks": namespace_masks} if namespace_masks else {}),
                },
                sort_keys=True,
            )
            + "\n"
        )
        base = [str(self.client), "--config", str(config), "--host", self.endpoint]
        dependencies = (
            self.client,
            self.guardian,
            Path(__file__),
            Path(__file__).with_name("container_image.py"),
            Path(__file__).with_name("container_policy.py"),
        )
        env = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}

        def run(stage, arguments, **kwargs):
            return invocation_record.run(
                [*base, *arguments],
                directory=evidence_root / stage,
                stage=stage,
                dependencies=dependencies,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s + 10,
                check=True,
                **kwargs,
            )

        token = uuid.uuid4().hex
        name = "merlin-owned-" + token
        args = [
            "create",
            "--name",
            name,
            "--label",
            "merlin.command-owner=" + token,
            "--cidfile",
            str(cid_file),
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges=true",
            "--pids-limit",
            "64",
            "--memory",
            "256m",
            "--cpus",
            "1",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--ipc",
            "private",
            "--cgroupns",
            "private",
            "--workdir",
            cwd,
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,noexec,size=64m",
            "--entrypoint",
            "/usr/bin/merlin-command-owner",
        ]
        # Image identity supplies an independently pinned minimal filesystem,
        # not an implicit grant of its tools/config/data. Only original explicit
        # file mounts reappear beneath these readonly, empty namespace scopes.
        for directory in masks:
            args += ["--tmpfs", directory + ":ro,nosuid,nodev,noexec,size=16m"]
        for row in final_mounts:
            args += [
                "--mount",
                f"type=bind,src={row.source},dst={row.destination},"
                "bind-propagation=rprivate,bind-recursive=disabled" + (",readonly" if row.read_only else ""),
            ]
        if environment is not None:
            for key, value in environment:
                args += ["--env", key + "=" + value]
        args += [closure["image_id"], str(timeout_s), "--", *command]
        receipt, native, container_id, failure, owned = {}, None, None, None, False
        try:
            run("container_create", args, inputs=(plan,), outputs=(cid_file,))
            original = run(
                "container_created_inspect",
                ["inspect", "--format", "{{json .}}", name],
                inputs=(cid_file, plan),
            )
            row = json.loads(original.stdout)
            container_id = verify_owned_container(row, image_id=closure["image_id"], name=name, token=token)
            owned = True
            if cid_file.read_text().strip() != container_id:
                raise StageGateError("service create output differs from exact owned container")
            _verify_created_container(
                row,
                image_id=closure["image_id"],
                mounts=final_mounts,
                command=command,
                cwd=cwd,
                timeout_s=timeout_s,
                masks=masks,
                environment=effective_environment,
            )
            result = run("container_command", ["start", "--attach", container_id], inputs=(cid_file, plan))
            if len(result.stdout) > 300000 or result.stderr:
                raise StageGateError("container owner returned unsupported output protocol")
            native = json.loads(result.stdout)
            fields = {
                "schema",
                "pid",
                "returncode",
                "timed_out",
                "output_exceeded",
                "reaped_after_command",
                "cleanup_wait_echild",
                "stdout_hex",
                "stderr_hex",
            }
            if (
                set(native) != fields
                or native["schema"] != "merlin.container_command.v1"
                or native["pid"] != 1
                or native["cleanup_wait_echild"] is not True
                or type(native["returncode"]) is not int
                or type(native["timed_out"]) is not bool
                or type(native["output_exceeded"]) is not bool
            ):
                raise StageGateError("container command lacks the original native owner/cleanup protocol")
            stdout, stderr = bytes.fromhex(native["stdout_hex"]), bytes.fromhex(native["stderr_hex"])
            for row in final_mounts:
                if row.read_only:
                    row.verify()
            self.verify()
            receipt["native_cleanup"] = "COMPLETE"
            if native["timed_out"] or native["output_exceeded"]:
                raise StageGateError("container command exceeded its bounded execution/output budget")
            return subprocess.CompletedProcess(command, native["returncode"], stdout, stderr)
        except BaseException as error:
            failure = error
            receipt["refusal"] = type(error).__name__ + ": " + str(error)[:500]
            raise
        finally:
            receipt.update(
                schema="merlin.container_transport_diagnostic.v1",
                container_id=container_id,
                native_result=native,
                native_cleanup=receipt.get("native_cleanup", "UNKNOWN"),
                delegated_service_authority="UNKNOWN",
                os_hard_deadline="UNKNOWN",
                compiler_runtime_physical_roles="UNISSUED",
            )
            if not owned:
                try:
                    recovered = run(
                        "container_partial_create_inspect", ["inspect", "--format", "{{json .}}", name], inputs=(plan,)
                    )
                    container_id = verify_owned_container(
                        json.loads(recovered.stdout), image_id=closure["image_id"], name=name, token=token
                    )
                    owned = True
                except Exception as error:
                    receipt["owned_container_identity"] = "UNKNOWN"
                    receipt["partial_create_diagnostic"] = type(error).__name__
            if owned:
                receipt["container_id"] = container_id
                try:
                    inspected = run(
                        "container_inspect",
                        ["inspect", "--format", "{{json .State}}", container_id],
                        inputs=(cid_file,),
                    )
                    state = json.loads(inspected.stdout)
                    receipt["observed_owned_state"] = state
                    if state["Running"]:
                        run("container_kill", ["kill", container_id], inputs=(cid_file,))
                    run("container_remove", ["rm", container_id], inputs=(cid_file,))
                    receipt["owned_container_removal"] = "OBSERVED"
                except Exception as error:
                    receipt["owned_container_removal"] = "UNKNOWN"
                    receipt["cleanup_error"] = type(error).__name__
                    if failure is None:
                        raise
                finally:
                    (evidence_root / "transport.json").write_text(json.dumps(receipt, sort_keys=True) + "\n")
            else:
                (evidence_root / "transport.json").write_text(json.dumps(receipt, sort_keys=True) + "\n")


def prepare_container_transport(
    *,
    client: Path,
    endpoint: str,
    configuration: Path,
    configuration_sha256: str,
    image_archive: Path,
    compiler: Path,
    evidence_root: Path,
) -> PreparedContainerTransport:
    """Build the original static command owner. No role or launch admission."""
    evidence_root.mkdir(parents=True)
    source = Path(__file__).with_name("container_guardian.c").resolve()
    guardian = evidence_root / "command-owner"
    result = invocation_record.run(
        [str(compiler.resolve()), "-static", "-O2", "-Wall", "-Werror", str(source), "-o", str(guardian)],
        directory=evidence_root / "build",
        stage="container_guardian_build",
        inputs=(source,),
        outputs=(guardian,),
        dependencies=(Path(__file__),),
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert result.returncode == 0
    records = list((evidence_root / "build").rglob("invocation.json"))
    if len(records) != 1:
        raise StageGateError("container owner native build observation is ambiguous")
    closure = verify_image_export(
        configuration=configuration, archive=image_archive, config_sha256=configuration_sha256
    )
    prepared = PreparedContainerTransport(
        client.resolve(),
        _sha(client),
        endpoint,
        configuration,
        configuration_sha256,
        image_archive,
        closure["archive_sha256"],
        guardian,
        _sha(guardian),
        records[0],
        object(),
    )
    _PREPARED[prepared._issuer] = prepared
    prepared.verify()
    return prepared
