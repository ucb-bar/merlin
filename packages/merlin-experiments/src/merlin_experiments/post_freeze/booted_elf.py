"""Import explicit lightweight boot-file observations without custody qualification.

The runtime host reports the file used by its simulator. This reproducibility
record is deliberately separate from the before/after, version-pinned manager
custody protocol. Neither record establishes complete model qualification.
"""

import posixpath
import shlex
from datetime import datetime, timedelta
from pathlib import PurePosixPath
from urllib.parse import urlsplit

from .elf_custody import identity, read

MODE = "booted_elf_hash_v1"
SCHEMA = "merlin.post_freeze.firesim.booted_elf_hash.v1"


def selected_mode(deployment):
    mode = deployment.get("provenance_mode")
    if mode not in (None, MODE):
        raise ValueError("unknown explicitly selected FireSim provenance mode")
    return mode == MODE


def _shell_words(text, depth=0):
    """Read nested shell -c strings as data; never expand or execute them."""
    if depth > 4 or len(text) > 1024 * 1024:
        raise ValueError("simulator script nesting/size exceeds the intake bound")
    lexer = shlex.shlex(text, posix=True, punctuation_chars=";&|()")
    lexer.whitespace_split = True
    words, previous = [], None
    for token in lexer:
        words.extend(_shell_words(token, depth + 1) if previous == "-c" else [token])
        previous = token
        if len(words) > 16384:
            raise ValueError("simulator script exceeds the token bound")
    return words


def _boot_path(record, script):
    argv = record.get("simulator_argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) and x and "\x00" not in x for x in argv):
        raise ValueError("runtime observation needs the actual simulator argv")
    values = [x.removeprefix("+prog0=") for x in argv if x.startswith("+prog0=")]
    words = _shell_words(script.read_text())
    if (
        len(values) != 1
        or not values[0]
        or sum(x.startswith("+prog0=") for x in words) != 1
        or sum(words[i : i + len(argv)] == argv for i in range(len(words))) != 1
    ):
        raise ValueError("sim-run.sh must contain exactly the reported single boot command")
    cwd = record.get("working_directory")
    consumed = record.get("consumed_elf", {}).get("path")
    if (
        not isinstance(cwd, str)
        or not PurePosixPath(cwd).is_absolute()
        or not isinstance(consumed, str)
        or not PurePosixPath(consumed).is_absolute()
        or any(c in values[0] + cwd + consumed for c in ("$", "`", "\x00"))
    ):
        raise ValueError("boot operand and runtime paths must be concrete")
    boot = posixpath.normpath(posixpath.join(cwd, values[0]))
    if boot != consumed or posixpath.normpath(cwd) != cwd:
        raise ValueError("reported consumed ELF is not the simulator boot operand at its cwd")
    return boot


def _upload(request, model, record, invocation, command_path, upload_path):
    # Use the existing exact versioned upload checks, without selecting the
    # strict manager protocol or manufacturing its manager-result document.
    from .firesim_upload import upload_identity

    deployment = request["deployment"]
    boot = [asset for asset in model["assets"] if asset["role"] == "boot_elf"]
    if len(boot) != 1:
        raise ValueError("one selected boot image is required")
    expected = identity(boot[0])
    from .firesim import load

    upload = load(upload_path)
    observed = {
        "elf_key": record["elf_key"],
        "consumed_elf": identity(record["consumed_elf"]),
        "uploaded_elf": {**expected, "version_id": upload.get("version_id")},
    }
    matches, evidence, reason = upload_identity(request, model, observed, invocation, command_path, upload_path)
    if (
        not matches
        or record["bucket"] != urlsplit(deployment["upload_prefix"]).netloc
        or record["ssm_command_id"] != invocation["CommandId"]
    ):
        raise ValueError(reason if not matches else "boot observation differs from selected SSM/upload")
    return evidence


def _model(request, record):
    expected = identity(record["consumed_elf"])
    matches = [
        model
        for model in request["models"]
        if any(asset["role"] == "boot_elf" and identity(asset) == expected for asset in model["assets"])
    ]
    if len(matches) != 1 or matches[0].get("transport") != "all_element_comparison_v1":
        raise ValueError("boot observation does not uniquely select a compact complete-chain image")
    return matches[0]


def collect(request_path, request, result_dir, invocation_path, command_path, upload_path):
    from .compact_chain import collect as compact_collect
    from .firesim import RESULT_SCHEMA, pin, reopen, verify_request

    artifacts, failures, compact = {}, [], None
    hardware_matches = uploaded_matches = binding = False
    record, model = None, None
    for name in ("booted-elf.json", "uartlog", "HW_CFG_SUMMARY", "sim-run.sh"):
        path = result_dir / name
        if path.is_file():
            artifacts[name] = pin(path)
        else:
            failures.append("missing " + name)
    try:
        if request["status"] != "prepared":
            raise ValueError("request still has unresolved payload prerequisites")
        if failures:
            raise ValueError("lightweight boot observation requires all four result artifacts")
        if any(path is None for path in (invocation_path, command_path, upload_path)):
            raise ValueError("original SSM invocation, command and upload evidence are required")
        record = read(result_dir / "booted-elf.json")
        invocation = read(invocation_path)
        if not isinstance(invocation.get("CommandId"), str) or not invocation["CommandId"]:
            raise ValueError("original SSM invocation needs its command identity")
        artifacts["ssm_invocation"] = pin(invocation_path)
        deployment = request["deployment"]
        if (
            invocation.get("InstanceId") != deployment["manager_instance"]
            or invocation.get("DocumentName") != deployment["ssm_document"]
            or invocation.get("Status") != "Success"
            or invocation.get("ResponseCode") != 0
        ):
            raise ValueError("SSM invocation did not succeed on the selected manager")
        if (
            record.get("schema") != SCHEMA
            or record.get("observation_stage") not in {"boot_file_before_run", "boot_file_after_run"}
            or not isinstance(record.get("runtime_host"), str)
            or not record["runtime_host"].strip()
        ):
            raise ValueError("expected a runtime-host boot-file hash observation")
        timestamp = datetime.fromisoformat(record["observed_utc"])
        if timestamp.utcoffset() != timedelta(0):
            raise ValueError("boot observation needs a UTC timestamp")
        model = _model(request, record)
        values = [
            line.partition(":")[2].strip()
            for line in (result_dir / "HW_CFG_SUMMARY").read_text().splitlines()
            if line.partition(":")[0].strip() == "AGFI"
        ]
        hardware_matches = values == [deployment["agfi"]] and record.get("agfi") == deployment["agfi"]
        if not hardware_matches:
            raise ValueError("boot observation or hardware summary AGFI differs from selection")
        if record.get("uartlog") != identity(artifacts["uartlog"]):
            raise ValueError("boot observation UART hash/size differs from the collected complete UART")
        _boot_path(record, result_dir / "sim-run.sh")
        artifacts.update(_upload(request, model, record, invocation, command_path, upload_path))
        uploaded_matches = binding = True
        compact = compact_collect(model, result_dir / "uartlog")
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append(str(error))
    for artifact in artifacts.values():
        if pin(reopen(artifact)) != artifact:
            raise ValueError("observation artifact changed during collection")
    verify_request(request)
    return {
        "schema": RESULT_SCHEMA,
        "request": pin(request_path),
        "artifacts": artifacts,
        "ssm_invocation": artifacts.get("ssm_invocation"),
        "status": "incomplete" if failures else "observation_imported",
        "failures": failures,
        "evidence_class": "lightweight_booted_elf_hash",
        "scientific_qualification": False,
        "model_validation_complete": False,
        "hardware_agfi_matches": hardware_matches,
        "reported_frozen_inputs_match": binding,
        "uploaded_elf_identity_matches": uploaded_matches,
        "consumed_file_custody_status": "lightweight_boot_file_hash_only",
        "model": model["id"] if model is not None else None,
        "boot_observation": record,
        "compact_comparison": compact,
        "numeric_status": (
            "guest_all_element_comparison_passed" if compact is not None else "pending_complete_compact_comparison"
        ),
        "placement_status": "pending_existing_source_and_final_elf_audits",
        "timing_status": "raw target-cycle evidence only; no target-latency conversion",
        "scope": "Nonadversarial reproducibility observation; no before/after custody, "
        "runtime isolation, scientific qualification or model completion.",
    }
