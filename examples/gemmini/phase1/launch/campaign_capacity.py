"""Read-only campaign accounting, separate from native service lifetimes.

This accounts trusted operator services; it grants no execution authority. Unknown
or unreadable ownership consumes the whole fleet budget until a later clean scan.
"""
from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import Path

PROC, CGROUP = Path("/proc"), Path("/sys/fs/cgroup")
UNKNOWN_CAMPAIGNS = 5
FULL_SIZE_CPU_MIN = 15
# These reviewed operator entry points perform postfreeze evaluation, not authoring.
# A live author descendant always overrides this classification.
OPERATOR_ENTRYPOINTS = frozenset({
    "coordinate-evaluation.py", "evaluate-worker.py", "run-physical-remaining-v1.py",
})


class ActiveCampaigns(set):
    """Real active service IDs, with independent campaign counts and evidence."""

    def __init__(self, observations):
        self.observations = observations
        self.author_counts = {unit: row["author_campaigns"] for unit, row in observations.items()}
        self.raw_author_counts = {
            unit: row.get("raw_author_campaigns", row["author_campaigns"]) for unit, row in observations.items()
        }
        super().__init__(observations)


def _value(argv, option):
    try:
        return argv[argv.index(option) + 1]
    except (ValueError, IndexError):
        return None


def _entrypoint(argv):
    """Identify an executed Python entrypoint; argument data grants no ownership."""
    argv = list(argv)
    if not argv:
        return None, None, []
    if Path(argv[0]).name == "env":
        cursor = 1
        while cursor < len(argv):
            value = argv[cursor]
            if value in {"-i", "--ignore-environment"} or re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", value):
                cursor += 1
            elif value in {"-u", "--unset"} and cursor + 1 < len(argv):
                cursor += 2
            elif value == "--":
                cursor += 1
                break
            else:
                break
        argv = argv[cursor:]
        if not argv:
            return None, None, []
    # The installed physical-run wrapper is operator execution, never an author.
    # Match its complete approved entrypoint, not a basename or argument data.
    if argv[0] == "/home/ubuntu/bin/fpga-run-elf":
        return "physical_operator", argv[0], argv[1:]
    executable = Path(argv[0]).name
    if executable == "codex":
        return "codex", executable, argv[1:]
    if re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)*)?", executable) is None:
        return None, None, []
    cursor = 1
    while cursor < len(argv):
        value = argv[cursor]
        if value == "-m" and cursor + 1 < len(argv):
            return "module", argv[cursor + 1], argv[cursor + 2:]
        if value == "-c":
            return None, None, []
        if value in {"-B", "-u", "-E", "-I", "-s", "-S", "-q", "-O", "-OO"}:
            cursor += 1
        elif value in {"-W", "-X"} and cursor + 1 < len(argv):
            cursor += 2
        elif (value.startswith("-W") or value.startswith("-X")) and len(value) > 2:
            cursor += 1
        elif value == "--" and cursor + 1 < len(argv):
            return "script", argv[cursor + 1], argv[cursor + 2:]
        elif not value.startswith("-"):
            return "script", value, argv[cursor + 1:]
        else:
            return None, None, []
    return None, None, []


def command_ownership(argv):
    """Return (controller identities, orphan Codex evidence, known operator)."""
    controllers = set()
    kind, entry, arguments = _entrypoint(argv)
    if kind == "module" and entry == "merlin_experiments.phase1":
        if "--preflight-only" in arguments:
            return controllers, False, True
        run_id = _value(arguments, "--run-id")
        if run_id:
            controllers.add(("native", run_id))
    if kind == "script":
        path = Path(entry)
        if path.name == "run.py" and path.parent.name == "kernel_abstractions":
            if arguments and arguments[0] == "launch" and _value(arguments, "--config"):
                controllers.add(("family", _value(arguments, "--config")))
            elif arguments and arguments[0] == "evaluate":
                return controllers, False, True
    codex = kind == "codex" and "exec" in arguments
    operator = kind == "physical_operator" or (
        kind == "script" and Path(entry).name in OPERATOR_ENTRYPOINTS)
    return controllers, codex, operator


def _exec_argv(text):
    match = re.search(r"argv\[\]=(.*?) ;", text)
    return shlex.split(match.group(1)) if match else []


def reservation_argv(unit, argv):
    """Use the same --state owner selected by native resource_reservation/run_watchdog."""
    kind, entry, arguments = _entrypoint(argv)
    if kind != "script" or Path(entry).name != "resource_reservation.py":
        return argv
    if not arguments or arguments[0] != "_child" or not _value(arguments, "--state"):
        raise ValueError("reservation has no selected child state")
    path = Path(_value(arguments, "--state"))
    if path.is_symlink() or path.resolve() != path or not path.is_file():
        raise ValueError("reservation state is not a direct canonical file")
    state = json.loads(path.read_text())
    if state.get("schema") != "merlin.operator.cpu_reservation.v1" or state.get("unit") != unit:
        raise ValueError("reservation state has different ownership")
    if not isinstance(state.get("argv"), list) or not all(isinstance(arg, str) for arg in state["argv"]):
        raise ValueError("reservation command is unavailable")
    return state["argv"]


def inspect_unit(unit):
    """Keep unreadable/racy ownership conservative; never silently drop a service."""
    result = subprocess.run(
        ["systemctl", "show", unit + ".service", "--property=ControlGroup,ActiveState,ExecStart"],
        capture_output=True, text=True, check=True,
    )
    properties = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    group_name = properties.get("ControlGroup", "")
    if not group_name.startswith("/") or group_name == "/":
        raise ValueError("active service has no selected cgroup")
    group = CGROUP / group_name.lstrip("/")
    if not group.is_dir():
        raise ValueError("active service cgroup disappeared")
    effective = (group / "cpuset.cpus.effective").read_text().strip()
    enforced = set()
    for part in effective.split(","):
        if re.fullmatch(r"[0-9]+(?:-[0-9]+)?", part) is None:
            raise ValueError("enforced cgroup CPU width is unavailable")
        ends = [int(value) for value in part.split("-")]
        lo, hi = ends[0], ends[-1]
        if hi < lo or hi >= 1000000:
            raise ValueError("invalid enforced cgroup CPU range")
        enforced.update(range(lo, hi + 1))
    if not enforced:
        raise ValueError("enforced cgroup CPU set is empty")
    small = len(enforced) < FULL_SIZE_CPU_MIN
    try:
        row = _inspect_ownership(unit, group, properties)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        row = dict(author_campaigns=UNKNOWN_CAMPAIGNS, classification="unknown_conservative", reason=str(error))
    row.update(raw_author_campaigns=row["author_campaigns"], raw_classification=row["classification"],
               effective_cpus=effective, enforced_cpu_width=len(enforced),
               campaign_scope="small_enforced_cpu_set" if small else "full_size_cpu_scope",
               full_size_cpu_min=FULL_SIZE_CPU_MIN)
    if small:
        row.update(author_campaigns=0, classification="small_cpu_scope_excluded")
    return row


def _inspect_ownership(unit, group, properties):
    """Raw author/operator evidence; a cgroup CPU scope is applied separately."""
    root_procs = group / "cgroup.procs"
    paths = [root_procs, *group.rglob("*/cgroup.procs")]
    pids = set()
    for path in paths:
        pids.update(int(value) for value in path.read_text().split())
    controllers, orphan, operator = set(), False, False
    command = _exec_argv(properties.get("ExecStart", ""))
    if command:
        selected = reservation_argv(unit, command)
        found, codex, known = command_ownership(selected)
        controllers.update(found)
        orphan |= codex
        operator |= known
    for pid in pids:
        path = PROC / str(pid) / "cmdline"
        try:
            argv = path.read_bytes().decode().rstrip("\0").split("\0")
        except FileNotFoundError:
            # A reaped child is ordinary. A live but hidden PID is not.
            if path.parent.exists():
                raise
            continue
        found, codex, known = command_ownership(argv)
        controllers.update(found)
        orphan |= codex
        operator |= known
    count = len(controllers) if controllers else int(orphan)
    if not count and not operator:
        raise ValueError("no known author or operator ownership")
    return dict(author_campaigns=count, classification="author_campaign" if count else "operator_only",
                controllers=[list(value) for value in sorted(controllers)], orphan_codex=orphan,
                pid_count=len(pids), control_group=properties["ControlGroup"])


def active_campaigns():
    result = subprocess.run(
        ["systemctl", "list-units", "--type=service", "--state=active,activating,reloading,deactivating",
         "--no-legend", "--plain", "--no-pager"],
        capture_output=True, text=True, check=True,
    )
    units = {line.split()[0].removesuffix(".service") for line in result.stdout.splitlines()
             if line.strip() and re.fullmatch(r"merlinphase1[a-z0-9]+\.service", line.split()[0])}
    observations = {}
    for unit in sorted(units):
        try:
            observations[unit] = inspect_unit(unit)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            observations[unit] = dict(author_campaigns=UNKNOWN_CAMPAIGNS,
                classification="unknown_conservative", raw_author_campaigns=UNKNOWN_CAMPAIGNS,
                enforced_cpu_width=None, campaign_scope="unknown_enforced_cpu_scope", reason=str(error))
    return ActiveCampaigns(observations)
