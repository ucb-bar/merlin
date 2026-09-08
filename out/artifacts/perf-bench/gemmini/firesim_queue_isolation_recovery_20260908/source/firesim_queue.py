#!/usr/bin/env python3
"""firesim-queue — single-FPGA job queue with per-user fair share.

Architecture (see /scratch2/agustin/firesim_queue/README.md):
  - SQLite-backed job table at /scratch2/agustin/firesim_queue/queue.db
  - One daemon process holds the FPGA flock and dispatches jobs
  - Clients call `submit` which inserts a row and polls until done
  - Scheduling: priority tier (10>5>0), then weighted round-robin by user

Subcommands:
  submit  --priority N [--background] [--cwd PATH] -- <command...>
  status  [--user USER] [--all]
  cancel  <job_id>
  tail    <job_id>
  daemon
  stop-daemon

Lifecycle states: QUEUED → RUNNING → {DONE | FAILED | CANCELLED}.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import getpass
import grp
import hashlib
import json
import os
import pathlib
import shlex
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import time
import textwrap

QUEUE_ROOT = pathlib.Path(
    os.environ.get("FIRESIM_QUEUE_ROOT", "/scratch/firesim_queue"))
DB_PATH = QUEUE_ROOT / "queue.db"
DAEMON_PID = QUEUE_ROOT / "daemon.pid"
DAEMON_LOG = QUEUE_ROOT / "daemon.log"
FPGA_LOCK = QUEUE_ROOT / "fpga.lock"
JOBS_DIR = QUEUE_ROOT / "jobs"
_runtime_root = pathlib.Path(
    os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.geteuid()}")
)
if not _runtime_root.is_dir():
    _runtime_root = pathlib.Path("/tmp")
_queue_identity = hashlib.sha256(str(QUEUE_ROOT.resolve()).encode()).hexdigest()[:16]
HWDB_SNAPSHOT_ROOT = pathlib.Path(
    os.environ.get(
        "FIRESIM_QUEUE_HWDB_SNAPSHOT_ROOT",
        str(_runtime_root / f"firesim-queue-hwdb-{_queue_identity}"),
    )
)
HEARTBEAT_KEY = "daemon_heartbeat"
DAEMON_STALE_SECONDS = 300        # if heartbeat older than this, daemon is dead
POLL_INTERVAL_SECONDS = 5
HEARTBEAT_INTERVAL_SECONDS = 30
# Control-plane operations normally complete in tens of seconds, but Fabric can
# wedge while killing a stale local simulator.  These phases must be bounded
# independently of the user workload timeout: a stuck leading/trailing kill
# otherwise blocks the shared FPGA queue forever, and cancellation cannot be
# observed while subprocess.run() is waiting.
FIRESIM_KILL_TIMEOUT_SECONDS = 90
FIRESIM_INFRASETUP_TIMEOUT_SECONDS = 900


# ---------------------------------------------------------------------------
# Shared-filesystem helpers
# ---------------------------------------------------------------------------
# Queue artifacts are created by whoever ran the command but written by
# whoever runs the daemon, and those are different people. Without a common
# group the daemon cannot write a submitter's job dir at all, so it skips the
# job forever (see the PermissionError arm of _daemon_loop) and the submitter
# watches their job sit QUEUED behind nothing. SHARED_GROUP is the group every
# queue user belongs to; job dirs are created setgid to it so write access is
# inherited no matter who submitted.
SHARED_GROUP = os.environ.get("FIRESIM_QUEUE_GROUP", "firesim")
SHARED_DIR_MODE = 0o2775          # setgid: children inherit SHARED_GROUP


def _shared_gid() -> int | None:
    """gid of SHARED_GROUP, or None if it doesn't exist on this host."""
    try:
        return grp.getgrnam(SHARED_GROUP).gr_gid
    except KeyError:
        return None


def _share_dir(path: pathlib.Path) -> None:
    """Make `path` group-writable and setgid, so the daemon can write it
    whoever submitted. Best-effort throughout: chgrp fails when we neither own
    the dir nor belong to the group, and neither is worth failing a submit
    over -- the daemon degrades to skipping the job, exactly as before."""
    gid = _shared_gid()
    try:
        if gid is not None and path.stat().st_gid != gid:
            os.chown(path, -1, gid)
    except OSError:
        pass
    try:
        path.chmod(SHARED_DIR_MODE)
    except OSError:
        pass


def _require_shared_group() -> None:
    """Refuse a submit from someone outside SHARED_GROUP, with instructions.

    This failure is silent by nature and costs days: a submitter outside the
    group creates a job dir the daemon cannot write, the daemon skips the job
    for its whole lifetime, and the submitter watches their job sit QUEUED
    behind an *empty* queue -- which reads as "someone is hogging the FPGA"
    rather than "you are not in a unix group". Refusing up front, naming the
    exact fix, is worth more than accepting a job we know will never run.
    """
    gid = _shared_gid()
    if gid is None:
        return                      # group not provisioned here; nothing to enforce
    if os.geteuid() == 0 or gid in os.getgroups():
        return
    who = os.environ.get("USER") or str(os.geteuid())
    print(
        f"error: {who} is not in the '{SHARED_GROUP}' group, so the queue daemon\n"
        f"       could not run this job -- it would sit QUEUED forever.\n"
        f"\n"
        f"       Fix (needs an admin):  sudo usermod -aG {SHARED_GROUP} {who}\n"
        f"       Then start a new login shell so the group takes effect.\n",
        file=sys.stderr)
    raise SystemExit(2)


def _hwdb_snapshot_parent_is_protected(*, owner_uid: int, mode: int) -> bool:
    """Whether a parent cannot be made replaceable by an untrusted owner."""
    trusted_owner = owner_uid in {0, os.geteuid()}
    daemon_private = owner_uid == os.geteuid() and mode & 0o077 == 0
    trusted_sticky = trusted_owner and bool(mode & 0o1000)
    trusted_not_writable = trusted_owner and mode & 0o022 == 0
    return daemon_private or trusted_sticky or trusted_not_writable


def _prepare_hwdb_snapshot_root() -> pathlib.Path:
    """Return a daemon-owned directory other queue users cannot replace.

    Job directories must be group-writable so a daemon can serve submissions
    from other users.  A read-only file directly beneath such a directory is
    not immutable: a group member can unlink or rename it.  Snapshot files
    therefore live beneath the daemon's private runtime directory.  The
    runtime parent is either the daemon-owned XDG runtime directory or /tmp
    (whose sticky bit prevents another uid from renaming our child).
    """
    root = HWDB_SNAPSHOT_ROOT.resolve()
    parent = root.parent
    try:
        parent_stat = parent.stat()
    except OSError as exc:
        raise RuntimeError(
            f"cannot stat HWDB snapshot parent {parent}: {exc}") from exc
    parent_mode = parent_stat.st_mode & 0o7777
    if not _hwdb_snapshot_parent_is_protected(
        owner_uid=parent_stat.st_uid, mode=parent_mode
    ):
        raise RuntimeError(
            f"HWDB snapshot parent is replaceable by another uid: {parent}")
    try:
        root.mkdir(mode=0o700, parents=False, exist_ok=True)
        root_stat = root.stat()
    except OSError as exc:
        raise RuntimeError(
            f"cannot prepare daemon-private HWDB snapshot root {root}: {exc}"
        ) from exc
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"HWDB snapshot root is not a real directory: {root}")
    if root_stat.st_uid != os.geteuid():
        raise RuntimeError(
            f"HWDB snapshot root is not owned by daemon uid {os.geteuid()}: {root}"
        )
    try:
        root.chmod(0o700)
    except OSError as exc:
        raise RuntimeError(
            f"cannot protect HWDB snapshot root {root}: {exc}") from exc
    if root.stat().st_mode & 0o077:
        raise RuntimeError(f"HWDB snapshot root is not private: {root}")
    return root


# ---------------------------------------------------------------------------
# DB schema and helpers
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user         TEXT    NOT NULL,
    priority     INTEGER NOT NULL DEFAULT 5,
    submitted_at REAL    NOT NULL,
    started_at   REAL,
    ended_at     REAL,
    state        TEXT    NOT NULL,
    exit_code    INTEGER,
    cmd          TEXT    NOT NULL,
    cwd          TEXT    NOT NULL,
    env_json     TEXT    NOT NULL,
    workdir      TEXT    NOT NULL,
    pid          INTEGER,
    project      TEXT    NOT NULL DEFAULT '(unknown)'
);

CREATE INDEX IF NOT EXISTS idx_jobs_state_prio
    ON jobs(state, priority DESC, submitted_at);
CREATE INDEX IF NOT EXISTS idx_jobs_user_state
    ON jobs(user, state);

CREATE TABLE IF NOT EXISTS kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _connect() -> sqlite3.Connection:
    QUEUE_ROOT.mkdir(parents=True, exist_ok=True)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    # Self-healing: a queue root deployed before SHARED_GROUP existed gets
    # fixed on the next command anyone runs.
    _share_dir(QUEUE_ROOT)
    _share_dir(JOBS_DIR)
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0,
                           isolation_level=None)  # autocommit; explicit BEGIN
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_SCHEMA)
    # In-place schema evolution: add columns introduced after initial release.
    # ALTER TABLE ADD COLUMN is idempotent if we catch the OperationalError.
    for col, ddl in (
        ("project", "ALTER TABLE jobs ADD COLUMN project TEXT NOT NULL "
                    "DEFAULT '(unknown)'"),
        # `kind` distinguishes legacy freeform bash submissions
        # (`bash` — daemon just runs the cmd string) from the new
        # structured atomic FireSim op (`runworkload-full` — daemon
        # owns the kill→infrasetup→runworkload→kill sequence).
        ("kind", "ALTER TABLE jobs ADD COLUMN kind TEXT NOT NULL "
                 "DEFAULT 'bash'"),
        # JSON-encoded structured args for non-bash kinds. For
        # `runworkload-full` this holds: chipyard, workload,
        # bootbinary, stage_from, timeout. NULL for `bash`.
        ("kind_args", "ALTER TABLE jobs ADD COLUMN kind_args TEXT"),
        # Per-job lifecycle phase. NULL for `bash`. For
        # `runworkload-full`: STAGING / INFRASETUP / RUNNING /
        # TEARDOWN / DONE / FAILED / CANCELLED / TIMEOUT. Lets the
        # TUI surface what the daemon is currently doing instead of
        # just "RUNNING" for the whole sequence.
        ("phase", "ALTER TABLE jobs ADD COLUMN phase TEXT"),
        # UID of the submitter. Used for owner-only cancel checks
        # (the new `cancel` rejects when caller UID != owner_uid).
        # Filled by cmd_submit / cmd_runworkload_full; legacy rows
        # have NULL → cancel falls back to user-name comparison.
        ("owner_uid", "ALTER TABLE jobs ADD COLUMN owner_uid INTEGER"),
        # Flag the daemon polls during a job's RUNNING phase. Non-NULL
        # value means "shut me down": daemon SIGTERMs the active
        # subprocess group, runs the cleanup kill, marks CANCELLED.
        # Cleared by the daemon once it acts on the request.
        ("cancel_requested", "ALTER TABLE jobs ADD COLUMN cancel_requested "
                             "INTEGER"),
    ):
        try:
            conn.execute(ddl)
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e).lower():
                raise
    return conn


# ---------------------------------------------------------------------------
# Project tag detection (auto-derived from cwd if --project not given)
# ---------------------------------------------------------------------------

def _infer_project(cwd: str, cmd: str) -> str:
    """Heuristic project name from cwd. Generic — no hardcoded list of
    projects (anyone can add a new repo under /scratch2/<user>/ without
    code changes).

    Strategy:
      1. If cwd matches `/scratch2/<user>/<project>/...` → return <project>.
      2. If cwd matches `/<vol>/<user>/<project>/...` (any volume) → return <project>.
      3. Else: basename of cwd.

    Returns "(none)" only if cwd is empty or root."""
    p = pathlib.Path(cwd or "").resolve()
    parts = p.parts
    # Look for the conventional /<volume>/<user>/<project>/... layout.
    # If $USER appears in parts, use the part right after it.
    user = os.environ.get("USER", "") or getpass.getuser()
    if user and user in parts:
        idx = parts.index(user)
        if idx + 1 < len(parts):
            return parts[idx + 1]
    # Fallback: try /scratch*/<user-like>/<project>/... → 3rd component
    if len(parts) >= 4 and parts[1].startswith("scratch"):
        return parts[3]
    # Last resort: basename of cwd.
    if parts:
        return parts[-1]
    return "(none)"


# Per-project color (dynamic — hashes the project name like _rich_user_color
# does for users). Distinct from user colors via a separate palette.
_RICH_PROJECT_COLORS = [
    "color(45)",   # teal-ish
    "color(214)",  # orange-ish
    "color(141)",  # violet
    "color(50)",   # bright teal
    "color(178)",  # mustard
    "color(99)",   # lavender
    "color(192)",  # lime
    "color(204)",  # rose
    "color(75)",   # sky
    "color(186)",  # tan
]


def _build_gantt_panel(conn: sqlite3.Connection, me: str, term_width: int):
    """Render a unicode Gantt chart of recent + running + projected queued
    jobs on a horizontal timeline. Per-user rows; per-job gradient color
    via _rich_user_job_color. Returns a Rich Panel.

    Time window: [now - 30 min, now + 60 min] (configurable via env
    FIRESIM_QUEUE_GANTT_PAST/_FUTURE in seconds — default 1800/3600).

    Past jobs use a solid fill char (█); projected/queued jobs use a
    striped fill (▒) to make uncertainty visually obvious. The vertical
    "NOW" marker is rendered with a bold reverse character.
    """
    from rich.panel import Panel
    from rich.text import Text
    from rich.box import ROUNDED

    past_window = int(os.environ.get("FIRESIM_QUEUE_GANTT_PAST", "1800"))
    future_window = int(os.environ.get("FIRESIM_QUEUE_GANTT_FUTURE", "3600"))
    now = time.time()
    lo = now - past_window
    hi = now + future_window
    span = hi - lo

    # Layout: <user label width> <bar cols> = term_width - panel_padding
    label_w = 14
    bar_w = max(20, term_width - label_w - 8)  # 8 = panel borders+padding

    # Gather rows in window.
    past_running = conn.execute(
        "SELECT id, user, started_at, ended_at, state, project, cmd"
        " FROM jobs WHERE started_at IS NOT NULL AND started_at >= ?"
        " ORDER BY user, started_at", (lo,)).fetchall()
    queued = conn.execute(
        "SELECT id, user, submitted_at, priority, project, cmd FROM jobs"
        " WHERE state='QUEUED' ORDER BY priority DESC, submitted_at"
    ).fetchall()

    # Simulate the scheduler for the queue to project (user, start_proj, end_proj)
    # cursor = max(now, end of currently running job).
    running_row = next((r for r in past_running if r[4] == "RUNNING"), None)
    if running_row:
        # Project running job's end using ETA helper.
        _id, _user, started_at, _ended, _state, _proj, rcmd = running_row
        eta, _kind, _n = _estimate_eta(conn, rcmd)
        running_end_proj = (started_at + eta) if eta else now + 300
        if running_end_proj < now:
            running_end_proj = now + 60
    else:
        running_end_proj = now

    # Mirror the daemon's scheduling rule: pick top-priority tier, then
    # round-robin by user. Default-duration fallback when no ETA found.
    last_user_kv = _kv_get(conn, "last_user_served")
    sim_cursor = running_end_proj
    projected: dict[int, tuple[float, float]] = {}  # job_id → (start, end)
    qpool = [list(q) for q in queued]  # mutable
    last_user = last_user_kv
    DEFAULT_DURATION = 300.0
    safety = 200
    while qpool and safety > 0:
        safety -= 1
        top_prio = qpool[0][3]
        tier = [j for j in qpool if j[3] == top_prio]
        pick = None
        if last_user is not None:
            for j in tier:
                if j[1] != last_user:
                    pick = j
                    break
        if pick is None:
            pick = tier[0]
        # Estimate duration
        eta, _kind, _n = _estimate_eta(conn, pick[5])
        dur = eta if eta else DEFAULT_DURATION
        projected[pick[0]] = (sim_cursor, sim_cursor + dur)
        sim_cursor += dur
        qpool.remove(pick)
        last_user = pick[1]
        if sim_cursor > hi:
            break  # past the visible window

    # Group all visible jobs by user, building (user → [(start, end, kind, job_id)])
    user_lanes: dict[str, list[tuple[float, float, str, int]]] = {}
    for r in past_running:
        (jid, user, st, en, state, proj, cmd) = r
        end = en if en is not None else now
        kind = "running" if state == "RUNNING" else "past"
        user_lanes.setdefault(user, []).append((st, end, kind, jid))
    for q in queued:
        (jid, user, sub, prio, proj, cmd) = q
        if jid in projected:
            st, en = projected[jid]
        else:
            # Beyond visible projection horizon — anchor at hi.
            st, en = hi, hi
        user_lanes.setdefault(user, []).append((st, en, "queued", jid))

    if not user_lanes:
        empty = Text.from_markup("[dim]no recent or queued jobs in window[/]")
        return Panel(empty, title="📈 Gantt · idle",
                     border_style="bright_blue", box=ROUNDED)

    def time_to_col(t: float) -> int:
        if t <= lo:
            return 0
        if t >= hi:
            return bar_w - 1
        return int((t - lo) / span * bar_w)

    now_col = time_to_col(now)

    lines: list[Text] = []
    # Top time axis with hour:minute labels at the 0%, 25%, 50%, 75%, 100% marks.
    axis_labels = [" " * bar_w]
    axis_chars = list(" " * bar_w)
    ticks_at_frac = [0.0, 0.25, 0.5, 0.75, 1.0]
    label_chunks: list[tuple[int, str]] = []
    for f in ticks_at_frac:
        col = min(bar_w - 1, int(f * bar_w))
        t = lo + f * span
        label = time.strftime("%H:%M", time.localtime(t))
        label_chunks.append((col, label))
    # Build axis line + tick line
    tick_chars = list(" " * bar_w)
    for col, _label in label_chunks:
        if 0 <= col < bar_w:
            tick_chars[col] = "│"
    label_row = list(" " * bar_w)
    for col, label in label_chunks:
        # place label so it begins at col (clip at end)
        for i, ch in enumerate(label):
            pos = col + i
            if 0 <= pos < bar_w:
                label_row[pos] = ch
    # NOW marker on label row replaces what's there with red "NOW" if room
    if 0 <= now_col < bar_w:
        tick_chars[now_col] = "┃"

    header_axis = Text.from_markup(
        " " * label_w +
        "[dim]" + "".join(label_row).replace("[", "(").replace("]", ")") +
        "[/]")
    header_ticks = Text.from_markup(
        " " * label_w + "[bright_yellow]" +
        "".join(tick_chars).replace("[", "(").replace("]", ")") + "[/]")

    # Compute occurrence per user (chronological within window) for gradient.
    body_lines: list[Text] = []
    for user, jobs in sorted(user_lanes.items(),
                              key=lambda kv: kv[0] != me and kv[0]):  # me-first sort
        jobs.sort(key=lambda j: j[0])
        # Build the cell array for this user.
        cells: list[tuple[str, str]] = [(" ", "")] * bar_w
        for occ, (st, en, kind, jid) in enumerate(jobs):
            c0 = time_to_col(st)
            c1 = time_to_col(en)
            if c1 < c0:
                c1 = c0
            # Always paint at least one cell so 0-duration jobs are visible.
            if c1 == c0 and c0 < bar_w - 1:
                c1 = c0 + 1
            color = _rich_user_job_color(user, occ)
            fill = "█" if kind == "past" else ("▓" if kind == "running" else "▒")
            for col in range(max(0, c0), min(bar_w, c1)):
                cells[col] = (fill, color)
        # Overlay the NOW marker (preserves cell content but switches the
        # foreground char to a vertical bar in bright yellow if empty).
        if 0 <= now_col < bar_w:
            cur_ch, _cur_clr = cells[now_col]
            if cur_ch == " ":
                cells[now_col] = ("│", "bright_yellow")
        # Build a Rich Text by chunking runs of identical color.
        markup_parts: list[str] = []
        i = 0
        while i < bar_w:
            ch, clr = cells[i]
            j = i + 1
            while j < bar_w and cells[j][1] == clr and cells[j][0] == ch:
                j += 1
            run = ch * (j - i)
            if clr:
                markup_parts.append(f"[{clr}]{run}[/]")
            else:
                markup_parts.append(run)
            i = j
        you = "*" if user == me else " "
        label = (f"[{_rich_user_color(user)}]{user[:label_w-2]:<{label_w-2}}[/]"
                 f"{you}")
        body_lines.append(Text.from_markup(label + "".join(markup_parts)))

    # Compose. Header rows + each user lane.
    legend = Text.from_markup(
        "[dim]legend:[/] [bright_white]█[/] past  "
        "[bright_white]▓[/] running  [bright_white]▒[/] projected  "
        "[bright_yellow]┃[/] NOW   "
        f"[dim]· window {past_window // 60}m past + "
        f"{future_window // 60}m projected[/]")

    from rich.console import Group
    body = Group(header_axis, header_ticks, *body_lines, Text(""), legend)
    return Panel(body, title="📈 Gantt", border_style="bright_blue",
                 box=ROUNDED, padding=(0, 1))


def _rich_project_color(project: str) -> str:
    """Stable color string for a project name. Any new project (not pre-
    registered anywhere) automatically gets a distinct color from the
    palette — hashed by name so the same project keeps the same color
    across sessions."""
    if not project or project == "(none)":
        return "dim"
    h = sum(ord(c) for c in project) % len(_RICH_PROJECT_COLORS)
    return _RICH_PROJECT_COLORS[h]


def _kv_get(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def _kv_set(conn: sqlite3.Connection, key: str, value: str):
    conn.execute("INSERT INTO kv(key, value) VALUES(?, ?) "
                 "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                 (key, value))


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

def _pick_next_job(conn: sqlite3.Connection,
                   skip: set[int] | None = None) -> dict | None:
    """Return next job to run (as dict) or None if queue empty.

    Selection rule:
      1. Highest-priority tier among QUEUED jobs.
      2. Within tier, weighted round-robin by user — pick the user with the
         oldest queued submission who is NOT the last user served (when
         multiple users share the tier).
      3. Ties → oldest submitted_at.
    """
    rows = conn.execute(
        "SELECT id, user, priority, submitted_at, cmd, cwd, env_json, workdir,"
        " kind, kind_args"
        " FROM jobs WHERE state='QUEUED' ORDER BY priority DESC, submitted_at"
    ).fetchall()
    if skip:
        # Jobs this daemon has already proven it cannot run (see the dispatch loop). Their state is
        # deliberately left QUEUED rather than mutated: the job is fine, THIS daemon just cannot
        # write its directory, and a daemon run by the owner will pick it up.
        rows = [r for r in rows if r[0] not in skip]
    if not rows:
        return None

    top_prio = rows[0][2]
    tier = [r for r in rows if r[2] == top_prio]
    last_user = _kv_get(conn, "last_user_served")

    # Find a user other than last_user with a queued job in this tier.
    pick = None
    if last_user is not None:
        for r in tier:
            if r[1] != last_user:
                pick = r
                break
    if pick is None:
        pick = tier[0]  # fall through: either last_user is None or only one
                         # user has jobs in this tier.

    return {
        "id": pick[0], "user": pick[1], "priority": pick[2],
        "submitted_at": pick[3], "cmd": pick[4], "cwd": pick[5],
        "env_json": pick[6], "workdir": pick[7],
        "kind": pick[8], "kind_args": pick[9],
    }


# ---------------------------------------------------------------------------
# Daemon liveness
# ---------------------------------------------------------------------------

def _daemon_alive(conn: sqlite3.Connection) -> tuple[bool, float | None]:
    """Return (alive, age_seconds_or_None)."""
    hb = _kv_get(conn, HEARTBEAT_KEY)
    if hb is None:
        return False, None
    try:
        age = time.time() - float(hb)
    except ValueError:
        return False, None
    return age < DAEMON_STALE_SECONDS, age


def _daemon_pid_alive() -> int | None:
    if not DAEMON_PID.exists():
        return None
    try:
        pid = int(DAEMON_PID.read_text().strip())
    except ValueError:
        return None
    try:
        os.kill(pid, 0)
        return pid
    except OSError:
        return None


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------

def cmd_submit(args: argparse.Namespace) -> int:
    _require_shared_group()
    conn = _connect()
    alive, age = _daemon_alive(conn)
    if not alive:
        print(f"[firesim-queue] warning: daemon not running"
              + (f" (stale heartbeat {age:.0f}s)" if age else "")
              + ". Submitting anyway — jobs will pick up once "
              "`firesim-queue daemon` is started.", file=sys.stderr)

    if not args.cmd:
        print("error: no command provided after --", file=sys.stderr)
        return 2

    now = time.time()
    user = args.user or getpass.getuser()
    cmd_str = " ".join(shlex.quote(c) for c in args.cmd)
    cwd = str(pathlib.Path(args.cwd or os.getcwd()).resolve())
    env_keep = {}
    for k in ("PATH", "HOME", "USER", "LOGNAME", "SHELL", "TERM",
              "CHIPYARD_ROOT", "ZEPHYR_BASE", "ZEPHYR_SDK_INSTALL_DIR",
              "Zephyr_SDK_DIR", "ZEPHYR_TOOLCHAIN_VARIANT",
              "FIRESIM_SOURCEME", "SSH_AUTH_SOCK"):
        if k in os.environ:
            env_keep[k] = os.environ[k]
    env_json = json.dumps(env_keep)

    project = (args.project if getattr(args, "project", None)
               else _infer_project(cwd, cmd_str))

    cur = conn.execute(
        "INSERT INTO jobs(user, priority, submitted_at, state, cmd, cwd,"
        " env_json, workdir, project, kind, owner_uid)"
        " VALUES(?, ?, ?, 'QUEUED', ?, ?, ?, '', ?, 'bash', ?)",
        (user, args.priority, now, cmd_str, cwd, env_json, project,
         os.geteuid()))
    job_id = cur.lastrowid
    workdir = JOBS_DIR / str(job_id)
    workdir.mkdir(parents=True, exist_ok=True)
    _share_dir(workdir)
    (workdir / "cmd.sh").write_text(
        f"#!/bin/bash\n# job_id={job_id} user={user} priority={args.priority}"
        f" project={project}\n"
        f"cd {shlex.quote(cwd)}\n{cmd_str}\n")
    (workdir / "cmd.sh").chmod(0o755)
    conn.execute("UPDATE jobs SET workdir=? WHERE id=?",
                 (str(workdir), job_id))

    print(f"[firesim-queue] job_id={job_id} user={user} "
          f"priority={args.priority} project={project} state=QUEUED")
    print(f"[firesim-queue]   cwd={cwd}")
    print(f"[firesim-queue]   cmd={cmd_str}")
    # Print ETA (if known) for confidence at submit time.
    eta, kind, n = _estimate_eta(conn, cmd_str)
    if eta is not None:
        print(f"[firesim-queue]   ETA {int(eta)}s "
              f"(median of {n} {kind} past run{'s' if n != 1 else ''})")
    else:
        print(f"[firesim-queue]   ETA unknown (no similar past jobs)")

    if args.background:
        return 0

    # Foreground: poll until job reaches a terminal state. Stream stdout.log
    # contents as they grow.
    stdout_path = workdir / "stdout.log"
    stderr_path = workdir / "stderr.log"
    stdout_pos = 0
    stderr_pos = 0
    last_state = None
    last_status_print = 0.0

    while True:
        row = conn.execute(
            "SELECT state, started_at, ended_at, exit_code FROM jobs WHERE id=?",
            (job_id,)).fetchone()
        if row is None:
            print(f"[firesim-queue] job_id={job_id} disappeared from queue",
                  file=sys.stderr)
            return 1
        state, started_at, ended_at, exit_code = row

        # Stream stdout.log incrementally so users see progress.
        for p, pos_attr in ((stdout_path, "stdout_pos"),
                            (stderr_path, "stderr_pos")):
            if p.exists():
                pos = stdout_pos if pos_attr == "stdout_pos" else stderr_pos
                size = p.stat().st_size
                if size > pos:
                    with p.open("rb") as f:
                        f.seek(pos)
                        chunk = f.read(size - pos)
                    sink = sys.stdout if pos_attr == "stdout_pos" else sys.stderr
                    sink.buffer.write(chunk)
                    sink.flush()
                    if pos_attr == "stdout_pos":
                        stdout_pos = size
                    else:
                        stderr_pos = size

        if state != last_state:
            print(f"[firesim-queue] job_id={job_id} state={state}",
                  file=sys.stderr)
            last_state = state

        if state in ("DONE", "FAILED", "CANCELLED"):
            wall = (ended_at - started_at) if (ended_at and started_at) else None
            wall_s = f" wall={wall:.1f}s" if wall else ""
            print(f"[firesim-queue] job_id={job_id} terminal state={state}"
                  f" exit_code={exit_code}{wall_s}",
                  file=sys.stderr)
            return int(exit_code or (0 if state == "DONE" else 1))

        # While queued, occasionally print the position summary.
        now2 = time.time()
        if state == "QUEUED" and now2 - last_status_print > 60:
            q_ahead = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE state='QUEUED'"
                " AND (priority > ? OR (priority = ? AND submitted_at < ?))",
                (args.priority, args.priority, now)).fetchone()[0]
            running = conn.execute(
                "SELECT id, user FROM jobs WHERE state='RUNNING'").fetchone()
            running_str = (f"running: id={running[0]} user={running[1]}"
                           if running else "no job running")
            print(f"[firesim-queue] job_id={job_id} still QUEUED: "
                  f"{q_ahead} jobs ahead; {running_str}", file=sys.stderr)
            last_status_print = now2

        time.sleep(POLL_INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# runworkload-full — atomic FireSim lifecycle (centralizes the kill →
# infrasetup → runworkload → kill dance under one queue-owned lock).
#
# Why this exists: the legacy `submit -- bash -c '...firesim infrasetup;
# firesim runworkload...'` flow lets clients hold the FPGA in an
# infrasetup'd state without releasing it (agent crash, OOM, forgotten
# kill), and races on shared state (deploy/config_runtime.yaml is
# rewritten by infrasetup — concurrent users overwrote each other's
# workload name, ending up running merlin's workload on agustin's
# submission). The atomic op removes both problems by construction:
#
#   1. Daemon owns every firesim CLI invocation, so the FPGA lock is
#      held continuously from kill through to the trailing kill — no
#      window in which a softlock can occur.
#   2. Daemon serializes config_runtime.yaml edits the same way it
#      serializes FPGA access, so two submissions can't clobber each
#      other.
#
# No `infrasetup` / `kill` subcommands are exposed to clients.
# Atomicity is the safety guarantee. The optional `--timeout` arg is
# the supported way to bound a run (it folds into the daemon's
# teardown — clients never need to call `firesim kill` themselves).
# ---------------------------------------------------------------------------


def cmd_runworkload_full(args: argparse.Namespace) -> int:
    _require_shared_group()
    conn = _connect()
    alive, age = _daemon_alive(conn)
    if not alive:
        print(f"[firesim-queue] warning: daemon not running"
              + (f" (stale heartbeat {age:.0f}s)" if age else "")
              + ". Submitting anyway — jobs will pick up once "
              "`firesim-queue daemon` is started.", file=sys.stderr)

    # Validate chipyard root has the firesim install we'll drive.
    chipyard = pathlib.Path(args.chipyard).resolve()
    sourceme = chipyard / "sims" / "firesim" / "sourceme-manager.sh"
    if not sourceme.exists():
        print(f"error: --chipyard {chipyard} doesn't look like a chipyard "
              f"install (missing {sourceme})", file=sys.stderr)
        return 2
    deploy = chipyard / "sims" / "firesim" / "deploy"
    if not deploy.exists():
        print(f"error: {deploy} does not exist", file=sys.stderr)
        return 2

    # Validate the workload JSON the daemon will hand to firesim runworkload.
    workload_json = deploy / "workloads" / f"{args.workload}.json"
    if not workload_json.exists():
        print(f"error: workload spec not found at {workload_json}",
              file=sys.stderr)
        return 2

    # Resolve bootbinary: caller arg > workload JSON's common_bootbinary.
    bootbinary = args.bootbinary
    if not bootbinary:
        try:
            wl_meta = json.loads(workload_json.read_text())
            bootbinary = wl_meta.get("common_bootbinary")
        except (json.JSONDecodeError, OSError):
            wl_meta = None
        if not bootbinary:
            print(f"error: no --bootbinary given and workload JSON has no "
                  f"common_bootbinary", file=sys.stderr)
            return 2

    # Optional staging source — daemon cp's this into
    # deploy/workloads/<workload>/<bootbinary> right before infrasetup.
    # We validate existence here so submit fails fast instead of the
    # daemon picking up a stale stage path.
    stage_from = None
    if args.stage_from:
        stage_from = str(pathlib.Path(args.stage_from).resolve())
        if not pathlib.Path(stage_from).exists():
            print(f"error: --stage-from {stage_from} doesn't exist",
                  file=sys.stderr)
            return 2

    # Optional exact hardware-database authority.  The submitter commits the
    # source bytes by hash now; the daemon re-hashes and snapshots them into
    # the job directory under the FPGA lock immediately before any FireSim
    # phase.  FireSim is then invoked with -a <snapshot>, never the mutable
    # shared config_hwdb.yaml.  Keeping this optional preserves existing
    # non-evidence jobs; provenance-sensitive callers require it downstream.
    hwdb_config_artifact = None
    hwdb_config_artifact_sha256 = None
    if getattr(args, "hwdb_config_artifact", None):
        if not getattr(args, "hw_config", None):
            print("error: --hwdb-config-artifact requires --hw-config",
                  file=sys.stderr)
            return 2
        hwdb_path = pathlib.Path(args.hwdb_config_artifact).resolve()
        if not hwdb_path.is_file():
            print(f"error: --hwdb-config-artifact is not a file: {hwdb_path}",
                  file=sys.stderr)
            return 2
        try:
            hwdb_raw = hwdb_path.read_bytes()
        except OSError as exc:
            print(f"error: cannot read --hwdb-config-artifact {hwdb_path}: {exc}",
                  file=sys.stderr)
            return 2
        if not hwdb_raw:
            print(f"error: --hwdb-config-artifact is empty: {hwdb_path}",
                  file=sys.stderr)
            return 2
        hwdb_config_artifact = str(hwdb_path)
        hwdb_config_artifact_sha256 = hashlib.sha256(hwdb_raw).hexdigest()

    now = time.time()
    user = args.user or getpass.getuser()
    # The "cmd" column gets a human-readable summary so existing TUIs +
    # `status` output don't break. Daemon dispatch reads kind_args, not cmd.
    cmd_summary = (f"runworkload-full workload={args.workload} "
                   f"bootbinary={bootbinary} "
                   f"timeout={args.timeout}s")
    if stage_from:
        cmd_summary += f" stage_from={os.path.basename(stage_from)}"
    if hwdb_config_artifact:
        cmd_summary += (" hwdb_config_artifact="
                        f"{os.path.basename(hwdb_config_artifact)}")

    cwd = str(deploy)
    env_keep = {}
    for k in ("PATH", "HOME", "USER", "LOGNAME", "SHELL", "TERM",
              "SSH_AUTH_SOCK"):
        if k in os.environ:
            env_keep[k] = os.environ[k]
    env_json = json.dumps(env_keep)

    project = (args.project if getattr(args, "project", None)
               else _infer_project(cwd, cmd_summary))

    kind_args = json.dumps({
        "chipyard": str(chipyard),
        "workload": args.workload,
        "bootbinary": bootbinary,
        "stage_from": stage_from,
        "timeout": int(args.timeout),
        # None = inherit the chipyard template's default_hw_config (previous behaviour).
        "hw_config": getattr(args, "hw_config", None),
        "hwdb_config_artifact": hwdb_config_artifact,
        "hwdb_config_artifact_sha256": hwdb_config_artifact_sha256,
    })

    cur = conn.execute(
        "INSERT INTO jobs(user, priority, submitted_at, state, cmd, cwd,"
        " env_json, workdir, project, kind, kind_args, owner_uid, phase)"
        " VALUES(?, ?, ?, 'QUEUED', ?, ?, ?, '', ?, 'runworkload-full',"
        " ?, ?, 'PENDING')",
        (user, args.priority, now, cmd_summary, cwd, env_json, project,
         kind_args, os.geteuid()))
    job_id = cur.lastrowid
    workdir = JOBS_DIR / str(job_id)
    workdir.mkdir(parents=True, exist_ok=True)
    _share_dir(workdir)
    # Drop a human-readable description of what the daemon will run —
    # mirrors the legacy `cmd.sh` so debugging stays similar.
    (workdir / "runworkload-full.json").write_text(
        json.dumps({"job_id": job_id, "user": user,
                    "priority": args.priority, "project": project,
                    **json.loads(kind_args)}, indent=2))
    conn.execute("UPDATE jobs SET workdir=? WHERE id=?",
                 (str(workdir), job_id))

    print(f"[firesim-queue] job_id={job_id} kind=runworkload-full "
          f"user={user} priority={args.priority} project={project} "
          f"state=QUEUED")
    print(f"[firesim-queue]   chipyard={chipyard}")
    print(f"[firesim-queue]   workload={args.workload} "
          f"bootbinary={bootbinary}")
    if stage_from:
        print(f"[firesim-queue]   stage_from={stage_from}")
    if hwdb_config_artifact:
        print("[firesim-queue]   hwdb_config_artifact="
              f"{hwdb_config_artifact} sha256={hwdb_config_artifact_sha256}")
    print(f"[firesim-queue]   timeout={args.timeout}s")
    eta, kind, n = _estimate_eta(conn, cmd_summary)
    if eta is not None:
        print(f"[firesim-queue]   ETA {int(eta)}s "
              f"(median of {n} {kind} past run{'s' if n != 1 else ''})")
    else:
        print(f"[firesim-queue]   ETA unknown (no similar past jobs)")

    if args.background:
        return 0

    # Foreground: same polling loop as cmd_submit. The daemon writes
    # progress into stdout.log so the user sees infrasetup output etc.
    stdout_path = workdir / "stdout.log"
    stderr_path = workdir / "stderr.log"
    stdout_pos = 0
    stderr_pos = 0
    last_state = None
    last_phase = None
    last_status_print = 0.0

    while True:
        row = conn.execute(
            "SELECT state, started_at, ended_at, exit_code, phase"
            " FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            print(f"[firesim-queue] job_id={job_id} disappeared from queue",
                  file=sys.stderr)
            return 1
        state, started_at, ended_at, exit_code, phase = row

        for p, pos_attr in ((stdout_path, "stdout_pos"),
                            (stderr_path, "stderr_pos")):
            if p.exists():
                pos = stdout_pos if pos_attr == "stdout_pos" else stderr_pos
                size = p.stat().st_size
                if size > pos:
                    with p.open("rb") as f:
                        f.seek(pos)
                        chunk = f.read(size - pos)
                    sink = sys.stdout if pos_attr == "stdout_pos" else sys.stderr
                    sink.buffer.write(chunk)
                    sink.flush()
                    if pos_attr == "stdout_pos":
                        stdout_pos = size
                    else:
                        stderr_pos = size

        if state != last_state or phase != last_phase:
            phase_s = f" phase={phase}" if phase else ""
            print(f"[firesim-queue] job_id={job_id} state={state}{phase_s}",
                  file=sys.stderr)
            last_state = state
            last_phase = phase

        if state in ("DONE", "FAILED", "CANCELLED", "TIMEOUT"):
            wall = (ended_at - started_at) if (ended_at and started_at) else None
            wall_s = f" wall={wall:.1f}s" if wall else ""
            print(f"[firesim-queue] job_id={job_id} terminal state={state}"
                  f" exit_code={exit_code}{wall_s}",
                  file=sys.stderr)
            return int(exit_code or (0 if state == "DONE" else 1))

        now2 = time.time()
        if state == "QUEUED" and now2 - last_status_print > 60:
            q_ahead = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE state='QUEUED'"
                " AND (priority > ? OR (priority = ? AND submitted_at < ?))",
                (args.priority, args.priority, now)).fetchone()[0]
            running = conn.execute(
                "SELECT id, user FROM jobs WHERE state='RUNNING'").fetchone()
            running_str = (f"running: id={running[0]} user={running[1]}"
                           if running else "no job running")
            print(f"[firesim-queue] job_id={job_id} still QUEUED: "
                  f"{q_ahead} jobs ahead; {running_str}", file=sys.stderr)
            last_status_print = now2

        time.sleep(POLL_INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# status / cancel / tail
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> int:
    conn = _connect()
    alive, age = _daemon_alive(conn)
    print(f"[firesim-queue] daemon: "
          + ("ALIVE" if alive else "DOWN")
          + (f" (heartbeat {age:.0f}s ago)" if age is not None else ""))

    where = "state IN ('QUEUED', 'RUNNING')"
    params: tuple = ()
    if args.all:
        where = "1=1"
    if args.user:
        where += " AND user=?"
        params = (args.user,)
    rows = conn.execute(
        f"SELECT id, user, priority, state, submitted_at, started_at, ended_at,"
        f" exit_code, cmd, phase FROM jobs WHERE {where}"
        f" ORDER BY CASE state WHEN 'RUNNING' THEN 0 WHEN 'QUEUED' THEN 1"
        f" ELSE 2 END, priority DESC, submitted_at",
        params).fetchall()
    if not rows:
        print("[firesim-queue] no jobs match")
        return 0
    print(f"{'id':>5} {'user':<12} {'prio':>4} {'state':<10} "
          f"{'phase':<11} {'wall_s':>8} {'rc':>3} cmd")
    now = time.time()
    for r in rows:
        (jid, user, prio, state, sub, started, ended, rc, cmd, phase) = r
        if state == "RUNNING" and started:
            wall = now - started
        elif started and ended:
            wall = ended - started
        else:
            wall = now - sub  # queued time
        cmd_short = cmd if len(cmd) <= 80 else cmd[:77] + "..."
        rc_str = str(rc) if rc is not None else "-"
        phase_s = phase or "-"
        print(f"{jid:>5} {user:<12} {prio:>4} {state:<10} "
              f"{phase_s:<11} {wall:>8.0f} {rc_str:>3} {cmd_short}")
    return 0


def cmd_cancel(args: argparse.Namespace) -> int:
    """Cancel a job by ID. Owner-only by default.

    Semantics by state:
      QUEUED  → atomic UPDATE to state='CANCELLED'; daemon skips it on
                next dispatch.
      RUNNING → set cancel_requested=now; the daemon's per-job watcher
                polls this flag, SIGTERMs the active subprocess group,
                runs the trailing `firesim kill` for cleanup, then
                marks state='CANCELLED'. The caller blocks until that
                transition is visible (or --no-wait is passed).
      anything else → already terminal; no-op + warning.

    Ownership: rejects when the requesting UID doesn't match the job's
    owner_uid. Legacy rows (no owner_uid) fall back to username
    comparison. `--force` bypasses the check, intended for ops use only.
    """
    conn = _connect()
    row = conn.execute(
        "SELECT user, state, owner_uid FROM jobs WHERE id=?",
        (args.job_id,)).fetchone()
    if row is None:
        print(f"error: job {args.job_id} not found", file=sys.stderr)
        return 1
    user, state, owner_uid = row
    me_uid = os.geteuid()
    me = getpass.getuser()
    # Owner check: prefer UID match when available, fall back to username
    # for legacy rows where owner_uid was NULL at insert time.
    is_owner = ((owner_uid is not None and int(owner_uid) == me_uid)
                or (owner_uid is None and user == me))
    if not is_owner and not args.force:
        print(f"error: job {args.job_id} owned by {user}"
              + (f" (uid={owner_uid})" if owner_uid is not None else "")
              + f", not you (uid={me_uid}). Pass --force to override.",
              file=sys.stderr)
        return 1

    if state == "QUEUED":
        conn.execute(
            "UPDATE jobs SET state='CANCELLED', ended_at=? WHERE id=?",
            (time.time(), args.job_id))
        print(f"[firesim-queue] cancelled job {args.job_id} (was QUEUED)")
        return 0

    if state == "RUNNING":
        # If no daemon is alive, there is no watcher to act on cancel_requested and no
        # live job process to tear down (the daemon owned it and is gone, releasing the
        # FPGA flock). Mark the job terminal directly so cancel can't hang for 120s on a
        # dead daemon (and so the orphan doesn't keep wedging the queue until a restart).
        alive, _age = _daemon_alive(conn)
        if not alive:
            conn.execute(
                "UPDATE jobs SET state='CANCELLED', ended_at=?, pid=NULL WHERE id=?",
                (time.time(), args.job_id))
            print(f"[firesim-queue] daemon is down — marked orphaned RUNNING job "
                  f"{args.job_id} CANCELLED directly (no live process to tear down)")
            return 0
        conn.execute(
            "UPDATE jobs SET cancel_requested=? WHERE id=?",
            (time.time(), args.job_id))
        print(f"[firesim-queue] cancel requested for RUNNING job "
              f"{args.job_id} — daemon will tear down (kill + firesim kill)")
        if args.no_wait:
            return 0
        # Block until the daemon picks up the flag and transitions us
        # to a terminal state. The daemon's watcher checks every
        # POLL_INTERVAL_SECONDS so this should take a few seconds at most
        # before the SIGTERM lands; the trailing `firesim kill` cleanup
        # can take ~10-30s depending on bitstream state.
        deadline = time.time() + 120  # 2-min grace
        while time.time() < deadline:
            r = conn.execute("SELECT state FROM jobs WHERE id=?",
                             (args.job_id,)).fetchone()
            if r and r[0] in ("DONE", "FAILED", "CANCELLED", "TIMEOUT"):
                print(f"[firesim-queue] job {args.job_id} now {r[0]}")
                return 0
            time.sleep(1.0)
        print(f"[firesim-queue] warning: job {args.job_id} still not "
              f"terminal after 120s; daemon may be stuck. Check "
              f"`firesim-queue status`.", file=sys.stderr)
        return 1

    # Already terminal.
    print(f"warning: job {args.job_id} is already {state}; nothing to do",
          file=sys.stderr)
    return 0


def cmd_tail(args: argparse.Namespace) -> int:
    conn = _connect()
    row = conn.execute(
        "SELECT workdir, state, cwd FROM jobs WHERE id=?",
        (args.job_id,)).fetchone()
    if row is None:
        print(f"error: job {args.job_id} not found", file=sys.stderr)
        return 1
    workdir, state, cwd = row

    if args.uartlog:
        # Stream the FireSim uartlog instead of the queue's captured stdout.
        # The uartlog is what users normally see when attaching via
        # `screen -r fsim0` — but it's a plain file (no per-UID socket
        # restriction), so any user with read perms can `tail -f` it.
        # Find the latest results-workload subdir under the job's cwd.
        results_root = pathlib.Path(cwd) / "results-workload"
        if not results_root.exists():
            # cwd might already be sims/firesim; try the canonical location.
            results_root = pathlib.Path(cwd).parent / "results-workload"
        if not results_root.exists():
            print(f"[firesim-queue] couldn't find results-workload/ under "
                  f"{cwd}. Set --uartlog-path explicitly.", file=sys.stderr)
            return 1
        candidates = [d for d in results_root.iterdir() if d.is_dir()]
        if not candidates:
            print(f"[firesim-queue] no result dirs under {results_root}",
                  file=sys.stderr)
            return 1
        latest = max(candidates, key=lambda d: d.stat().st_mtime)
        # Find the first uartlog file inside the latest result dir tree.
        uartlogs = list(latest.rglob("uartlog"))
        if not uartlogs:
            print(f"[firesim-queue] no uartlog found under {latest}",
                  file=sys.stderr)
            return 1
        p = uartlogs[0]
        print(f"[firesim-queue] tailing FireSim uartlog: {p}",
              file=sys.stderr)
    else:
        p = pathlib.Path(workdir) / "stdout.log"
        if not p.exists():
            print(f"[firesim-queue] job {args.job_id} state={state}; "
                  "no stdout yet (pass --uartlog if the FireSim run has "
                  "started)")
            return 0
    try:
        proc = subprocess.run(["tail", "-f" if args.follow else "-n",
                               "+1" if args.follow else "200", str(p)])
        return proc.returncode
    except KeyboardInterrupt:
        return 130


# ---------------------------------------------------------------------------
# daemon
# ---------------------------------------------------------------------------

def _write_pid():
    DAEMON_PID.write_text(f"{os.getpid()}\n")


def _heartbeat(conn: sqlite3.Connection):
    _kv_set(conn, HEARTBEAT_KEY, f"{time.time()}")


def _run_one_job(conn: sqlite3.Connection, job: dict) -> int:
    """Run the picked job. Holds the FPGA flock for the duration.

    Dispatches on `kind`:
      - 'bash'              → legacy freeform: just exec cmd.sh
      - 'runworkload-full'  → atomic FireSim lifecycle (daemon owns
                              stage → kill → infrasetup → runworkload
                              → kill, with per-job config_runtime.yaml
                              and timeout enforcement).
    """
    kind = job.get("kind") or "bash"
    if kind == "runworkload-full":
        return _run_one_job_runworkload_full(conn, job)
    return _run_one_job_bash(conn, job)


def _run_one_job_bash(conn: sqlite3.Connection, job: dict) -> int:
    """Legacy freeform path: execute cmd.sh under the FPGA lock."""
    workdir = pathlib.Path(job["workdir"])
    stdout_path = workdir / "stdout.log"
    stderr_path = workdir / "stderr.log"
    env = json.loads(job["env_json"])
    final_env = dict(os.environ)
    final_env.update(env)

    started = time.time()
    conn.execute(
        "UPDATE jobs SET state='RUNNING', started_at=? WHERE id=?",
        (started, job["id"]))

    FPGA_LOCK.touch(exist_ok=True)
    with FPGA_LOCK.open("w") as lock_f:
        try:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        except OSError as e:
            print(f"[daemon] failed to acquire FPGA lock: {e}", flush=True)
            conn.execute(
                "UPDATE jobs SET state='FAILED', ended_at=?, exit_code=-1"
                " WHERE id=?", (time.time(), job["id"]))
            return -1

        try:
            with stdout_path.open("wb") as so, stderr_path.open("wb") as se:
                proc = subprocess.Popen(
                    ["bash", str(workdir / "cmd.sh")],
                    cwd=job["cwd"], env=final_env,
                    stdout=so, stderr=se,
                    start_new_session=True)
                conn.execute("UPDATE jobs SET pid=? WHERE id=?",
                             (proc.pid, job["id"]))
                # Watch loop: heartbeat + check for owner-requested cancel.
                # Legacy bash path treats cancel as "SIGTERM the process
                # group". No firesim kill cleanup (the bash blob is opaque
                # to us — caller should have included it).
                while True:
                    rc = proc.poll()
                    if rc is not None:
                        break
                    cancel_row = conn.execute(
                        "SELECT cancel_requested FROM jobs WHERE id=?",
                        (job["id"],)).fetchone()
                    if cancel_row and cancel_row[0] is not None:
                        print(f"[daemon] job_id={job['id']} cancel requested"
                              " — SIGTERM process group", flush=True)
                        try:
                            os.killpg(proc.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        try:
                            rc = proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                            rc = proc.wait()
                        ended = time.time()
                        conn.execute(
                            "UPDATE jobs SET state='CANCELLED', ended_at=?,"
                            " exit_code=?, pid=NULL,"
                            " cancel_requested=NULL WHERE id=?",
                            (ended, rc, job["id"]))
                        _kv_set(conn, "last_user_served", job["user"])
                        return rc
                    _heartbeat(conn)
                    time.sleep(min(HEARTBEAT_INTERVAL_SECONDS,
                                   POLL_INTERVAL_SECONDS))
                ended = time.time()
                state = "DONE" if rc == 0 else "FAILED"
                conn.execute(
                    "UPDATE jobs SET state=?, ended_at=?, exit_code=?,"
                    " pid=NULL WHERE id=?",
                    (state, ended, rc, job["id"]))
                _kv_set(conn, "last_user_served", job["user"])
                wall = ended - started
                print(f"[daemon] job_id={job['id']} user={job['user']} "
                      f"prio={job['priority']} state={state} rc={rc} "
                      f"wall={wall:.1f}s", flush=True)
                return rc
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# runworkload-full daemon-side dispatcher
# ---------------------------------------------------------------------------

def _set_phase(conn: sqlite3.Connection, job_id: int, phase: str):
    """Update the phase column + heartbeat. Phase transitions are how
    the TUI surfaces what the daemon is currently doing inside the
    atomic op."""
    conn.execute("UPDATE jobs SET phase=? WHERE id=?", (phase, job_id))


def _render_per_job_runtime_yaml(template_path: pathlib.Path,
                                 out_path: pathlib.Path,
                                 workload_name: str, suffix_tag: str,
                                 hw_config: str | None = None,
                                 simulation_dir: str | None = None):
    """Write a per-job config_runtime.yaml for this run.

    We don't parse YAML (avoids a dep). Instead we copy the chipyard
    template and rewrite the two `workload:` keys we care about with a
    line-oriented edit. The keys we touch:

      workload:
          workload_name: <X>
          suffix_tag: <Y>
      target_config:
          default_hw_config: <Z>      # only when --hw-config is given
      run_farm.recipe_arg_overrides:
          default_simulation_dir: <job-private path>

    `default_hw_config` names the BITSTREAM the job flashes. Left to the shared
    template it is global mutable state: two users wanting different designs
    race, and the loser silently runs the other's hardware. Passing it per job
    makes the design a property of the job, like the workload already is.

    The template's other settings (target_config, tracing, host_debug,
    etc.) flow through unchanged — that's intentional. Anyone wanting
    different settings can either edit the chipyard template or extend
    this renderer.
    """
    # Best-effort: if the template doesn't exist (fresh chipyard install
    # with no config_runtime.yaml yet), write a minimal one. FireSim's
    # firesim CLI populates config_runtime.yaml on first sourceme run,
    # so this should usually exist.
    if not template_path.exists():
        hw = f"target_config:\n    default_hw_config: {hw_config}\n" if hw_config else ""
        farm = ("run_farm:\n  recipe_arg_overrides:\n"
                f"    default_simulation_dir: {simulation_dir}\n"
                if simulation_dir else "")
        out_path.write_text(
            farm + hw
            + f"workload:\n"
            f"    workload_name: {workload_name}.json\n"
            f"    terminate_on_completion: no\n"
            f"    suffix_tag: {suffix_tag}\n"
        )
        return

    lines = template_path.read_text().splitlines()
    out_lines: list[str] = []
    in_workload = False
    in_target = False
    saw_name = False
    saw_suffix = False
    saw_hw = False
    for line in lines:
        stripped = line.lstrip()
        if simulation_dir and stripped.startswith("default_simulation_dir:"):
            indent = line[:len(line) - len(stripped)]
            out_lines.append(f"{indent}default_simulation_dir: {simulation_dir}")
            continue
        # `target_config:` is tracked exactly like `workload:` below: enter on the top-level key,
        # leave on the next one. Only the default_hw_config line inside it is rewritten.
        if line.startswith("target_config:"):
            in_target = True
            out_lines.append(line)
            continue
        if in_target and line and not line[0].isspace() and not line.startswith("#"):
            if hw_config and not saw_hw:
                out_lines.append(f"    default_hw_config: {hw_config}")
                saw_hw = True
            in_target = False
        if in_target and hw_config and stripped.startswith("default_hw_config:"):
            indent = line[:len(line) - len(stripped)]
            out_lines.append(f"{indent}default_hw_config: {hw_config}")
            saw_hw = True
            continue
        # Track when we enter / exit the `workload:` mapping. A new
        # top-level key (no indent) ends the section.
        if line.startswith("workload:"):
            in_workload = True
            out_lines.append(line)
            continue
        if in_workload and line and not line[0].isspace() and not line.startswith("#"):
            # New top-level key — emit any missing fields before leaving.
            if not saw_name:
                out_lines.append(f"    workload_name: {workload_name}.json")
                saw_name = True
            if not saw_suffix:
                out_lines.append(f"    suffix_tag: {suffix_tag}")
                saw_suffix = True
            in_workload = False
        if in_workload:
            if stripped.startswith("workload_name:"):
                indent = line[:len(line) - len(stripped)]
                out_lines.append(f"{indent}workload_name: {workload_name}.json")
                saw_name = True
                continue
            if stripped.startswith("suffix_tag:"):
                indent = line[:len(line) - len(stripped)]
                out_lines.append(f"{indent}suffix_tag: {suffix_tag}")
                saw_suffix = True
                continue
        out_lines.append(line)
    if in_target and hw_config and not saw_hw:
        out_lines.append(f"    default_hw_config: {hw_config}")
    # Trailing flush if `workload:` was the last top-level key.
    if in_workload:
        if not saw_name:
            out_lines.append(f"    workload_name: {workload_name}.json")
        if not saw_suffix:
            out_lines.append(f"    suffix_tag: {suffix_tag}")
    out_path.write_text("\n".join(out_lines) + "\n")


def _deploy_overlay(workdir: pathlib.Path, deploy: pathlib.Path,
                    workload: str, bootbinary: str) -> pathlib.Path:
    """Build a per-job stand-in for the submitter's `deploy/` directory.

    FireSim resolves exactly three things relative to its cwd, and it *writes*
    all three: the workload inputs (`workloads/`, runtools/workload.py:113),
    the results (`results-workload/`, workload.py:169) and its own logs
    (`logs/`, deploy/firesim:505). Everything else is either absolute or comes
    in through a -c/-a/-b/-r flag.

    When the daemon and the submitter are different people, those three writes
    land in a tree the daemon has no write bit on. That is what killed jobs
    419 and 422 -- and no amount of chmod on the *queue* fixes it, because the
    directory being written belongs to the submitter's chipyard checkout.

    So: symlink the deploy dir into the job dir and replace those three
    entries with real directories the daemon owns. FireSim sees an ordinary
    deploy tree; the writes land in the queue root, under the job that caused
    them. Read-only state (config_hwdb.yaml, build recipes, the sim binaries)
    is reached through the symlinks and stays shared, as it should be.
    """
    overlay = workdir / "deploy_overlay"
    overlay.mkdir(parents=True, exist_ok=True)

    def _link(src: pathlib.Path, dst: pathlib.Path) -> None:
        if not dst.is_symlink() and not dst.exists():
            dst.symlink_to(src)

    # Everything except the three cwd-relative writers is a symlink.
    private = {"workloads", "logs", "results-workload"}
    for entry in deploy.iterdir():
        if entry.name not in private:
            _link(entry, overlay / entry.name)
    for name in ("logs", "results-workload"):
        (overlay / name).mkdir(exist_ok=True)

    # `workloads/` is half theirs, half ours: the daemon must be able to write
    # the staged bootbinary, but every other workload JSON in the submitter's
    # tree has to stay visible or FireSim cannot resolve the run at all. Mirror
    # the directory as symlinks, then carve out just the one workload.
    wl = overlay / "workloads"
    wl.mkdir(exist_ok=True)
    src_wl = deploy / "workloads"
    if src_wl.is_dir():
        for entry in src_wl.iterdir():
            if entry.name != workload:
                _link(entry, wl / entry.name)

    # Same trick one level down: keep the workload's rootfs and friends
    # reachable, own only the bootbinary we are about to stage over.
    dst_wl = wl / workload
    dst_wl.mkdir(exist_ok=True)
    src_wl_dir = src_wl / workload
    if src_wl_dir.is_dir():
        for entry in src_wl_dir.iterdir():
            if entry.name != bootbinary:
                _link(entry, dst_wl / entry.name)
    return overlay


def _run_one_job_runworkload_full(conn: sqlite3.Connection,
                                  job: dict) -> int:
    """Atomic FireSim lifecycle. Daemon owns every firesim CLI call.

    Phases:
      STAGING     — cp --stage-from into deploy/workloads/<workload>/<bin>
      INFRASETUP  — firesim kill (best-effort) + firesim -c <yaml> infrasetup
      RUNNING     — firesim -c <yaml> runworkload (with timeout watcher)
      TEARDOWN    — firesim kill (always, regardless of upstream phase rc)

    State machine outcomes:
      DONE       — every phase rc=0
      FAILED     — any phase rc != 0 (TEARDOWN still ran)
      CANCELLED  — owner set cancel_requested mid-RUNNING
      TIMEOUT    — wall-clock --timeout fired mid-RUNNING

    The FPGA lock is held continuously across all four phases. xdma
    rmmod/insmod inside firesim's deploy manager is therefore safe
    against concurrent users (FireSim itself has zero serialization
    for this — confirmed by surveying the firesim repo).
    """
    workdir = pathlib.Path(job["workdir"])
    stdout_path = workdir / "stdout.log"
    stderr_path = workdir / "stderr.log"
    job_id = job["id"]
    kind_args = json.loads(job["kind_args"] or "{}")
    chipyard = pathlib.Path(kind_args["chipyard"])
    workload = kind_args["workload"]
    bootbinary = kind_args["bootbinary"]
    stage_from = kind_args.get("stage_from")
    hwdb_config_artifact = kind_args.get("hwdb_config_artifact")
    hwdb_config_artifact_sha256 = kind_args.get(
        "hwdb_config_artifact_sha256")
    # timeout==0 means "no cap" — let firesim runworkload exit on its
    # own. Anything > 0 arms the daemon's wall-clock SIGTERM timer.
    timeout_s = int(kind_args.get("timeout", 0))

    firesim_root = chipyard / "sims" / "firesim"
    deploy = firesim_root / "deploy"
    env_script = chipyard / "env.sh"
    sourceme = firesim_root / "sourceme-manager.sh"
    template_yaml = deploy / "config_runtime.yaml"

    # Cross-user dispatch: when the daemon cannot write the submitter's
    # deploy tree, run against a per-job overlay of it instead. Same-user
    # jobs keep the exact pre-overlay behaviour -- this path only engages
    # where the old one already failed outright, so it cannot regress the
    # common case.
    if os.access(deploy, os.W_OK):
        run_dir = deploy
    else:
        run_dir = _deploy_overlay(workdir, deploy, workload, bootbinary)
        print(f"[daemon] job_id={job['id']} user={job['user']}: {deploy} is not"
              f" writable by this daemon; running under overlay {run_dir}",
              flush=True)

    # Per-job artifacts under the queue's jobs dir. The per-job
    # config_runtime.yaml is what we hand to `firesim -c` so we never
    # mutate the shared deploy/config_runtime.yaml.
    per_job_yaml = workdir / "config_runtime.yaml"
    per_job_hwdb = None
    suffix_tag = f"q{job_id}"
    simulation_dir = workdir / "simulation"
    simulation_dir.mkdir(exist_ok=True)
    _render_per_job_runtime_yaml(
        template_yaml,
        per_job_yaml,
        workload,
        suffix_tag,
        hw_config=kind_args.get("hw_config"),
        simulation_dir=str(simulation_dir),
    )

    env = json.loads(job["env_json"])
    final_env = dict(os.environ)
    final_env.update(env)

    started = time.time()
    conn.execute(
        "UPDATE jobs SET state='RUNNING', started_at=? WHERE id=?",
        (started, job_id))

    # Open the FPGA lock; HOLD across STAGING → TEARDOWN. Releasing
    # before TEARDOWN finishes would let the next job's infrasetup
    # race with our trailing kill.
    FPGA_LOCK.touch(exist_ok=True)

    so = stdout_path.open("wb")
    se = stderr_path.open("wb")
    # A prelaunch validation/staging failure must not touch FireSim at all.
    # Once the leading kill begins, however, trailing kill remains mandatory.
    firesim_started = False
    hwdb_snapshot_ready = not bool(hwdb_config_artifact)

    def _write_banner(label: str):
        so.write(f"\n=== [firesim-queue] phase={label} "
                 f"job_id={job_id} ===\n".encode())
        so.flush()

    # Wrapper that sources chipyard env + sourceme then runs the
    # given firesim subcommand. We use bash -c so the env sourcing
    # works the same way it does interactively.
    def _firesim_cmd(*subcmd: str, extra_setup: str = "") -> list[str]:
        # Note: we pass our per-job YAML via `-c <abs path>`. That's
        # the FireSim CLI's documented per-invocation config override
        # (deploy/firesim:398). Eliminates the shared-YAML race.
        sub = " ".join(shlex.quote(s) for s in subcmd)
        per_yaml = shlex.quote(str(per_job_yaml))
        env_q = shlex.quote(str(env_script))
        sm_q = shlex.quote(str(sourceme))
        # run_dir is `deploy` for same-user jobs and the per-job overlay for
        # cross-user ones (see _deploy_overlay).
        deploy_q = shlex.quote(str(run_dir))
        # Drop any inherited conda state so chipyard's activate is a
        # clean operation. Same pattern as the legacy bash blob, just
        # owned by the daemon now.
        if per_job_hwdb is not None and not hwdb_snapshot_ready:
            raise RuntimeError(
                "refusing FireSim command before immutable HWDB snapshot")
        hwdb_arg = (f"-a {shlex.quote(str(per_job_hwdb))} "
                    if per_job_hwdb is not None else "")
        inner = (
            f"set -e; "
            f"unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER "
            f"CONDA_PYTHON_EXE CONDA_SHLVL CONDA_EXE _CE_M _CE_CONDA; "
            f"export PATH=/scratch2/agustin/miniforge3/condabin:$PATH; "
            f"source {env_q}; "
            f"cd {shlex.quote(str(firesim_root))}; "
            f"source {sm_q} --skip-ssh-setup; "
            f"cd {deploy_q}; "
            f"{extra_setup}"
            f"firesim -c {per_yaml} {hwdb_arg}{sub}"
        )
        return ["bash", "-c", inner]

    def _verify_hwdb_snapshot(phase: str) -> str:
        if per_job_hwdb is None and not hwdb_config_artifact:
            return ""
        if per_job_hwdb is None or not hwdb_snapshot_ready:
            raise RuntimeError(
                "refusing FireSim command before immutable HWDB snapshot")
        try:
            observed = hashlib.sha256(per_job_hwdb.read_bytes()).hexdigest()
        except OSError as exc:
            raise RuntimeError(
                f"cannot verify immutable HWDB snapshot before {phase}: {exc}"
            ) from exc
        if observed != hwdb_config_artifact_sha256:
            raise RuntimeError(
                f"immutable HWDB snapshot changed before {phase}: expected "
                f"{hwdb_config_artifact_sha256}, observed {observed}"
            )
        so.write(
            f"hwdb verify phase={phase} sha256={observed} "
            f"path={per_job_hwdb}\n".encode()
        )
        so.flush()
        return observed

    def _check_cancel() -> bool:
        row = conn.execute("SELECT cancel_requested FROM jobs WHERE id=?",
                           (job_id,)).fetchone()
        return bool(row and row[0] is not None)

    def _run_phase(phase: str, argv: list[str],
                   poll_cancel: bool = False,
                   wall_timeout: float | None = None) -> tuple[int, str]:
        """Run argv as a subprocess, streaming output to so/se. Returns
        (rc, final_state_hint). final_state_hint is 'DONE'/'CANCELLED'/
        'TIMEOUT' to indicate why the phase ended; the caller maps to
        terminal state."""
        _set_phase(conn, job_id, phase)
        _write_banner(phase)
        deadline = (time.time() + wall_timeout) if wall_timeout else None
        proc = subprocess.Popen(
            argv, cwd=str(run_dir), env=final_env,
            stdout=so, stderr=se, start_new_session=True)
        conn.execute("UPDATE jobs SET pid=? WHERE id=?", (proc.pid, job_id))
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            if poll_cancel and _check_cancel():
                print(f"[daemon] job_id={job_id} cancel requested in "
                      f"phase={phase} — SIGTERM", flush=True)
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    rc = proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    rc = proc.wait()
                return rc, "CANCELLED"
            if deadline and time.time() > deadline:
                print(f"[daemon] job_id={job_id} timeout in phase={phase} "
                      f"({wall_timeout}s elapsed) — SIGTERM", flush=True)
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    rc = proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    rc = proc.wait()
                return rc, "TIMEOUT"
            _heartbeat(conn)
            time.sleep(min(HEARTBEAT_INTERVAL_SECONDS,
                           POLL_INTERVAL_SECONDS))
        return rc, "DONE"

    final_state = "DONE"
    overall_rc = 0
    try:
        with FPGA_LOCK.open("w") as lock_f:
            try:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            except OSError as e:
                print(f"[daemon] failed to acquire FPGA lock: {e}",
                      flush=True)
                conn.execute(
                    "UPDATE jobs SET state='FAILED', ended_at=?,"
                    " exit_code=-1, phase='LOCK_FAILED' WHERE id=?",
                    (time.time(), job_id))
                return -1

            try:
                # ------ STAGING ------
                _set_phase(conn, job_id, "STAGING")
                _write_banner("STAGING")
                if bool(hwdb_config_artifact) != bool(
                        hwdb_config_artifact_sha256):
                    se.write(b"incomplete HWDB artifact commitment in job "
                             b"spec\n")
                    overall_rc = 1
                    final_state = "FAILED"
                    raise RuntimeError("incomplete HWDB artifact commitment")
                if hwdb_config_artifact:
                    if (not isinstance(hwdb_config_artifact_sha256, str)
                            or len(hwdb_config_artifact_sha256) != 64
                            or any(c not in "0123456789abcdef"
                                   for c in hwdb_config_artifact_sha256)):
                        se.write(b"invalid committed HWDB artifact SHA-256 "
                                 b"in job spec\n")
                        overall_rc = 1
                        final_state = "FAILED"
                        raise RuntimeError("invalid HWDB artifact SHA-256")
                    hwdb_source = pathlib.Path(
                        str(hwdb_config_artifact)).resolve()
                    try:
                        hwdb_raw = hwdb_source.read_bytes()
                    except OSError as exc:
                        se.write(f"hwdb snapshot read failed: {exc}\n".encode())
                        overall_rc = 1
                        final_state = "FAILED"
                        raise RuntimeError("hwdb snapshot read failed")
                    observed_hwdb_sha256 = hashlib.sha256(hwdb_raw).hexdigest()
                    if observed_hwdb_sha256 != hwdb_config_artifact_sha256:
                        se.write(
                            b"hwdb source changed after submission: expected "
                            + str(hwdb_config_artifact_sha256).encode()
                            + b" observed "
                            + observed_hwdb_sha256.encode()
                            + b"\n"
                        )
                        overall_rc = 1
                        final_state = "FAILED"
                        raise RuntimeError("hwdb source changed after submission")
                    snapshot_root = _prepare_hwdb_snapshot_root()
                    per_job_hwdb = snapshot_root / f"job-{job_id}.yaml"
                    try:
                        # Exclusive creation makes a pre-created/reused job
                        # snapshot a hard failure rather than silently
                        # overwriting potentially untrusted bytes.
                        with per_job_hwdb.open("xb") as snapshot_f:
                            snapshot_f.write(hwdb_raw)
                            snapshot_f.flush()
                            os.fsync(snapshot_f.fileno())
                        per_job_hwdb.chmod(0o444)
                        snapshot_sha256 = hashlib.sha256(
                            per_job_hwdb.read_bytes()).hexdigest()
                    except OSError as exc:
                        se.write(f"hwdb snapshot write failed: {exc}\n".encode())
                        overall_rc = 1
                        final_state = "FAILED"
                        raise RuntimeError("hwdb snapshot write failed")
                    if snapshot_sha256 != hwdb_config_artifact_sha256:
                        se.write(b"hwdb snapshot hash differs after write\n")
                        overall_rc = 1
                        final_state = "FAILED"
                        raise RuntimeError("hwdb snapshot hash differs after write")
                    hwdb_snapshot_ready = True
                    so.write(
                        f"hwdb snapshot {hwdb_source} sha256="
                        f"{snapshot_sha256} -> {per_job_hwdb}\n".encode()
                    )
                    so.flush()
                if stage_from:
                    stage_dst_dir = run_dir / "workloads" / workload
                    stage_dst_dir.mkdir(parents=True, exist_ok=True)
                    stage_dst = stage_dst_dir / bootbinary
                    try:
                        shutil.copy2(stage_from, stage_dst)
                        so.write(f"staged {stage_from} -> {stage_dst}\n"
                                 .encode())
                    except OSError as e:
                        se.write(f"stage failed: {e}\n".encode())
                        overall_rc = 1
                        final_state = "FAILED"
                        raise RuntimeError("staging failed")
                else:
                    so.write(b"no --stage-from; assuming binary already "
                             b"staged by caller\n")

                # ------ INFRASETUP ------
                # Leading kill is best-effort: cleans any prior sim's
                # half-configured xdma / screen sessions. Then infrasetup
                # under our per-job YAML.
                _set_phase(conn, job_id, "INFRASETUP")
                _write_banner("INFRASETUP")
                # Best-effort kill, ignore its ordinary nonzero rc (there may
                # simply be no simulator).  It is still process-group bounded
                # and cancellation-aware: Fabric has been observed to wedge in
                # its local pkill path, and an unbounded subprocess.run here
                # made both the queue heartbeat and cancel request ineffective.
                _verify_hwdb_snapshot("before_leading_kill")
                kill_argv = _firesim_cmd("kill")
                firesim_started = True
                rc_kill, hint_kill = _run_phase(
                    "LEADING_KILL", kill_argv, poll_cancel=True,
                    wall_timeout=FIRESIM_KILL_TIMEOUT_SECONDS)
                if hint_kill == "CANCELLED":
                    overall_rc = rc_kill or 1
                    final_state = "CANCELLED"
                    raise RuntimeError("leading kill ended CANCELLED")
                if hint_kill == "TIMEOUT":
                    se.write(
                        f"warning: leading FireSim kill exceeded "
                        f"{FIRESIM_KILL_TIMEOUT_SECONDS}s; its process group "
                        f"was terminated and infrasetup will re-establish the "
                        f"requested image\n".encode())
                    se.flush()
                _verify_hwdb_snapshot("before_infrasetup")
                infra_argv = _firesim_cmd("infrasetup")
                rc_infra, hint_infra = _run_phase(
                    "INFRASETUP", infra_argv, poll_cancel=True,
                    wall_timeout=FIRESIM_INFRASETUP_TIMEOUT_SECONDS)
                if hint_infra != "DONE":
                    overall_rc = rc_infra or 1
                    final_state = hint_infra
                    raise RuntimeError(f"infrasetup ended {hint_infra}")
                if rc_infra != 0:
                    overall_rc = rc_infra
                    final_state = "FAILED"
                    raise RuntimeError(f"infrasetup rc={rc_infra}")

                # ------ RUNNING ------
                # timeout_s == 0 → no wall-clock cap (pass None to
                # _run_phase). > 0 → enforce.
                _verify_hwdb_snapshot("before_runworkload")
                run_argv = _firesim_cmd("runworkload")
                rc_run, hint_run = _run_phase(
                    "RUNNING", run_argv, poll_cancel=True,
                    wall_timeout=(float(timeout_s) if timeout_s > 0 else None))
                if hint_run == "DONE":
                    if rc_run != 0:
                        overall_rc = rc_run
                        final_state = "FAILED"
                else:
                    overall_rc = rc_run or 1
                    final_state = hint_run
            except RuntimeError as exc:
                # Phase signalled failure / cancel / timeout — fall
                # through to TEARDOWN (always runs).
                se.write(f"FireSim lifecycle failed: {exc}\n".encode())
                se.flush()
            finally:
                # ------ TEARDOWN ------
                # Always runs, regardless of upstream state. Best-effort.
                # Without this, the FPGA is left in an infrasetup'd state
                # → next job's leading kill picks it up, but we'd rather
                # release cleanly under our own lock.
                _set_phase(conn, job_id, "TEARDOWN")
                _write_banner("TEARDOWN")
                if firesim_started:
                    try:
                        _verify_hwdb_snapshot("before_trailing_kill")
                        kill_argv = _firesim_cmd("kill")
                        _rc_teardown, _hint_teardown = _run_phase(
                            "TEARDOWN", kill_argv, poll_cancel=False,
                            wall_timeout=FIRESIM_KILL_TIMEOUT_SECONDS)
                        if _hint_teardown == "TIMEOUT":
                            se.write(
                                f"FireSim teardown exceeded "
                                f"{FIRESIM_KILL_TIMEOUT_SECONDS}s; its process "
                                f"group was terminated\n".encode())
                            se.flush()
                            overall_rc = overall_rc or 1
                            if final_state == "DONE":
                                final_state = "FAILED"
                        _verify_hwdb_snapshot("after_trailing_kill")
                    except RuntimeError as exc:
                        se.write(f"FireSim teardown HWDB check failed: {exc}\n"
                                 .encode())
                        se.flush()
                        overall_rc = overall_rc or 1
                        final_state = "FAILED"
                else:
                    so.write(b"FireSim teardown skipped: no FireSim phase "
                             b"started\n")
                    so.flush()
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
    finally:
        so.close()
        se.close()

    ended = time.time()
    conn.execute(
        "UPDATE jobs SET state=?, ended_at=?, exit_code=?, pid=NULL,"
        " phase=?, cancel_requested=NULL WHERE id=?",
        (final_state, ended, overall_rc, final_state, job_id))
    _kv_set(conn, "last_user_served", job["user"])
    wall = ended - started
    print(f"[daemon] job_id={job_id} user={job['user']} "
          f"kind=runworkload-full state={final_state} rc={overall_rc} "
          f"wall={wall:.1f}s", flush=True)
    return overall_rc


def cmd_daemon(args: argparse.Namespace) -> int:
    # Refuse if another daemon is running.
    pid = _daemon_pid_alive()
    if pid is not None and pid != os.getpid():
        print(f"error: daemon already running (pid={pid})", file=sys.stderr)
        return 1
    _write_pid()
    conn = _connect()
    _heartbeat(conn)

    print(f"[daemon] firesim-queue daemon up pid={os.getpid()} "
          f"root={QUEUE_ROOT}", flush=True)

    # Crash recovery: a fresh daemon holds the only FPGA and has run nothing yet, so
    # any job left in RUNNING is orphaned — its daemon died mid-job (the FPGA flock is
    # released on process exit, so it is free now). Reconcile such jobs to FAILED so a
    # crashed-mid-job state can't wedge the queue or its usage stats, and so a pending
    # `cancel` isn't blocked forever on a watcher that will never run. (This is exactly
    # what stranded job 290 in RUNNING for ~4 days after the daemon died.)
    orphaned = conn.execute("SELECT id FROM jobs WHERE state='RUNNING'").fetchall()
    for (jid,) in orphaned:
        conn.execute(
            "UPDATE jobs SET state='FAILED', ended_at=?, exit_code=-1, pid=NULL"
            " WHERE id=?", (time.time(), jid))
        print(f"[daemon] reconciled orphaned RUNNING job {jid} -> FAILED "
              f"(no daemon was alive to run it)", flush=True)

    stop = {"flag": False}

    def _sig(s, f):
        print(f"[daemon] caught signal {s}, draining...", flush=True)
        stop["flag"] = True
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    last_idle_log = 0.0
    skip_ids: set[int] = set()
    while not stop["flag"]:
        _heartbeat(conn)
        job = _pick_next_job(conn, skip=skip_ids)
        if job is None:
            now = time.time()
            if now - last_idle_log > 120:
                print("[daemon] idle, queue empty", flush=True)
                last_idle_log = now
            time.sleep(POLL_INTERVAL_SECONDS)
            continue
        print(f"[daemon] dispatching job_id={job['id']} user={job['user']} "
              f"prio={job['priority']}", flush=True)
        try:
            _run_one_job(conn, job)
        except PermissionError as exc:
            # A daemon can only run jobs whose directory it can write, and job dirs are owned by
            # their submitters (mode 775). Before this, such a job took the DAEMON down -- the whole
            # queue stopped, for every user, on one unwritable file, and it happened again on every
            # restart because the same job was still at the head. Skip it for this daemon's lifetime
            # and keep serving: do not touch its state, because the job is not what is broken.
            print(f"[daemon] job_id={job['id']} user={job['user']} is not runnable by this daemon "
                  f"(cannot write its job dir: {exc}); skipping it and continuing. Its owner's "
                  f"daemon can still run it.", flush=True)
            skip_ids.add(job["id"])
        except Exception as exc:                                     # noqa: BLE001
            # Same reasoning for anything else unexpected: one bad job must not end the queue.
            import traceback
            print(f"[daemon] job_id={job['id']} raised {type(exc).__name__}: {exc}; skipping it "
                  f"and continuing", flush=True)
            traceback.print_exc()
            skip_ids.add(job["id"])

    print("[daemon] exiting", flush=True)
    try:
        DAEMON_PID.unlink()
    except FileNotFoundError:
        pass
    return 0


def cmd_stop_daemon(args: argparse.Namespace) -> int:
    pid = _daemon_pid_alive()
    if pid is None:
        print("[firesim-queue] daemon not running")
        return 0
    print(f"[firesim-queue] sending SIGTERM to daemon pid={pid}")
    os.kill(pid, signal.SIGTERM)
    return 0


# ---------------------------------------------------------------------------
# Usage stats helpers (used by both `usage` subcommand and the interactive TUI)
# ---------------------------------------------------------------------------

def _usage_stats(conn: sqlite3.Connection, since_seconds: float) -> list[dict]:
    """Per-user usage rollup over the trailing `since_seconds`.

    Returns rows sorted by total_seconds descending. Includes terminal-state
    jobs only (DONE/FAILED) since CANCELLED/QUEUED don't represent FPGA usage.
    """
    cutoff = time.time() - since_seconds
    rows = conn.execute(
        "SELECT user, COUNT(*) AS jobs, "
        " COALESCE(SUM(ended_at - started_at), 0) AS total_s, "
        " COALESCE(AVG(ended_at - started_at), 0) AS avg_s, "
        " SUM(CASE WHEN state='DONE' THEN 1 ELSE 0 END) AS done_n, "
        " SUM(CASE WHEN state='FAILED' THEN 1 ELSE 0 END) AS failed_n "
        " FROM jobs"
        " WHERE state IN ('DONE','FAILED') AND ended_at >= ?"
        " GROUP BY user ORDER BY total_s DESC", (cutoff,)).fetchall()
    return [{"user": r[0], "jobs": r[1], "total_s": r[2], "avg_s": r[3],
             "done": r[4], "failed": r[5]} for r in rows]


def _fmt_dur(seconds: float) -> str:
    if seconds is None:
        return "-"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}:{seconds % 60:02d}"
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


_DIGIT_RUN_RE = __import__("re").compile(r"\d+")


def _cmd_fingerprint(cmd: str) -> tuple[str, str]:
    """Return (exact, loose) fingerprints for ETA similarity matching.

    `exact`  — the raw command string (used for exact-match lookups).
    `loose`  — last 4 tokens joined by space, lowercased, with digit runs
               normalized to `<N>`. Examples:
                 `bash -c 'echo job-3; sleep 2'`  → `echo job-<n>; sleep <n>`
                 `run_all.sh dronet gemmini`      → `run_all.sh dronet gemmini`
                 `run_all.sh yolov8n rvv`         → `run_all.sh yolov<n>n rvv`
               (the yolov8n→yolov<n>n drift is intentional — runs of the
               same model/backend collapse to the same loose fingerprint
               regardless of unique numeric suffixes).
    """
    exact = cmd.strip()
    try:
        toks = [t for t in shlex.split(cmd, posix=True) if t]
    except ValueError:
        toks = cmd.split()
    # Further split bash -c '...' style: if a token contains internal
    # whitespace (single-token inner script), break it open so the loose
    # match sees the actual command tokens.
    flattened: list[str] = []
    for t in toks:
        if " " in t or ";" in t or "&" in t:
            flattened.extend(t.replace(";", " ; ").split())
        else:
            flattened.append(t)
    normalized = [_DIGIT_RUN_RE.sub("<n>", t.lower()) for t in flattened]
    loose = " ".join(normalized[-4:]) if normalized else ""
    return exact, loose


def _estimate_eta(conn: sqlite3.Connection, cmd: str) -> tuple[float | None, str, int]:
    """Estimate wall-time for `cmd` from past DONE jobs.

    Returns (median_seconds_or_None, match_kind, sample_size).
    match_kind ∈ {"exact", "loose", "none"}.
    """
    exact, loose = _cmd_fingerprint(cmd)
    rows = conn.execute(
        "SELECT cmd, ended_at - started_at AS wall FROM jobs"
        " WHERE state='DONE' AND ended_at IS NOT NULL AND started_at IS NOT NULL"
        " ORDER BY ended_at DESC LIMIT 200").fetchall()
    exact_walls = [w for c, w in rows if c == exact and w is not None]
    if len(exact_walls) >= 1:
        return _median(exact_walls), "exact", len(exact_walls)
    if loose:
        loose_walls = [
            w for c, w in rows
            if w is not None and _cmd_fingerprint(c)[1] == loose
        ]
        if len(loose_walls) >= 1:
            return _median(loose_walls), "loose", len(loose_walls)
    return None, "none", 0


def _median(xs: list[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


# 6-color palette for per-user coloring in the TUI. Excludes red/green/yellow
# (already used for daemon-status / job-state). Hashed by username so each
# user keeps the same color across sessions.
_USER_PALETTE = [
    "blue", "magenta", "cyan", "bright_blue", "bright_magenta", "bright_cyan",
]


def _user_color_pair(user: str) -> int:
    """Return curses color-pair number 10..15 for a given user.

    The TUI initializes 6 user color pairs (10..15) at startup; this helper
    picks one deterministically by hashing the username.
    """
    h = sum(ord(c) for c in user) % len(_USER_PALETTE)
    return 10 + h


def cmd_usage(args: argparse.Namespace) -> int:
    """Print per-user FPGA usage stats over a window."""
    conn = _connect()
    window = args.window
    print(f"FireSim queue usage over the last "
          f"{_fmt_dur(window)} ({DB_PATH}):")
    rows = _usage_stats(conn, window)
    if not rows:
        print("  (no terminal-state jobs in window)")
        return 0
    total_user_s = sum(r["total_s"] for r in rows)
    print(f"  {'user':<14} {'jobs':>5} {'done':>5} {'failed':>6} "
          f"{'total':>10} {'avg':>8} {'share':>6}")
    for r in rows:
        share = (r["total_s"] / total_user_s * 100.0
                 if total_user_s > 0 else 0.0)
        print(f"  {r['user']:<14} {r['jobs']:>5} {r['done']:>5} "
              f"{r['failed']:>6} {_fmt_dur(r['total_s']):>10} "
              f"{_fmt_dur(r['avg_s']):>8} {share:>5.1f}%")
    print(f"  {'TOTAL':<14} {sum(r['jobs'] for r in rows):>5} "
          f"{sum(r['done'] for r in rows):>5} "
          f"{sum(r['failed'] for r in rows):>6} "
          f"{_fmt_dur(total_user_s):>10}")
    return 0


# ---------------------------------------------------------------------------
# Interactive TUI
# ---------------------------------------------------------------------------

def cmd_interactive(args: argparse.Namespace) -> int:
    """Rich-based dashboard with live queue, usage stats, ETAs, per-user
    colors, and keystroke commands.

    Matches the ModelBlaster `mb-cost` TUI aesthetic — rounded panels, big
    headline metrics, sortable tables, sticky control bar.

    Falls back to the bare-curses TUI if Rich isn't importable.
    """
    try:
        from rich.console import Console        # noqa: F401
    except ImportError:
        print("[firesim-queue] rich library not available, falling back to "
              "the bare curses TUI", file=sys.stderr)
        import curses
        try:
            return curses.wrapper(_interactive_loop, args)
        except KeyboardInterrupt:
            return 130

    try:
        return _rich_interactive_loop(args)
    except KeyboardInterrupt:
        return 130


def _interactive_loop(stdscr, args: argparse.Namespace) -> int:
    import curses
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)     # daemon alive / DONE
    curses.init_pair(2, curses.COLOR_RED, -1)       # daemon dead / FAILED
    curses.init_pair(3, curses.COLOR_YELLOW, -1)    # RUNNING
    curses.init_pair(4, curses.COLOR_CYAN, -1)      # headers
    curses.init_pair(5, curses.COLOR_MAGENTA, -1)   # own-user marker

    # Per-user color palette (pairs 10..15). Hashed in _user_color_pair().
    # Even when terminals lack the "bright" variants, the base colors are
    # distinguishable enough on dim defaults.
    palette_codes = [
        curses.COLOR_BLUE, curses.COLOR_MAGENTA, curses.COLOR_CYAN,
        curses.COLOR_BLUE | (8 if curses.COLORS >= 16 else 0),
        curses.COLOR_MAGENTA | (8 if curses.COLORS >= 16 else 0),
        curses.COLOR_CYAN | (8 if curses.COLORS >= 16 else 0),
    ]
    for i, c in enumerate(palette_codes):
        try:
            curses.init_pair(10 + i, c, -1)
        except curses.error:
            curses.init_pair(10 + i, curses.COLOR_WHITE, -1)

    refresh_interval = 2.0
    last_refresh = 0.0
    me = getpass.getuser()
    message = ""
    message_until = 0.0
    usage_window = args.window

    conn = _connect()

    while True:
        now = time.time()
        if now - last_refresh >= refresh_interval:
            _interactive_render(stdscr, conn, me, usage_window,
                                message if now < message_until else "")
            last_refresh = now

        try:
            ch = stdscr.getch()
        except curses.error:
            ch = -1
        if ch == -1:
            time.sleep(0.1)
            continue

        key = chr(ch).lower() if 0 <= ch < 256 else None
        if key == "q":
            return 0
        if key == "r":
            last_refresh = 0.0          # force redraw
            continue
        if key in ("s",):
            curses.echo()
            curses.curs_set(1)
            stdscr.nodelay(False)
            try:
                _interactive_submit(stdscr, conn, me)
                message = "Job submitted."
                message_until = time.time() + 3
            except Exception as e:
                message = f"Submit failed: {e}"
                message_until = time.time() + 5
            finally:
                curses.noecho()
                curses.curs_set(0)
                stdscr.nodelay(True)
                last_refresh = 0.0
            continue
        if key == "c":
            curses.echo()
            curses.curs_set(1)
            stdscr.nodelay(False)
            try:
                _interactive_cancel(stdscr, conn, me)
                message = "Cancel requested."
                message_until = time.time() + 3
            except Exception as e:
                message = f"Cancel failed: {e}"
                message_until = time.time() + 5
            finally:
                curses.noecho()
                curses.curs_set(0)
                stdscr.nodelay(True)
                last_refresh = 0.0
            continue
        if key == "t":
            curses.echo()
            curses.curs_set(1)
            stdscr.nodelay(False)
            try:
                _interactive_tail(stdscr, conn)
            finally:
                curses.noecho()
                curses.curs_set(0)
                stdscr.nodelay(True)
                last_refresh = 0.0
            continue
        if key == "u":
            # cycle usage window: 1h → 6h → 24h → 7d → 1h ...
            sequence = [3600, 6 * 3600, 24 * 3600, 7 * 24 * 3600]
            try:
                idx = sequence.index(int(usage_window))
            except ValueError:
                idx = 2
            usage_window = sequence[(idx + 1) % len(sequence)]
            last_refresh = 0.0
            continue


def _interactive_render(stdscr, conn: sqlite3.Connection, me: str,
                        usage_window: int, message: str):
    import curses
    stdscr.erase()
    h, w = stdscr.getmaxyx()

    def addstr(y, x, s, attr=0):
        if y < 0 or y >= h or x < 0:
            return
        s = s[: max(0, w - x - 1)]
        try:
            stdscr.addstr(y, x, s, attr)
        except curses.error:
            pass

    title = "  firesim-queue — interactive dashboard"
    addstr(0, 0, title.ljust(w - 1), curses.color_pair(4) | curses.A_BOLD)

    # ------ Daemon status ------
    alive, age = _daemon_alive(conn)
    age_str = f" (heartbeat {age:.0f}s ago)" if age is not None else ""
    daemon_line = f"  Daemon: {'ALIVE' if alive else 'DOWN '}{age_str}"
    addstr(2, 0, daemon_line,
           curses.color_pair(1 if alive else 2) | curses.A_BOLD)
    addstr(2, max(0, w - 20), f"  user: {me}", curses.color_pair(5))

    # ------ Running ------
    row = 4
    running = conn.execute(
        "SELECT id, user, priority, started_at, cmd, workdir FROM jobs"
        " WHERE state='RUNNING' LIMIT 1").fetchone()
    addstr(row, 0, "  Currently running",
           curses.color_pair(4) | curses.A_BOLD)
    row += 1
    if running:
        (jid, ruser, prio, st, cmd, workdir) = running
        wall = (time.time() - st) if st else 0
        own = " *" if ruser == me else ""
        addstr(row, 4, f"job_id={jid}{own} user={ruser} priority={prio} "
                       f"wall={_fmt_dur(wall)}",
               curses.color_pair(3) | curses.A_BOLD)
        row += 1
        addstr(row, 4, f"cmd: {cmd[:w - 10]}")
        row += 1
        addstr(row, 4, f"workdir: {workdir}",
               curses.color_pair(0) | curses.A_DIM)
        row += 1
    else:
        addstr(row, 4, "(none)", curses.A_DIM)
        row += 1
    row += 1

    # ------ Queue ------
    queued = conn.execute(
        "SELECT id, user, priority, submitted_at, cmd FROM jobs"
        " WHERE state='QUEUED'"
        " ORDER BY priority DESC, submitted_at").fetchall()
    addstr(row, 0, f"  Queue ({len(queued)} jobs)",
           curses.color_pair(4) | curses.A_BOLD)
    row += 1
    addstr(row, 4, f"{'pos':>3} {'id':>5} {'user':<12} {'prio':>4} "
                   f"{'wait':>8}  cmd",
           curses.color_pair(0) | curses.A_DIM)
    row += 1
    now = time.time()
    for pos, q in enumerate(queued, 1):
        if row >= h - 8:
            addstr(row, 4, f"… {len(queued) - pos + 1} more",
                   curses.A_DIM)
            row += 1
            break
        (jid, user, prio, sub, cmd) = q
        wait = now - sub
        own = (curses.color_pair(5) | curses.A_BOLD) if user == me else 0
        cmd_short = cmd[: max(0, w - 38)]
        addstr(row, 4, f"{pos:>3} {jid:>5} {user:<12} {prio:>4} "
                       f"{_fmt_dur(wait):>8}  {cmd_short}", own)
        row += 1
    row += 1

    # ------ Per-user usage ------
    stats = _usage_stats(conn, usage_window)
    addstr(row, 0, f"  Per-user usage (last {_fmt_dur(usage_window)})",
           curses.color_pair(4) | curses.A_BOLD)
    row += 1
    if not stats:
        addstr(row, 4, "(no completed jobs in window)", curses.A_DIM)
        row += 1
    else:
        total_user_s = sum(s["total_s"] for s in stats) or 1
        addstr(row, 4, f"{'user':<14} {'jobs':>5} {'done':>5} {'fail':>5} "
                       f"{'total':>10} {'avg':>8} {'share':>6}",
               curses.color_pair(0) | curses.A_DIM)
        row += 1
        for s in stats:
            if row >= h - 4:
                break
            share = s["total_s"] / total_user_s * 100.0
            own = (curses.color_pair(5) | curses.A_BOLD) if s["user"] == me else 0
            addstr(row, 4, f"{s['user']:<14} {s['jobs']:>5} {s['done']:>5} "
                           f"{s['failed']:>5} {_fmt_dur(s['total_s']):>10} "
                           f"{_fmt_dur(s['avg_s']):>8} {share:>5.1f}%", own)
            row += 1

    # ------ Message and key bar ------
    if message:
        addstr(h - 3, 0, f"  ► {message}", curses.color_pair(3))
    addstr(h - 2, 0,
           "  [s]ubmit  [c]ancel  [t]ail  [u]sage window  [r]efresh  [q]uit",
           curses.color_pair(4) | curses.A_BOLD)
    stdscr.refresh()


def _prompt(stdscr, y: int, x: int, prompt: str, max_len: int = 200) -> str:
    import curses
    stdscr.addstr(y, x, prompt)
    stdscr.refresh()
    buf = ""
    while True:
        ch = stdscr.getch()
        if ch in (10, 13):
            break
        if ch == 27:  # ESC
            return ""
        if ch in (curses.KEY_BACKSPACE, 127, 8):
            buf = buf[:-1]
            y2, _ = stdscr.getyx()
            stdscr.move(y, x + len(prompt))
            stdscr.clrtoeol()
            stdscr.addstr(y, x + len(prompt), buf)
            stdscr.refresh()
            continue
        if 32 <= ch < 127 and len(buf) < max_len:
            buf += chr(ch)
            stdscr.addch(y, x + len(prompt) + len(buf) - 1, ch)
            stdscr.refresh()
    return buf


def _interactive_submit(stdscr, conn: sqlite3.Connection, me: str):
    import curses
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    stdscr.addstr(0, 0, "  Submit a new FireSim job (ESC to cancel)",
                  curses.color_pair(4) | curses.A_BOLD)
    stdscr.addstr(2, 0,
        "  The command runs under bash. Quoting works as in a shell. "
        "ENV vars CHIPYARD_ROOT etc. are preserved from your current "
        "shell — make sure they are set BEFORE you start the TUI.",
        curses.A_DIM)
    cmd = _prompt(stdscr, 5, 2, "command: ", max_len=500)
    if not cmd:
        return
    prio_s = _prompt(stdscr, 7, 2, "priority [5]: ", max_len=3) or "5"
    try:
        prio = int(prio_s)
    except ValueError:
        prio = 5
    cwd = _prompt(stdscr, 9, 2, f"cwd [{os.getcwd()}]: ", max_len=300) \
        or os.getcwd()
    args_ns = argparse.Namespace(
        priority=prio, user=None, cwd=cwd, background=True,
        cmd=["bash", "-c", cmd])
    cmd_submit(args_ns)


def _interactive_cancel(stdscr, conn: sqlite3.Connection, me: str):
    import curses
    stdscr.erase()
    stdscr.addstr(0, 0, "  Cancel a QUEUED job (ESC to cancel)",
                  curses.color_pair(4) | curses.A_BOLD)
    stdscr.addstr(2, 0,
        "  Only QUEUED jobs (no preemption — RUNNING jobs complete on their own).",
        curses.A_DIM)
    jid_s = _prompt(stdscr, 5, 2, "job_id: ", max_len=10)
    if not jid_s:
        return
    try:
        jid = int(jid_s)
    except ValueError:
        return
    args_ns = argparse.Namespace(job_id=jid, force=False)
    cmd_cancel(args_ns)


_RICH_USER_COLORS = [
    "bright_blue", "bright_magenta", "bright_cyan",
    "blue", "magenta", "cyan",
    "spring_green3", "deep_pink3", "dark_orange3", "purple4",
]


# Per-user color families: each user gets a base hue + a 3-shade gradient
# (light, base, dark) so their N jobs cycle through visually distinct shades.
# Built from ANSI 256-color codes; selected to be distinct from red/green/
# yellow (reserved for status).
_RICH_USER_FAMILIES = [
    ("color(33)",  "color(27)",  "color(21)"),    # blue gradient
    ("color(177)", "color(135)", "color(93)"),    # magenta/purple
    ("color(87)",  "color(45)",  "color(31)"),    # cyan/teal
    ("color(208)", "color(202)", "color(166)"),   # orange
    ("color(213)", "color(207)", "color(165)"),   # pink/magenta
    ("color(159)", "color(123)", "color(81)"),    # pale blue
    ("color(229)", "color(228)", "color(220)"),   # straw yellow (light)
    ("color(157)", "color(120)", "color(78)"),    # lime
]


def _rich_user_color(user: str) -> str:
    """Stable per-user BASE color string for Rich markup (one shade per
    user — used for the per-user usage table). Avoids red/green/yellow.

    For per-job display in queue/recent tables, use
    `_rich_user_job_color(user, occurrence)` to get a gradient shade."""
    h = sum(ord(c) for c in user) % len(_RICH_USER_FAMILIES)
    return _RICH_USER_FAMILIES[h][1]  # base shade


def _rich_user_job_color(user: str, occurrence: int) -> str:
    """Per-user job style: same base color, but odd-occurrence jobs get
    a "dim" modifier so a single user's stack of N queued jobs reads
    visually as one block with alternating intensity (the row_styles=
    ["", "dim"] pattern from mb-cost's per-kernel / per-model tables).

    The result is a Rich markup style string that can include both a
    color and a modifier ("bold", "dim"), e.g. "color(33) dim".
    """
    h = sum(ord(c) for c in user) % len(_RICH_USER_FAMILIES)
    base = _RICH_USER_FAMILIES[h][1]   # always use the base shade
    return base if (occurrence % 2 == 0) else f"{base} dim"


def _project_badge(project: str) -> str:
    """Render a project label colored by its hash. No emoji or hardcoded
    list — works for any project name."""
    color = _rich_project_color(project)
    return f"[{color}]{project}[/]"


_FIRE_FRAMES = ("🔥", "🔥💥", "💥🔥", "🔥✨", "✨🔥", "🔥💢")


class _AsciiFire:
    """Procedural ASCII fire à la mhearse/asciifire — cellular-automaton
    flicker rendered with a red→orange→yellow→white intensity ramp.

    Each cell holds an integer intensity 0..max_v. Per tick:
    new[x,y] = avg(neighbors_below_3 + below2) - random_fade
    The bottom rows are reseeded with high values each tick so the
    flame keeps burning. The narrow "torch" shape (taper toward the
    top, wider base) matches the FireSim logo's pillar look.

    Width is square-ish (e.g. 12x8) so the flame reads as a single
    contained block rather than a field of fire. Rendered as a list
    of rich-markup strings, one per row, top-to-bottom for display.
    """

    # Character ramp by intensity (low → high). Sparse low values give
    # the wispy top, dense high values give the solid base.
    _CHARS = " .':^^**sxSXZ$#@@"

    def __init__(self, width: int = 12, height: int = 8, max_v: int = 16):
        self.W = width
        self.H = height
        self.max_v = max_v
        # Two extra rows below the visible area act as the ember source.
        self.grid = [[0] * width for _ in range(height + 2)]
        self._reseed()

    def _reseed(self):
        import random
        for x in range(self.W):
            # Taper the ember intensity at the edges so the flame
            # narrows toward the top instead of being a uniform slab.
            edge_dist = min(x, self.W - 1 - x)
            cap = self.max_v if edge_dist >= 2 else self.max_v - (2 - edge_dist) * 3
            self.grid[self.H + 1][x] = cap
            self.grid[self.H][x] = random.randint(max(0, cap - 4), cap)

    def tick(self):
        import random
        new_grid = [row[:] for row in self.grid]
        for y in range(self.H):
            for x in range(self.W):
                xm = max(0, x - 1)
                xp = min(self.W - 1, x + 1)
                # Average four cells below for a wider, calmer flame.
                avg = (self.grid[y + 1][xm] + self.grid[y + 1][x]
                       + self.grid[y + 1][xp] + self.grid[y + 2][x]) // 4
                fade = random.randint(0, 3)
                # Higher fade at the top so the flame tapers.
                fade += (self.H - y) // 4
                new_grid[y][x] = max(0, avg - fade)
        # Reseed the bottom rows so the fire keeps burning.
        for x in range(self.W):
            edge_dist = min(x, self.W - 1 - x)
            cap = self.max_v if edge_dist >= 2 else self.max_v - (2 - edge_dist) * 3
            new_grid[self.H + 1][x] = cap
            new_grid[self.H][x] = random.randint(max(0, cap - 4), cap)
        self.grid = new_grid

    def render(self) -> str:
        """Return a multi-line rich-markup string. Each row's characters
        are colored by their intensity using a heat-map ramp."""
        rows = []
        for y in range(self.H):
            parts = []
            for x in range(self.W):
                v = self.grid[y][x]
                if v <= 0:
                    parts.append(" ")
                    continue
                idx = min(len(self._CHARS) - 1, v)
                ch = self._CHARS[idx]
                if v >= 14:
                    color = "bold bright_white"
                elif v >= 11:
                    color = "bold bright_yellow"
                elif v >= 8:
                    color = "yellow"
                elif v >= 5:
                    color = "bright_red"
                elif v >= 2:
                    color = "red"
                else:
                    color = "color(52)"   # dark red ember
                parts.append(f"[{color}]{ch}[/]")
            rows.append("".join(parts))
        return "\n".join(rows)

# Per-panel border + header colors so each section reads as a distinct
# zone instead of a wall of bright_blue. The header_style is applied to
# the table column headers inside each panel.
_PANEL_COLORS = {
    "headline": {"border": "bright_blue",    "header": "bold bright_blue"},
    "gantt":    {"border": "bright_magenta", "header": "bold bright_magenta"},
    "queue":    {"border": "bright_yellow",  "header": "bold bright_yellow"},
    "usage":    {"border": "bright_green",   "header": "bold bright_green"},
    "recent":   {"border": "bright_cyan",    "header": "bold bright_cyan"},
    "controls": {"border": "bright_white",   "header": "bold bright_white"},
}


# Candidate paths the simulator's UART output (the `screen -r fsim0`
# equivalent) might land at, in priority order. The actual path is set
# via config_runtime.yaml::default_simulation_dir on the running firesim
# install; users override that per-host, so we probe a few well-known
# locations + the legacy default. First file that exists wins.
_UARTLOG_CANDIDATES = (
    "/scratch2/agustin/FIRESIM_RUNS_DIR/sim_slot_0/uartlog",
    # Add other hosts' default_simulation_dir overrides here as the queue
    # gets deployed on new machines. The legacy fallback (chipyard install
    # firesim_rundir) is checked via workdir-based discovery below.
)


def _live_uartlog_path(running_workdir: Optional[str]) -> Optional[str]:
    """Best-effort locate the UART output of the currently-RUNNING job.
    Returns None when nothing's running or no uartlog found."""
    for p in _UARTLOG_CANDIDATES:
        if os.path.exists(p):
            return p
    # Fallback: derive from the job's workdir
    # (workdir = <firesim_root>/deploy → uartlog = <firesim_root>/firesim_rundir/sim_slot_0/uartlog).
    if running_workdir:
        root = os.path.dirname(running_workdir.rstrip("/"))
        candidate = os.path.join(root, "firesim_rundir/sim_slot_0/uartlog")
        if os.path.exists(candidate):
            return candidate
    return None


# Wholesome filler shown when the FPGA is busy with a job owned by a
# different user and the UART tail isn't readable for the current viewer.
# `screen -r fsim0` only works for the UID that started the screen, so
# the practical answer for non-owners is "ping that person". Keep these
# light — they appear in the TUI for ~minutes at a stretch.
_WHOLESOME_PHRASES = (
    "the FPGA is busy cooking — kindly wait your turn",
    "someone else's bits are dancing right now",
    "patience, the silicon is doing its little jig",
    "your future self will thank you for not preempting this",
    "the bitstream is having its moment, let it have its moment",
    "do not disturb the gates, they are deep in thought",
    "you could go grab a coffee — the queue won't move without you",
    "this is the way of the shared FPGA",
    "another timeline is being explored on the chip",
    "DRAM cells are being lovingly written to",
    "the LUTs are vibing",
    "FireSim's mascot agrees: take a breath",
    "your job is queued and well-loved",
)


def _phrase_for(now: float, period_s: float = 8.0) -> str:
    """Pick a wholesome phrase that rotates every `period_s` seconds.
    Deterministic on now so the picker is stable while the TUI re-renders
    multiple times per second."""
    idx = int(now / period_s) % len(_WHOLESOME_PHRASES)
    return _WHOLESOME_PHRASES[idx]


def _tail_lines(path: str, n: int = 8) -> list[str]:
    """Read the last `n` lines of a file efficiently. Returns [] on any
    error so the caller can degrade silently."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            # Read up to 16 KB from the end -- enough for ~80 lines of
            # typical UART output, fast on any file size.
            chunk = min(size, 16 * 1024)
            f.seek(size - chunk)
            data = f.read(chunk).decode("utf-8", errors="replace")
    except OSError:
        return []
    lines = data.splitlines()
    return lines[-n:]


def _short_cmd(cmd: Optional[str], width: int = 80) -> str:
    """Compact a queued/run cmd for a one-line table cell.

    The submitted commands often arrive as multi-line bash heredocs
    (merlin's `bash -c 'set -e\\n        unset CONDA_PREFIX...\\n
    source ...env.sh\\n        firesim infrasetup\\n        ...'`). Rich's
    overflow="ellipsis" can't help when the source string itself has
    newlines: every \\n forces a new visual row, blowing up the cell's
    vertical span and making the whole table near-illegible.

    Strategy: collapse ALL runs of whitespace to a single space, drop
    the conda-cleanup boilerplate ModelBlaster + merlin both prepend
    (set -e; unset CONDA_*; export PATH=...), and ellipsize to `width`.
    The full text is still in the DB and reachable via `tail <jid>`.
    """
    if not cmd:
        return ""
    # 1) Collapse all whitespace runs (newlines, tabs, multiple spaces)
    #    to single spaces. This is the load-bearing step for the
    #    multi-line heredoc problem.
    s = " ".join(cmd.split())
    # 2) Strip the verbose conda-cleanup prologue both ModelBlaster's
    #    firesim_runner and merlin's run_hetero.sh prepend before
    #    sourceme-manager.sh. The block looks like:
    #      bash -c 'set -e; unset CONDA_PREFIX ... CE_CONDA;
    #      export PATH=/scratch2/.../miniforge3/condabin:$PATH;
    #      source /scratch2/.../env.sh; cd /scratch2/.../firesim;
    #      source ./sourceme-manager.sh --skip-ssh-setup; ...'
    #    None of which the human reader cares about — they want to see
    #    the *firesim infrasetup / runworkload* part.
    import re as _re
    s = _re.sub(
        r"^bash -c ['\"]?set -e;?\s*"
        r"(unset CONDA[^;]*;\s*)?"
        r"(export PATH=[^;]*;\s*)?"
        r"(source [^;]*env\.sh;?\s*)?"
        r"(cd [^;]*;\s*)?"
        r"(source [^;]*sourceme-manager\.sh[^;]*;?\s*)?"
        r"(cd [^;]*;\s*)?",
        "", s)
    # 3) Drop trailing close-quote if our prefix strip left it.
    s = s.rstrip("' ").rstrip('" ')
    # 4) Hard cap on width with ellipsis.
    if len(s) > width:
        s = s[:width - 1] + "…"
    return s


def _rich_interactive_loop(args: argparse.Namespace) -> int:
    """Live TUI with pinned controls + scrollable queue + fire animation.

    Layout (rich.Layout split_column, three rows):
      TOP    — headline panel (fixed ~7 lines): daemon status + running
               job + a 🔥 fire animation that flickers each frame while a
               job is RUNNING.
      MIDDLE — three stacked panels (queue / per-user usage / recent
               jobs). The queue panel scrolls when there are more jobs
               than fit; j/k or arrow-down/up paginate by half a screen.
      BOTTOM — controls + status message (fixed 5 lines, ALWAYS visible
               regardless of how many queued jobs exist). Fixes the
               "vertical overflow swallowed the controls" issue.
    """
    import queue as _queue
    import threading
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.align import Align
    from rich.box import ROUNDED

    me = getpass.getuser()
    usage_window = args.window
    message_state = {"text": "", "until": 0.0}
    action_q: "_queue.Queue[str]" = _queue.Queue()
    stop = {"flag": False}
    scroll = {"queue": 0}
    fire_frame = {"i": 0}
    # Square-ish block: 14 cols x 7 rows. Terminal cells are roughly 2:1
    # tall so width=2*height looks visually square. Compact enough to
    # sit in the top-right corner without dominating.
    fire = _AsciiFire(width=14, height=7)
    conn = _connect()

    console = Console()

    def _read_keys():
        """Background thread: cbreak + os.read for single-byte immediate
        keystrokes (no ENTER needed). Falls back to line-input when
        stdin isn't a tty (piped input, CI). The single-key path is
        the lessons-learned path from mb-cost: 60Hz polling, escape-
        sequence decoding for arrows."""
        if not sys.stdin.isatty():
            while not stop["flag"]:
                try:
                    line = input("")
                except (EOFError, KeyboardInterrupt):
                    stop["flag"] = True
                    action_q.put("q")
                    return
                line = line.strip().lower()
                if line:
                    action_q.put(line[0])
            return
        import termios, tty, select
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not stop["flag"]:
                r, _, _ = select.select([fd], [], [], 0.1)
                if not r:
                    continue
                try:
                    b = os.read(fd, 1)
                except (BlockingIOError, OSError):
                    continue
                if not b:
                    continue
                ch = b.decode("utf-8", errors="replace")
                if ch == "\x1b":
                    # Escape sequence (likely arrow). Peek the next 2.
                    r2, _, _ = select.select([fd], [], [], 0.01)
                    if r2:
                        seq = os.read(fd, 2)
                        if seq == b"[A":
                            ch = "k"   # UP -> scroll-up
                        elif seq == b"[B":
                            ch = "j"   # DOWN -> scroll-down
                        elif seq == b"[C":
                            ch = ""    # RIGHT (ignored)
                        elif seq == b"[D":
                            ch = ""    # LEFT (ignored)
                        else:
                            continue
                    else:
                        ch = "q"   # bare ESC -> quit
                if ch in ("\x03",):
                    ch = "q"
                if not ch:
                    continue
                ch = ch.lower()
                if ch in ("q", "r", "u", "s", "c", "t", "j", "k", "g", "G".lower()):
                    action_q.put(ch)
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except Exception:
                pass

    threading.Thread(target=_read_keys, daemon=True).start()

    def _build_render():
        """Returns a rich.Layout (NOT a Group). The layout is split into
        TOP / MIDDLE / BOTTOM with explicit sizes so the controls bar at
        the bottom never gets pushed off-screen, no matter how many
        queued jobs there are."""
        now = time.time()
        alive, age = _daemon_alive(conn)
        daemon_color = "bright_green" if alive else "bright_red"
        daemon_status = "ALIVE" if alive else "DOWN"
        daemon_age = (f"  ·  heartbeat {_fmt_dur(age)} ago"
                      if age is not None else "")

        # Tick the procedural fire only when there's room to show it.
        # On narrow terminals (<100 cols) we drop the fire column
        # entirely so the headline status doesn't get squeezed; ticking
        # also stops, so there's no wasted CPU when invisible.
        # The CA itself is cheap (~112 cells x 4 Hz = ~450 cell-ops/sec)
        # so visibility is the only real cost lever.
        show_fire = console.size.width >= 100
        if show_fire:
            fire.tick()
            fire_markup = fire.render()
        else:
            fire_markup = ""

        running = conn.execute(
            "SELECT id, user, priority, started_at, cmd, workdir, project,"
            " phase, kind FROM jobs WHERE state='RUNNING' LIMIT 1").fetchone()
        queued = conn.execute(
            "SELECT id, user, priority, submitted_at, cmd, project, kind"
            " FROM jobs WHERE state='QUEUED'"
            " ORDER BY priority DESC, submitted_at"
        ).fetchall()
        recent = conn.execute(
            "SELECT id, user, priority, state, started_at, ended_at,"
            " exit_code, cmd, project FROM jobs WHERE state IN ('DONE',"
            "'FAILED','CANCELLED') ORDER BY ended_at DESC LIMIT 10").fetchall()
        stats = _usage_stats(conn, usage_window)
        # Per-user job-count among queued/running, used as the `occurrence`
        # index for the gradient. We tally as we iterate below.
        user_occurrence: dict[str, int] = {}

        # ---- Headline panel ----
        if running:
            (jid, ruser, prio, st, cmd, _wd, project,
             rphase, rkind) = running
            elapsed = (now - st) if st else 0
            eta, kind, n = _estimate_eta(conn, cmd)
            if eta is not None:
                pct = min(99, int(elapsed * 100 / eta)) if eta > 0 else 0
                eta_str = (f"  ·  ETA {_fmt_dur(eta)} ({kind}, n={n})  "
                           f"·  {pct}%")
            else:
                eta_str = "  ·  ETA unknown (no similar past jobs)"
            # For runworkload-full jobs, surface the current daemon-side
            # phase (STAGING / INFRASETUP / RUNNING / TEARDOWN) so the
            # viewer sees what the daemon is actually doing inside the
            # atomic op — not just "RUNNING" for the whole 5+ minutes.
            phase_str = ""
            if rkind == "runworkload-full" and rphase:
                phase_color = {
                    "STAGING": "yellow",
                    "INFRASETUP": "color(214)",  # orange
                    "RUNNING": "bright_green",
                    "TEARDOWN": "magenta",
                }.get(rphase, "white")
                phase_str = f"  ·  [bold {phase_color}]phase={rphase}[/]"
            user_clr = _rich_user_color(ruser)
            you = " [bold magenta](you)[/]" if ruser == me else ""
            # Live UART tail (the `screen -r fsim0` equivalent). The
            # underlying uartlog file is usually group-readable, so we
            # can tail it cross-user even though `screen -r fsim0` is
            # per-UID and won't attach for a non-owner.
            uart_path = _live_uartlog_path(_wd)
            uart_tail = _tail_lines(uart_path, n=6) if uart_path else []
            is_owner = (ruser == me)
            if uart_tail:
                # Strip rich markup in case the UART output has [...] strings
                # (rich would try to interpret them as markup).
                from rich.markup import escape
                tail_block = "\n".join(
                    f"  [bright_black]│[/] {escape(line)[:160]}"
                    for line in uart_tail)
                if is_owner:
                    hint = (f" [dim](attach live: [/]"
                            f"[bold]screen -r fsim0[/][dim])[/]")
                else:
                    hint = (f" [dim](this is [bold {user_clr}]{ruser}[/]'s "
                            f"session — `screen -r` is per-UID, "
                            f"ping them if you need the live view)[/]")
                tail_section = (f"\n[dim]uart tail ({os.path.basename(uart_path)}):[/]{hint}\n"
                                f"{tail_block}")
            elif is_owner:
                # Owner, no UART yet -> still in infrasetup. Show cmd.
                tail_section = (f"\n[dim]cmd:[/] {_short_cmd(cmd, width=140)}")
            else:
                # Cross-user viewer with no readable uart. Don't dump the
                # other person's conda boilerplate — instead, a rotating
                # wholesome line + a clear pointer to them.
                phrase = _phrase_for(now)
                tail_section = (
                    f"\n[dim italic]{phrase}[/]\n"
                    f"  [bright_black]│[/] [dim]for live UART, ping "
                    f"[bold {user_clr}]@{ruser}[/][dim] — "
                    f"`screen -r fsim0` is per-UID and won't attach for you[/]"
                )
            headline = Text.from_markup(
                f"[bold {daemon_color}]Daemon: {daemon_status}[/]"
                f"[dim]{daemon_age}[/]\n"
                f"[bold yellow]▶ RUNNING[/]  "
                f"job_id={jid}  user=[bold {user_clr}]{ruser}[/]{you}  "
                f"priority={prio}  project={_project_badge(project)}  "
                f"[bold]elapsed[/] {_fmt_dur(elapsed)}{eta_str}{phase_str}"
                f"{tail_section}",
                justify="left")
        else:
            headline = Text.from_markup(
                f"[bold {daemon_color}]Daemon: {daemon_status}[/]"
                f"[dim]{daemon_age}[/]\n\n"
                f"[dim]no job running[/]\n",
                justify="left")

        # Headline layout: daemon-status + running-job info on the LEFT
        # (priority), live ASCII fire as a square block in the TOP-RIGHT
        # corner. On narrow terminals (<100 cols, set above) the fire
        # column is dropped entirely so the status info gets the full
        # width.
        if show_fire:
            headline_grid = Table.grid(expand=True, padding=(0, 1))
            headline_grid.add_column(ratio=1, justify="left", vertical="top")
            headline_grid.add_column(width=fire.W, justify="right",
                                      vertical="top")
            headline_grid.add_row(
                headline,
                Text.from_markup(fire_markup),
            )
            headline_content = headline_grid
        else:
            # Narrow terminal: skip the fire, give all space to status.
            headline_content = headline
        headline_panel = Panel(
            headline_content,
            title="FireSim queue",
            border_style=_PANEL_COLORS["headline"]["border"],
            box=ROUNDED, padding=(0, 1))

        # ---- Queue table ----
        q_table = Table(box=ROUNDED, show_header=True,
                        header_style=_PANEL_COLORS["queue"]["header"],
                        title_justify="left", expand=True, pad_edge=False,
                        row_styles=["", "on grey11"])
        q_table.add_column("pos", justify="right", width=4)
        q_table.add_column("id", justify="right", width=5)
        q_table.add_column("user", width=14, overflow="fold")
        q_table.add_column("project", width=18, overflow="ellipsis", no_wrap=True)
        q_table.add_column("prio", justify="right", width=4)
        q_table.add_column("wait", justify="right", width=8)
        q_table.add_column("ETA", justify="right", width=12)
        # `kind` makes it visually obvious which queued jobs are
        # routed through the atomic FireSim lifecycle (runworkload-full
        # → "rwf") vs the legacy freeform bash blob ("bash"). One
        # narrow column to keep the cmd column wide.
        q_table.add_column("kind", width=4)
        q_table.add_column("cmd", overflow="ellipsis", no_wrap=True, ratio=1)
        # Derive a visible-row cap from the terminal height so the queue
        # panel doesn't blow past the screen + push the controls off.
        # Heuristic: leave ~22 rows for headline/recent/usage/controls;
        # the rest is queue. Minimum 4 to keep the panel useful.
        term_h = max(20, console.size.height)
        queue_visible = max(4, term_h - 22)
        total_queued = len(queued)
        # Clamp scroll so we never scroll past the last row.
        max_scroll = max(0, total_queued - queue_visible)
        scroll["queue"] = min(max(0, scroll["queue"]), max_scroll)
        view_lo = scroll["queue"]
        view_hi = min(total_queued, view_lo + queue_visible)
        # Per-user gradient occurrence is computed across ALL queued jobs
        # so colors stay consistent when you scroll (a user's 3rd job in
        # the full list keeps its shade when scrolled into view).
        full_user_occ: dict[str, int] = {}
        rendered_rows = 0
        for pos, q in enumerate(queued, 1):
            (qid, quser, qprio, qsub, qcmd, qproject, qkind) = q
            occ = full_user_occ.get(quser, 0)
            full_user_occ[quser] = occ + 1
            if not (view_lo < pos <= view_hi):
                continue
            wait = now - qsub
            user_clr = _rich_user_job_color(quser, occ)
            you_marker = " *" if quser == me else "  "
            eta_v, kind, n = _estimate_eta(conn, qcmd)
            if eta_v is not None:
                eta_disp = (f"[bright_green]{_fmt_dur(eta_v)}[/]" if kind == "exact"
                            else f"[yellow]~{_fmt_dur(eta_v)}[/]")
            else:
                eta_disp = "[dim]unknown[/]"
            prio_disp = (f"[bold red]{qprio}[/]" if qprio >= 10
                         else (f"[yellow]{qprio}[/]" if qprio >= 5
                               else f"[dim]{qprio}[/]"))
            # "rwf" = runworkload-full (atomic, daemon-managed lifecycle).
            # "bash" = legacy freeform submission.
            kind_disp = (f"[bright_green]rwf[/]"
                         if qkind == "runworkload-full"
                         else f"[dim]bash[/]")
            q_table.add_row(
                str(pos), str(qid),
                f"[{user_clr}]{quser}[/]{you_marker}",
                _project_badge(qproject),
                prio_disp,
                _fmt_dur(wait),
                eta_disp,
                kind_disp,
                _short_cmd(qcmd),
            )
            rendered_rows += 1
        if view_lo > 0:
            q_table.add_row(f"[dim]↑ {view_lo} above (k / ↑)[/]",
                            "", "", "", "", "", "", "", "")
        if view_hi < total_queued:
            q_table.add_row(f"[dim]↓ {total_queued - view_hi} below (j / ↓)[/]",
                            "", "", "", "", "", "", "", "")
        queue_title = (f"📋 Queue · {len(queued)} job"
                       + ("s" if len(queued) != 1 else "")
                       + (f"  · showing {view_lo+1}-{view_hi}"
                          if total_queued > queue_visible else ""))
        if not queued:
            queue_panel = Panel(Align.center("[dim]queue empty[/]",
                                              vertical="middle"),
                                title=queue_title,
                                border_style=_PANEL_COLORS["queue"]["border"],
                                box=ROUNDED)
        else:
            queue_panel = Panel(q_table, title=queue_title,
                                border_style=_PANEL_COLORS["queue"]["border"],
                                box=ROUNDED)

        # ---- Per-user usage ----
        u_table = Table(box=ROUNDED, show_header=True,
                        header_style=_PANEL_COLORS["usage"]["header"],
                        expand=True, row_styles=["", "on grey11"])
        u_table.add_column("user", width=14)
        u_table.add_column("jobs", justify="right")
        u_table.add_column("done", justify="right")
        u_table.add_column("fail", justify="right")
        u_table.add_column("total", justify="right")
        u_table.add_column("avg", justify="right")
        u_table.add_column("share", justify="right")
        total_user_s = sum(s["total_s"] for s in stats) or 1
        for s in stats:
            user_clr = _rich_user_color(s["user"])
            you_marker = " *" if s["user"] == me else ""
            share = s["total_s"] / total_user_s * 100.0
            u_table.add_row(
                f"[{user_clr}]{s['user']}[/]{you_marker}",
                str(s["jobs"]), str(s["done"]),
                (f"[red]{s['failed']}[/]" if s["failed"] else "0"),
                _fmt_dur(s["total_s"]),
                _fmt_dur(s["avg_s"]),
                f"{share:.1f}%")
        usage_panel = Panel(
            u_table if stats else Align.center(
                "[dim](no completed jobs in window)[/]",
                vertical="middle"),
            title=f"📊 Per-user usage · last {_fmt_dur(usage_window)}",
            border_style=_PANEL_COLORS["usage"]["border"], box=ROUNDED)

        # ---- Recent terminal ----
        r_table = Table(box=ROUNDED, show_header=True,
                        header_style=_PANEL_COLORS["recent"]["header"],
                        expand=True, pad_edge=False,
                        row_styles=["", "on grey11"])
        r_table.add_column("id", justify="right", width=5)
        r_table.add_column("user", width=14)
        r_table.add_column("project", width=18, overflow="ellipsis", no_wrap=True)
        r_table.add_column("state", width=10)
        r_table.add_column("prio", justify="right", width=4)
        r_table.add_column("wall", justify="right", width=10)
        r_table.add_column("rc", justify="right", width=4)
        r_table.add_column("cmd", overflow="ellipsis", no_wrap=True, ratio=1)
        recent_occ: dict[str, int] = {}
        for r in recent:
            (rid, ruser, rprio, rstate, rstart, rend, rrc, rcmd, rproject) = r
            wall = (rend - rstart) if (rend and rstart) else 0
            state_color = ("bright_green" if rstate == "DONE"
                           else ("bright_red" if rstate == "FAILED"
                                 else "dim"))
            occ = recent_occ.get(ruser, 0)
            recent_occ[ruser] = occ + 1
            user_clr = _rich_user_job_color(ruser, occ)
            r_table.add_row(
                str(rid),
                f"[{user_clr}]{ruser}[/]",
                _project_badge(rproject),
                f"[{state_color}]{rstate}[/]",
                str(rprio),
                _fmt_dur(wall),
                str(rrc if rrc is not None else "-"),
                _short_cmd(rcmd),
            )
        recent_panel = Panel(
            r_table if recent else Align.center("[dim](no history)[/]",
                                                 vertical="middle"),
            title="📜 Recent jobs · last 10 terminal",
            border_style=_PANEL_COLORS["recent"]["border"], box=ROUNDED)

        # ---- Message and control bar (ALWAYS visible) ----
        message_txt = message_state["text"] if now < message_state["until"] else ""
        # In cbreak mode the keys take effect immediately (no ENTER); the
        # hint reflects that.
        hint = ("[dim]Single-key input — keys take effect immediately.[/]"
                if sys.stdin.isatty()
                else "[dim]Press a key and ENTER. "
                     "(stdin-line input — works in any terminal.)[/]")
        controls = Panel(
            Text.from_markup(
                (f"[bold yellow]►[/] {message_txt}\n" if message_txt else "")
                + "[bold]s[/] submit   [bold]c[/] cancel   "
                  "[bold]t[/] tail   [bold]u[/] cycle usage window   "
                  "[bold]r[/] refresh   "
                  "[bold]j[/]/[bold]k[/] scroll queue   "
                  "[bold]q[/] quit\n"
                + hint,
                justify="left"),
            title=f"🎛  Controls · user=[bold]{me}[/]",
            border_style=_PANEL_COLORS["controls"]["border"], box=ROUNDED)

        gantt_panel = _build_gantt_panel(conn, me, console.width)

        # ---- Layout: TOP (fixed) / MIDDLE (flexible) / BOTTOM (fixed)
        # Bottom is sized so controls always fit regardless of how many
        # queued jobs are visible above.
        middle = Group(gantt_panel, queue_panel, usage_panel, recent_panel)
        layout = Layout()
        layout.split_column(
            Layout(headline_panel, name="top", size=11),
            Layout(middle, name="middle", ratio=1),
            Layout(controls, name="bottom", size=6),
        )
        return layout

    # screen=True uses the alternate screen buffer so the Layout's
    # split_column pins the bottom controls in place. With screen=False
    # (the prior setting) + vertical_overflow="visible", rich just
    # scrolled the whole frame past the terminal bottom whenever there
    # were too many queued jobs.
    refresh_hz = 4.0
    with Live(_build_render(), console=console,
              refresh_per_second=refresh_hz, screen=True) as live:
        last_render = 0.0
        while not stop["flag"]:
            try:
                key = action_q.get(timeout=0.1)
            except _queue.Empty:
                key = None
            if key == "q":
                break
            if key == "j":
                # Page down: scroll the queue panel by half a screen.
                scroll["queue"] += max(4, console.size.height // 4)
                live.update(_build_render())
                continue
            if key == "k":
                scroll["queue"] = max(0, scroll["queue"]
                                      - max(4, console.size.height // 4))
                live.update(_build_render())
                continue
            if key == "g":
                scroll["queue"] = 0
                live.update(_build_render())
                continue
            if key == "r":
                scroll["queue"] = 0
                live.update(_build_render())
                continue
            if key == "u":
                seq = [3600, 6 * 3600, 24 * 3600, 7 * 24 * 3600]
                try:
                    idx = seq.index(int(usage_window))
                except ValueError:
                    idx = 2
                usage_window = seq[(idx + 1) % len(seq)]
                message_state["text"] = (
                    f"usage window → {_fmt_dur(usage_window)}")
                message_state["until"] = time.time() + 3
                live.update(_build_render())
                continue
            if key == "s":
                # Pause live for the interactive prompt sequence so input
                # echo isn't clobbered by the redraw.
                live.stop()
                try:
                    _rich_submit_prompt(console, conn)
                    message_state["text"] = "submitted."
                except Exception as e:
                    message_state["text"] = f"submit failed: {e}"
                message_state["until"] = time.time() + 5
                live.start(_build_render())
                continue
            if key == "c":
                live.stop()
                try:
                    _rich_cancel_prompt(console, conn)
                    message_state["text"] = "cancel requested."
                except Exception as e:
                    message_state["text"] = f"cancel failed: {e}"
                message_state["until"] = time.time() + 5
                live.start(_build_render())
                continue
            if key == "t":
                live.stop()
                try:
                    _rich_tail_prompt(console, conn)
                finally:
                    live.start(_build_render())
                continue

            # Periodic auto-refresh when no key pressed. ~4 Hz so the
            # fire frame animates smoothly + queued/running ETAs update
            # without feeling sluggish.
            now = time.time()
            if now - last_render >= 0.25:
                live.update(_build_render())
                last_render = now

    stop["flag"] = True
    return 0


def _rich_submit_prompt(console, conn):
    from rich.prompt import Prompt, IntPrompt
    console.print("\n[bold]Submit a new job[/] (Ctrl-C to cancel)\n")
    cmd = Prompt.ask("command (bash -c '...' equivalent)")
    if not cmd.strip():
        console.print("[red]empty command, aborted[/]")
        return
    prio = IntPrompt.ask("priority", default=5)
    cwd = Prompt.ask("cwd", default=os.getcwd())
    # Pre-show ETA before actually submitting.
    eta, kind, n = _estimate_eta(conn, cmd)
    if eta is not None:
        console.print(f"[green]Estimated wall:[/] {_fmt_dur(eta)} "
                      f"(based on {n} {kind} past run"
                      f"{'s' if n != 1 else ''})")
    else:
        console.print("[yellow]No similar past jobs found — duration unknown.[/]")
    args_ns = argparse.Namespace(
        priority=prio, user=None, cwd=cwd, background=True,
        cmd=["bash", "-c", cmd])
    cmd_submit(args_ns)


def _rich_cancel_prompt(console, conn):
    from rich.prompt import IntPrompt
    console.print("\n[bold]Cancel a QUEUED job[/] "
                  "(no preemption — RUNNING jobs finish on their own)\n")
    jid = IntPrompt.ask("job_id")
    args_ns = argparse.Namespace(job_id=jid, force=False)
    cmd_cancel(args_ns)


def _rich_tail_prompt(console, conn):
    from rich.prompt import IntPrompt
    console.print("\n[bold]Tail a job's stdout[/] "
                  "(Ctrl-C to exit tail and return to dashboard)\n")
    jid = IntPrompt.ask("job_id")
    row = conn.execute(
        "SELECT workdir, state FROM jobs WHERE id=?", (jid,)).fetchone()
    if row is None:
        console.print(f"[red]job {jid} not found[/]")
        return
    p = pathlib.Path(row[0]) / "stdout.log"
    if not p.exists():
        console.print(f"[yellow](no stdout yet for job {jid})[/]")
        time.sleep(2)
        return
    try:
        subprocess.run(["tail", "-f", str(p)])
    except KeyboardInterrupt:
        pass


def _interactive_tail(stdscr, conn: sqlite3.Connection):
    import curses
    stdscr.erase()
    stdscr.addstr(0, 0,
                  "  Tail a job's stdout (ESC at prompt to cancel; "
                  "Ctrl-C to exit tail)",
                  curses.color_pair(4) | curses.A_BOLD)
    jid_s = _prompt(stdscr, 3, 2, "job_id: ", max_len=10)
    if not jid_s:
        return
    try:
        jid = int(jid_s)
    except ValueError:
        return
    row = conn.execute(
        "SELECT workdir FROM jobs WHERE id=?", (jid,)).fetchone()
    if row is None:
        return
    p = pathlib.Path(row[0]) / "stdout.log"
    if not p.exists():
        stdscr.addstr(5, 2, f"(no stdout yet for job {jid})")
        stdscr.refresh()
        time.sleep(2)
        return
    # Drop out of curses, run plain `tail -f`, restore curses on exit.
    curses.endwin()
    try:
        subprocess.run(["tail", "-f", str(p)])
    except KeyboardInterrupt:
        pass
    finally:
        # curses.wrapper will re-init on return; nothing to do here.
        pass


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    # Group-writable by default. The inherited 022 umask is precisely what
    # made every cross-user job undispatchable: files landed 644 in a dir the
    # daemon could not write.
    os.umask(0o002)
    p = argparse.ArgumentParser(
        prog="firesim-queue", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_sub = sub.add_parser("submit", help="enqueue a FireSim command")
    p_sub.add_argument("--priority", type=int, default=5,
                       help="0=low, 5=normal, 10=high (default 5)")
    p_sub.add_argument("--user", default=None,
                       help="override user (defaults to $USER)")
    p_sub.add_argument("--cwd", default=None,
                       help="working directory (defaults to current)")
    p_sub.add_argument("--project", default=None,
                       help="explicit project tag (else auto-inferred "
                            "from cwd: merlin / modelblaster / chipyard / etc.)")
    p_sub.add_argument("--background", action="store_true",
                       help="return immediately after submission")
    p_sub.add_argument("cmd", nargs=argparse.REMAINDER,
                       help="command to run; pass after `--`")
    p_sub.set_defaults(func=cmd_submit)

    p_st = sub.add_parser("status", help="show queue + running jobs")
    p_st.add_argument("--user", default=None)
    p_st.add_argument("--all", action="store_true",
                      help="include terminal-state jobs (DONE/FAILED)")
    p_st.set_defaults(func=cmd_status)

    p_c = sub.add_parser("cancel",
        help="cancel a job by id; QUEUED -> removed, RUNNING -> "
             "SIGTERM'd + cleaned up. Owner-only (uid match) unless "
             "--force.")
    p_c.add_argument("job_id", type=int)
    p_c.add_argument("--force", action="store_true",
                     help="allow cancelling another user's job "
                          "(ops use; bypasses the owner-UID check)")
    p_c.add_argument("--no-wait", action="store_true",
                     help="don't block waiting for the daemon to "
                          "finish tearing down a RUNNING job; just "
                          "set the cancel flag and return")
    p_c.set_defaults(func=cmd_cancel)

    # New atomic FireSim lifecycle op (see cmd_runworkload_full for the
    # full design rationale). Replaces the legacy bash-blob submit for
    # FireSim use cases. Legacy `submit` still works for non-FireSim
    # commands.
    p_rwf = sub.add_parser("runworkload-full",
        help="atomic FireSim lifecycle: daemon owns the full "
             "kill -> infrasetup -> runworkload -> kill sequence, "
             "with per-job config_runtime.yaml + timeout enforcement "
             "(eliminates softlock + YAML race classes by construction)")
    p_rwf.add_argument("--chipyard", required=True,
        help="absolute path to the chipyard root containing "
             "sims/firesim/sourceme-manager.sh + env.sh")
    p_rwf.add_argument("--workload", required=True,
        help="FireSim workload name; must match a "
             "deploy/workloads/<name>.json on the chipyard install")
    p_rwf.add_argument("--bootbinary", default=None,
        help="ELF basename inside deploy/workloads/<workload>/; "
             "defaults to the workload JSON's common_bootbinary")
    p_rwf.add_argument("--stage-from", default=None,
        help="if given, daemon copies this file into "
             "deploy/workloads/<workload>/<bootbinary> "
             "before infrasetup (atomic — happens under the FPGA lock)")
    p_rwf.add_argument("--priority", type=int, default=5,
        help="0=low, 5=normal, 10=high (default 5)")
    p_rwf.add_argument("--user", default=None,
        help="override user (defaults to $USER)")
    p_rwf.add_argument("--project", default=None,
        help="explicit project tag (else auto-inferred from cwd)")
    p_rwf.add_argument("--timeout", type=int, default=0,
        help="optional wall-clock cap on the RUNNING phase. 0 (the "
             "default) = no timeout, run until firesim runworkload "
             "exits on its own. Pass a positive integer to bound it "
             "(daemon SIGTERMs + cleans up on expiry; TEARDOWN always "
             "runs). Common case: `--timeout 600` for a 10-min smoke "
             "or anything where you'd rather lose the run than hold "
             "the FPGA past N seconds.")
    p_rwf.add_argument("--hw-config", dest="hw_config", default=None,
        help="hwdb entry naming the BITSTREAM this job flashes (a key in "
             "deploy/config_hwdb.yaml). Omitted = inherit the chipyard "
             "template's default_hw_config. Pass it whenever more than one "
             "design is in use: left to the shared template, two users "
             "wanting different bitstreams race and the loser silently runs "
             "the other's hardware.")
    p_rwf.add_argument("--hwdb-config-artifact", default=None,
        help="optional immutable one-entry HWDB file. Its bytes are hash-"
             "committed at submission, snapshotted under the FPGA lock, and "
             "passed to every FireSim phase with -a. Requires --hw-config. "
             "Fit-evidence jobs must provide it; omitted preserves legacy "
             "shared-config behavior for non-evidence jobs.")
    p_rwf.add_argument("--background", action="store_true",
        help="return immediately after submission")
    p_rwf.set_defaults(func=cmd_runworkload_full)

    p_t = sub.add_parser("tail", help="stream a job's stdout.log "
                                     "(or the FireSim uartlog with --uartlog)")
    p_t.add_argument("job_id", type=int)
    p_t.add_argument("-f", "--follow", action="store_true")
    p_t.add_argument("--uartlog", action="store_true",
        help="stream the FireSim uartlog (the live UART output from the "
             "simulated chip — same content as `screen -r fsim0` from the "
             "daemon's user, but works for any user with read perms)")
    p_t.set_defaults(func=cmd_tail)

    p_d = sub.add_parser("daemon", help="run the dispatch daemon (blocking)")
    p_d.set_defaults(func=cmd_daemon)

    p_sd = sub.add_parser("stop-daemon", help="signal the daemon to exit")
    p_sd.set_defaults(func=cmd_stop_daemon)

    p_u = sub.add_parser("usage",
        help="per-user FPGA usage stats over a trailing window")
    p_u.add_argument("--window", type=int, default=24 * 3600,
        help="trailing window in seconds (default 86400 = 24h)")
    p_u.set_defaults(func=cmd_usage)

    p_i = sub.add_parser("interactive",
        help="curses-based dashboard with live queue + usage stats")
    p_i.add_argument("--window", type=int, default=24 * 3600,
        help="initial usage window in seconds (default 86400 = 24h); "
             "press 'u' in the TUI to cycle through windows")
    p_i.set_defaults(func=cmd_interactive)

    args = p.parse_args(argv)
    # Strip the leading '--' from REMAINDER args if present.
    if hasattr(args, "cmd") and isinstance(args.cmd, list) and args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
