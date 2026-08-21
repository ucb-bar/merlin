"""Render a :class:`~merlin.targetgen.sandbox.plan.MountPlan` as a ``docker run`` command.

The second isolation backend, for hosts where bwrap cannot run. That is not hypothetical: Ubuntu 24.04+
ships ``kernel.apparmor_restrict_unprivileged_userns=1``, under which a non-setuid bwrap creates a user
namespace but is denied the ``uid_map`` write — so an unprivileged bwrap (a conda build, say) installs
fine and then dies at ``setting up uid map: Permission denied``. Only the distro package works, because
it ships an AppArmor profile permitting it. Docker sidesteps this entirely: the daemon runs as root and
does the namespacing, so nothing unprivileged needs the capability.

HOST-MOUNT MODEL (a deliberate choice): this binds ``/usr``, ``/bin``, ``/lib``, ``/etc`` from the host
exactly as bwrap does, rather than relying on an image's own filesystem. It keeps the mount table — and
therefore the isolation proof in :mod:`.plan` — identical across both backends, and means there is no
image to build or keep in sync with the host toolchain. The base image supplies only a shell.

ORDERING — the one real semantic gap. bwrap applies mounts in argv order and the plan depends on that
("deny wins … masks AFTER the allowed binds"; "workspace LAST"). Docker does NOT honour flag order: it
sorts by destination depth, so a nested mount always wins over its parent regardless of where it appears.
For the shape this plan actually produces — deny paths are sub-paths of allowed ones — the two agree.
Where they cannot agree is a mask at the SAME destination as an earlier mount: bwrap's last-wins would
apply it, while docker rejects the duplicate outright. :func:`_dedupe_last_wins` resolves that here, in
the renderer, so the plan keeps one ordering contract and the backends both honour it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from merlin.targetgen.sandbox.plan import Mount, MountPlan


def _config_dir() -> str:
    from merlin.targetgen.sandbox.answer_surfaces import claude_config_dir
    return str(claude_config_dir())

#: A shell and coreutils, nothing else — every real tool is bound from the host (see the module note).
DEFAULT_IMAGE = "ubuntu:24.04"


def _dedupe_last_wins(mounts: tuple[Mount, ...]) -> list[Mount]:
    """One mount per destination, keeping the LAST — bwrap's tie-break, which docker cannot express.

    Docker refuses duplicate destinations, so without this a plan whose mask lands on an already-mounted
    path would fail to start rather than mask. Keeping the last preserves exactly what `plan.is_exposed`
    predicts, so the isolation proof stays true of what actually runs.
    """
    by_dst: dict[str, Mount] = {}
    for m in mounts:
        by_dst[m.dst] = m
    return [by_dst[d] for d in sorted(by_dst, key=lambda d: (d.count("/"), d))]


def daemon_can_reach(src: str) -> bool:
    """Can the docker DAEMON set up a bind from ``src``?

    Not the same question as "can I read it". The daemon runs as root, which ignores permissions on a
    local filesystem — but a network home directory is typically exported with **root_squash**, mapping
    root to ``nobody``. A ``0700`` dir under an NFS home is then invisible to the daemon, which reports
    the mount source as missing and tries to ``mkdir`` it: ``error while creating mount source path …
    permission denied``. Observed here on ``~/.local/share/claude`` (0700, NFS) while every ``/scratch``
    path (local ext4, real root) mounted fine.

    So: local paths are always reachable; under a network mount every directory component needs ``o+x``
    (and a file leaf needs ``o+r``).
    """
    p = Path(src)
    if not _under_network_fs(p):
        return True
    for comp in [p, *p.parents]:
        try:
            st = comp.stat()
        except OSError:
            return False
        need = 0o001 if comp.is_dir() else 0o004
        if not st.st_mode & need:
            return False
        if str(comp) == "/":
            break
    return True


def _under_network_fs(p: Path) -> bool:
    """True if ``p`` sits on a network filesystem (nfs/cifs/lustre/afs) — where root gets squashed."""
    try:
        mounts = Path("/proc/mounts").read_text().splitlines()
    except OSError:
        return False
    best, kind = "", ""
    for line in mounts:
        parts = line.split()
        if len(parts) < 3:
            continue
        mp, fstype = parts[1], parts[2]
        if (str(p) == mp or mp == "/" or str(p).startswith(mp.rstrip("/") + "/")) and len(mp) > len(best):
            best, kind = mp, fstype
    return kind.startswith(("nfs", "cifs", "lustre", "afs", "fuse.sshfs"))


def unreachable_sources(plan: MountPlan) -> list[str]:
    """Mount sources the daemon cannot bind — the actionable form of a cryptic docker startup error."""
    return sorted({m.src for m in plan.mounts
                   if m.src and m.op in ("ro", "rw") and not daemon_can_reach(m.src)})


def stage_unreachable(plan: MountPlan, staging: Path) -> MountPlan:
    """Copy daemon-unreachable sources onto a local staging dir and re-point their mounts at the copies.

    The DESTINATIONS are unchanged — the agent still sees ``~/.local/share/claude`` where it expects it;
    only the host side moves to somewhere the daemon can actually read. Staging goes under the run's own
    workspace (local disk), so nothing is made world-readable to work around the NFS squash: the daemon
    is real root there and traverses a 0700 dir fine.
    """
    import shutil as _sh
    out: list[Mount] = []
    for m in plan.mounts:
        if not (m.src and m.op in ("ro", "rw")) or daemon_can_reach(m.src):
            out.append(m)
            continue
        dest = staging / Path(m.src).relative_to("/")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            src = Path(m.src)
            if src.is_dir():
                _sh.copytree(src, dest, symlinks=True, dirs_exist_ok=True)
            else:
                _sh.copy2(src, dest)
        out.append(Mount(m.op, m.dst, src=str(dest), optional=m.optional))
    return MountPlan(mounts=tuple(out), unsetenv=plan.unsetenv, workdir=plan.workdir)


def render(plan: MountPlan, *, image: str = DEFAULT_IMAGE, cidfile: str | None = None) -> list[str]:
    """The full ``docker run`` argv for a plan (without the trailing command)."""
    home = os.path.expanduser("~")
    argv = [
        "docker", "run", "--rm",
        # Same uid as the caller: the default root-in-container would leave every file the agent writes
        # root-owned on the host, and grading could then neither read nor clean up the run dir.
        "--user", f"{os.getuid()}:{os.getgid()}",
        # Docker inherits NO host environment. That is a feature for the nested-session vars (they are
        # absent by construction, where bwrap must strip them) but it silently breaks two things bwrap
        # got for free, and both are fatal rather than degraded:
        #
        #   HOME — with --user and no matching passwd entry the container gets HOME=/, so `~/.claude`
        #          resolves to `//.claude` and the agent's credentials/config are simply not there.
        #   PATH — the container default has no `~/.local/bin`, which is where the `claude` launcher
        #          lives, so the agent command dies with `claude: command not found` (rc 127) and the
        #          QA loop burns through its rounds doing nothing.
        #
        # Both are set explicitly here so the container's environment matches what the plan's mounts
        # already assume. `sandbox_env` then prepends the venv/LLVM/sim dirs on top of this PATH.
        "-e", f"HOME={home}",
        # Point the agent's CLI at the same config the plan binds. Without it the CLI falls back to
        # ~/.claude inside the container, which on a host that relocates the config is a different (and
        # here, expired) session — the run then dies at auth having looked entirely healthy up to that
        # point. Derived, never the literal home path.
        "-e", f"CLAUDE_CONFIG_DIR={_config_dir()}",
        "-e", f"PATH={home}/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        # Host networking, for two reasons that point the same way.
        #
        # FIDELITY: bwrap never unshares the network (there is no --unshare-net anywhere in the plan),
        # so the agent has always run with the host's network. Matching that keeps the two backends
        # comparable; giving docker its own netns would make this backend STRICTER than the one the
        # measured runs used, which is its own kind of incomparability.
        #
        # NECESSITY: the host-mount model binds the host's /etc, which masks the per-container
        # /etc/resolv.conf docker injects. The host's copy is systemd-resolved's 127.0.0.53 stub —
        # correct in the host netns, meaningless in a private one. The agent reached the API layer and
        # died with `API Error: Unable to connect to API (ENOTIMP)`: DNS, not credentials.
        "--network", "host",
    ]
    # NO `--init`, and the reason is specific to the host-mount model rather than to this daemon.
    # Docker injects its reaping shim at /sbin/docker-init; on a modern Ubuntu image /sbin is a symlink
    # to /usr/sbin, and we bind the HOST's /usr over the image — which masks the injected binary. The
    # container then dies at startup with `exec: "/sbin/docker-init": no such file or directory`, an
    # error that names nothing about mounts. `docker run --init ubuntu:24.04 true` succeeds on its own,
    # so a probe cannot see this: it is the interaction that breaks. The shell becomes pid 1 instead and
    # reaps its own children, which is all this workload creates.
    if cidfile:
        # bwrap has --die-with-parent; docker has nothing equivalent, so the launcher must be able to
        # find and kill the container if it outlives its parent. A 4-hour orphan holding the workspace
        # is the failure this exists to make recoverable.
        argv += ["--cidfile", cidfile]
    for m in _dedupe_last_wins(plan.mounts):
        if m.op in ("proc", "dev"):
            continue                      # docker provides /proc and /dev itself
        if m.op == "ro":
            argv += ["-v", f"{m.src}:{m.dst}:ro"]
        elif m.op == "rw":
            argv += ["-v", f"{m.src}:{m.dst}"]
        elif m.op == "devnull":
            argv += ["-v", f"/dev/null:{m.dst}:ro"]
        elif m.op == "hide":
            argv += ["--tmpfs", m.dst]
    if plan.workdir:
        argv += ["-w", plan.workdir]
    # NOTE: plan.unsetenv needs no rendering. Docker does not inherit the host environment, so the
    # nested-session vars are absent by construction — the opposite of bwrap, which must strip them.
    # The isolation test asserts that rather than trusting it.
    argv.append(image)
    return argv


def available() -> bool:
    """True if docker can actually run a container here (never raises)."""
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def wrap(te, ws, inner: str, bundle: dict | None = None, *, image: str = DEFAULT_IMAGE,
         cidfile: str | None = None) -> str:
    """A ready-to-run shell string: ``docker run`` + the sandbox env exports + the inner command."""
    from merlin.targetgen.sandbox import toolchain as TC
    from merlin.targetgen.sandbox.plan import full_plan
    plan = full_plan(te, ws, bundle)
    if unreachable_sources(plan):
        plan = stage_unreachable(plan, Path(ws) / ".sandbox-staging")
    argv = render(plan, image=image, cidfile=cidfile)
    return " ".join(argv) + f" bash -c '{TC.sandbox_env(te, ws)} {inner}'"
