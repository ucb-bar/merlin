"""Render the isolated candidate filesystem permission profile."""

from __future__ import annotations

import json
from pathlib import Path

_CANDIDATE_PERMISSION_PROFILE = "merlin-candidate"


def _candidate_permission_config(codex_home: Path, *, read_paths: tuple[str, ...] | None = None, real_home) -> str:
    """Freeze the candidate's filesystem policy into the isolated Codex config.

    The outer bwrap masks /scratch and /scratch2 before remounting only declared
    inputs and tools. Granting those mount destinations read-only here therefore
    cannot reveal an unmounted host path; the writable workspace gets its own
    narrower rule. The even narrower CODEX_HOME denial is essential: its auth
    file is writable to the Codex *parent* for refresh, never to candidate tools.
    """
    if not codex_home.is_absolute():
        raise ValueError("isolated CODEX_HOME must be absolute")
    if read_paths is not None:
        if (
            not isinstance(read_paths, tuple)
            or not read_paths
            or len(set(read_paths)) != len(read_paths)
            or any(
                not isinstance(path, str)
                or not Path(path).is_absolute()
                or ".." in Path(path).parts
                or Path(path) == Path("/")
                or codex_home.is_relative_to(Path(path))
                or Path(path).is_relative_to(codex_home)
                for path in read_paths
            )
        ):
            raise ValueError("explicit candidate read grants must be exact absolute paths without control credentials")
        grants = "".join(f'{json.dumps(path)} = "read"\n' for path in read_paths)
        return (
            f"[permissions.{_CANDIDATE_PERMISSION_PROFILE}]\n"
            'extends = ":workspace"\n'
            f"[permissions.{_CANDIDATE_PERMISSION_PROFILE}.filesystem]\n"
            '":root" = "deny"\n'
            '":minimal" = "read"\n' + grants + f'{json.dumps(str(codex_home))} = "deny"\n'
            f'[permissions.{_CANDIDATE_PERMISSION_PROFILE}.filesystem.":workspace_roots"]\n'
            '"." = "write"\n'
            f"[permissions.{_CANDIDATE_PERMISSION_PROFILE}.network]\n"
            "enabled = false\n"
        )
    launcher_dir = Path.home() / ".local" / "bin"
    package_roots = dict.fromkeys((real_home() / "packages", Path.home() / ".codex" / "packages"))
    runtime_grants = "".join(
        f'{json.dumps(str(path))} = "read"\n' for path in (launcher_dir, *package_roots) if path.exists()
    )
    return (
        f"[permissions.{_CANDIDATE_PERMISSION_PROFILE}]\n"
        'extends = ":workspace"\n'
        f"[permissions.{_CANDIDATE_PERMISSION_PROFILE}.filesystem]\n"
        '":root" = "deny"\n'
        '":minimal" = "read"\n'
        '"/scratch" = "read"\n'
        '"/scratch2" = "read"\n'
        f"{runtime_grants}"
        f'{json.dumps(str(codex_home))} = "deny"\n'
        f'[permissions.{_CANDIDATE_PERMISSION_PROFILE}.filesystem.":workspace_roots"]\n'
        '"." = "write"\n'
        f"[permissions.{_CANDIDATE_PERMISSION_PROFILE}.network]\n"
        "enabled = false\n"
    )
