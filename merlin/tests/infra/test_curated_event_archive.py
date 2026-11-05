"""Compressed event evidence restores the exact bytes pinned by its original studies."""

import importlib.util
import json
from pathlib import Path

from merlin.common.paths import repo_root


def test_archived_events_match_unchanged_historical_manifests(tmp_path):
    root = repo_root()
    spec = importlib.util.spec_from_file_location(
        "restore_reference_archive", root / "build_tools/scripts/restore_reference_archive.py"
    )
    restorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(restorer)
    evidence = root / "experiments/reference-data/public-event-traces-20261022"
    archive = evidence / "event-traces.tar.gz"
    _, rows = restorer.load_members(evidence / "manifest.json", archive)
    restorer.verify(archive, rows)
    assert restorer.restore(archive, tmp_path, rows) == 2
    assert restorer.restore(archive, tmp_path, rows) == 0
    for name, row in rows.items():
        relative = Path(name)
        study = relative.parent if relative.parent.name != "matmul-trace" else relative.parent.parent
        manifest = json.loads((root / study / "manifest.json").read_bytes())
        original = next(
            member for member in manifest["members"] if member["path"] == relative.relative_to(study).as_posix()
        )
        assert original["sha256"] == row["sha256"] == restorer.sha256_file(tmp_path / name)
        assert original["bytes"] == row["bytes"] == (tmp_path / name).stat().st_size
        assert (tmp_path / name).stat().st_mode & 0o777 == int(row["mode"], 8)
