"""Protected, answer-free build inputs for Merlin-owned public host compilation."""

import hashlib
import json
import os
from pathlib import Path

import yaml

from merlin.common.strict_json import loads

from .private_source_freeze import pinned_file

ENV = "MERLIN_MODEL_OWNED_HOST_BUILD_SELECTION"


def _public_facts_pin(private_pin, *, target, path, sha256):
    """Change only serialization/placement, never the complete original fact view."""
    if path is None and sha256 is None:
        return private_pin
    if (
        path is None
        or type(sha256) is not str
        or len(sha256) != 64
        or any(char not in "0123456789abcdef" for char in sha256)
    ):
        raise ValueError("public host facts require an exact admitted byte pin")
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or path.is_symlink() or not path.is_file():
        raise ValueError("public host facts require an ordinary admitted absolute path")
    limit = 16 * 1024 * 1024
    private = Path(private_pin["path"])
    if path.stat().st_size > limit or private.stat().st_size > limit:
        raise ValueError("public host facts exceed their bounded document")
    raw, original = path.read_bytes(), private.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256 or hashlib.sha256(original).hexdigest() != private_pin["sha256"]:
        raise ValueError("public or private host facts changed during projection")
    public_data, private_data = loads(raw, max_bytes=limit), loads(original, max_bytes=limit)
    inputs = public_data.get("inputs") if type(public_data) is dict else None
    if (
        type(inputs) is not dict
        or inputs.get("target") != target
        or json.dumps(public_data, sort_keys=True, allow_nan=False, separators=(",", ":"))
        != json.dumps(private_data, sort_keys=True, allow_nan=False, separators=(",", ":"))
    ):
        raise ValueError("public host facts differ from the complete original hardware/provenance view")
    if path.read_bytes() != raw or private.read_bytes() != original:
        raise ValueError("host facts changed before public projection publication")
    return {"path": str(path), "sha256": sha256}


def prepare(spec, declaration, *, target, output, selected_facts=None, selected_facts_sha256=None):
    spec = Path(spec)
    document = yaml.safe_load(spec.read_bytes())
    if document.get("target") != target or not document.get("models"):
        raise ValueError("owned host selection requires matching private launch configuration")
    rows = document["models"]
    selected = None
    arenas = []
    for row in rows:
        board = row.get("board")
        arena = row.get("arena_mb")
        if not isinstance(board, str) or not board or type(arena) is not int or arena <= 0:
            raise ValueError("owned host selection requires explicit board and positive arena_mb")
        current = {"board": board}
        for key, source in [("board_catalog", "board_catalog"), ("dts", "host_dts"), ("rtl_facts", "rtl_facts")]:
            path = pinned_file(spec, row.get(source), row.get(source + "_sha256"))
            current[key] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if selected is not None and current != selected:
            raise ValueError("public host inputs require one common selected catalog, DTS, board and RTL extraction")
        selected = current
        arenas.append(arena)
    selected["rtl_facts"] = _public_facts_pin(
        selected["rtl_facts"], target=target, path=selected_facts, sha256=selected_facts_sha256
    )
    declaration = Path(declaration)
    if declaration.is_symlink() or not declaration.is_file():
        raise ValueError("owned host selection requires exact operator instruction declaration")
    selected["instruction_declaration"] = {
        "path": str(declaration.absolute()),
        "sha256": hashlib.sha256(declaration.read_bytes()).hexdigest(),
    }
    selected["arena_mb"] = max(arenas)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selected, sort_keys=True, indent=2) + "\n")
    output.chmod(0o600)
    return {"path": str(output.absolute()), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def install_for_session(spec, declaration, *, target, run_dir, selected_facts=None, selected_facts_sha256=None):
    """Publish pinned operator inputs before authoring and broker subprocess startup."""
    os.environ.pop(ENV, None)
    if spec is None or declaration is None:
        return None
    record = prepare(
        spec,
        declaration,
        target=target,
        selected_facts=selected_facts,
        selected_facts_sha256=selected_facts_sha256,
        output=Path(run_dir).absolute() / "public_owned_host_build_selection.json",
    )
    os.environ[ENV] = json.dumps(record, sort_keys=True)
    return record
