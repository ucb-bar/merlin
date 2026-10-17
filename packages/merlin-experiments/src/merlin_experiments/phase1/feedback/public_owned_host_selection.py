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


def _public_rtl_board(selected, private_arena_mb, *, descriptor, target):
    """Select the declared RTL board for public host leaves, with a bounded arena.

    Private whole models may use a larger functional Spike memory span. Public
    host leaves need an ELF that can execute on the selected RTL board as well.
    The separate private build selection is never rewritten here.
    """
    from merlin.runtime.boards import load_boards
    from merlin.targetgen.owned_host_model_execution import _selected_facts
    from merlin.targetgen.target_experiment import load_target_experiment

    experiment = load_target_experiment(descriptor)
    if experiment.target != target or not experiment.host_board:
        raise ValueError("public host leaves require the selected target's RTL host board")
    catalog = experiment.selected_board_catalog()
    if catalog is None or (
        hashlib.sha256(catalog.read_bytes()).hexdigest() != selected["board_catalog"]["sha256"]
    ):
        raise ValueError("public host board catalog differs from the frozen private input")
    board = load_boards(catalog).get(experiment.host_board)
    facts = _selected_facts(selected["rtl_facts"]["path"], target=target)
    if (
        board is None or board.target != target or board.rtl_sim_config != facts["config"]
        or board.host_dts_sha256 != selected["dts"]["sha256"]
        or not board.code_reserve or board.dram_bytes <= board.code_reserve
    ):
        raise ValueError("public host board, DTS and RTL facts identify different configurations")
    # Keep half of the space after the board's code floor available for actual
    # code, static I/O and alignment. The final build checks the exact layout.
    arena_ceiling_mb = (board.dram_bytes - board.code_reserve) // (2 * 1024 * 1024)
    if arena_ceiling_mb < 1:
        raise ValueError("selected RTL board has no bounded public host arena")
    selected["board"] = experiment.host_board
    selected["arena_mb"] = min(private_arena_mb, arena_ceiling_mb)


def prepare(spec, declaration, *, target, output, selected_facts=None, selected_facts_sha256=None,
            descriptor=None):
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
    # Pure provided-host leaves use only the pinned board/DTS/facts. Device
    # model builds separately require the complete instruction declaration.
    if declaration is not None:
        declaration = Path(declaration)
        if declaration.is_symlink() or not declaration.is_file():
            raise ValueError("owned host selection requires exact operator instruction declaration")
        selected["instruction_declaration"] = {
            "path": str(declaration.absolute()),
            "sha256": hashlib.sha256(declaration.read_bytes()).hexdigest(),
        }
    selected["arena_mb"] = max(arenas)
    if descriptor is not None:
        _public_rtl_board(selected, selected["arena_mb"], descriptor=descriptor, target=target)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selected, sort_keys=True, indent=2) + "\n")
    output.chmod(0o600)
    return {"path": str(output.absolute()), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def install_for_session(spec, declaration, *, target, run_dir, selected_facts=None, selected_facts_sha256=None,
                        descriptor=None):
    """Publish pinned operator inputs before authoring and broker subprocess startup."""
    os.environ.pop(ENV, None)
    if spec is None:
        return None
    record = prepare(
        spec,
        declaration,
        target=target,
        selected_facts=selected_facts,
        selected_facts_sha256=selected_facts_sha256,
        descriptor=descriptor,
        output=Path(run_dir).absolute() / "public_owned_host_build_selection.json",
    )
    os.environ[ENV] = json.dumps(record, sort_keys=True)
    return record
