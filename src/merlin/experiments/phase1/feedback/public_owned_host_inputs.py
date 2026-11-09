"""Explicit judge-only host build inputs, independent of private model campaigns.

These selections supply build inputs, never execution credit or author grants.
The original host-leaf/model owners retain their source, ELF and native gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from merlin.common.digest import is_sha256
from merlin.common.strict_json import loads

SCHEMA = "merlin.provided_host_build_inputs.v1"
MAX_RECORD_BYTES = 1024 * 1024
FIELDS = {"schema", "target", "descriptor", "board_catalog", "dts", "rtl_facts", "board", "arena_mb"}
KEY = "public_owned_host_inputs"
FROZEN_SCHEMA = "merlin.frozen_public_owned_host_inputs.v1"


def _ordinary(path):
    path = Path(path)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or any(member.is_symlink() for member in (path, *path.parents))
        or not path.is_file()
    ):
        raise ValueError("host build input must be an ordinary absolute file")
    return path


def _pinned(record):
    if (
        type(record) is not dict
        or set(record) != {"path", "sha256"}
        or type(record["path"]) is not str
        or not is_sha256(str(record["sha256"]))
    ):
        raise ValueError("host build input needs an exact path and SHA256 pin")
    path = _ordinary(record["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError("selected host build input changed")
    return path


def verify(selection, *, target, descriptor, repo=None):
    """Validate explicit byte pins and the descriptor/board/DTS/FIRRTL join.

    No directory search, target-name default, ambient host-selection fallback,
    synthetic model record, arena clamp or author-visible projection is used.
    """
    from merlin.runtime.boards import load_boards
    from merlin.runtime.modelinputs import selected_firrtl
    from merlin.targetgen.target_experiment import load_target_experiment

    request = _pinned(selection)
    if request.stat().st_size > MAX_RECORD_BYTES:
        raise ValueError("host build input record exceeds its bounded document")
    raw = request.read_bytes()
    record = loads(raw, max_bytes=MAX_RECORD_BYTES)
    if (
        type(record) is not dict
        or not FIELDS <= set(record)
        or set(record) - FIELDS - {"instruction_declaration"}
        or record["schema"] != SCHEMA
        or record["target"] != target
    ):
        raise ValueError("host build input record has the wrong schema, target or fields")
    selected_descriptor = _pinned(record["descriptor"])
    if selected_descriptor != _ordinary(descriptor):
        raise ValueError("host build descriptor differs from the selected experiment")
    experiment = load_target_experiment(selected_descriptor, source_root=repo)
    catalog, dts, facts = (_pinned(record[key]) for key in ("board_catalog", "dts", "rtl_facts"))
    if experiment.target != target or experiment.selected_board_catalog() != catalog:
        raise ValueError("host catalog differs from the selected target descriptor")
    name, arena = record["board"], record["arena_mb"]
    if type(name) is not str or not name or type(arena) is not int or arena < 1:
        raise ValueError("host build requires an explicit board and positive integer arena")
    board = load_boards(catalog).get(name)
    if (
        board is None
        or board.target != target
        or not board.rtl_sim_config
        or board.host_dts_sha256 != record["dts"]["sha256"]
        or not board.code_reserve
        or board.dram_bytes <= board.code_reserve
    ):
        raise ValueError("host board, target and exact DTS pins do not agree")
    selected_firrtl(facts, target=target, config=board.rtl_sim_config)
    ceiling = (board.dram_bytes - board.code_reserve) // (2 * 1024 * 1024)
    if arena > ceiling:
        raise ValueError("host arena exceeds the selected board's bounded public build space")
    selected = {key: dict(record[key]) for key in ("board_catalog", "dts", "rtl_facts")}
    selected.update(board=name, arena_mb=arena)
    if "instruction_declaration" in record:
        from merlin.experiments.phase1.feedback.private_instruction_declaration import read

        declaration = _pinned(record["instruction_declaration"])
        read(declaration, target=target).verify()
        selected["instruction_declaration"] = dict(record["instruction_declaration"])
    # Recheck all pins after semantic validation before publishing any selection.
    for key in ("descriptor", "board_catalog", "dts", "rtl_facts", "instruction_declaration"):
        if key in record:
            _pinned(record[key])
    if request.read_bytes() != raw or _pinned(selection) != request:
        raise ValueError("host build input selection changed before publication")
    return selected


def prepare(selection, *, target, descriptor, output, repo=None):
    """Emit the existing five-/six-field native selection, privately and once."""
    selected = verify(selection, target=target, descriptor=descriptor, repo=repo)
    output = Path(output)
    if (
        not output.is_absolute()
        or ".." in output.parts
        or output.exists()
        or any(member.is_symlink() for member in (output, *output.parents))
    ):
        raise ValueError("host build selection needs a fresh ordinary absolute output")
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(selected, sort_keys=True, indent=2) + "\n").encode()
    import os

    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
    return {"path": str(output), "sha256": hashlib.sha256(raw).hexdigest()}


def _file_pin(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _private_root(run_dir):
    return Path(run_dir).absolute() / "private_public_host_inputs"


def verify_record(record, *, target, descriptor, run_dir):
    """Reopen only the normal session's exact source and private copied bytes."""
    root = _private_root(run_dir)
    if record is None:
        if root.exists():
            raise ValueError("run-owned host input selection was removed")
        return None
    if (
        type(record) is not dict
        or set(record) != {"schema", "target", "source", "request", "inputs", "selection"}
        or record["schema"] != FROZEN_SCHEMA
        or record["target"] != target
    ):
        raise ValueError("run-owned host input record is malformed")
    source = _pinned(record["source"])
    request = _pinned(record["request"])
    if (
        request != root / "request.json"
        or source.read_bytes() != request.read_bytes()
        or record["request"]["sha256"] != record["source"]["sha256"]
    ):
        raise ValueError("run-owned host input source differs from its private copy")
    original = loads(request.read_bytes(), max_bytes=MAX_RECORD_BYTES)
    if (
        type(original) is not dict
        or not FIELDS <= set(original)
        or set(original) - FIELDS - {"instruction_declaration"}
        or original["schema"] != SCHEMA
        or original["target"] != target
        or _pinned(original["descriptor"]) != _ordinary(descriptor)
    ):
        raise ValueError("run-owned host descriptor or source selection changed")
    roles = {"descriptor", "board_catalog", "dts", "rtl_facts"}
    if "instruction_declaration" in original:
        roles.add("instruction_declaration")
    if set(record["inputs"]) != roles:
        raise ValueError("run-owned host input membership changed")
    native = {"board": original["board"], "arena_mb": original["arena_mb"]}
    for role in roles:
        selected = _pinned(original[role])
        copied = _pinned(record["inputs"][role])
        if (
            copied != root / "files" / (role + ".input")
            or record["inputs"][role]["sha256"] != original[role]["sha256"]
            or selected.read_bytes() != copied.read_bytes()
        ):
            raise ValueError("run-owned host input bytes or placement changed")
        if role != "descriptor":
            native[role] = record["inputs"][role]
    path = _pinned(record["selection"])
    if path != root / "selection.json" or loads(path.read_bytes()) != native:
        raise ValueError("native host selection differs from the run-owned copied inputs")
    if "instruction_declaration" in native:
        from merlin.experiments.phase1.feedback.private_instruction_declaration import read

        read(native["instruction_declaration"]["path"], target=target).verify()
    return dict(record["selection"])


def freeze_for_run(path, sha256, *, target, descriptor, run_dir, repo=None, prior=None, resuming=False):
    """Validate before spend; copy every direct input privately; bind resume."""
    if not path and not sha256:
        if resuming and prior is not None:
            raise ValueError("resume refused: public host inputs were removed")
        return None, None
    if not path or not sha256:
        raise ValueError("public host inputs require both explicit path and SHA256")
    source = {"path": str(path), "sha256": sha256}
    selected = verify(source, target=target, descriptor=descriptor, repo=repo)
    root = _private_root(run_dir)
    if resuming:
        if prior is None or prior.get("source") != source:
            raise ValueError("resume refused: selected public host input source changed")
        return verify_record(prior, target=target, descriptor=descriptor, run_dir=run_dir), prior
    if root.exists() or root.is_symlink():
        raise ValueError("public host inputs require a fresh private run directory")
    from merlin.experiments.corpus.preparation import copy_input

    original = loads(_pinned(source).read_bytes(), max_bytes=MAX_RECORD_BYTES)
    root.mkdir(parents=True, mode=0o700)
    files = root / "files"
    files.mkdir(mode=0o700)
    request = root / "request.json"
    copy_input(Path(source["path"]), request, private=True, expected_sha256=sha256)
    request.chmod(0o400)
    frozen = {}
    for role in ("descriptor", "board_catalog", "dts", "rtl_facts", "instruction_declaration"):
        if role not in original:
            continue
        destination = files / (role + ".input")
        copy_input(Path(original[role]["path"]), destination, private=True, expected_sha256=original[role]["sha256"])
        destination.chmod(0o400)
        frozen[role] = _file_pin(destination)
        if role != "descriptor":
            selected[role] = frozen[role]
    selection = root / "selection.json"
    selection.write_text(json.dumps(selected, sort_keys=True, indent=2) + "\n")
    selection.chmod(0o400)
    record = dict(
        schema=FROZEN_SCHEMA,
        target=target,
        source=source,
        request=_file_pin(request),
        inputs=frozen,
        selection=_file_pin(selection),
    )
    verify_record(record, target=target, descriptor=descriptor, run_dir=run_dir)
    return record["selection"], record


def from_run(run_dir, *, target, descriptor):
    """Formal grading reopens the protected native record, not inherited env."""
    import yaml

    path = Path(run_dir).absolute() / "environment.yaml"
    if not path.exists() and not path.is_symlink():
        return verify_record(None, target=target, descriptor=descriptor, run_dir=run_dir)
    environment = yaml.safe_load(_ordinary(path).read_bytes())
    if type(environment) is not dict:
        raise ValueError("host build inputs require an original native environment")
    record = environment.get(KEY)
    if (None if record is None else record.get("source")) != (environment.get("run_config") or {}).get(KEY):
        raise ValueError("host inputs differ from the original native invocation")
    return verify_record(record, target=target, descriptor=descriptor, run_dir=run_dir)


def install_from_run(run_dir, *, target, descriptor):
    """Formal operator child trusts the owned record, never an inherited pin."""
    import os

    from .public_owned_host_selection import ENV

    os.environ.pop(ENV, None)
    selected = from_run(run_dir, target=target, descriptor=descriptor)
    if selected is not None:
        os.environ[ENV] = json.dumps(selected, sort_keys=True)
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare(
        {"path": str(args.inputs), "sha256": args.inputs_sha256},
        target=args.target,
        descriptor=args.descriptor,
        output=args.output,
        repo=args.repo,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
