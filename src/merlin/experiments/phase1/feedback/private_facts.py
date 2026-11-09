"""Select private input facts without substituting the public effective view."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import yaml

from merlin.experiments.phase1.runtime_environment import PreparedEnvironment, applied_environment
from merlin.runtime.modelinputs import file_sha256, selected_firrtl

from . import private_full_models as models


@contextmanager
def selected_input_facts(spec: Path, *, target: str, required_models: tuple[str, ...]):
    """Bind the private roster's exact facts and restore the public readers.

    The public effective view and private raw extraction may have different
    bytes. They must still name the same verified FIRRTL target/config. This
    establishes input roles, not equivalence of extra host/SDK facts or code.
    The ordinary private gate retains its exact ambient-facts check.
    """
    if spec.is_symlink() or not spec.is_file():
        raise ValueError("private facts have no ordinary model specification")
    spec_sha = file_sha256(spec)
    document = yaml.safe_load(spec.read_bytes())
    rows = document.get("models") if isinstance(document, dict) else None
    if (
        not isinstance(document, dict)
        or document.get("schema") != models.SCHEMA
        or document.get("target") != target
        or not isinstance(rows, list)
        or not rows
        or any(not isinstance(row, dict) for row in rows)
        or len(rows) != len(required_models)
        or {row.get("id") for row in rows} != set(required_models)
    ):
        raise ValueError("private facts differ from the required model roster")
    selections = []
    for row in rows:
        path = models._file(spec, row.get("rtl_facts"), row.get("rtl_facts_sha256"))
        selections.append(selected_firrtl(path, target=target, config=str(row.get("rtl_config") or "")))
    raw = selections[0]
    identity = (raw["sha256"], raw["target"], raw["config"], raw["firrtl_sha256"])
    if any((item["sha256"], item["target"], item["config"], item["firrtl_sha256"]) != identity for item in selections):
        raise ValueError("private model roster selects conflicting RTL facts")
    ambient_path = os.environ.get("MERLIN_RTL_FACTS", "").strip()
    if not ambient_path:
        raise ValueError("formal private handoff has no selected public RTL facts")
    public = selected_firrtl(Path(ambient_path), target=target, config=raw["config"])
    if public["firrtl_sha256"] != raw["firrtl_sha256"]:
        raise ValueError("private facts do not describe the public selected FIRRTL")
    environment = dict(os.environ, MERLIN_RTL_FACTS=raw["path"])
    with applied_environment(PreparedEnvironment(environment=environment, account={})):
        yield {"public_effective": public, "private_input": raw}
        if (
            os.environ.get("MERLIN_RTL_FACTS") != raw["path"]
            or file_sha256(spec) != spec_sha
            or any(selected_firrtl(item["path"], target=target, config=item["config"]) != item for item in selections)
            or selected_firrtl(public["path"], target=target, config=public["config"]) != public
        ):
            raise ValueError("selected private/public RTL facts changed during model builds")
