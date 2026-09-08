"""Bind pure short builds to the host's existing immutable sandbox namespace.

This resolves obligations only. It never adds a mount or runs a tool/target.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024*1024):
            result.update(chunk)
    return result.hexdigest()


def _exports(prefix):
    records = {}
    for index, arg in enumerate(prefix[:-2]):
        if Path(arg).name != "bash" or prefix[index+1] != "-c":
            continue
        words = shlex.split(prefix[index+2])
        for i, word in enumerate(words[:-1]):
            if word != "export":
                continue
            key, sep, value = words[i+1].rstrip(";").partition("=")
            if sep and key in {"PYTHONPATH", "MERLIN_GEMMINI_HARNESS_DIR"}:
                if key in records:
                    raise ValueError("ambiguous native policy build environment")
                records[key] = value
    return records


def short_program_environment(sandbox):
    from . import gemmini
    from merlin.targetgen.gsim_emulator import citation
    from merlin.targetgen.sandbox.bwrap import is_exposed

    prefix = sandbox["command_prefix"]
    if not prefix or Path(prefix[0]).name != "bwrap" or "--clearenv" not in prefix:
        raise ValueError("short build environment requires the existing clear-environment policy")
    exports = _exports(prefix)
    python_path = exports.get("PYTHONPATH", "")
    suffix = "${PYTHONPATH:+:$PYTHONPATH}"
    if python_path.endswith(suffix):
        python_path = python_path[:-len(suffix)]
    if "$" in python_path or ":" in python_path:
        raise ValueError("unsupported or ambiguous compiler Python namespace")
    namespace = Path(python_path)
    if (not namespace.is_absolute() or tuple(namespace.parts[-2:]) != ("merlin", "python")
            or not namespace.is_dir()):
        raise ValueError("native policy lacks an exact existing compiler Python namespace")
    project = namespace.parent.parent
    interpreter = project/".venv/bin/python"
    if not interpreter.is_file() or not is_exposed(prefix, interpreter):
        raise ValueError("native policy lacks the selected existing interpreter")
    recipe = gemmini.harness_build_recipe()
    harness = gemmini.rocc_tests_dir().resolve()
    alias = Path(exports.get("MERLIN_GEMMINI_HARNESS_DIR", str(harness)))
    if not alias.is_absolute() or not alias.is_dir() or "$" in str(alias):
        raise ValueError("native policy lacks an existing literal harness path")
    evidence = {}
    for source in (*recipe.support_sources, recipe.link_script):
        source = Path(source)
        destination = alias/source.relative_to(harness)
        if (not destination.is_file() or not is_exposed(prefix, destination)
                or _sha(source) != _sha(destination)):
            raise ValueError("existing harness view differs from the selected target build recipe")
        evidence[str(destination)] = _sha(destination)
    engine = citation("gemmini", env_var=gemmini.GSIM_EMU_ENV)
    if (engine.get("available") is not True or engine.get("refused") is not False
            or engine.get("receipt_status") != "bound" or engine.get("flavour") != "binary"):
        raise ValueError("configured short runtime engine lacks an exact bound build receipt")
    return {"schema": "existing_short_program_environment_v1", "python_executable": str(interpreter),
        "build_namespace_root": str(project), "build_path_bindings": ((str(harness), str(alias)),),
        "engine_provenance": engine, "existing_support_pins": evidence,
        "adapter_source_pins": {str(Path(__file__).resolve()): _sha(__file__),
                                str(Path(gemmini.__file__).resolve()): _sha(gemmini.__file__)},
        "sandbox_policy_sha256": hashlib.sha256(json.dumps(sandbox, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "mount_grants": False, "scope": "existing namespace and byte-identical recipe paths only"}
