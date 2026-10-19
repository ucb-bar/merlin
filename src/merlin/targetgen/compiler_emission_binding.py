"""Bind original capability qualification to an actual generic compiler run.

Final grade/layer consumers have their own original package/tree admission.
This check is for original host-produced qualification pairs, not a mechanism
for granting authority from a saved successful compiler or native JSON record.
"""

import json
from pathlib import Path
from types import MethodType

from merlin.common import invocation_record
from merlin.targetgen.compiler_pass_selection import _pin, _tree


def check_compiler_emission(owner, arm, *, argument_templates, required_sources):
    package = arm.get("compiler_package")
    invocation = arm.get("ordinary_invocation")
    if (
        type(package) is not dict
        or set(package) != {"path", "sha256"}
        or type(invocation) is not dict
        or set(invocation) != {"path", "sha256"}
        or _pin(invocation["path"]) != {key: invocation[key] for key in ("path", "sha256")}
        or _tree(Path(package["path"])) != package["sha256"]
    ):
        raise ValueError("qualification lacks the original exact compiler tree and invocation")
    binding = getattr(owner, "check_compiler_binding", None)
    if type(binding) is not MethodType or binding.__self__ is not owner:
        raise ValueError("qualification requires its original bound compiler-library checker")
    binding(Path(package["path"]))
    observed = invocation_record.verify(Path(invocation["path"]))
    cb, lowered, native = (arm[key] for key in ("command_buffer_artifact", "lowered", "native_result"))
    for pin in (cb, lowered, native):
        if _pin(pin["path"]) != {key: pin[key] for key in ("path", "sha256")}:
            raise ValueError("qualification compilation/native output custody changed")
    source = json.loads(Path(native["path"]).read_bytes())["source"]["interface"]
    source = {key: source[key] for key in ("path", "sha256")}
    expected = [
        str(Path(package["path"]) / "compiler.py"),
        *(part.format(input_mlir=source["path"], output_json=cb["path"]) for part in argument_templates),
    ]
    if (
        observed.get("kind") != "subprocess"
        or observed["status"] != "completed"
        or observed["returncode"] != 0
        or observed["argv"][1:] != expected
        or observed["cwd"] != package["path"]
        or observed["inputs"] != [source]
        or {key: cb[key] for key in ("path", "sha256")} not in observed["outputs"]
        or observed["stdout"]["sha256"] != lowered["sha256"]
    ):
        raise ValueError("qualification artifacts differ from the actual original selected compiler invocation")
    dependencies = {row["path"]: row["sha256"] for row in observed["dependencies"]}
    for relative in required_sources:
        path = Path(package["path"]) / relative
        if dependencies.get(str(path)) != _pin(path)["sha256"]:
            raise ValueError("qualification compiler run omitted its actual selected source dependency")
    owner.revalidate()
    return {
        "status": "checked_original_selected_compiler_emission",
        "compiler_package": package,
        "ordinary_invocation": invocation,
    }


def revalidate_compiler_emission(record, *, checked_packages=None):
    """Reopen byte custody inside an already-issued original owner's lifetime.

    This does not issue an owner, call a qualification factory or claim a new
    compilation. An original owner must first run ``check_compiler_emission``.
    """
    if (
        type(record) is not dict
        or set(record) != {"status", "compiler_package", "ordinary_invocation"}
        or record["status"] != "checked_original_selected_compiler_emission"
    ):
        raise ValueError("original compiler emission custody record is malformed")
    package, invocation = record["compiler_package"], record["ordinary_invocation"]
    key = (package["path"], package["sha256"])
    if checked_packages is None or key not in checked_packages:
        if _tree(Path(package["path"])) != package["sha256"]:
            raise ValueError("original issued compiler source tree changed")
        if checked_packages is not None:
            checked_packages.add(key)
    if _pin(Path(invocation["path"])) != invocation:
        raise ValueError("original issued compiler invocation changed")
    invocation_record.verify(Path(invocation["path"]))
