"""Frozen source attribution must not fall back to editable owners or bytecode."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import pytest

from merlin.common.paths import data_path, module_source_path


def _write(root, relative, source):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def _run(root, code, *, roots=("src",), names=(), before="", sources=None):
    config = {
        "snapshot_root": str(root),
        "import_roots": [str(root / relative) for relative in roots],
        "sources": sources
        if sources is not None
        else {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*.py")
        },
        "legacy_names": names,
    }
    script = (
        "import json,runpy,sys\n"
        "from pathlib import Path\n"
        "config=json.loads(sys.argv[2])\n" + before + "\nrunpy.run_path(sys.argv[1])['activate'](**config)\n" + code
    )
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            script,
            str(module_source_path("merlin.common.frozen_imports")),
            json.dumps(config),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )


def test_split_frozen_owners_and_root_precedence(tmp_path):
    _write(
        tmp_path, "src/merlin/__init__.py", "from pkgutil import extend_path\n__path__=extend_path(__path__,__name__)\n"
    )
    _write(tmp_path, "src/merlin/value.py", "VALUE='first'\n")
    _write(tmp_path, "extra/merlin/value.py", "VALUE='second'\n")
    _write(tmp_path, "extra/merlin/optional.py", "VALUE='extension'\n")
    _write(tmp_path, "extra/merlin_experiments/example.py", "VALUE=3\n")
    result = _run(
        tmp_path,
        "from merlin import value,optional\nfrom merlin_experiments import example\n"
        "assert (value.VALUE,optional.VALUE,example.VALUE)==('first','extension',3)",
        roots=("src", "extra"),
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("owner", ["merlin", "merlin_experiments", "merlin_analysis", "merlin_dse", "merlin_mining"])
def test_preloaded_owner_is_rejected(tmp_path, owner):
    _write(tmp_path, f"src/{owner}/__init__.py", "")
    result = _run(tmp_path, "", before=f"import types\nsys.modules[{owner!r}]=types.ModuleType({owner!r})")
    assert result.returncode != 0
    assert "imported before frozen source isolation" in result.stderr


def test_live_namespace_and_editable_finder_cannot_supply_missing_child(tmp_path):
    frozen = tmp_path / "frozen"
    live = tmp_path / "live"
    init = "from pkgutil import extend_path\n__path__=extend_path(__path__,__name__)\n"
    _write(frozen, "src/merlin/__init__.py", init)
    _write(live, "merlin/__init__.py", init)
    _write(live, "merlin/missing.py", "raise AssertionError('live source executed')\n")
    before = f"sys.path.append({str(live)!r})"
    code = f"""
import merlin
assert not any(str(path).startswith({str(live)!r}) for path in merlin.__path__)
class Editable:
    def find_spec(self, fullname, path=None, target=None):
        raise AssertionError('editable finder reached')
sys.meta_path.append(Editable())
merlin.__path__.append({str(live / "merlin")!r})
import merlin.missing
"""
    result = _run(frozen, code, before=before)
    assert result.returncode != 0
    assert "module unavailable in frozen source receipt: merlin.missing" in result.stderr
    assert "live source executed" not in result.stderr
    assert "editable finder reached" not in result.stderr


def test_source_hash_and_map_are_defensively_owned(tmp_path):
    path = _write(tmp_path, "src/merlin/value.py", "VALUE=1\n")
    code = f"""
Path({str(path)!r}).write_text('VALUE=2\\n')
config['sources']['src/merlin/value.py']=__import__('hashlib').sha256(b'VALUE=2\\n').hexdigest()
import merlin.value
"""
    result = _run(tmp_path, code)
    assert result.returncode != 0
    assert "frozen source hash mismatch" in result.stderr


def test_unlisted_source_is_not_authorized(tmp_path):
    _write(tmp_path, "src/merlin/__init__.py", "")
    _write(tmp_path, "src/merlin/unlisted.py", "VALUE=1\n")
    result = _run(
        tmp_path, "import merlin.unlisted", sources={"src/merlin/__init__.py": hashlib.sha256(b"").hexdigest()}
    )
    assert result.returncode != 0
    assert "absent from the verified receipt" in result.stderr


def test_stale_bytecode_is_ignored_and_bytecode_only_is_denied(tmp_path):
    import py_compile

    path = _write(tmp_path, "src/merlin/value.py", "raise AssertionError('stale bytecode executed')\n")
    py_compile.compile(str(path), cfile=str(path.with_suffix(".pyc")), doraise=True)
    path.write_text("VALUE=42\n")
    result = _run(tmp_path, "from merlin.value import VALUE\nassert VALUE==42")
    assert result.returncode == 0, result.stderr
    path.unlink()
    _write(tmp_path, "src/merlin/__init__.py", "")
    result = _run(tmp_path, "import merlin.value")
    assert result.returncode != 0
    assert "module unavailable in frozen source receipt" in result.stderr


@pytest.mark.parametrize("where", ["snapshot", "root", "package", "source"])
def test_symlink_import_paths_are_refused(tmp_path, where):
    frozen = tmp_path / "frozen"
    original = _write(frozen, "src/merlin/value.py", "VALUE=1\n")
    if where == "snapshot":
        alias = tmp_path / "alias"
        alias.symlink_to(frozen, target_is_directory=True)
        frozen = alias
    elif where == "root":
        (frozen / "alias").symlink_to(frozen / "src", target_is_directory=True)
    elif where == "package":
        (frozen / "src/merlin").rename(frozen / "actual")
        (frozen / "src/merlin").symlink_to(frozen / "actual", target_is_directory=True)
    else:
        original.rename(frozen / "value.py")
        original.symlink_to(frozen / "value.py")
    result = _run(frozen, "import merlin.value", roots=("alias",) if where == "root" else ("src",))
    assert result.returncode != 0
    assert "VALUE=1" not in result.stdout


def test_exact_legacy_helper_and_missing_helper(tmp_path):
    _write(tmp_path, "scripts/legacy_helper.py", "VALUE=19\n")
    result = _run(
        tmp_path,
        "import legacy_helper\nassert legacy_helper.VALUE==19",
        roots=("scripts",),
        names=("legacy_helper", "missing_helper"),
    )
    assert result.returncode == 0, result.stderr
    result = _run(tmp_path, "import missing_helper", roots=("scripts",), names=("legacy_helper", "missing_helper"))
    assert result.returncode != 0
    assert "module unavailable in frozen source receipt: missing_helper" in result.stderr
    result = _run(
        tmp_path,
        "",
        roots=("scripts",),
        names=("legacy_helper",),
        before="import types\nsys.modules['legacy_helper']=types.ModuleType('legacy_helper')",
    )
    assert result.returncode != 0
    assert "imported before frozen source isolation" in result.stderr


def test_package_and_namespace_resources_are_frozen(tmp_path):
    _write(tmp_path, "src/merlin/__init__.py", "")
    _write(tmp_path, "src/merlin/data/value.json", '{"value": 4}')
    _write(tmp_path, "src/merlin_experiments/first.json", '"first"')
    _write(tmp_path, "extra/merlin_experiments/second.json", '"second"')
    pins = {
        path.relative_to(tmp_path).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    code = """
from importlib.resources import files,as_file
assert files('merlin').joinpath('data/value.json').read_bytes() == b'{"value": 4}'
with as_file(files('merlin').joinpath('data/value.json')) as path:
    assert path.read_text() == '{"value": 4}'
resources=files('merlin_experiments')
assert {path.name for path in resources.iterdir()} == {'first.json','second.json'}
assert resources.joinpath('first.json').read_text() == '"first"'
assert resources.joinpath('second.json').read_text() == '"second"'
"""
    result = _run(tmp_path, code, roots=("src", "extra"), sources=pins)
    assert result.returncode == 0, result.stderr
    changed = tmp_path / "src/merlin/data/value.json"
    result = _run(
        tmp_path, f"Path({str(changed)!r}).write_text('changed')\n" + code, roots=("src", "extra"), sources=pins
    )
    assert result.returncode != 0
    assert "frozen source hash mismatch" in result.stderr


def test_nested_namespace_resources_merge_owners(tmp_path):
    _write(tmp_path, "src/merlin_experiments/common/a.json", '"first"')
    _write(tmp_path, "extra/merlin_experiments/common/b.json", '"second"')
    pins = {
        path.relative_to(tmp_path).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    code = """
from importlib.resources import files,as_file
root=files('merlin_experiments')
common,=root.iterdir()
assert common.name=='common' and common.is_dir()
assert {entry.name for entry in common.iterdir()}=={'a.json','b.json'}
with as_file(common) as directory:
    assert {path.name for path in directory.iterdir()}=={'a.json','b.json'}
    assert (directory/'a.json').read_text()=='"first"'
    assert (directory/'b.json').read_text()=='"second"'
for filename,value in [('a.json','"first"'),('b.json','"second"')]:
    resource=root.joinpath('common',filename)
    assert resource.read_text()==value
    with as_file(resource) as path:
        assert path.read_text()==value
"""
    result = _run(tmp_path, code, roots=("src", "extra"), sources=pins)
    assert result.returncode == 0, result.stderr


def test_resource_replaced_by_fifo_is_refused_before_read(tmp_path):
    _write(tmp_path, "src/merlin/__init__.py", "")
    resource = _write(tmp_path, "src/merlin/value.json", "42")
    pins = {
        path.relative_to(tmp_path).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    code = f"""
from importlib.resources import files
import os
path=Path({str(resource)!r})
path.unlink()
os.mkfifo(path)
files('merlin').joinpath('value.json').read_bytes()
"""
    result = _run(tmp_path, code, sources=pins)
    assert result.returncode != 0
    assert "not a regular file" in result.stderr


@pytest.mark.parametrize("changed", [None, "changed", "unlisted", "missing", "symlink"])
def test_real_data_path_exposes_only_verified_physical_resources(tmp_path, changed):
    for module in ("merlin", "merlin.common", "merlin.common.paths"):
        source = module_source_path(module)
        relative = module.replace(".", "/")
        relative += "/__init__.py" if source.name == "__init__.py" else ".py"
        _write(tmp_path, f"src/{relative}", source.read_text())
    schema = data_path("contract", "schemas", "command_buffer.schema.json").read_text()
    directory = tmp_path / "src/merlin/_data/contract/schemas"
    resource = _write(tmp_path, "src/merlin/_data/contract/schemas/command_buffer.schema.json", schema)
    pins = {
        path.relative_to(tmp_path).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    mutation = {
        None: "",
        "changed": f"Path({str(resource)!r}).write_text('{{}}')\n",
        "unlisted": f"Path({str(directory / 'unlisted.json')!r}).write_text('{{}}')\n",
        "missing": f"Path({str(resource)!r}).unlink()\n",
        "symlink": f"Path({str(directory / 'alias.json')!r}).symlink_to({str(resource)!r})\n",
    }[changed]
    code = f"""
import os
os.environ['MERLIN_REPO_ROOT']={str(tmp_path / "outside-checkout")!r}
from merlin.common.paths import data_path
{mutation}
directory=data_path('contract','schemas')
assert directory==Path({str(directory)!r})
schema=json.loads((directory/'command_buffer.schema.json').read_text())
assert schema['type']=='object' and 'commands' in schema['required']
assert data_path('contract','schemas','command_buffer.schema.json')==directory/'command_buffer.schema.json'
"""
    result = _run(tmp_path, code, sources=pins)
    if changed is None:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        expected = {
            "changed": "hash mismatch",
            "unlisted": "absent from the verified receipt",
            "missing": "subtree membership mismatch",
            "symlink": "escapes its declared roots",
        }[changed]
        assert expected in result.stderr
