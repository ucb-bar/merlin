"""The libyaml loader builds the documents, types and errors PyYAML's pure SafeLoader builds."""

from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from merlin.common.yaml import fast_safe_load

SAMPLES = [
    "b: [1, 2.5, 1e3, 1.0e+3, .inf, -.inf, null, ~, yes, No, on, OFF, 0x1f, 0o7, 017, 0b101, 1_000, 190:20:30, '1', \"x\\ty\"]\n",
    "a: {z: 1, y: 2, 3: three, 1.5: f, true: t, null: n}\n",
    "t: [2026-10-28, 2026-10-28T11:12:52Z, 2026-10-28 11:12:52.5 +02:00, 2026-1-8]\n",
    "c: !!set {p, q}\nd: !!omap [a: 1, b: 2]\ne: !!binary aGk=\nf: !!str 12\ng: !!int '12'\nh: !!float 3\n",
    "base: &b {x: 1, y: 2}\nuse: {<<: *b, y: 3}\nalias: *b\n",
    "dup: 1\ndup: 2\nother: {k: 1, k: 2}\n",
    "f: |\n  multi\n  line\ng: >-\n  folded\n  text\nh: 'é ü \u2028 \x85'\ni: \"\\u00e9\\x41\\n\"\n",
    "- - 1\n  - 2\n- {a: [b, {c: d}]}\n- ? complex\n  : value\n",
    "empty:\nlist: []\nmap: {}\nstr: ''\n",
    "---\na: 1\n...\n",
    "# only a comment\n",
    "",
]
MALFORMED = [
    "a: [",
    "a: b: c",
    "- a\nb: 1\n",
    "a: 'unterminated",
    "a: !!int notanint\n",
    "a: *undefined\n",
    "\tkey: value\n",
    "a: 1\n---\nb: 2\n",  # two documents: safe_load refuses
    "!!python/object:os.system {}\n",
    'a: "\\q"\n',
    "? [list, key]\n: value\n",
]


def identical(left, right, path="$"):
    assert type(left) is type(right), f"{path}: {type(left)} vs {type(right)}"
    if isinstance(left, dict):
        assert list(left) == list(right), f"{path}: keys or order"
        assert [type(k) for k in left] == [type(k) for k in right], f"{path}: key types"
        for key in left:
            identical(left[key], right[key], f"{path}.{key!r}")
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), f"{path}: length"
        for index, (a, b) in enumerate(zip(left, right)):
            identical(a, b, f"{path}[{index}]")
    elif isinstance(left, float):
        assert (math.isnan(left) and math.isnan(right)) or repr(left) == repr(right), f"{path}: {left!r} {right!r}"
    else:
        assert left == right, f"{path}: {left!r} vs {right!r}"


def outcome(load, text):
    try:
        return "ok", load(text)
    except Exception as error:  # noqa: BLE001 -- the test compares whatever the loader raised
        return "error", (type(error), str(error))


def forms(text):
    return [text, text.encode("utf-8")]


def test_the_fast_loader_is_pyyamls_safe_loader_over_libyaml_scanning():
    fast = yaml.CSafeLoader
    assert issubclass(fast, yaml.constructor.SafeConstructor) and issubclass(fast, yaml.resolver.Resolver)
    assert fast.yaml_constructors == yaml.SafeLoader.yaml_constructors
    assert fast.yaml_implicit_resolvers == yaml.SafeLoader.yaml_implicit_resolvers
    assert fast.yaml_path_resolvers == yaml.SafeLoader.yaml_path_resolvers


@pytest.mark.parametrize("text", SAMPLES)
def test_documents_and_their_types_are_identical(text):
    for form in forms(text):
        identical(yaml.safe_load(form), fast_safe_load(form))


@pytest.mark.parametrize("text", MALFORMED)
def test_malformed_input_is_refused_with_the_same_error_text(text):
    for form in forms(text):
        slow = outcome(yaml.safe_load, form)
        assert slow[0] == "error", "fixture must be refused by PyYAML"
        assert outcome(fast_safe_load, form) == slow


def _tracked_yaml():
    root = Path(__file__).resolve().parents[2]
    listed = subprocess.run(
        ["git", "ls-files", "*.yaml", "*.yml"], cwd=root, capture_output=True, text=True
    ).stdout.split()
    return [root / name for name in listed if (root / name).is_file() and (root / name).stat().st_size < 4_000_000]


def _extra_yaml():
    return [Path(p) for p in os.environ.get("MERLIN_FAST_YAML_FILES", "").split(os.pathsep) if p]


def test_every_tracked_and_selected_yaml_file_loads_identically():
    files = _tracked_yaml() + _extra_yaml()
    assert files
    for path in files:
        raw = path.read_bytes()
        for form in (raw, raw.decode("utf-8")):
            slow, fast = outcome(yaml.safe_load, form), outcome(fast_safe_load, form)
            assert slow[0] == fast[0], path
            if slow[0] == "ok":
                identical(slow[1], fast[1], str(path))
            else:
                assert slow == fast, path
