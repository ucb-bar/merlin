"""Minimal-install CLI/resource behavior without requiring optional validators."""

from __future__ import annotations

import importlib.resources
import json
import os
import subprocess
import sys

import pytest

from merlin.common import paths
from merlin.targetgen import corpora


def test_corpus_registry_uses_bundled_resource_without_checkout(tmp_path, monkeypatch):
    bundled = tmp_path / "installed/merlin"
    registry = bundled / "_data/contract/corpora.yaml"
    registry.parent.mkdir(parents=True)
    registry.write_text("kernel_corpora:\n  example:\n    layout: standalone_benchmarks\n")
    monkeypatch.setattr(paths, "merlin_dir", lambda: tmp_path / "absent-checkout")
    monkeypatch.setattr(corpora, "merlin_dir", lambda: tmp_path / "absent-checkout")
    original = importlib.resources.files
    monkeypatch.setattr(importlib.resources, "files", lambda name: bundled if name == "merlin" else original(name))
    assert corpora.kernel_corpora() == {"example": {"layout": "standalone_benchmarks"}}


@pytest.mark.parametrize("contents", [None, "- not-a-mapping\n"])
def test_unavailable_or_malformed_registry_never_becomes_empty(tmp_path, monkeypatch, contents):
    registry = tmp_path / "corpora.yaml"
    if contents is not None:
        registry.write_text(contents)
    monkeypatch.setattr(corpora, "data_path", lambda *parts: registry, raising=False)
    with pytest.raises(FileNotFoundError if contents is None else ValueError):
        corpora.kernel_corpora()


@pytest.mark.parametrize("arguments", [["--help"], ["publish", "--help"]])
def test_publish_help_without_jsonschema(tmp_path, arguments):
    result = _without_jsonschema(tmp_path, f"publish.main({arguments!r})")
    assert result.returncode == 0, result.stderr
    assert "usage: merlin-target-publish" in result.stdout
    assert "Traceback" not in result.stderr


def test_publish_validation_requires_extra_without_suppressing_validation(tmp_path):
    result = _without_jsonschema(
        tmp_path,
        """
try:
    publish._rewrite_manifest(None, layout_version='fixture', hoisted_tree=False)
except publish.PublishError as error:
    assert "pip install 'merlin[targetgen]'" in str(error)
    print('missing-validator-refused')
else:
    raise AssertionError('validation did not require its dependency')
""",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "missing-validator-refused"


def _without_jsonschema(tmp_path, body):
    script = (
        """
import importlib.abc, json, sys
sys.path[:0] = json.loads(sys.argv[1])
class MissingValidator(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'jsonschema' or fullname.startswith('jsonschema.'):
            raise ModuleNotFoundError('optional validator absent', name='jsonschema')
sys.meta_path.insert(0, MissingValidator())
from merlin.targetgen import publish
"""
        + body
    )
    environment = {key: value for key, value in os.environ.items() if not key.startswith("MERLIN_")}
    return subprocess.run(
        [sys.executable, "-I", "-c", script, json.dumps(list(map(str, paths.python_import_roots())))],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
    )
