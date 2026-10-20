"""The existing sealed worker retains its new direct requirement dependency."""

import os

import pytest
from merlin_experiments.phase1.feedback import compiler_profile_child as CHILD
from merlin_experiments.phase2.contracts import StageGateError
from test_compiler_profile_child import _scope
from test_compiler_profile_child import original as original_fixture

from merlin.common import paths

original = original_fixture


def test_worker_reopens_its_original_requirement_dependency(original, monkeypatch):
    resolver = paths.module_source_path
    module = "merlin.targetgen.shared_pass_context"
    source = original.private / "original_requirement_owner.py"
    raw = resolver(module).read_bytes()
    source.write_bytes(raw)

    def selected_owner(name):
        return source if name == module else resolver(name)

    # Select a disposable source owner; never mutate the installed host library.
    # This exercises source custody only, not a replacement execution authority.
    monkeypatch.setattr(paths, "module_source_path", selected_owner)
    with _scope(original):
        with CHILD.prepare_model_child(
            spec_path=original.spec, environment=dict(os.environ), package_dir=original.candidate
        ) as transport:
            assert CHILD._pin(source) in transport.document["host_sources"]
            transport._verify_original()
            try:
                source.write_bytes(raw + b"\n# changed after original worker selection\n")
                with pytest.raises(StageGateError, match="source or execution owner changed"):
                    transport._verify_original()
            finally:
                source.write_bytes(raw)
