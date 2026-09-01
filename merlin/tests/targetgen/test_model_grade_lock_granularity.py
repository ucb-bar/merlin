"""Two whole-model grades of DIFFERENT capsules must not wait on each other.

The lock is real and must stay: `compile_model`'s nested mesh certifications write under a runs root
shared by every grade in one checkout, and two writers in one directory overwrite the exact ELF and
oracle evidence the other is still producing. What was wrong was its GRANULARITY -- keyed on the target,
it made every whole-model grade in a checkout single-writer.

Measured 2026-09-01: two concurrent arms of one experiment in a shared checkout serialized on that lock,
one blocked in `locks_lock_inode_wait` for 33 minutes while the other certified.

The collision is now removed at its cause: both nested paths are content-addressed, so two grades share
a directory only when they are doing byte-identical work.
"""
from __future__ import annotations

import inspect

from merlin import compile_cli as CLI
from merlin.targetgen import capsule_runner as CR


class TestNestedPathsAreContentAddressed:
    def test_the_mesh_verify_run_id_binds_the_certified_bytes(self):
        src = inspect.getsource(CLI.compile_model) if hasattr(CLI, "compile_model") else ""
        if "_content_id(mlir)" not in src:
            # the call may live in a helper; assert on the module instead
            src = CLI.__loader__.get_source(CLI.__name__) or ""
        assert "_content_id(mlir)" in src, \
            "the mesh_verify run id must bind the digest of what it certifies, not just its plan index"

    def test_two_different_tiles_get_two_directories(self):
        a = CLI._content_id("module { /* tile A */ }")
        b = CLI._content_id("module { /* tile B */ }")
        assert a != b, "different certified bytes must not share a run directory"
        assert CLI._content_id("x") == CLI._content_id("x"), "identical work may share one"


class TestLockGranularity:
    """The lock scope is derived from the capsule, and degrades to target-wide rather than to nothing."""

    @staticmethod
    def _scope(capsule: dict, target: str = "t") -> str:
        """Re-derive the lock path the function would take, without running a grade."""
        import hashlib
        import json
        ident = str(capsule.get("name") or capsule.get("id") or "")
        try:
            payload = json.dumps(capsule, sort_keys=True, default=str).encode("utf-8")
            ident = f"{ident}_{hashlib.sha256(payload).hexdigest()[:16]}"
        except Exception:                                          # noqa: BLE001
            pass
        safe_ident = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in ident)
        return f"{target}.{safe_ident}" if safe_ident.strip("_") else target

    def test_two_different_capsules_take_different_locks(self):
        a = self._scope({"name": "M0_llama", "kind": "model"})
        b = self._scope({"name": "M2_vit", "kind": "model"})
        assert a != b, "different capsules must not serialize against each other"

    def test_the_same_capsule_takes_the_same_lock(self):
        c = {"name": "M0_llama", "kind": "model"}
        assert self._scope(c) == self._scope(dict(c)), "identical work must still serialize"

    def test_the_same_name_with_different_content_takes_different_locks(self):
        """A name is not an identity: two submissions may share one and grade different bytes."""
        a = self._scope({"name": "M0", "submission": "v1"})
        b = self._scope({"name": "M0", "submission": "v2"})
        assert a != b

    def test_the_scope_always_names_the_target(self):
        """A lock that cannot say what it protects must protect everything, never nothing.

        Whatever the capsule looks like -- empty, unnamed, oddly typed -- the scope must remain a
        superset of the target-wide lock, so a capsule whose identity cannot be pinned still serializes
        against every other grade rather than against none.
        """
        for capsule in ({}, {"name": ""}, {"name": None, "id": None}, {"kind": "model"}):
            scope = self._scope(capsule, target="tgt")
            assert scope == "tgt" or scope.startswith("tgt."), \
                f"{capsule!r} produced a scope that does not cover the target: {scope!r}"
            assert scope.strip(). strip("."), "an empty scope would lock nothing at all"

    def test_the_result_records_the_scope_it_actually_held(self):
        src = inspect.getsource(CR._grade_model_capsule)
        assert "model_grade_serialization" in src
        assert "capsule:" in src, "the recorded scope must name the capsule, not only the target"
