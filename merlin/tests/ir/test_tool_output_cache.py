"""Exact tool reuse retains every original application publication and replay."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from merlin.compile.semantic_application import CompiledCRegion, compile_semantic_application
from merlin.semantic_compiler import materialization
from merlin.semantic_compiler.materialization import (
    cached_tool_output, canonical_request_record, materialization_session, materialization_key, materialization_region,
)
from merlin.tests.ir.test_semantic_regions import graph, plan_for


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def tool(invoke, *, data=b"input", identity=None, namespace="synthetic.compiler"):
    return cached_tool_output(
        namespace=namespace, input_bytes=data,
        compilation_identity=identity or {"executable_sha256": "tool-1", "argv": ["--compile"]},
        invoke=invoke,
    )


def test_exact_inputs_identity_namespace_and_empty_output():
    calls = []

    def invoke():
        calls.append(True)
        return b""

    with materialization_session(enabled=True) as stats:
        assert tool(invoke) == tool(invoke) == b""
        assert len(calls) == 1
        tool(invoke, data=b"input\0changed")
        tool(invoke, namespace="other.compiler")
        for field in ("executable_sha256", "argv", "environment", "limits", "options", "provider_revision"):
            tool(invoke, identity={"executable_sha256": "tool-1", "argv": ["--compile"], field: "changed"})
    assert stats["hits"] == 1
    assert stats["misses"] == len(calls) == 9
    assert stats["calls"] == 10
    assert stats["elapsed_seconds"] >= 0


def test_sessions_are_fresh_nested_and_disabled_outside_application():
    calls = []

    def invoke():
        calls.append(True)
        return b"output"

    tool(invoke)
    tool(invoke)
    with materialization_session(enabled=False) as disabled:
        tool(invoke)
        tool(invoke)
    assert disabled["hits"] == 0 and disabled["misses"] == 2
    with materialization_session(enabled=True) as outer:
        tool(invoke)
        with materialization_session(enabled=True) as inner:
            tool(invoke)
            tool(invoke)
        tool(invoke)
    assert outer["hits"] == inner["hits"] == 1
    with materialization_session(enabled=True) as fresh:
        tool(invoke)
    assert fresh["hits"] == 0
    assert len(calls) == 7


def test_failures_and_nonbytes_results_are_not_cached():
    with materialization_session(enabled=True) as stats:
        with pytest.raises(RuntimeError, match="failed"):
            tool(lambda: (_ for _ in ()).throw(RuntimeError("failed")))
        with pytest.raises(TypeError, match="output"):
            tool(lambda: "not bytes")
        assert tool(lambda: b"valid") == b"valid"
        assert tool(lambda: pytest.fail("hit must not invoke")) == b"valid"
    assert stats["misses"] == 3 and stats["hits"] == 1


def test_tool_hash_collisions_compare_complete_identity(monkeypatch):
    monkeypatch.setattr(materialization, "sha256", lambda _: SimpleNamespace(hexdigest=lambda: "collision"))
    with materialization_session(enabled=True) as stats:
        assert tool(lambda: b"first", data=b"first") == b"first"
        assert tool(lambda: b"second", data=b"second") == b"second"
        assert tool(lambda: pytest.fail("missing first"), data=b"first") == b"first"
        assert tool(lambda: pytest.fail("missing second"), data=b"second") == b"second"
    assert stats["misses"] == stats["hits"] == 2


class SyntheticAdapter:
    """A deterministic test compiler with source-bound per-region exports."""

    supports_materialization_cache = True

    def materialization_identity(self, region, alternative):
        return {
            "target_choice": {"placement": alternative.placement, "implementation": "synthetic"},
            "compilation_identity": {"provider_revision": "synthetic-1"},
            "source_identity_policy": "provenance",
        }

    def __init__(self):
        self.tool_calls = self.compiles = self.replays = 0

    def plan(self, source):
        plan = plan_for(source, ((0,), (1,), (2,), (3,)))
        return replace(plan, selected=tuple(replace(row, placement="host") for row in plan.selected))

    def compile_region(self, snapshot, region, alternative, name, segments, target_source, limits):
        self.compiles += 1

        def invoke():
            self.tool_calls += 1
            return b"return 0;"

        body = tool(
            invoke, data=getattr(self, "exact_tool_input", None) or encoded(
                canonical_request_record(region, source_identity_policy="provenance")),
            identity={"executable_sha256": "synthetic-1", "limits": limits},
        )
        return self._export(region, name, body)

    def export_region(self, region, alternative, name, segments, target_source):
        self.replays += 1
        expected = self._export(region, name, b"return 0;")
        for filename, data in expected.files.items():
            assert (segments.parent / "c-package/regions" / name / filename).read_bytes() == data
        return expected

    def _export(self, region, name, body):
        declaration = f"int {name}(void *,const void *const *,void *const *);\n"
        implementation = (
            f"/* {region.digest()} */\n"
            f"int {name}(void *c,const void *const *i,void *const *o){{{body.decode()}}}\n"
        )
        return CompiledCRegion(
            name, name + ".h", tuple(key for key, _ in region.input_storages), region.outputs,
            {name + ".h": declaration.encode(), name + ".c": implementation.encode()}, (), (),
            {"request_digest": region.digest(), "original_outputs": list(region.outputs)},
        )


@pytest.mark.parametrize("supported", [True, False])
@pytest.mark.parametrize("constant_tool_input", [False, True])
@pytest.mark.parametrize("identity_hook", [False, True])
def test_application_cached_and_fresh_publications_match_every_byte(tmp_path, supported, constant_tool_input, identity_hook):
    source = graph()
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    (snapshot_root / "manifest.json").write_bytes(b"synthetic snapshot")
    snapshot = SimpleNamespace(root=snapshot_root)
    results = []
    for enabled in (False, True):
        adapter = SyntheticAdapter()
        adapter.supports_materialization_cache = supported
        if not identity_hook:
            adapter.materialization_identity = None
        if constant_tool_input:
            # Different semantics must partition even coincident tool inputs.
            adapter.exact_tool_input = b"same exact tool input for all regions"
        stats = {}
        output = tmp_path / ("cached" if enabled else "fresh")
        compile_semantic_application(
            source, adapter=adapter, snapshot=snapshot, target_source=tmp_path,
            destination=output, limits={"max_candidates": 5}, name="test_application", capacity=4096,
            allow_host_only=True, reuse_materializations=enabled, materialization_stats=stats,
        )
        assert adapter.compiles == adapter.replays == stats["region_calls"] == stats["calls"] == 4
        assert adapter.tool_calls == stats["misses"] == (3 if enabled and supported and identity_hook else 4)
        assert stats["hits"] == (1 if enabled and supported and identity_hook else 0)
        results.append({path.relative_to(output).as_posix(): path.read_bytes()
                        for path in output.rglob("*") if path.is_file()})
    assert results[0] == results[1]


def test_cli_exposes_fresh_application_baseline():
    from merlin.targetgen.cli import build_parser

    arguments = [
        "native-compile", "--engine", "merlin_native", "--support", "synthetic", "--snapshot", "snapshot",
        "--application", "--compile-inputs", "inputs", "--abi", "auto", "--target-source", "target",
        "--mode", "diagnostic", "--out", "output",
    ]
    parser = build_parser()
    defaults = parser.parse_args(arguments)
    assert defaults.no_materialization_cache is False
    assert defaults.materialization_cache is False
    assert defaults.region_materialization_cache is False
    assert parser.parse_args([*arguments, "--materialization-cache"]).materialization_cache is True
    assert parser.parse_args([*arguments, "--region-materialization-cache"]).region_materialization_cache is True
    assert parser.parse_args([*arguments, "--no-materialization-cache"]).no_materialization_cache is True


@pytest.mark.parametrize("collide", [False, True])
@pytest.mark.parametrize("identity_hook", [False, True])
def test_region_materialization_reduces_real_compile_calls_and_retains_publications(tmp_path, monkeypatch, collide, identity_hook):
    identities, canonical_sources = [], []

    class RebindingAdapter(SyntheticAdapter):
        def materialization_identity(self, region, alternative):
            identities.append(region.digest())
            return super().materialization_identity(region, alternative)

        def supports_region_materialization(self, region, alternative):
            return True

        def materialize_region(self, snapshot, region, alternative, name, segments, target_source, limits):
            canonical_sources.append(region)
            checked = self.compile_region(snapshot, region, alternative, name, segments, target_source, limits)
            return checked.files[name + ".c"].split(b"{")[1].rsplit(b"}", 1)[0]

        def bind_materialization(self, materialized, region, alternative, name, segments, target_source):
            assert materialized == b"return 0;"
            return self._export(region, name, materialized)

    if collide:
        monkeypatch.setattr(materialization, "sha256", lambda _: SimpleNamespace(hexdigest=lambda: "collision"))
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    (snapshot_root / "manifest.json").write_bytes(b"snapshot")
    publications = []
    for enabled in (False, True):
        adapter, stats = RebindingAdapter(), {}
        identities.clear()
        canonical_sources.clear()
        if not identity_hook:
            adapter.materialization_identity = None
        output = tmp_path / str(enabled)
        compile_semantic_application(
            graph(), adapter=adapter, snapshot=SimpleNamespace(root=snapshot_root), target_source=tmp_path,
            destination=output, limits={"max_candidates": 5}, name="test_application", capacity=4096,
            allow_host_only=True, reuse_region_materializations=enabled, materialization_stats=stats)
        shared = enabled and identity_hook
        assert adapter.compiles == stats["region_compiles"] == (3 if shared else 4)
        assert stats["region_calls"] == adapter.replays == 4
        assert stats["region_hits"] == (1 if shared else 0)
        assert stats["region_materializations"] == (3 if shared else 4 if enabled else 0)
        # Compilation and automatic replay each query the contract exactly
        # once per original region. The same contract normalizes the source.
        assert len(identities) == (8 if identity_hook else 0)
        if enabled:
            assert all(bool(source.source_identity) == (not identity_hook) for source in canonical_sources)
        publications.append({path.relative_to(output).as_posix(): path.read_bytes()
                             for path in output.rglob("*") if path.is_file()})
    assert all(publication == publications[0] for publication in publications)


def test_target_can_decline_checked_materialization_without_repeating_probe(tmp_path):
    class DecliningAdapter(SyntheticAdapter):
        def supports_region_materialization(self, region, alternative):
            return True

        def materialize_region(self, *args):
            self.compile_region(*args)
            return None

        def bind_materialization(self, *args):
            pytest.fail("a declined canonical construction must use original compilation")

    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    (snapshot_root / "manifest.json").write_bytes(b"snapshot")
    adapter, stats = DecliningAdapter(), {}
    compile_semantic_application(
        graph(), adapter=adapter, snapshot=SimpleNamespace(root=snapshot_root), target_source=tmp_path,
        destination=tmp_path / "output", limits={}, name="test_application", capacity=4096,
        allow_host_only=True, reuse_region_materializations=True, materialization_stats=stats)
    assert adapter.compiles == stats["region_compiles"] == 7  # Three canonical probes, four original requests.
    assert adapter.replays == stats["region_calls"] == 4
    assert stats["region_materializations"] == 3 and stats["region_hits"] == 0


def test_canonical_region_partition_semantics_provenance_and_collisions(monkeypatch):
    from merlin.tests.ir.test_materialization import request
    from merlin.semantic_compiler.codegen import CodegenOptions

    def region_key(source, **overrides):
        kwargs = dict(target_choice={"engine": "synthetic"}, options=CodegenOptions(),
                      compilation_identity={"revision": "test-1"}, source_identity_policy="provenance")
        kwargs.update(overrides)
        return materialization_key(source, **kwargs)

    # Identical raw tool inputs must never bypass distinct semantic regions.
    source = request()
    changed_node = replace(source.nodes[-1], attrs=(("numeric_semantics", "changed"),))
    changed = replace(source, nodes=(*source.nodes[:-1], changed_node))
    monkeypatch.setattr(materialization, "sha256", lambda _: SimpleNamespace(hexdigest=lambda: "collision"))
    with materialization_session(enabled=True) as stats:
        for region, output in ((source, b"first"), (changed, b"second")):
            with materialization_region(region_key(region)):
                assert tool(lambda output=output: output) == output
        # Alpha names, prov.* and explicitly approved source provenance share.
        for region, output in ((request("b", source_identity="other"), b"first"), (changed, b"second")):
            with materialization_region(region_key(region)):
                assert tool(lambda: pytest.fail("canonical region missed")) == output
        for overrides in ({"source_identity_policy": "preserve"},
                          {"compilation_identity": {"revision": "test-2"}},
                          {"target_choice": {"engine": "other"}},
                          {"options": CodegenOptions(io_memory_bytes=4096)}):
            with materialization_region(region_key(source, **overrides)):
                assert tool(lambda: b"distinct") == b"distinct"
    assert stats["misses"] == 6 and stats["hits"] == 2


def test_region_scope_resets_after_exception():
    from merlin.tests.ir.test_materialization import key, request

    with materialization_session(enabled=True) as stats:
        assert tool(lambda: b"unscoped") == b"unscoped"
        with pytest.raises(RuntimeError):
            with materialization_region(key(request())):
                assert tool(lambda: b"scoped") == b"scoped"
                raise RuntimeError("leave scope")
        assert tool(lambda: pytest.fail("scope leaked")) == b"unscoped"
    assert stats["misses"] == 2 and stats["hits"] == 1


class SubprocessAdapter(SyntheticAdapter):
    """Count real child processes during compilation and source replay."""

    def __init__(self, child_log):
        super().__init__()
        self.child_log = child_log

    def _run_tool(self):
        import os
        import subprocess
        import sys
        program = ("from pathlib import Path;import sys;"
                   "p=Path(sys.argv[1]);"
                   "f=p.open('ab');f.write(b'child\\n');f.close();"
                   "sys.stdout.buffer.write(b'return 0;')")
        command = [sys.executable, "-c", program, str(self.child_log)]

        def invoke():
            result = subprocess.run(command, input=b"same exact input", capture_output=True, check=True)
            self.tool_calls += 1
            return result.stdout

        return cached_tool_output(
            namespace="synthetic.child.compiler", input_bytes=b"same exact input",
            compilation_identity={"command": command, "environment": dict(os.environ)}, invoke=invoke,
        )

    def compile_region(self, snapshot, region, alternative, name, segments, target_source, limits):
        self.compiles += 1
        return self._export(region, name, self._run_tool())

    def export_region(self, region, alternative, name, segments, target_source):
        assert self._run_tool() == b"return 0;"
        return super().export_region(region, alternative, name, segments, target_source)


@pytest.mark.parametrize("supported", [False, True])
def test_cli_compile_report_includes_actual_source_replay_tool_processes(tmp_path, monkeypatch, supported):
    from merlin.targetgen import native_application
    from merlin.compile.semantic_application import verify_semantic_application

    source = graph()
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    (snapshot_root / "manifest.json").write_bytes(b"snapshot")
    snapshot = SimpleNamespace(root=snapshot_root, profile=SimpleNamespace(target_identity=source.target_identity))
    results = []
    original_compile = compile_semantic_application

    def host_compile(*args, **kwargs):
        # The complete report/compiler/verification path uses a host synthetic
        # provider so no target binaries are needed for this process-count gate.
        return original_compile(*args, **kwargs, allow_host_only=True)

    monkeypatch.setattr(native_application, "compile_semantic_application", host_compile)
    monkeypatch.setattr(native_application, "preparation_counts", lambda _: {})
    for enabled in (False, True):
        child_log = tmp_path / ("cached-children" if enabled else "fresh-children")
        adapter = SubprocessAdapter(child_log)
        adapter.supports_materialization_cache = supported
        staged = SimpleNamespace(program_sha256="source")
        monkeypatch.setattr(native_application, "_source", lambda *args: (staged, source, adapter, None, None, None))
        output = tmp_path / ("cached" if enabled else "fresh")
        args = SimpleNamespace(abi="auto", application_memory_bytes=4096, out=str(output),
                               status_file=None, target_source=str(tmp_path), application_name="process_test",
                               materialization_cache=enabled, no_materialization_cache=False,
                               engine="merlin_native", support="synthetic")
        report = native_application.compile_application_report(
            args, binding=object(), snapshot=snapshot, limits={"candidates": 5},
            write_status=lambda *args: None,
        )
        expected = 7 if enabled and supported else 8
        assert len(child_log.read_bytes().splitlines()) == adapter.tool_calls == expected
        stats = report["materialization_cache"]
        assert stats["region_calls"] == adapter.compiles == 4
        assert stats["region_replays"] == adapter.replays == 4
        assert stats["calls"] == 8 and stats["misses"] == expected
        assert stats["hits"] == 8 - expected
        assert stats["compilation"]["calls"] == stats["verification"]["calls"] == 4
        assert stats["verification"]["hits"] == 0
        assert stats["verification"]["misses"] == 4
        results.append({path.relative_to(output).as_posix(): path.read_bytes()
                        for path in output.rglob("*") if path.is_file()})
        # A later verification owns a fresh cache and reauthenticates all four
        # exports. It never borrows a previous compile invocation's storage.
        later_stats = {}
        verify_semantic_application(
            source, adapter=adapter, snapshot=snapshot, target_source=tmp_path, artifact=output,
            allow_host_only=True, reuse_materializations=enabled, materialization_stats=later_stats,
        )
        later_expected = 4
        assert len(child_log.read_bytes().splitlines()) == expected + later_expected
        assert later_stats["calls"] == later_stats["region_replays"] == 4
        assert later_stats["region_calls"] == 0 and later_stats["misses"] == later_expected
    assert results[0] == results[1]


def test_verify_cli_exposes_fresh_application_replay_baseline():
    from merlin.targetgen.cli import build_parser

    arguments = ["native-verify", "--engine", "merlin_native", "--support", "synthetic",
                 "--snapshot", "snapshot", "--application", "--compile-inputs", "inputs",
                 "--mode", "diagnostic", "--artifact", "artifact", "--out", "status"]
    parser = build_parser()
    defaults = parser.parse_args(arguments)
    assert defaults.no_materialization_cache is False
    assert parser.parse_args([*arguments, "--no-materialization-cache"]).no_materialization_cache is True


def test_region_and_tool_reuse_default_off_and_enable_independently(tmp_path):
    class DefaultAdapter(SyntheticAdapter):
        def supports_region_materialization(self, region, alternative):
            return True

        def materialize_region(self, *arguments):
            self.compile_region(*arguments)
            return b"return 0;"

        def bind_materialization(self, materialized, region, alternative, name, segments, target_source):
            return self._export(region, name, materialized)

    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    (snapshot_root / "manifest.json").write_bytes(b"snapshot")
    publications = []
    for label, options, region_reuse in (
        ("default", {}, False),
        ("tools", {"reuse_materializations": True}, False),
        ("regions", {"reuse_region_materializations": True}, True),
        ("both", {"reuse_materializations": True, "reuse_region_materializations": True}, True),
    ):
        adapter, stats = DefaultAdapter(), {}
        output = tmp_path / label
        compile_semantic_application(
            graph(), adapter=adapter, snapshot=SimpleNamespace(root=snapshot_root), target_source=tmp_path,
            destination=output, limits={}, name="test_application", capacity=4096, allow_host_only=True,
            **options, materialization_stats=stats)
        assert adapter.compiles == stats["region_compiles"] == (3 if region_reuse else 4)
        assert stats["region_materializations"] == (3 if region_reuse else 0)
        assert stats["region_hits"] == (1 if region_reuse else 0)
        publications.append({path.relative_to(output).as_posix(): path.read_bytes()
                             for path in output.rglob("*") if path.is_file()})
    assert all(publication == publications[0] for publication in publications)


@pytest.mark.parametrize("reuse", [False, True])
def test_verification_cannot_join_or_populate_an_enabled_compile_cache(tmp_path, monkeypatch, reuse):
    from merlin.compile import semantic_application

    calls = []

    def invoke():
        calls.append(True)
        return b"checked"

    def verify(**kwargs):
        assert tool(invoke) == tool(invoke) == b"checked"
        return {"status": "verified"}

    monkeypatch.setattr(semantic_application, "_verify_semantic_application", verify)
    stats = {}
    with materialization_session(enabled=True) as compile_stats:
        assert tool(invoke) == b"checked"
        semantic_application.verify_semantic_application(
            object(), adapter=SimpleNamespace(supports_materialization_cache=True), snapshot=None,
            target_source=tmp_path, artifact=tmp_path, reuse_materializations=reuse,
            materialization_stats=stats,
        )
        assert tool(lambda: pytest.fail("verification replaced compile cache")) == b"checked"
    assert len(calls) == 3
    assert compile_stats["calls"] == 2 and compile_stats["hits"] == 1
    assert stats["calls"] == stats["misses"] == 2 and stats["hits"] == 0


@pytest.mark.parametrize("disabled", [False, True])
def test_cli_cache_flags_are_independent_and_global_disable_wins(tmp_path, monkeypatch, disabled):
    from merlin.targetgen import native_application

    source, staged = graph(), SimpleNamespace(program_sha256="source")
    monkeypatch.setattr(native_application, "_source", lambda *args: (staged, source, object(), None, None, None))
    monkeypatch.setattr(native_application, "preparation_counts", lambda _: {})
    selected = []

    def compile_source(source, **kwargs):
        kwargs["destination"].mkdir()
        selected.append((kwargs["reuse_materializations"], kwargs["reuse_region_materializations"]))
        return {"segments": [], "regions": [], "application_interface": {}}

    monkeypatch.setattr(native_application, "compile_semantic_application", compile_source)
    for index, (tools, regions) in enumerate(((False, False), (True, False), (False, True), (True, True))):
        args = SimpleNamespace(
            abi="auto", application_memory_bytes=4096, out=str(tmp_path / str(index)), status_file=None,
            target_source=str(tmp_path), application_name="flags", engine="merlin_native", support="synthetic",
            materialization_cache=tools, region_materialization_cache=regions, no_materialization_cache=disabled,
        )
        native_application.compile_application_report(
            args, binding=object(), snapshot=SimpleNamespace(profile=SimpleNamespace(target_identity="test")),
            limits={}, write_status=lambda *args: None,
        )
    assert selected == ([(False, False)] * 4 if disabled else
                        [(False, False), (True, False), (False, True), (True, True)])


@pytest.mark.parametrize("enabled", [False, True])
def test_deterministic_receipts_are_explicit_target_scope(tmp_path, monkeypatch, enabled):
    from contextlib import contextmanager
    from merlin.compile import semantic_application

    active = []

    @contextmanager
    def publication_options(*, deterministic_receipts=False):
        active.append(deterministic_receipts)
        try:
            yield
        finally:
            active.pop()

    def compile_source(**kwargs):
        assert active == [enabled]
        assert "deterministic_receipts" not in kwargs
        return "published"

    monkeypatch.setattr(semantic_application, "_compile_semantic_application", compile_source)
    assert semantic_application.compile_semantic_application(
        object(), adapter=SimpleNamespace(publication_options=publication_options), snapshot=None,
        target_source=tmp_path, destination=tmp_path, limits={}, name="scope", capacity=1,
        deterministic_receipts=enabled) == "published"
    assert active == []


def test_deterministic_receipts_require_target_capability(tmp_path):
    from merlin.compile.semantic_application import compile_semantic_application

    with pytest.raises(ValueError, match="target does not support"):
        compile_semantic_application(
            object(), adapter=object(), snapshot=None, target_source=tmp_path, destination=tmp_path,
            limits={}, name="scope", capacity=1, deterministic_receipts=True)


def test_publication_scope_failure_preserves_error_and_clears_stats(tmp_path):
    from contextlib import contextmanager
    from merlin.compile.semantic_application import compile_semantic_application

    @contextmanager
    def publication_options(**kwargs):
        raise ValueError("publication refused")
        yield

    stats = {"stale": True}
    with pytest.raises(ValueError, match="publication refused"):
        compile_semantic_application(
            object(), adapter=SimpleNamespace(publication_options=publication_options), snapshot=None,
            target_source=tmp_path, destination=tmp_path, limits={}, name="scope", capacity=1,
            deterministic_receipts=True, materialization_stats=stats)
    assert stats == {}
