"""Minimal fixed-Qv9 fixture utilities only; original test methods excluded.
Custody/kernel are mocked by explicit setup/activation; no native authority.
"""

import unittest
from pathlib import Path
from unittest.mock import patch

SCALAR = """builtin.module {
  func.func private @observe(i8) -> i8
  func.func @main(%a: i8, %b: i8) -> i8 {
    %x = arith.addi %a, %b : i8
    %y = arith.addi %a, %b : i8
    %z = arith.muli %x, %y : i8
    %r = func.call @observe(%z) : (i8) -> i8
    func.return %r : i8
  }
  func.func @other(%a: i8) -> i8 {
    func.return %a : i8
  }
}"""

LOOP = """builtin.module {
  func.func @main(%a: i8, %b: i8) -> i8 {
    %lb = arith.constant 0 : index
    %ub = arith.constant 3 : index
    %st = arith.constant 1 : index
    %r = scf.for %i = %lb to %ub step %st iter_args(%c = %a) -> (i8) {
      %x = arith.addi %a, %b : i8
      %v = arith.muli %c, %x : i8
      scf.yield %v : i8
    }
    func.return %r : i8
  }
}"""


class WholeBoundary(unittest.TestCase):
    def setUp(self):
        from merlin.transforms.certificates import parse_ir
        from merlin.transforms.pipeline import PipelineOptions, apply_common_pipeline
        from merlin.verify import lean_checker
        from merlin.verify import lean_graph_pipeline_gate as gate
        from merlin.verify import lean_graph_pipeline_gate_projection as projection

        self.gate, self.mapping, self.checker = gate, projection, lean_checker
        self.parse, self.pipeline = parse_ir, apply_common_pipeline
        self.options = PipelineOptions(
            index_bits=64,
            normalize_shapes=False,
            simplify_integers=False,
            reuse_buffers=False,
            eliminate_redundancy=True,
        )
        self.change = self.pipeline(self.parse(SCALAR), self.options, plan_only=True).changes[0]

    def context_fixture(self):
        from merlin.verify import lean_graph_pipeline_gate_provider as provider

        controller, data, _ = provider.selected()
        Context, CheckedQuery, FileRef = controller.Context, controller.CheckedQuery, data.FileRef
        prepared = self.mapping.prepare(self.change)
        request = self.gate.native.request(prepared)
        context = object.__new__(Context)  # No from_review, filesystem or SDK reads.
        context.context_id = "1" * 64
        context.held = False
        context.workspace = Path("/mock/graph-context")
        context.compile_audit_roots = dict(provider._COMPILE_ROOTS)
        context.review = {
            role: FileRef(str(Path(provider.__file__).resolve().parents[3] / path), digest, size)
            for role, (path, digest, size) in provider._REVIEW_PINS.items()
        }
        context.controls = tuple(
            FileRef(str(provider._DIRECTORY / (name + ".py")), digest, size) for name, digest, size in provider._PINS
        )
        context.originals = {name: FileRef(path, digest, size) for name, path, digest, size in request.project_sources}

        def checked(module, query, *, final_root, support_roots):
            return CheckedQuery(
                context.context_id,
                module,
                FileRef("/mock/query.lean", self.mapping.sha(query), len(query)),
                FileRef("/mock/query.olean", "2" * 64, 1),
                final_root,
                support_roots,
                tuple((name, ("propext",)) for name in (final_root, *support_roots)),
                FileRef("/mock/audit.json", "3" * 64, 1),
                FileRef("/mock/types.txt", "4" * 64, 1),
                FileRef("/mock/imports.json", "5" * 64, 1),
                (),
                object(),
            )

        return Context, prepared, context, checked


class ServiceBoundary(unittest.TestCase):
    def setUp(self):
        import contextlib
        import tempfile

        from merlin.targetgen import common_transforms as service
        from merlin.verify import lean_graph_pipeline_gate_service as selection

        self.case = WholeBoundary()
        self.case.setUp()
        self.Context, _, self.context, self.checked = self.case.context_fixture()
        self.service_module, self.selection = service, selection
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.manifest = directory / "host-bundle.yaml"
        self.manifest.write_text("{}\n")
        self.workspace = directory / "workspace"
        self.profile = {
            "sha256": "a" * 64,
            "host_index_bits": 64,
            "common_substrate": "merlin.common_compiler_substrate.v2",
            "common_sources": [],
            "automatic_checker_sources": [],
        }
        self.stack.enter_context(patch.object(service, "verify_tool_profile", return_value=self.profile))
        self.stack.enter_context(
            patch.object(service.BW, "verify_bundle_snapshot", return_value={"content_sha256": "b" * 64})
        )
        self.stack.enter_context(patch.object(service.BW, "snapshot_marker_digest", return_value="c" * 64))
        self.stack.enter_context(
            patch.object(service.BW, "verified_snapshot_scope", side_effect=lambda *a, **k: contextlib.nullcontext())
        )
        self.stack.enter_context(
            patch.object(
                service.CommonTransformService, "_toolchain", side_effect=AssertionError("no legacy SDK fallback")
            )
        )
        self.stack.enter_context(
            patch.object(self.Context, "from_review", side_effect=AssertionError("no service context construction"))
        )
        self.stack.enter_context(patch.object(self.Context, "_guard", side_effect=AssertionError("no runtime reads")))
        self.binding = selection.bind_service_context(
            self.context, compiler_profile_sha256=self.profile["sha256"], timeout_s=120
        )
        self.service = self.make_service(self.binding)
        self.request = {
            "schema": service.SCHEMA,
            "source_ir": SCALAR,
            "options": {
                "normalize_shapes": False,
                "simplify_integers": False,
                "reuse_buffers": False,
                "eliminate_redundancy": True,
            },
        }

    def make_service(self, binding):
        return self.service_module.CommonTransformService(
            workspace=self.workspace,
            bundle_manifest=self.manifest,
            repo=Path(self.case.mapping.__file__).resolve().parents[3],
            graph_binding=binding,
        )

    def activate_mock_methods(self, check=None):
        self.stack.enter_context(patch.object(self.Context, "check_query", side_effect=check or self.checked))
        self.stack.enter_context(patch.object(self.Context, "revalidate", side_effect=lambda value: value))
        self.stack.enter_context(
            patch.object(self.case.gate.native, "check", self.case.gate.native._check_with_context)
        )
        self.stack.enter_context(
            patch.object(self.case.gate.native, "revalidate", self.case.gate.native._revalidate_with_context)
        )
