"""Six NEW SDK-free handoff tests; real service/gate/pipeline, explicit mocks.

Fixed Q fixture setup utilities are loaded by the isolated driver. None of Q's
original46 tests or Prepare's service matrix are run. Positive kernel/custody
completions are mocked and the compiler callback only records a string.
"""

import copy
import json
import unittest
from dataclasses import replace
from unittest.mock import patch

import fixed_q_fixture as Q

from merlin.targetgen.graph_consumer_bridge import ConsumerRefused, GraphConsumerBridge


class ConsumerTests(unittest.TestCase):
    def fixture(self):
        case = Q.ServiceBoundary()
        case.setUp()
        self.addCleanup(case.doCleanups)
        return case

    def bridge(self, case):
        calls = []
        bridge = GraphConsumerBridge(
            service=case.service,
            binding=case.binding,
            compiler=lambda text: calls.append(text) or "mock_compiler_recorded",
        )
        return bridge, calls

    def test_01_supported_cse_delivers_exact_final_once(self):
        case = self.fixture()
        case.activate_mock_methods()
        bridge, calls = self.bridge(case)
        source = case.request["source_ir"]
        result = bridge.consume(case.request)
        self.assertEqual(result, "mock_compiler_recorded")
        self.assertEqual(len(calls), 1)
        recorded = next(row for row in reversed(case.selection._DELIVERIES.values()) if row[2] is case.binding)
        document = recorded[0]
        self.assertEqual(calls[0], document["transformed_ir"])
        self.assertNotEqual(calls[0], source)
        self.assertEqual(recorded[3], source)
        self.assertEqual(recorded[4][-1].after_ir, calls[0])
        self.assertIs(document, recorded[0])
        self.assertEqual(document["compiler_pass_chain"], "not_established")
        # Observation only: this is not a real compiler output or proof receipt.
        globals()["WORKED_OBSERVATION"] = {
            "original_ir": source,
            "final_ir": calls[0],
            "service_document": document,
            "callback_calls": len(calls),
            "same_live_document": True,
            "mocks": [
                "original profile/custody",
                "Context.check_query",
                "Context.revalidate",
                "native dispatch temporarily mapped to inactive adapter with mocked Context",
            ],
            "real_compiler_or_kernel_executed": False,
            "native_authorization": False,
        }

    def test_02_service_and_second_boundary_refusal_call_compiler_zero(self):
        case = self.fixture()
        bridge, calls = self.bridge(case)
        with self.assertRaisesRegex(ValueError, "held"):
            bridge.consume(case.request)
        self.assertEqual(calls, [])
        checked_calls = []
        provider = __import__("merlin.verify.lean_graph_pipeline_gate_provider", fromlist=["selected"])

        def checked(*args, **kwargs):
            checked_calls.append(args[0])
            if len(checked_calls) == 2:
                raise provider.selected()[0].ProofRefused("mocked second-boundary refusal")
            return case.checked(*args, **kwargs)

        case.activate_mock_methods(checked)
        source = Q.LOOP.replace(
            "%x = arith.addi %a, %b : i8",
            """%x = arith.addi %a, %b : i8
      %duplicate = arith.addi %a, %b : i8
      %dead = arith.muli %x, %duplicate : i8""",
        )
        request = {
            **case.request,
            "source_ir": source,
            "options": {**case.request["options"], "hoist_invariants": True},
        }
        with self.assertRaisesRegex(ValueError, "second-boundary"):
            bridge.consume(request)
        self.assertEqual(len(checked_calls), 2)
        self.assertEqual(calls, [])

    def test_03_copied_or_altered_service_output_call_compiler_zero(self):
        case = self.fixture()
        case.activate_mock_methods()
        bridge, calls = self.bridge(case)
        owner = type(case.service)
        apply = owner.apply
        for mode in ("copy", "alter-final", "alter-description"):

            def intercepted(service, request, **kwargs):
                document = apply(service, request, **kwargs)
                if mode == "copy":
                    return json.loads(json.dumps(document))
                if mode == "alter-final":
                    document["transformed_ir"] += "\n"
                else:
                    document["status"] = "spoofed"
                return document

            with self.subTest(mode=mode), patch.object(owner, "apply", intercepted):
                with self.assertRaises(ConsumerRefused):
                    bridge.consume(case.request)
            self.assertEqual(calls, [])

    def test_04_original_or_final_change_during_handoff_call_compiler_zero(self):
        case = self.fixture()
        case.activate_mock_methods()
        bridge, calls = self.bridge(case)
        owner = type(case.service)
        eligible = owner.compiler_eligible
        for mode in ("original", "final"):
            request = copy.deepcopy(case.request)

            def intercepted(service, document, *, source_ir):
                accepted = eligible(service, document, source_ir=source_ir)
                self.assertIs(accepted, True)
                if mode == "original":
                    request["source_ir"] += "\n"
                else:
                    document["transformed_ir"] += "\n"
                return accepted

            with self.subTest(mode=mode), patch.object(owner, "compiler_eligible", intercepted):
                with self.assertRaisesRegex(ConsumerRefused, "during eligibility"):
                    bridge.consume(request)
            self.assertEqual(calls, [])

    def test_05_held_foreign_or_recreated_binding_call_compiler_zero(self):
        case = self.fixture()
        case.activate_mock_methods()
        bridge, calls = self.bridge(case)
        case.context.held = True
        with self.assertRaises(ValueError):
            bridge.consume(case.request)
        case.context.held = False
        foreign = case.selection.bind_service_context(
            case.context, compiler_profile_sha256=case.profile["sha256"], timeout_s=120
        )
        case.service.graph_binding = foreign
        with self.assertRaisesRegex(ConsumerRefused, "binding changed"):
            bridge.consume(case.request)
        case.service.graph_binding = case.binding
        for selected in (replace(case.binding), case.binding.describe(), None):
            with self.subTest(binding=type(selected).__name__), self.assertRaises(ValueError):
                GraphConsumerBridge(service=case.service, binding=selected, compiler=calls.append)
        self.assertEqual(calls, [])

    def test_06_plan_legacy_unsupported_or_request_capability_call_compiler_zero(self):
        case = self.fixture()
        case.activate_mock_methods()
        bridge, calls = self.bridge(case)
        for options in ({"plan_only": True}, {"normalize_shapes": True}, {"tile_iterations": [2]}):
            request = {**case.request, "options": {**case.request["options"], **options}}
            with self.subTest(options=options), self.assertRaises(ValueError):
                bridge.consume(request)
        for field in ("graph_binding", "context", "compiler", "callback"):
            with self.subTest(request_field=field), self.assertRaises(ConsumerRefused):
                bridge.consume({**case.request, field: "request-selected-capability"})
        with self.assertRaises(ValueError):
            bridge.consume({**case.request, "schema": "merlin.common_check_request.v1"})
        # An actual plan_only result is DATA, not the SAME service output dict.
        planned = case.case.pipeline(case.case.parse(Q.SCALAR), case.case.options, plan_only=True)
        with patch.object(type(case.service), "apply", return_value=planned):
            with self.assertRaises(ConsumerRefused):
                bridge.consume(case.request)
        legacy = object.__new__(case.case.checker.LeanTransformReceipt)
        with patch.object(case.case.checker, "verify_pipeline_change", return_value=legacy):
            with self.assertRaisesRegex(ValueError, "not established"):
                bridge.consume(case.request)
        self.assertEqual(calls, [])
