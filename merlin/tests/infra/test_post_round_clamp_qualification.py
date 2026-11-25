"""Synthetic custody/gate tests; actual RV and model numerics are separate evidence."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from _post_round_clamp_fixture import ASSEMBLER, BASE, ROOT, SUPPLIER, pin

from merlin.llvmlower import post_round_clamp_stage as S
from merlin.llvmlower.host_transform_chain import HostLLVMArtifact as A
from merlin.llvmlower.host_transform_chain import HostLLVMTransformChain


class QualificationTests(unittest.TestCase):
    def fixture(self):
        d = Path(tempfile.mkdtemp(prefix="qualification-", dir=ROOT))
        control = d / "link.ld"
        control.write_text("unit link-control custody only")
        review = d / "review.json"
        review.write_text('{"scope":"unit context custody only; no numerical qualification"}')
        support = d / "support.json"
        support.write_text(
            json.dumps(
                {
                    "schema": "merlin.host_closed_support_object_policy.v1",
                    "objects": [pin(SUPPLIER)],
                    "link_control_inputs": [pin(control)],
                    "source_effect_review": pin(review),
                }
            )
        )
        q = S.ClosedCalleeQualification(BASE, A.capture(ROOT / "contract.json"), A.capture(support), A.capture(review))
        stage = S.make_stage(
            enabled=True,
            contract=BASE,
            contract_document=ROOT / "contract.json",
            assembler=ASSEMBLER,
            semantic_proofs=(ROOT / "proof.json",),
        )
        chain = HostLLVMTransformChain((stage.contract,), (stage,))
        return d, q, chain

    def test_exact_selection(self):
        _, q, chain = self.fixture()
        self.assertEqual(S.qualification_record(chain, q), q.record())

    def test_missing_context_refuses(self):
        _, _, chain = self.fixture()
        with self.assertRaises(TypeError):
            S.qualification_record(chain, None)

    def test_absence_unchanged(self):
        self.assertIsNone(S.qualification_record(None, None))

    def test_extra_context_refuses(self):
        _, q, _ = self.fixture()
        with self.assertRaises(ValueError):
            S.qualification_record(None, q)

    def test_changed_review_refuses_before_audit(self):
        _, q, chain = self.fixture()
        q.context_review.path.write_text("changed")
        with self.assertRaises(ValueError):
            S.qualification_record(chain, q)

    def test_changed_support_supplier_refuses(self):
        d, q, chain = self.fixture()
        other = d / "other.o"
        other.write_bytes(b"original")
        support = json.loads(q.support_policy.path.read_bytes())
        support["objects"].append(pin(other))
        q.support_policy.path.write_text(json.dumps(support))
        q = replace(q, support_policy=A.capture(q.support_policy.path))
        q.record()
        other.write_bytes(b"different")
        with self.assertRaises(ValueError):
            S.qualification_record(chain, q)

    def test_stage_contract_mismatch_refuses(self):
        _, q, chain = self.fixture()
        q = replace(q, contract=replace(BASE, host_isa="portable"))
        with self.assertRaises(ValueError):
            S.qualification_record(chain, q)

    def test_changed_final_nonround_provider_refuses_before_trace(self):
        d, q, _ = self.fixture()
        sup = json.loads(q.support_policy.path.read_bytes())
        other = d / "other.o"
        other.write_bytes(b"different provider")
        recipe = d / "changed.json"
        recipe.write_text(json.dumps({"commands": [{"inputs": [pin(other), *sup["link_control_inputs"]]}]}))
        with patch.object(S, "trace_completed_link") as traced:
            with self.assertRaisesRegex(ValueError, "support closure changed"):
                S.audit_closed_callee_suppliers(
                    BASE,
                    recipe,
                    d / "audit",
                    support_policy_path=q.support_policy.path,
                    support_policy_sha256=q.support_policy.sha256,
                )
            traced.assert_not_called()

    def test_missing_actual_provider_refuses_before_trace(self):
        d, q, _ = self.fixture()
        sup = json.loads(q.support_policy.path.read_bytes())
        recipe = d / "missing.json"
        recipe.write_text(json.dumps({"commands": [{"inputs": sup["link_control_inputs"]}]}))
        with patch.object(S, "trace_completed_link") as traced:
            with self.assertRaises(ValueError):
                S.audit_closed_callee_suppliers(
                    BASE,
                    recipe,
                    d / "audit",
                    support_policy_path=q.support_policy.path,
                    support_policy_sha256=q.support_policy.sha256,
                )
            traced.assert_not_called()

    def test_qualification_brackets_actual_recipe_and_context(self):
        d, q, _ = self.fixture()
        elf = d / "fixture.elf"
        elf.write_bytes(b"unit executable custody only")
        recipe = d / "compilation_recipe.json"
        recipe.write_text("{}")
        calls = []

        def verify(*a, **kw):
            calls.append(("verify", kw["executable"]))
            return {}

        def audit(*a, **kw):
            calls.append(("audit", kw["support_policy_sha256"]))
            return {"unit": "current synthetic audit"}

        with (
            patch.object(S, "verify_completed_recipe", side_effect=verify),
            patch.object(S, "audit_closed_callee_suppliers", side_effect=audit),
        ):
            r = q.audit(
                recipe_path=recipe,
                recipe_sha256=pin(recipe)["sha256"],
                executable=elf,
                executable_sha256=pin(elf)["sha256"],
                destination=d / "qualification",
            )
            self.assertTrue(Path(r["path"]).is_file())
        self.assertEqual([x[0] for x in calls], ["verify", "audit", "verify"])

    def test_changed_context_during_audit_refuses(self):
        d, q, _ = self.fixture()
        elf = d / "fixture.elf"
        elf.write_bytes(b"unit")
        recipe = d / "compilation_recipe.json"
        recipe.write_text("{}")

        def audit(*a, **kw):
            q.context_review.path.write_text("changed during audit")
            return {}

        with (
            patch.object(S, "verify_completed_recipe", return_value={}),
            patch.object(S, "audit_closed_callee_suppliers", side_effect=audit),
        ):
            with self.assertRaises(ValueError):
                q.audit(
                    recipe_path=recipe,
                    recipe_sha256=pin(recipe)["sha256"],
                    executable=elf,
                    executable_sha256=pin(elf)["sha256"],
                    destination=d / "qualification",
                )

    def test_failed_final_provider_audit_publishes_no_qualification(self):
        d, q, _ = self.fixture()
        elf = d / "fixture.elf"
        elf.write_bytes(b"unit")
        recipe = d / "compilation_recipe.json"
        recipe.write_text("{}")
        with (
            patch.object(S, "verify_completed_recipe", return_value={}),
            patch.object(S, "audit_closed_callee_suppliers", side_effect=ValueError("changed provider")),
        ):
            with self.assertRaisesRegex(ValueError, "changed provider"):
                q.audit(
                    recipe_path=recipe,
                    recipe_sha256=pin(recipe)["sha256"],
                    executable=elf,
                    executable_sha256=pin(elf)["sha256"],
                    destination=d / "qualification",
                )
        self.assertFalse((d / "qualification/qualification.json").exists())


if __name__ == "__main__":
    unittest.main()
