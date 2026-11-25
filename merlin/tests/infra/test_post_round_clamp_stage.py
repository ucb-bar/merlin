import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _post_round_clamp_fixture import ASSEMBLER, ROOT
from test_post_round_clamp import BASE

from merlin.llvmlower import post_round_clamp_stage as stage


class StageTests(unittest.TestCase):
    def test_absent_unchanged(self):
        self.assertIsNone(stage.make_stage())
        self.assertEqual(stage.supplier_link_flags(), ())

    def test_enable_requires_contract(self):
        with self.assertRaises(ValueError):
            stage.make_stage(enabled=True)

    def test_contract_requires_selection(self):
        with self.assertRaises(ValueError):
            stage.make_stage(contract=BASE)

    def test_boolean_exact(self):
        with self.assertRaises(TypeError):
            stage.make_stage(enabled=1)

    def test_proofs_required(self):
        with self.assertRaises(ValueError):
            stage.make_stage(enabled=True, contract=BASE)

    def test_real_typed_stage(self):
        s = stage.make_stage(
            enabled=True,
            contract=BASE,
            contract_document=ROOT / "contract.json",
            assembler=ASSEMBLER,
            semantic_proofs=(ROOT / "proof.json",),
        )
        s.validate()
        self.assertEqual(s.contract.name, "post_round_clamp_rne")

    def test_actual_link_trace_required(self):
        import json

        p = Path(BASE.supplier_recipe_path)
        r = json.loads(p.read_bytes())["commands"][-1]
        with self.assertRaises(ValueError):
            stage.verify_final_supplier(
                BASE, actual_recipe_path=p, link_argv=r["argv"], link_inputs=r["inputs"], link_stderr=""
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
