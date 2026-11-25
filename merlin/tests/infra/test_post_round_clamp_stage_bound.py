import dataclasses
import unittest

from _post_round_clamp_fixture import ASSEMBLER, ROOT
from test_post_round_clamp import BASE

from merlin.llvmlower import post_round_clamp_stage as S


class BoundStageTests(unittest.TestCase):
    def test_explicit_complete_stage(self):
        root = ROOT
        s = S.make_stage(
            enabled=True,
            contract=BASE,
            contract_document=root / "contract.json",
            assembler=ASSEMBLER,
            semantic_proofs=(root / "proof.json",),
        )
        s.validate()
        self.assertIn((root / "contract.json").resolve(), {p.path for p in s.contract.semantic_proofs})
        self.assertTrue(any(p.path.name == "late_quant_rne.py" for p in s.contract.sources))

    def test_selected_owner_cannot_change_isa(self):
        root = ROOT
        s = S.make_stage(
            enabled=True,
            contract=BASE,
            contract_document=root / "contract.json",
            assembler=ASSEMBLER,
            semantic_proofs=(root / "proof.json",),
        )
        with self.assertRaises(AttributeError):
            s.transform.__self__.contract = dataclasses.replace(BASE, host_isa="portable")

    def test_contract_knob_not_in_document_refused(self):
        root = ROOT
        with self.assertRaises(ValueError):
            S.make_stage(
                enabled=True,
                contract=dataclasses.replace(BASE, host_isa="portable"),
                contract_document=root / "contract.json",
                assembler=ASSEMBLER,
                semantic_proofs=(root / "proof.json",),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
