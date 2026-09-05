"""A member's shape class must be DERIVED from the capsule, and a gap must read as a gap.

The failure this guards is not a crash. It is a member that looks representative because nobody could
cheaply ask whether real models present its shape -- which was true of 27 of the 29 classifiable
OBJECTIVE members on this corpus, every one of them in a geometric class the target's measured census
does not contain at all.
"""
from __future__ import annotations

from merlin.perf import member_geometry as MG


def _matmul_capsule(*, m: int, k: int, n: int, op: str = "matmul") -> dict:
    return {"operation": {"op": op, "attributes": {"weight": "W", "lhs": "A0"}},
            "inputs": [{"name": "W", "role": "weight", "shape": [k, n], "dtype": "i8"},
                       {"name": "A0", "role": "input", "shape": [m, k], "dtype": "i8"}]}


class TestGeometryComesFromTheCapsuleNotTheEntry:
    def test_mkn_is_read_off_the_declared_operands(self):
        g = MG.declared_geometry(_matmul_capsule(m=16, k=2048, n=16))
        assert g["status"] == "derived"
        assert (g["M"], g["K"], g["N"]) == (16, 2048, 16)
        assert "operand shapes" in g["basis"]

    def test_a_resident_reuse_member_reads_its_first_activation(self):
        cap = _matmul_capsule(m=16, k=32, n=16, op="resident_reuse")
        cap["inputs"].append({"name": "A1", "role": "input", "shape": [16, 32], "dtype": "i8"})
        g = MG.declared_geometry(cap)
        assert (g["M"], g["K"], g["N"]) == (16, 32, 16)

    def test_operands_that_do_not_describe_one_contraction_are_refused(self):
        """The activation reducing over a different extent than the weight is not a shape to classify."""
        cap = _matmul_capsule(m=16, k=32, n=16)
        cap["inputs"][1]["shape"] = [16, 64]
        g = MG.declared_geometry(cap)
        assert g["status"] == "refused" and "one contraction" in g["reason"]

    def test_an_op_with_no_contraction_operands_is_refused_with_its_reason(self):
        """An elementwise member has no (M,K,N); inventing one would misattribute it."""
        cap = {"operation": {"op": "bias_add", "attributes": {}},
               "inputs": [{"name": "X", "role": "input", "shape": [16, 16], "dtype": "i8"}]}
        g = MG.declared_geometry(cap)
        assert g["status"] == "refused" and "bias_add" in g["reason"]

    def test_a_capsule_declaring_no_inputs_is_refused_rather_than_defaulted(self):
        g = MG.declared_geometry({"operation": {"op": "matmul"}})
        assert g["status"] == "refused" and "no inputs" in g["reason"]


class TestAGapReadsAsAGap:
    def test_unreadable_geometry_and_an_absent_class_are_different_answers(self):
        """`None` says the shape could not be read; `in_census: false` says it was and no model has it.

        Collapsing them turns an unpriced op into a coverage claim.
        """
        elementwise = {"operation": {"op": "bias_add", "attributes": {}}, "inputs": []}
        assert MG.stamp_for(elementwise, target="nonexistent_target") is None

        block = MG.stamp_for(_matmul_capsule(m=16, k=16, n=16), target="nonexistent_target")
        assert block is not None
        assert block["in_census"] is None, "no census must be UNKNOWN, never False"
        assert "unknown rather than false" in block["census_note"]

    def test_the_class_is_recomputed_from_the_shape_not_carried(self):
        """Changing a member's extents must change its recorded class with no edit anywhere."""
        square = MG.stamp_for(_matmul_capsule(m=64, k=64, n=64), target="nonexistent_target")
        wide = MG.stamp_for(_matmul_capsule(m=16, k=64, n=256), target="nonexistent_target")
        assert square["geometry_class"] != wide["geometry_class"]
        assert wide["geometry_class"] == "wide_skinny"

    def test_the_stamp_records_output_size_which_needs_no_threshold(self):
        """Output elements are comparable to the census without trusting any classifier threshold."""
        block = MG.stamp_for(_matmul_capsule(m=16, k=4096, n=16), target="nonexistent_target")
        assert block["out_elements"] == 256
        assert block["classifier"].endswith("classify_geometry")
