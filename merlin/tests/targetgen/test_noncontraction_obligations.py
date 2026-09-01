"""A non-contraction op owes instructions too. An empty requirement is satisfied by emitting anything."""
import struct

from merlin.targetgen import isa_disasm, isa_taxonomy as IT
from merlin.targetgen.isa_model import IsaModel


def _model():
    """A target with a memory role and both vector-compute roles, plus a systolic path."""
    return IsaModel(target="t", by_mnemonic={
        "MOVE": {"class": "Move", "mnemonic": "MOVE", "role": "memory",
                 "fixed_mask": 0xFF, "fixed_value": 0x01, "fields": {}},
        "VBIN": {"class": "VBin", "mnemonic": "VBIN", "role": "tensor_compute_binary",
                 "fixed_mask": 0xFF, "fixed_value": 0x02, "fields": {}},
        "VRED": {"class": "VUn", "mnemonic": "VRED", "role": "tensor_compute_unary",
                 "fixed_mask": 0xFF, "fixed_value": 0x03, "fields": {}},
    }, roles={"memory": ["Move"], "tensor_compute_binary": ["VBin"], "tensor_compute_unary": ["VUn"]})


def test_elementwise_op_owes_move_and_vector_compute():
    """The hole this closes: these slots used to be empty, so nothing about the kernel was checkable."""
    assert IT.required_role_slots(op="add") == [
        ("memory",), ("tensor_compute_binary", "tensor_compute_unary")]


def test_reduction_prefers_the_unary_role_but_accepts_either():
    """Which role a reduction lands in is a target's choice, so both are offered rather than assumed."""
    assert IT.required_role_slots(op="reduce_sum") == [
        ("memory",), ("tensor_compute_unary", "tensor_compute_binary")]


def test_composite_family_owes_both_of_its_primitives():
    slots = IT.required_role_slots(op="softmax")
    assert ("tensor_compute_unary", "tensor_compute_binary") in slots      # its reduction
    assert ("tensor_compute_binary", "tensor_compute_unary") in slots      # its elementwise map


def test_contraction_sequence_is_unchanged():
    """The systolic path must keep its exact previous obligation -- this change adds, never rewrites."""
    assert IT.required_role_slots(op="matmul") == [
        ("memory",), ("weight_load",), ("matmul",), ("acc_readout", "acc_readout_scaled")]
    assert IT.required_role_slots(op="matmul", output_dtype="fp8_e4m3")[-1] == (
        "acc_readout_scaled", "acc_readout")


def test_unrecognised_op_still_owes_nothing():
    """Inventing a demand for an op the vocabulary does not know is the other direction of error."""
    assert IT.required_role_slots(op="not_a_known_op") == []


def test_movement_op_owes_only_memory():
    assert IT.required_role_slots(op="movement") == [("memory",)]


def _kernel_words(words):
    return list(words)


def test_obligation_is_falsifiable_on_a_kernel_that_skips_the_compute():
    """A kernel that moves data and never computes must now FAIL an elementwise capsule."""
    m = _model()
    recs = isa_disasm.disassemble(m, _kernel_words([0x01, 0x01]))       # moves only
    cov = isa_disasm.coverage(m, recs, op="add")
    assert cov["missing"] == ["VBin"]


def test_obligation_is_satisfied_by_a_kernel_that_does_compute():
    m = _model()
    recs = isa_disasm.disassemble(m, _kernel_words([0x01, 0x02]))       # move + vector binary
    assert isa_disasm.coverage(m, recs, op="add")["missing"] == []


def test_target_without_the_compute_role_adds_no_impossible_demand():
    """A target declaring no vector role must not be handed a requirement it cannot ever satisfy."""
    m = IsaModel(target="t", by_mnemonic={
        "MOVE": {"class": "Move", "mnemonic": "MOVE", "role": "memory",
                 "fixed_mask": 0xFF, "fixed_value": 0x01, "fields": {}}},
        roles={"memory": ["Move"]})
    recs = isa_disasm.disassemble(m, _kernel_words([0x01]))
    assert isa_disasm.coverage(m, recs, op="add")["missing"] == []
