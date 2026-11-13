"""Atlas's architectural delay slot in the physical functional loop."""
import pytest

from npu_model.configs import isa_definition as isa
from npu_model.functional.program import FunctionalInstruction, FunctionalProgram

from functional_runtime import PhysicalFunctionalInterpreter


def run(*instructions, **kwargs):
    program = FunctionalProgram([FunctionalInstruction(index, instruction, index + 1, str(instruction))
                                 for index, instruction in enumerate(instructions)])
    return PhysicalFunctionalInterpreter().run(program, **kwargs)


def test_redirects_execute_one_delay_slot_and_link_to_it():
    result = run(
        isa.BEQ(rs1=0, rs2=0, imm=6),   # 0: taken, target 3
        isa.ADDI(rd=1, rs1=0, imm=7),   # 1: delay slot
        isa.ADDI(rd=2, rs1=0, imm=9),   # 2: skipped
        isa.BNE(rs1=0, rs2=0, imm=6),   # 3: not taken, so the JAL is legal
        isa.JAL(rd=3, imm=6),           # 4: target 7, link 5
        isa.ADDI(rd=4, rs1=0, imm=11),  # 5: delay slot
        isa.ADDI(rd=5, rs1=0, imm=13),  # 6: skipped
        isa.ADDI(rd=6, rs1=0, imm=11),  # 7: JALR target
        isa.JALR(rd=7, rs1=6, imm=0),   # 8: target 11, link 9
        isa.ADDI(rd=8, rs1=0, imm=17),  # 9: delay slot
        isa.ADDI(rd=9, rs1=0, imm=19),  # 10: skipped
        isa.ECALL(),                    # 11
    )
    assert [result.state.read_xrf(index) for index in range(1, 10)] == [7, 0, 5, 11, 0, 11, 9, 17, 0]
    assert (result.instructions_executed, result.final_pc, result.halt_reason) == (9, 12, "ecall")


@pytest.mark.parametrize("halt", [isa.ECALL(), isa.EBREAK()])
def test_halt_in_delay_slot_stops_before_the_target(halt):
    result = run(isa.JAL(rd=0, imm=4), halt, isa.ADDI(rd=1, rs1=0, imm=1), max_instructions=2)
    assert (result.instructions_executed, result.final_pc, result.halt_reason) == (2, 2, halt.mnemonic)
    assert result.state.read_xrf(1) == 0


@pytest.mark.parametrize("slot", [isa.BNE(rs1=0, rs2=0, imm=4), isa.JAL(rd=0, imm=4), isa.JALR(rd=0, rs1=0, imm=3)])
def test_control_flow_in_delay_slot_is_illegal(slot, monkeypatch):
    executed = []
    monkeypatch.setattr(slot, "exec", executed.append)
    with pytest.raises(RuntimeError, match=f"illegal control flow '{slot.mnemonic}' in the delay slot at PC 1"):
        run(isa.JAL(rd=0, imm=4), slot, isa.ECALL(), isa.ECALL())
    assert not executed

