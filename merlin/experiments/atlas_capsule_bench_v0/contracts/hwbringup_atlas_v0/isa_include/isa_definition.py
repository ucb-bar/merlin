from typing import TYPE_CHECKING
import torch
from npu_model.isa import (
    CSRType,
    IType,
    RType,
    SBType,
    SType,
    UJType,
    UType,
    VIType,
    VLSType,
    VRType,
)
from npu_model.isa_patterns import (
    DirectImm,
    DMARegUnary,
    ExponentImm,
    ExponentOffsetLoad,
    Nullary,
    ScalarBaseOffsetStore,
    ScalarComputeImm,
    ScalarComputeShamt,
    ScalarBranchImm,
    ScalarComputeReg,
    ScalarImm,
    ScalarOffsetLoad,
    JalrPattern,
    TensorBaseOffset,
    TensorComputeBinary,
    TensorComputeMixed,
    TensorComputeUnary,
    MXUWeightPush,
    MXUAccumulatorPush,
    MXUAccumulatorPopE1,
    MXUAccumulatorPop,
    MXUMatMul,
    UnaryImm,
)
from npu_model.isa_types import (
    ScalarReg,
    ExponentReg,
    MatrixReg,
    WeightBuffer,
    Accumulator,
    SBImm12,
    Imm12,
    EXU,
)

if TYPE_CHECKING:
    from npu_model.hardware.arch_state import ArchState

PIPELINE_LATENCY = 2

# Mask for 64-bit unsigned comparison (RISC-V RV64)
_MASK64 = 0xFFFFFFFFFFFFFFFF


# =============================================================================
# Helper Functions
# =============================================================================


def _sign_extend(value: int, length: int):
    """Sign-extends a value of a given bit length to the Python integer width."""
    value &= (1 << length) - 1
    if value & (1 << (length - 1)):
        value -= 1 << length
    return value


def _int_to_le_bytes(data: int, length: int) -> torch.Tensor:
    type_map = {1: torch.uint8, 2: torch.int16, 4: torch.int32}
    if length not in type_map:
        raise ValueError("Length must be 1, 2, or 4 bytes.")
    return torch.tensor([data], dtype=type_map[length]).view(torch.uint8).clone()


def _le_bytes_to_int(tensor: torch.Tensor) -> int:
    length = tensor.numel()
    type_map = {1: torch.uint8, 2: torch.int16, 4: torch.int32}
    if length not in type_map:
        raise ValueError("Tensor length must be 1, 2, or 4 bytes.")
    raw_val = tensor.contiguous().view(type_map[length]).item()
    masks = {1: 0xFF, 2: 0xFFFF, 4: 0xFFFFFFFF}
    return int(raw_val) & masks[length]


def _tensor_register_bytes(state: ArchState) -> int:
    return state.cfg.mrf_depth * state.cfg.mrf_width // torch.uint8.itemsize


def _vmatmul(
    state: ArchState,
    unit: str,
    vd: Accumulator,
    vs1: MatrixReg,
    vs2: WeightBuffer,
    accumulate: bool,
) -> None:
    activation_fp16 = state.read_mrf_fp8(vs1).to(torch.float16)
    weight_fp16 = state.read_wb_fp8(unit, vs2).to(torch.float16)
    result_fp16 = activation_fp16 @ weight_fp16
    if accumulate:
        result_fp16 = result_fp16 + state.read_acc_bf16(unit, vd).to(torch.float16)
    state.write_acc_bf16(unit, vd, result_fp16.to(torch.bfloat16))


def _assert_bf16_pair(state: ArchState, reg: int) -> None:
    assert reg < state.cfg.num_m_registers - 1


def _read_mrf_bf16_pair(state: ArchState, reg: int) -> torch.Tensor:
    _assert_bf16_pair(state, reg)
    return state.read_mrf_bf16_tile(reg)


def _write_mrf_bf16_pair(state: ArchState, reg: int, value: torch.Tensor) -> None:
    _assert_bf16_pair(state, reg)
    state.write_mrf_bf16_tile(reg, value.to(torch.bfloat16).contiguous())


class LB(ScalarOffsetLoad, IType, exu=EXU.LSU, opcode=0b0000011, funct3=0b000):
    def exec(self, state: ArchState) -> None:
        imm = _sign_extend(self.imm & 0xFFF, 12)
        value = _le_bytes_to_int(state.read_vmem(state.read_xrf(self.rs1), imm, 1))
        state.write_xrf(self.rd, _sign_extend(value, 8))


class LH(ScalarOffsetLoad, IType, exu=EXU.LSU, opcode=0b0000011, funct3=0b001):
    def exec(self, state: ArchState) -> None:
        imm = _sign_extend(self.imm & 0xFFF, 12)
        value = _le_bytes_to_int(state.read_vmem(state.read_xrf(self.rs1), imm, 2))
        state.write_xrf(self.rd, _sign_extend(value, 16))


class LW(ScalarOffsetLoad, IType, exu=EXU.LSU, opcode=0b0000011, funct3=0b010):
    def exec(self, state: ArchState) -> None:
        imm = _sign_extend(self.imm & 0xFFF, 12)
        value = _le_bytes_to_int(state.read_vmem(state.read_xrf(self.rs1), imm, 4))
        state.write_xrf(self.rd, value)


class LBU(ScalarOffsetLoad, IType, exu=EXU.LSU, opcode=0b0000011, funct3=0b100):
    def exec(self, state: ArchState) -> None:
        imm = _sign_extend(self.imm & 0xFFF, 12)
        value = _le_bytes_to_int(state.read_vmem(state.read_xrf(self.rs1), imm, 1))
        state.write_xrf(self.rd, value)


class LHU(ScalarOffsetLoad, IType, exu=EXU.LSU, opcode=0b0000011, funct3=0b101):
    def exec(self, state: ArchState) -> None:
        imm = _sign_extend(self.imm & 0xFFF, 12)
        value = _le_bytes_to_int(state.read_vmem(state.read_xrf(self.rs1), imm, 2))
        state.write_xrf(self.rd, value)


class SELD(
    ExponentOffsetLoad, IType[ExponentReg], exu=EXU.LSU, opcode=0b0000011, funct3=0b110
):
    def exec(self, state: ArchState) -> None:
        imm = _sign_extend(self.imm & 0xFFF, 12)
        state.write_erf(
            self.rd,
            int(state.read_vmem(state.read_xrf(self.rs1), imm, 1).view(torch.uint8)),
        )


class SELI(
    ExponentImm, IType[ExponentReg], exu=EXU.SCALAR, opcode=0b0000011, funct3=0b111
):
    def exec(self, state: ArchState):
        state.write_erf(self.rd, _sign_extend(self.imm & 0xFFF, 12))


class VLOAD(TensorBaseOffset, VLSType, exu=EXU.LSU, opcode=0b0000111, funct2=0b00):
    def exec(self, state: ArchState) -> None:
        addr = state.read_xrf(self.rs1) + (self.imm << 5)
        data = state.read_vmem(addr, 0, _tensor_register_bytes(state)).view(torch.uint8)
        state.write_mrf_u8(self.vd, data)


class VSTORE(TensorBaseOffset, VLSType, exu=EXU.LSU, opcode=0b0000111, funct2=0b01):
    def exec(self, state: ArchState) -> None:
        addr = state.read_xrf(self.rs1) + (self.imm << 5)
        data = state.read_mrf_fp8(self.vd).view(torch.uint8)
        state.write_vmem(addr, 0, data)


class FENCE(Nullary, IType, exu=EXU.SCALAR, opcode=0b0001111, funct3=0b000):
    def exec(self, state: ArchState) -> None:
        pass


class ADDI(ScalarComputeImm, IType, exu=EXU.SCALAR, opcode=0b0010011, funct3=0b000):
    def exec(self, state: ArchState) -> None:
        state.write_xrf(
            self.rd, state.xrf[self.rs1] + _sign_extend(self.imm & 0xFFF, 12)
        )


class SLLI(ScalarComputeShamt, IType, exu=EXU.SCALAR, opcode=0b0010011, funct3=0b001):
    def exec(self, state: ArchState) -> None:
        state.write_xrf(self.rd, state.xrf[self.rs1] << (self.imm & 0x3F))


class SLTI(ScalarComputeImm, IType, exu=EXU.SCALAR, opcode=0b0010011, funct3=0b010):
    def exec(self, state: ArchState) -> None:
        imm = _sign_extend(self.imm & 0xFFF, 12)
        state.write_xrf(self.rd, 1 if state.xrf[self.rs1] < imm else 0)


class SLTIU(ScalarComputeImm, IType, exu=EXU.SCALAR, opcode=0b0010011, funct3=0b011):
    def exec(self, state: ArchState) -> None:
        a = state.xrf[self.rs1] & _MASK64
        b = _sign_extend(self.imm & 0xFFF, 12) & _MASK64
        state.write_xrf(self.rd, 1 if a < b else 0)


class XORI(ScalarComputeImm, IType, exu=EXU.SCALAR, opcode=0b0010011, funct3=0b100):
    def exec(self, state: ArchState) -> None:
        state.write_xrf(
            self.rd, state.xrf[self.rs1] ^ _sign_extend(self.imm & 0xFFF, 12)
        )


class SRLI(ScalarComputeShamt, IType, exu=EXU.SCALAR, opcode=0b0010011, funct3=0b101):
    def exec(self, state: ArchState) -> None:
        state.write_xrf(self.rd, state.xrf[self.rs1] >> (self.imm & 0x3F))


class SRAI(ScalarComputeShamt, IType, exu=EXU.SCALAR, opcode=0b0010011, funct3=0b101):
    UPPER_IMM = 0b0100000

    def exec(self, state: ArchState) -> None:
        src = _sign_extend(state.xrf[self.rs1] & 0xFFFFFFFF, 32)
        state.write_xrf(self.rd, (src >> (self.imm & 0x1F)) & 0xFFFFFFFF)


class ORI(ScalarComputeImm, IType, exu=EXU.SCALAR, opcode=0b0010011, funct3=0b110):
    def exec(self, state: ArchState) -> None:
        state.write_xrf(
            self.rd, state.xrf[self.rs1] | _sign_extend(self.imm & 0xFFF, 12)
        )


class ANDI(ScalarComputeImm, IType, exu=EXU.SCALAR, opcode=0b0010011, funct3=0b111):
    def exec(self, state: ArchState) -> None:
        state.write_xrf(
            self.rd, state.xrf[self.rs1] & _sign_extend(self.imm & 0xFFF, 12)
        )


class AUIPC(ScalarImm, UType, exu=EXU.SCALAR, opcode=0b0010111):
    def exec(self, state: ArchState) -> None:
        state.write_xrf(
            self.rd, ((self.imm << 12) & 0xFFFFFFFF) + state.pc - PIPELINE_LATENCY * 4
        )


class SB(ScalarBaseOffsetStore, SType, exu=EXU.LSU, opcode=0b0100011, funct3=0b000):
    def exec(self, state: ArchState) -> None:
        imm = _sign_extend(self.imm & 0xFFF, 12)
        state.write_vmem(
            state.read_xrf(self.rs1),
            imm,
            _int_to_le_bytes(state.read_xrf(self.rs2) & 0xFF, 1),
        )


class SH(ScalarBaseOffsetStore, SType, exu=EXU.LSU, opcode=0b0100011, funct3=0b001):
    def exec(self, state: ArchState) -> None:
        imm = _sign_extend(self.imm & 0xFFF, 12)
        state.write_vmem(
            state.read_xrf(self.rs1),
            imm,
            _int_to_le_bytes(state.read_xrf(self.rs2) & 0xFFFF, 2),
        )


class SW(ScalarBaseOffsetStore, SType, exu=EXU.LSU, opcode=0b0100011, funct3=0b010):
    def exec(self, state: ArchState) -> None:
        imm = _sign_extend(self.imm & 0xFFF, 12)
        state.write_vmem(
            state.read_xrf(self.rs1),
            imm,
            _int_to_le_bytes(state.read_xrf(self.rs2) & 0xFFFFFFFF, 4),
        )


class ADD(
    ScalarComputeReg,
    RType,
    exu=EXU.SCALAR,
    opcode=0b0110011,
    funct3=0b000,
    funct7=0b0000000,
):
    def exec(self, state: ArchState) -> None:
        state.write_xrf(self.rd, state.xrf[self.rs1] + state.xrf[self.rs2])


class SUB(
    ScalarComputeReg,
    RType,
    exu=EXU.SCALAR,
    opcode=0b0110011,
    funct3=0b000,
    funct7=0b0100000,
):
    def exec(self, state: ArchState) -> None:
        state.write_xrf(self.rd, state.xrf[self.rs1] - state.xrf[self.rs2])


class SLL(
    ScalarComputeReg,
    RType,
    exu=EXU.SCALAR,
    opcode=0b0110011,
    funct3=0b001,
    funct7=0b0000000,
):
    def exec(self, state: ArchState) -> None:
        state.write_xrf(self.rd, state.xrf[self.rs1] << state.xrf[self.rs2])


class SLT(
    ScalarComputeReg,
    RType,
    exu=EXU.SCALAR,
    opcode=0b0110011,
    funct3=0b010,
    funct7=0b0000000,
):
    def exec(self, state: ArchState) -> None:
        state.write_xrf(self.rd, 1 if state.xrf[self.rs1] < state.xrf[self.rs2] else 0)


class SLTU(
    ScalarComputeReg,
    RType,
    exu=EXU.SCALAR,
    opcode=0b0110011,
    funct3=0b011,
    funct7=0b0000000,
):
    def exec(self, state: ArchState) -> None:
        a = state.xrf[self.rs1] & _MASK64
        b = state.xrf[self.rs2] & _MASK64
        state.write_xrf(self.rd, 1 if a < b else 0)


class XOR(
    ScalarComputeReg,
    RType,
    exu=EXU.SCALAR,
    opcode=0b0110011,
    funct3=0b100,
    funct7=0b0000000,
):
    def exec(self, state: ArchState) -> None:
        state.write_xrf(self.rd, state.xrf[self.rs1] ^ state.xrf[self.rs2])


class SRL(
    ScalarComputeReg,
    RType,
    exu=EXU.SCALAR,
    opcode=0b0110011,
    funct3=0b101,
    funct7=0b0000000,
):
    def exec(self, state: ArchState) -> None:
        state.write_xrf(self.rd, state.xrf[self.rs1] >> state.xrf[self.rs2])


class SRA(
    ScalarComputeReg,
    RType,
    exu=EXU.SCALAR,
    opcode=0b0110011,
    funct3=0b101,
    funct7=0b0100000,
):
    def exec(self, state: ArchState) -> None:
        src = _sign_extend(state.xrf[self.rs1] & 0xFFFFFFFF, 32)
        state.write_xrf(self.rd, (src >> (state.xrf[self.rs2] & 0x1F)) & 0xFFFFFFFF)


class OR(
    ScalarComputeReg,
    RType,
    exu=EXU.SCALAR,
    opcode=0b0110011,
    funct3=0b110,
    funct7=0b0000000,
):
    def exec(self, state: ArchState) -> None:
        state.write_xrf(self.rd, state.xrf[self.rs1] | state.xrf[self.rs2])


class AND(
    ScalarComputeReg,
    RType,
    exu=EXU.SCALAR,
    opcode=0b0110011,
    funct3=0b111,
    funct7=0b0000000,
):
    def exec(self, state: ArchState) -> None:
        state.write_xrf(self.rd, state.xrf[self.rs1] & state.xrf[self.rs2])


class LUI(ScalarImm, UType, exu=EXU.SCALAR, opcode=0b0110111):
    def exec(self, state: ArchState) -> None:
        state.write_xrf(self.rd, (self.imm << 12) & _MASK64)


class VADD_BF16(
    TensorComputeBinary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b0000000
):
    def exec(self, state: ArchState) -> None:
        a = _read_mrf_bf16_pair(state, self.vs1)
        b = _read_mrf_bf16_pair(state, self.vs2)
        _write_mrf_bf16_pair(state, self.vd, a + b)


class VREDSUM_BF16(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b0000001
):
    def exec(self, state: ArchState) -> None:
        x = _read_mrf_bf16_pair(state, self.vs1)
        result = x.sum(dim=0, keepdim=True).to(torch.bfloat16).expand_as(x).contiguous()
        _write_mrf_bf16_pair(state, self.vd, result)


class VSUB_BF16(
    TensorComputeBinary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b0000010
):
    def exec(self, state: ArchState) -> None:
        a = _read_mrf_bf16_pair(state, self.vs1)
        b = _read_mrf_bf16_pair(state, self.vs2)
        _write_mrf_bf16_pair(state, self.vd, (a - b).to(torch.bfloat16))


class VMUL_BF16(
    TensorComputeBinary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b0000011
):
    def exec(self, state: ArchState) -> None:
        a = _read_mrf_bf16_pair(state, self.vs1)
        b = _read_mrf_bf16_pair(state, self.vs2)
        _write_mrf_bf16_pair(state, self.vd, a * b)


class VMINIMUM_BF16(
    TensorComputeBinary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b0000100
):
    def exec(self, state: ArchState) -> None:
        a = _read_mrf_bf16_pair(state, self.vs1)
        b = _read_mrf_bf16_pair(state, self.vs2)
        _write_mrf_bf16_pair(state, self.vd, torch.minimum(a, b))


class VREDMIN_BF16(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b0000101
):
    def exec(self, state: ArchState) -> None:
        x = _read_mrf_bf16_pair(state, self.vs1)
        result = (
            x.min(dim=0, keepdim=True)
            .values.to(torch.bfloat16)
            .expand_as(x)
            .contiguous()
        )
        _write_mrf_bf16_pair(state, self.vd, result)


class VMAXIMUM_BF16(
    TensorComputeBinary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b0000110
):
    def exec(self, state: ArchState) -> None:
        a = _read_mrf_bf16_pair(state, self.vs1)
        b = _read_mrf_bf16_pair(state, self.vs2)
        _write_mrf_bf16_pair(state, self.vd, torch.maximum(a, b))


class VREDMAX_BF16(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b0000111
):
    def exec(self, state: ArchState) -> None:
        x = _read_mrf_bf16_pair(state, self.vs1)
        result = (
            x.max(dim=0, keepdim=True)
            .values.to(torch.bfloat16)
            .expand_as(x)
            .contiguous()
        )
        _write_mrf_bf16_pair(state, self.vd, result)


class VREDSUM_ROW_BF16(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b0100001
):
    def exec(self, state: ArchState) -> None:
        x = _read_mrf_bf16_pair(state, self.vs1)
        result = x.sum(dim=1, keepdim=True).to(torch.bfloat16).expand_as(x).contiguous()
        _write_mrf_bf16_pair(state, self.vd, result)


class VREDMIN_ROW_BF16(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b0100100
):
    def exec(self, state: ArchState) -> None:
        x = _read_mrf_bf16_pair(state, self.vs1)
        result = (
            x.min(dim=1, keepdim=True)
            .values.to(torch.bfloat16)
            .expand_as(x)
            .contiguous()
        )
        _write_mrf_bf16_pair(state, self.vd, result)


class VREDMAX_ROW_BF16(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b0100110
):
    def exec(self, state: ArchState) -> None:
        x = _read_mrf_bf16_pair(state, self.vs1)
        result = (
            x.max(dim=1, keepdim=True)
            .values.to(torch.bfloat16)
            .expand_as(x)
            .contiguous()
        )
        _write_mrf_bf16_pair(state, self.vd, result)


class VMOV(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b1000000
):
    def exec(self, state: ArchState) -> None:
        state.write_mrf_bf16(self.vd, state.read_mrf_bf16(self.vs1))


class VRECIP_BF16(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b1000001
):
    def exec(self, state: ArchState) -> None:
        _write_mrf_bf16_pair(
            state, self.vd, torch.reciprocal(_read_mrf_bf16_pair(state, self.vs1))
        )


class VEXP_BF16(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b1000010
):
    def exec(self, state: ArchState) -> None:
        x = _read_mrf_bf16_pair(state, self.vs1)
        _write_mrf_bf16_pair(state, self.vd, torch.exp(x))


class VEXP2_BF16(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b1000011
):
    def exec(self, state: ArchState) -> None:
        x = _read_mrf_bf16_pair(state, self.vs1)
        _write_mrf_bf16_pair(state, self.vd, torch.exp2(x))


class VPACK_BF16_FP8(
    TensorComputeMixed, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b1000100
):
    def exec(self, state: ArchState) -> None:
        assert self.vs2 != state.cfg.num_m_registers - 1
        scale = state.read_erf(self.es1)
        reg_low = state.read_mrf_bf16(self.vs2)
        reg_high = state.read_mrf_bf16(self.vs2 + 1)
        combined_bf16 = torch.cat([reg_low, reg_high], dim=1)
        quantized_fp8 = (combined_bf16 * scale).to(torch.float8_e4m3fn)
        state.write_mrf_fp8(self.vd, quantized_fp8)


class VUNPACK_FP8_BF16(
    TensorComputeMixed, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b1000101
):
    def exec(self, state: ArchState) -> None:
        assert self.vd != state.cfg.num_m_registers - 1
        scale = state.read_erf(self.es1)
        source_fp8 = state.read_mrf_fp8(self.vs2)
        dequantized_bf16 = source_fp8.to(torch.bfloat16)
        scaled_bf16 = dequantized_bf16 / scale
        reg_low, reg_high = torch.chunk(scaled_bf16, chunks=2, dim=1)
        state.write_mrf_bf16(self.vd, reg_low)
        state.write_mrf_bf16(self.vd + 1, reg_high)


class VRELU_BF16(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b1001000
):
    def exec(self, state: ArchState) -> None:
        _write_mrf_bf16_pair(
            state, self.vd, torch.relu(_read_mrf_bf16_pair(state, self.vs1))
        )


class VSIN_BF16(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b1001001
):
    def exec(self, state: ArchState) -> None:
        x = _read_mrf_bf16_pair(state, self.vs1)
        _write_mrf_bf16_pair(state, self.vd, torch.sin(x))


class VCOS_BF16(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b1001010
):
    def exec(self, state: ArchState) -> None:
        x = _read_mrf_bf16_pair(state, self.vs1)
        _write_mrf_bf16_pair(state, self.vd, torch.cos(x))


class VTANH_BF16(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b1001011
):
    def exec(self, state: ArchState) -> None:
        x = _read_mrf_bf16_pair(state, self.vs1)
        _write_mrf_bf16_pair(state, self.vd, torch.tanh(x))


class VLOG2_BF16(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b1001100
):
    def exec(self, state: ArchState) -> None:
        x = _read_mrf_bf16_pair(state, self.vs1)
        _write_mrf_bf16_pair(state, self.vd, torch.log2(x))


class VSQRT_BF16(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b1001101
):
    def exec(self, state: ArchState) -> None:
        x = _read_mrf_bf16_pair(state, self.vs1)
        _write_mrf_bf16_pair(state, self.vd, torch.sqrt(x))


class VSQUARE_BF16(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b1001110
):
    def exec(self, state: ArchState) -> None:
        x = _read_mrf_bf16_pair(state, self.vs1)
        _write_mrf_bf16_pair(state, self.vd, x * x)


class VCUBE_BF16(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b1001111
):
    def exec(self, state: ArchState) -> None:
        x = _read_mrf_bf16_pair(state, self.vs1)
        _write_mrf_bf16_pair(state, self.vd, x * x * x)


class VLI_ALL(DirectImm, VIType, exu=EXU.VECTOR, opcode=0b1011111, funct3=0b000):
    def exec(self, state: ArchState) -> None:
        shape = state.read_mrf_bf16(0).shape
        raw_bits = torch.full(shape, self.imm, dtype=torch.uint16)
        bf16_tensor = raw_bits.view(torch.bfloat16)
        state.write_mrf_bf16(self.vd, bf16_tensor)


class VLI_ROW(DirectImm, VIType, exu=EXU.VECTOR, opcode=0b1011111, funct3=0b001):
    def exec(self, state: ArchState) -> None:
        shape = state.read_mrf_bf16(0).shape
        x = torch.zeros(shape, dtype=torch.bfloat16)
        x[0, :] = self.imm
        bf16_tensor = x.view(torch.bfloat16)
        state.write_mrf_bf16(self.vd, bf16_tensor)


class VLI_COL(DirectImm, VIType, exu=EXU.VECTOR, opcode=0b1011111, funct3=0b010):
    def exec(self, state: ArchState) -> None:
        shape = state.read_mrf_bf16(0).shape
        x = torch.zeros(shape, dtype=torch.bfloat16)
        x[:, 0] = self.imm
        bf16_tensor = x.view(torch.bfloat16)
        state.write_mrf_bf16(self.vd, bf16_tensor)


class VLI_ONE(DirectImm, VIType, exu=EXU.VECTOR, opcode=0b1011111, funct3=0b011):
    def exec(self, state: ArchState) -> None:
        shape = state.read_mrf_bf16(0).shape
        x = torch.zeros(shape, dtype=torch.bfloat16)
        x[0, 0] = self.imm
        bf16_tensor = x.view(torch.bfloat16)
        state.write_mrf_bf16(self.vd, bf16_tensor)


class BEQ(ScalarBranchImm, SBType, exu=EXU.SCALAR, opcode=0b1100011, funct3=0b000):
    def exec(self, state: ArchState) -> None:
        imm = _sign_extend(self.imm & 0x1FFF, 13)
        if state.xrf[self.rs1] == state.xrf[self.rs2]:
            state.set_npc(state.pc + imm - PIPELINE_LATENCY * 4)


class BNE(ScalarBranchImm, SBType, exu=EXU.SCALAR, opcode=0b1100011, funct3=0b001):
    def exec(self, state: ArchState) -> None:
        imm = _sign_extend(self.imm & 0x1FFF, 13)
        if state.xrf[self.rs1] != state.xrf[self.rs2]:
            state.set_npc(state.pc + imm - PIPELINE_LATENCY * 4)


class BLT(ScalarBranchImm, SBType, exu=EXU.SCALAR, opcode=0b1100011, funct3=0b100):
    def exec(self, state: ArchState) -> None:
        imm = _sign_extend(self.imm & 0x1FFF, 13)
        if state.xrf[self.rs1] < state.xrf[self.rs2]:
            state.set_npc(state.pc + imm - PIPELINE_LATENCY * 4)


class BGE(ScalarBranchImm, SBType, exu=EXU.SCALAR, opcode=0b1100011, funct3=0b101):
    def exec(self, state: ArchState) -> None:
        imm = _sign_extend(self.imm & 0x1FFF, 13)
        if state.xrf[self.rs1] >= state.xrf[self.rs2]:
            state.set_npc(state.pc + imm - PIPELINE_LATENCY * 4)


class BLTU(ScalarBranchImm, SBType, exu=EXU.SCALAR, opcode=0b1100011, funct3=0b110):
    def exec(self, state: ArchState) -> None:
        imm = _sign_extend(self.imm & 0x1FFF, 13)
        a = state.xrf[self.rs1] & _MASK64
        b = state.xrf[self.rs2] & _MASK64
        if a < b:
            state.set_npc(state.pc + imm - PIPELINE_LATENCY * 4)


class BGEU(ScalarBranchImm, SBType, exu=EXU.SCALAR, opcode=0b1100011, funct3=0b111):
    def exec(self, state: ArchState) -> None:
        imm = _sign_extend(self.imm & 0x1FFF, 13)
        a = state.xrf[self.rs1] & _MASK64
        b = state.xrf[self.rs2] & _MASK64
        if a >= b:
            state.set_npc(state.pc + imm - PIPELINE_LATENCY * 4)


class JALR(
    JalrPattern,
    IType[ScalarReg, SBImm12],
    exu=EXU.SCALAR,
    opcode=0b1100111,
    funct3=0b000,
):
    def exec(self, state: ArchState) -> None:
        imm = _sign_extend(self.imm & 0x1FFF, 13)
        state.write_xrf(self.rd, state.pc - PIPELINE_LATENCY * 4 + 4)
        state.set_npc(state.read_xrf(self.rs1) + imm)


class DELAY(UnaryImm, IType, exu=EXU.SCALAR, opcode=0b1100111, funct3=0b001):
    def exec(self, state: ArchState) -> None:
        pass


class VTRPOSE_XLU(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1101011, funct7=0b0000000
):
    def exec(self, state: ArchState) -> None:
        reg_in = state.read_mrf_fp8(self.vs1)
        transposed = reg_in.view(32, 32).t().contiguous().reshape(-1)
        state.write_mrf_fp8(self.vd, transposed)


class JAL(ScalarImm, UJType, exu=EXU.SCALAR, opcode=0b1101111):
    def exec(self, state: ArchState) -> None:
        imm = _sign_extend(self.imm & 0xFFFFF, 20)
        state.write_xrf(self.rd, state.pc - PIPELINE_LATENCY * 4 + 4)
        state.set_npc(state.pc + imm - PIPELINE_LATENCY * 4)


class CSRRW(ScalarComputeImm, CSRType, exu=EXU.SCALAR, opcode=0b1110011, funct3=0b001):
    def exec(self, state: ArchState) -> None:
        old = state.read_csrf(self.imm)
        state.write_csrf(self.imm, state.read_xrf(self.rs1))
        state.write_xrf(self.rd, old)


class CSRRS(ScalarComputeImm, CSRType, exu=EXU.SCALAR, opcode=0b1110011, funct3=0b010):
    def exec(self, state: ArchState) -> None:
        old = state.read_csrf(self.imm)
        state.write_csrf(self.imm, old | state.read_xrf(self.rs1))
        state.write_xrf(self.rd, old)


class CSRRC(ScalarComputeImm, CSRType, exu=EXU.SCALAR, opcode=0b1110011, funct3=0b011):
    def exec(self, state: ArchState) -> None:
        old = state.read_csrf(self.imm)
        state.write_csrf(self.imm, old & ~state.read_xrf(self.rs1))
        state.write_xrf(self.rd, old)


class CSRRWI(ScalarComputeImm, CSRType, exu=EXU.SCALAR, opcode=0b1110011, funct3=0b101):
    def exec(self, state: ArchState) -> None:
        old = state.read_csrf(self.imm)
        state.write_csrf(self.imm, self.rs1 & 0b11111)
        state.write_xrf(self.rd, old)


class CSRRSI(ScalarComputeImm, CSRType, exu=EXU.SCALAR, opcode=0b1110011, funct3=0b110):
    def exec(self, state: ArchState) -> None:
        old = state.read_csrf(self.imm)
        state.write_csrf(self.imm, old | (self.rs1 & 0b11111))
        state.write_xrf(self.rd, old)


class CSRRCI(ScalarComputeImm, CSRType, exu=EXU.SCALAR, opcode=0b1110011, funct3=0b100):
    def exec(self, state: ArchState) -> None:
        old = state.read_csrf(self.imm)
        state.write_csrf(self.imm, old & ~(self.rs1 & 0b11111))
        state.write_xrf(self.rd, old)


class ECALL(Nullary, IType, exu=EXU.SCALAR, opcode=0b1110011, funct3=0b000):
    imm: Imm12 = Imm12(0)

    def exec(self, state: ArchState) -> None:
        state.halted = True


class EBREAK(Nullary, IType, exu=EXU.SCALAR, opcode=0b1110011, funct3=0b000):
    imm: Imm12 = Imm12(1)

    def exec(self, state: ArchState) -> None:
        state.halted = True


class VMATPUSH_WEIGHT_MXU0(
    MXUWeightPush,
    VRType[WeightBuffer, MatrixReg],
    exu=EXU.MATRIX_SYSTOLIC,
    opcode=0b1110111,
    funct7=0b0000000,
):
    def exec(self, state: ArchState) -> None:
        state.write_wb_u8("mxu0", self.vd, state.mrf[self.vs1].view(torch.uint8))


class VMATPUSH_WEIGHT_MXU1(
    MXUWeightPush,
    VRType[WeightBuffer, MatrixReg],
    exu=EXU.MATRIX_INNER,
    opcode=0b1110111,
    funct7=0b0000001,
):
    def exec(self, state: ArchState) -> None:
        state.write_wb_u8("mxu1", self.vd, state.mrf[self.vs1].view(torch.uint8))


class VMATPUSH_ACC_FP8_MXU0(
    MXUAccumulatorPush,
    VRType[Accumulator, MatrixReg],
    exu=EXU.MATRIX_SYSTOLIC,
    opcode=0b1110111,
    funct7=0b0000010,
):
    def exec(self, state: ArchState) -> None:
        state.write_acc_bf16(
            "mxu0", self.vd, state.read_mrf_fp8(self.vs1).to(torch.bfloat16)
        )


class VMATPUSH_ACC_FP8_MXU1(
    MXUAccumulatorPush,
    VRType[Accumulator, MatrixReg],
    exu=EXU.MATRIX_INNER,
    opcode=0b1110111,
    funct7=0b0000011,
):
    def exec(self, state: ArchState) -> None:
        state.write_acc_bf16(
            "mxu1", self.vd, state.read_mrf_fp8(self.vs1).to(torch.bfloat16)
        )


class VMATPUSH_ACC_BF16_MXU0(
    MXUAccumulatorPush,
    VRType[Accumulator, MatrixReg],
    exu=EXU.MATRIX_SYSTOLIC,
    opcode=0b1110111,
    funct7=0b0000100,
):
    def exec(self, state: ArchState) -> None:
        state.write_acc_bf16("mxu0", self.vd, state.read_mrf_bf16_tile(self.vs1))


class VMATPUSH_ACC_BF16_MXU1(
    MXUAccumulatorPush,
    VRType[Accumulator, MatrixReg],
    exu=EXU.MATRIX_INNER,
    opcode=0b1110111,
    funct7=0b0000101,
):
    def exec(self, state: ArchState) -> None:
        state.write_acc_bf16("mxu1", self.vd, state.read_mrf_bf16_tile(self.vs1))


class VMATPOP_FP8_ACC_MXU0(
    MXUAccumulatorPopE1,
    VRType[MatrixReg, Accumulator],
    exu=EXU.MATRIX_SYSTOLIC,
    opcode=0b1110111,
    funct7=0b0000110,
):
    def exec(self, state: ArchState) -> None:
        quantized = torch.div(
            state.read_acc_bf16("mxu0", self.vs2),
            state.read_erf(self.es1),
            rounding_mode="trunc",
        ).to(torch.float8_e4m3fn)
        state.write_mrf_u8(self.vd, quantized.view(torch.uint8))


class VMATPOP_FP8_ACC_MXU1(
    MXUAccumulatorPopE1,
    VRType[MatrixReg, Accumulator],
    exu=EXU.MATRIX_INNER,
    opcode=0b1110111,
    funct7=0b0000111,
):
    def exec(self, state: ArchState) -> None:
        quantized = torch.div(
            state.read_acc_bf16("mxu1", self.vs2),
            state.read_erf(self.es1),
            rounding_mode="trunc",
        ).to(torch.float8_e4m3fn)
        state.write_mrf_u8(self.vd, quantized.view(torch.uint8))


class VMATPOP_BF16_ACC_MXU0(
    MXUAccumulatorPop,
    VRType[MatrixReg, Accumulator],
    exu=EXU.MATRIX_SYSTOLIC,
    opcode=0b1110111,
    funct7=0b0001000,
):
    def exec(self, state: ArchState) -> None:
        state.write_mrf_bf16_tile(self.vd, state.read_acc_bf16("mxu0", self.vs2))


class VMATPOP_BF16_ACC_MXU1(
    MXUAccumulatorPop,
    VRType[MatrixReg, Accumulator],
    exu=EXU.MATRIX_INNER,
    opcode=0b1110111,
    funct7=0b0001001,
):
    def exec(self, state: ArchState) -> None:
        state.write_mrf_bf16_tile(self.vd, state.read_acc_bf16("mxu1", self.vs2))


class VMATMUL_MXU0(
    MXUMatMul,
    VRType[Accumulator, WeightBuffer],
    exu=EXU.MATRIX_SYSTOLIC,
    opcode=0b1110111,
    funct7=0b0001010,
):
    def exec(self, state: ArchState) -> None:
        _vmatmul(state, "mxu0", self.vd, self.vs1, self.vs2, accumulate=False)


class VMATMUL_MXU1(
    MXUMatMul,
    VRType[Accumulator, WeightBuffer],
    exu=EXU.MATRIX_INNER,
    opcode=0b1110111,
    funct7=0b0001011,
):
    def exec(self, state: ArchState) -> None:
        _vmatmul(state, "mxu1", self.vd, self.vs1, self.vs2, accumulate=False)


class VMATMUL_ACC_MXU0(
    MXUMatMul,
    VRType[Accumulator, WeightBuffer],
    exu=EXU.MATRIX_SYSTOLIC,
    opcode=0b1110111,
    funct7=0b0001100,
):
    def exec(self, state: ArchState) -> None:
        _vmatmul(state, "mxu0", self.vd, self.vs1, self.vs2, accumulate=True)


class VMATMUL_ACC_MXU1(
    MXUMatMul,
    VRType[Accumulator, WeightBuffer],
    exu=EXU.MATRIX_INNER,
    opcode=0b1110111,
    funct7=0b0001101,
):
    def exec(self, state: ArchState) -> None:
        _vmatmul(state, "mxu1", self.vd, self.vs1, self.vs2, accumulate=True)


class _DMA_LOAD_CHN(ScalarComputeReg):
    def exec(self, state: ArchState) -> None:
        length = state.read_xrf(self.rs2)
        data = state.read_dram(state.read_xrf(self.rs1), length)
        state.write_vmem(state.read_xrf(self.rd), 0, data)


class DMA_LOAD_CH0(
    _DMA_LOAD_CHN, RType, exu=EXU.DMA, opcode=0b1111011, funct3=0b000, funct7=0b0000000
):
    pass


class DMA_LOAD_CH1(
    _DMA_LOAD_CHN, RType, exu=EXU.DMA, opcode=0b1111011, funct3=0b001, funct7=0b0000000
):
    pass


class DMA_LOAD_CH2(
    _DMA_LOAD_CHN, RType, exu=EXU.DMA, opcode=0b1111011, funct3=0b010, funct7=0b0000000
):
    pass


class DMA_LOAD_CH3(
    _DMA_LOAD_CHN, RType, exu=EXU.DMA, opcode=0b1111011, funct3=0b011, funct7=0b0000000
):
    pass


class DMA_LOAD_CH4(
    _DMA_LOAD_CHN, RType, exu=EXU.DMA, opcode=0b1111011, funct3=0b100, funct7=0b0000000
):
    pass


class DMA_LOAD_CH5(
    _DMA_LOAD_CHN, RType, exu=EXU.DMA, opcode=0b1111011, funct3=0b101, funct7=0b0000000
):
    pass


class DMA_LOAD_CH6(
    _DMA_LOAD_CHN, RType, exu=EXU.DMA, opcode=0b1111011, funct3=0b110, funct7=0b0000000
):
    pass


class DMA_LOAD_CH7(
    _DMA_LOAD_CHN, RType, exu=EXU.DMA, opcode=0b1111011, funct3=0b111, funct7=0b0000000
):
    pass


class _DMA_STORE_CHN(ScalarComputeReg):
    def exec(self, state: ArchState) -> None:
        length = state.read_xrf(self.rs2)
        data = state.read_vmem(state.read_xrf(self.rs1), 0, length)
        state.write_dram(state.read_xrf(self.rd), data)


class DMA_STORE_CH0(
    _DMA_STORE_CHN, RType, exu=EXU.DMA, opcode=0b1111011, funct3=0b000, funct7=0b0000001
):
    pass


class DMA_STORE_CH1(
    _DMA_STORE_CHN, RType, exu=EXU.DMA, opcode=0b1111011, funct3=0b001, funct7=0b0000001
):
    pass


class DMA_STORE_CH2(
    _DMA_STORE_CHN, RType, exu=EXU.DMA, opcode=0b1111011, funct3=0b010, funct7=0b0000001
):
    pass


class DMA_STORE_CH3(
    _DMA_STORE_CHN, RType, exu=EXU.DMA, opcode=0b1111011, funct3=0b011, funct7=0b0000001
):
    pass


class DMA_STORE_CH4(
    _DMA_STORE_CHN, RType, exu=EXU.DMA, opcode=0b1111011, funct3=0b100, funct7=0b0000001
):
    pass


class DMA_STORE_CH5(
    _DMA_STORE_CHN, RType, exu=EXU.DMA, opcode=0b1111011, funct3=0b101, funct7=0b0000001
):
    pass


class DMA_STORE_CH6(
    _DMA_STORE_CHN, RType, exu=EXU.DMA, opcode=0b1111011, funct3=0b110, funct7=0b0000001
):
    pass


class DMA_STORE_CH7(
    _DMA_STORE_CHN, RType, exu=EXU.DMA, opcode=0b1111011, funct3=0b111, funct7=0b0000001
):
    pass


class _DMA_CONFIG_CHN(DMARegUnary):
    def exec(self, state: ArchState) -> None:
        state.base = state.read_xrf(self.rs1)


class DMA_CONFIG_CH0(
    _DMA_CONFIG_CHN,
    RType,
    exu=EXU.DMA,
    opcode=0b1111111,
    funct3=0b000,
    funct7=0b0000001,
):
    pass


class DMA_CONFIG_CH1(
    _DMA_CONFIG_CHN,
    RType,
    exu=EXU.DMA,
    opcode=0b1111111,
    funct3=0b001,
    funct7=0b0000001,
):
    pass


class DMA_CONFIG_CH2(
    _DMA_CONFIG_CHN,
    RType,
    exu=EXU.DMA,
    opcode=0b1111111,
    funct3=0b010,
    funct7=0b0000001,
):
    pass


class DMA_CONFIG_CH3(
    _DMA_CONFIG_CHN,
    RType,
    exu=EXU.DMA,
    opcode=0b1111111,
    funct3=0b011,
    funct7=0b0000001,
):
    pass


class DMA_CONFIG_CH4(
    _DMA_CONFIG_CHN,
    RType,
    exu=EXU.DMA,
    opcode=0b1111111,
    funct3=0b100,
    funct7=0b0000001,
):
    pass


class DMA_CONFIG_CH5(
    _DMA_CONFIG_CHN,
    RType,
    exu=EXU.DMA,
    opcode=0b1111111,
    funct3=0b101,
    funct7=0b0000001,
):
    pass


class DMA_CONFIG_CH6(
    _DMA_CONFIG_CHN,
    RType,
    exu=EXU.DMA,
    opcode=0b1111111,
    funct3=0b110,
    funct7=0b0000001,
):
    pass


class DMA_CONFIG_CH7(
    _DMA_CONFIG_CHN,
    RType,
    exu=EXU.DMA,
    opcode=0b1111111,
    funct3=0b111,
    funct7=0b0000001,
):
    pass


class _DMA_WAIT_CHN(Nullary):
    imm = 1

    def exec(self, state: ArchState) -> None:
        pass


class DMA_WAIT_CH0(
    _DMA_WAIT_CHN, RType, exu=EXU.DMA, opcode=0b1111111, funct3=0b000, funct7=0b0000001
):
    pass


class DMA_WAIT_CH1(
    _DMA_WAIT_CHN, RType, exu=EXU.DMA, opcode=0b1111111, funct3=0b001, funct7=0b0000001
):
    pass


class DMA_WAIT_CH2(
    _DMA_WAIT_CHN, RType, exu=EXU.DMA, opcode=0b1111111, funct3=0b010, funct7=0b0000001
):
    pass


class DMA_WAIT_CH3(
    _DMA_WAIT_CHN, RType, exu=EXU.DMA, opcode=0b1111111, funct3=0b011, funct7=0b0000001
):
    pass


class DMA_WAIT_CH4(
    _DMA_WAIT_CHN, RType, exu=EXU.DMA, opcode=0b1111111, funct3=0b100, funct7=0b0000001
):
    pass


class DMA_WAIT_CH5(
    _DMA_WAIT_CHN, RType, exu=EXU.DMA, opcode=0b1111111, funct3=0b101, funct7=0b0000001
):
    pass


class DMA_WAIT_CH6(
    _DMA_WAIT_CHN, RType, exu=EXU.DMA, opcode=0b1111111, funct3=0b110, funct7=0b0000001
):
    pass


class DMA_WAIT_CH7(
    _DMA_WAIT_CHN, RType, exu=EXU.DMA, opcode=0b1111111, funct3=0b111, funct7=0b0000001
):
    pass
