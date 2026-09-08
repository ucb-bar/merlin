"""Atlas instruction encoders, transcribed from the shipped ISA format definitions."""


def i_type(opcode, funct3, rd, rs1, imm):
    return ((imm & 0xFFF) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def u_type(opcode, rd, imm):
    return ((imm & 0xFFFFF) << 12) | (rd << 7) | opcode


def r_type(opcode, funct3, funct7, rd, rs1, rs2):
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def b_type(opcode, funct3, rs1, rs2, imm):
    value = imm & 0x1FFF
    return (((value >> 12) & 1) << 31) | (((value >> 5) & 0x3F) << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (((value >> 1) & 0xF) << 8) | (((value >> 11) & 1) << 7) | opcode


def j_type(opcode, rd, imm):
    value = imm & 0x1FFFFF
    return (((value >> 20) & 1) << 31) | (((value >> 1) & 0x3FF) << 21) | (((value >> 11) & 1) << 20) | (((value >> 12) & 0xFF) << 12) | (rd << 7) | opcode


def s_type(opcode, funct3, rs1, rs2, imm):
    value = imm & 0xFFF
    return ((value >> 5) << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | ((value & 31) << 7) | opcode


def vls(opcode, funct2, vd, rs1, imm=0):
    return ((imm & 0xFFF) << 20) | (rs1 << 15) | (funct2 << 13) | (vd << 7) | opcode


def vr(opcode, funct7, vd, vs1=0, vs2=0):
    return (funct7 << 25) | (vs2 << 19) | (vs1 << 13) | (vd << 7) | opcode


def vi(opcode, funct3, vd, imm):
    return ((imm & 0xFFFF) << 16) | (funct3 << 13) | (vd << 7) | opcode


def li(rd, value):
    value &= 0xFFFFFFFF
    upper = ((value + 0x800) >> 12) & 0xFFFFF
    lower = value & 0xFFF
    return [u_type(0x37, rd, upper), i_type(0x13, 0, rd, rd, lower)]


def addi(rd, rs1, imm): return i_type(0x13, 0, rd, rs1, imm)
def andi(rd, rs1, imm): return i_type(0x13, 7, rd, rs1, imm)
def add(rd, rs1, rs2): return r_type(0x33, 0, 0, rd, rs1, rs2)
def or_(rd, rs1, rs2): return r_type(0x33, 6, 0, rd, rs1, rs2)
def srl(rd, rs1, rs2): return r_type(0x33, 5, 0, rd, rs1, rs2)
def bne(rs1, rs2, imm): return b_type(0x63, 1, rs1, rs2, imm)
def beq(rs1, rs2, imm): return b_type(0x63, 0, rs1, rs2, imm)
def jal(rd, imm): return j_type(0x6F, rd, imm)
def slli(rd, rs1, shamt): return i_type(0x13, 1, rd, rs1, shamt)
def lbu(rd, rs1, imm=0): return i_type(0x03, 4, rd, rs1, imm)
def lhu(rd, rs1, imm=0): return i_type(0x03, 5, rd, rs1, imm)
def lw(rd, rs1, imm=0): return i_type(0x03, 2, rd, rs1, imm)
def sb(rs2, rs1, imm=0): return s_type(0x23, 0, rs1, rs2, imm)
def sh(rs2, rs1, imm=0): return s_type(0x23, 1, rs1, rs2, imm)
def sw(rs2, rs1, imm=0): return s_type(0x23, 2, rs1, rs2, imm)
def delay(cycles): return i_type(0x67, 1, 0, 0, cycles)
def seli(ed, imm): return i_type(0x03, 7, ed, 0, imm)
def dma_config(rs1, channel=0): return r_type(0x7F, channel, 0, 0, rs1, 0)
def dma_load(vmem_rd, dram_rs1, size_rs2, channel=0): return r_type(0x7B, channel, 0, vmem_rd, dram_rs1, size_rs2)
def dma_store(dram_rd, vmem_rs1, size_rs2, channel=0): return r_type(0x7B, channel, 1, dram_rd, vmem_rs1, size_rs2)
def dma_wait(channel=0): return r_type(0x7F, channel, 1, 0, 0, 0)
def vload(vd, base, imm=0): return vls(0x07, 0, vd, base, imm)
def vstore(vd, base, imm=0): return vls(0x07, 1, vd, base, imm)
def vli_all(vd, imm): return vi(0x5F, 0, vd, imm)
def transpose(vd, vs1): return vr(0x6B, 0, vd, vs1)
def weight_push(vd, vs1, mxu=0): return vr(0x77, mxu, vd, vs1)
def matmul(vd, vs1, vs2, accumulate=False, mxu=0): return vr(0x77, 10 + mxu + (2 if accumulate else 0), vd, vs1, vs2)
def pop_bf16(vd, acc, mxu=0): return vr(0x77, 8 + mxu, vd, 0, acc)
def vunpack(vd, src, scale=0): return vr(0x57, 0x45, vd, scale, src)
def vpack(vd, src, scale=0): return vr(0x57, 0x44, vd, scale, src)
def vrelu(vd, src): return vr(0x57, 0x48, vd, src)
def vadd(vd, lhs, rhs): return vr(0x57, 0x00, vd, lhs, rhs)
def vsub(vd, lhs, rhs): return vr(0x57, 0x02, vd, lhs, rhs)
def vmul(vd, lhs, rhs): return vr(0x57, 0x03, vd, lhs, rhs)
def vmax(vd, lhs, rhs): return vr(0x57, 0x06, vd, lhs, rhs)
def vredsum(vd, src): return vr(0x57, 0x01, vd, src)
def vredsum_row(vd, src): return vr(0x57, 0x21, vd, src)
def vredmax_row(vd, src): return vr(0x57, 0x26, vd, src)
def vrecip(vd, src): return vr(0x57, 0x41, vd, src)
def vexp(vd, src): return vr(0x57, 0x42, vd, src)
def vsqrt(vd, src): return vr(0x57, 0x4D, vd, src)
def vsquare(vd, src): return vr(0x57, 0x4E, vd, src)
def ecall(): return 0x00000073


def validate_pair_banks(words):
    """Mirror Atlas RTL's even-base contract for every emitted MREG pair.

    ``ScalarDecoder`` defines VR/VI ``vd``, ``vs1``, and ``vs2`` as the six-bit
    instruction fields [12:7], [18:13], and [24:19]. ``VectorEngineTop`` then
    requires even primary/secondary pair reads and pair writes.  FP8 pack writes
    one bank, FP8 unpack does not perform the normal primary-pair read, and
    VLI_COL/VLI_ONE write one bank. MXU BF16 pops also write a pair. VLOAD,
    VSTORE, and XLU transpose are single-bank operations and deliberately absent.
    """

    def require_even(index, word, role, bank):
        if bank & 1:
            raise ValueError(
                f"Atlas pair-{role} instruction {index} targets odd MREG bank "
                f"{bank}: 0x{word:08x}"
            )

    for index, word in enumerate(words):
        opcode = word & 0x7F
        destination = (word >> 7) & 0x3F
        primary = (word >> 13) & 0x3F
        secondary = (word >> 19) & 0x3F
        function = (word >> 25) & 0x7F
        if opcode == 0x57:
            if function != 0x45:  # FP8 unpack has no normal primary-pair read.
                require_even(index, word, "read-primary", secondary if function == 0x44 else primary)
            if function in (0x00, 0x02, 0x03, 0x04, 0x06):
                require_even(index, word, "read-secondary", secondary)
            if function != 0x44:  # FP8 pack writes one bank.
                require_even(index, word, "write", destination)
        elif opcode == 0x5F:
            vli_kind = (word >> 13) & 0x7
            if vli_kind not in (0, 1, 2, 3):
                raise ValueError(f"unknown Atlas VLI kind {vli_kind} at instruction {index}")
            if vli_kind not in (2, 3):  # VLI_ALL/ROW write pairs; COL/ONE do not.
                require_even(index, word, "write", destination)
        elif opcode == 0x77 and function in (8, 9):
            require_even(index, word, "write", destination)
