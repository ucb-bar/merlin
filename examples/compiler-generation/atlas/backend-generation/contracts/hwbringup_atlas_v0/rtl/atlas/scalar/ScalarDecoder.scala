/*
ScalarDecoder.scala — Instruction decoder.
Handles RV32I formats + custom VR/VI/VLS tensor formats.
*/

package atlas.scalar

import chisel3._
import chisel3.util._

class DecodedInstr extends Bundle {
  val legal    = Bool()
  val alu_fn   = UInt(4.W)
  val br_type  = UInt(3.W)
  val op1_sel  = UInt(2.W)
  val op2_sel  = UInt(2.W)
  val imm_sel  = UInt(3.W)
  val rd_wen   = Bool()
  val is_jal   = Bool()
  val is_jalr  = Bool()
  val csr_cmd  = UInt(3.W)
  val dma_cmd  = UInt(3.W)
  val mxu_cmd  = UInt(4.W)
  val mxu_sel  = UInt(1.W)
  val vpu_cmd  = UInt(5.W)
  val xlu_cmd  = UInt(1.W)
  val mem_cmd  = UInt(4.W)   // scalar load/store type
  val lsu_cmd  = UInt(2.W)   // vload / vstore

  // Standard RV32I field extraction
  val rd       = UInt(5.W)
  val rs1      = UInt(5.W)
  val rs2      = UInt(5.W)
  val funct7   = UInt(7.W)
  val funct3   = UInt(3.W)
  val csr_addr = UInt(12.W)
  val imm      = SInt(32.W)

  // VR format: vd[12:7], vs1[18:13], vs2[24:19]
  val vd       = UInt(6.W)
  val vs1      = UInt(6.W)
  val vs2      = UInt(6.W)

  // VI format: imm16[31:16]
  val vi_imm   = UInt(16.W)

  // Derived
  val is_csr   = Bool()
  val illegal  = Bool()
  val is_seli  = Bool()
  val is_seld  = Bool()
  val is_mem_load  = Bool()   // any scalar load (LB/LH/LW/LBU/LHU/SELD)
  val is_mem_store = Bool()   // any scalar store (SB/SH/SW)
  val is_lsu       = Bool()   // vload or vstore
  val is_ecall     = Bool()   // environment call (halt)
  val is_ebreak    = Bool()   // breakpoint (halt)
  val is_delay     = Bool()   // frontend delay (stall for imm cycles)
}

/** Combinational decoder for the scalar ISA plus Atlas custom extensions. */
class ScalarDecoder extends Module {
  val io = IO(new Bundle {
    val instr   = Input(UInt(32.W))
    val decoded = Output(new DecodedInstr)
  })

  import ScalarISA._

  // Standard RV32I fields
  val rd     = io.instr(11, 7)
  val rs1    = io.instr(19, 15)
  val rs2    = io.instr(24, 20)
  val funct7 = io.instr(31, 25)
  val funct3 = io.instr(14, 12)

  // VR format fields (all register-register tensor ops, including
  // unary VR instructions such as VSQUARE.BF16 / VCUBE.BF16).
  val vr_vd  = io.instr(12, 7)   // 6 bits
  val vr_vs1 = io.instr(18, 13)  // 6 bits
  val vr_vs2 = io.instr(24, 19)  // 6 bits

  // VI format
  val vi_imm = io.instr(31, 16)  // 16 bits

  // RV32I Immediate generation (sign-extended to 32 bits)
  val imm_i = Cat(Fill(20, io.instr(31)), io.instr(31, 20)).asSInt
  val imm_s = Cat(Fill(20, io.instr(31)), io.instr(31, 25), io.instr(11, 7)).asSInt
  val imm_b = Cat(Fill(20, io.instr(31)), io.instr(7), io.instr(30, 25),
                  io.instr(11, 8), 0.U(1.W)).asSInt
  val imm_u = Cat(io.instr(31, 12), 0.U(12.W)).asSInt
  val imm_j = Cat(Fill(12, io.instr(31)), io.instr(19, 12), io.instr(20),
                  io.instr(30, 21), 0.U(1.W)).asSInt
  val imm_z = Cat(0.U(27.W), rs1).asSInt

  // VLS format: imm12 << 5 (word offset for vload/vstore)
  //   imm12 = instr[31:20], shift left by 5 to give word address offset
  val imm_vls = Cat(Fill(15, io.instr(31)), io.instr(31, 20), 0.U(5.W)).asSInt

  // Decode table lookup
  val cs = ListLookup(io.instr, IDecode.default, IDecode.table)

  val dec = Wire(new DecodedInstr)
  dec.legal    := cs(0).asBool
  dec.alu_fn   := cs(1)
  dec.br_type  := cs(2)
  dec.op1_sel  := cs(3)
  dec.op2_sel  := cs(4)
  dec.imm_sel  := cs(5)
  dec.rd_wen   := cs(6).asBool
  dec.is_jal   := cs(7).asBool
  dec.is_jalr  := cs(8).asBool
  dec.csr_cmd  := cs(9)
  dec.dma_cmd  := cs(10)
  dec.mxu_cmd  := cs(11)
  dec.mxu_sel  := cs(12)
  dec.vpu_cmd  := cs(13)
  dec.xlu_cmd  := cs(14)
  dec.mem_cmd  := cs(15)
  dec.lsu_cmd  := cs(16)

  dec.rd      := rd
  dec.rs1     := rs1
  dec.rs2     := rs2
  dec.funct7  := funct7
  dec.funct3  := funct3
  dec.csr_addr := io.instr(31, 20)
  dec.vd      := vr_vd
  dec.vs1     := vr_vs1
  dec.vs2     := vr_vs2
  dec.vi_imm  := vi_imm

  dec.imm := MuxLookup(dec.imm_sel, imm_i)(Seq(
    IMM_I   -> imm_i,
    IMM_S   -> imm_s,
    IMM_B   -> imm_b,
    IMM_U   -> imm_u,
    IMM_J   -> imm_j,
    IMM_Z   -> imm_z,
    IMM_VLS -> imm_vls
  ))

  dec.is_csr  := dec.csr_cmd =/= CSR_NONE
  dec.illegal := !dec.legal

  // seli: opcode=0000011, funct3=111
  dec.is_seli := dec.legal &&
    io.instr(6, 0) === "b0000011".U && funct3 === "b111".U

  // seld: opcode=0000011, funct3=110  (decoded as MEM_SELD)
  dec.is_seld := dec.mem_cmd === MEM_SELD

  // Derived convenience flags
  dec.is_mem_load  := dec.mem_cmd =/= MEM_NONE &&
    (dec.mem_cmd === MEM_LB  || dec.mem_cmd === MEM_LH  || dec.mem_cmd === MEM_LW ||
     dec.mem_cmd === MEM_LBU || dec.mem_cmd === MEM_LHU || dec.mem_cmd === MEM_SELD)
  dec.is_mem_store := dec.mem_cmd === MEM_SB || dec.mem_cmd === MEM_SH || dec.mem_cmd === MEM_SW
  dec.is_lsu       := dec.lsu_cmd =/= LSU_NONE

  // ecall:  fixed encoding 0x00000073
  dec.is_ecall  := dec.legal && io.instr === "h00000073".U
  // ebreak: fixed encoding 0x00100073
  dec.is_ebreak := dec.legal && io.instr === "h00100073".U
  // delay:  opcode=1100111, funct3=001
  dec.is_delay  := dec.legal &&
    io.instr(6, 0) === "b1100111".U && funct3 === "b001".U

  io.decoded := dec
}
