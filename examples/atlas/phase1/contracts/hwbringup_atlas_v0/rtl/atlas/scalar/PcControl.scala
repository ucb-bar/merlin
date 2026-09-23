/*
PcControl.scala
Program counter control for the scalar pipeline frontend.

The PC is a word index (not byte address) starting at 0.
It directly addresses IMEM words. Increments by 1 per cycle.

Implements 1 architectural delay slot:
  Cycle N  : branch/jump evaluated in execute (at branch_pc).
             Instruction at branch_pc+1 is already in-flight (delay slot).
             Fetch redirects to the target for the next cycle.
  Cycle N+1: delay slot executes. Target instruction is in-flight.
  Cycle N+2: target instruction executes.
*/

package atlas.scalar

import chisel3._
import chisel3.util._

/** Program-counter controller with architectural delay-slot handling.
  *
  * @param resetPC  Initial IMEM word index after reset.
  */
class PcControl(resetPC: Int = 0) extends Module {
  val io = IO(new Bundle {
    val redirect        = Input(Bool())
    val redirect_target = Input(UInt(32.W))
    val stall           = Input(Bool())
    val halted          = Input(Bool())
    val restart         = Input(Bool())
    val pc              = Output(UInt(32.W))   // word index for IMEM fetch
    val s1_pc           = Output(UInt(32.W))   // PC of instruction in execute
    val s1_valid        = Output(Bool())
    val in_delay_slot   = Output(Bool())
  })

  val pc_reg       = RegInit(resetPC.U(32.W))
  val fetch_pc_reg = RegInit(resetPC.U(32.W))
  val s1_valid_reg = RegInit(false.B)

  val delay_slot_pending = RegInit(false.B)

  when(io.restart) {
    pc_reg        := resetPC.U
    fetch_pc_reg  := resetPC.U
    s1_valid_reg  := false.B
    delay_slot_pending := false.B
  }.elsewhen(io.halted) {
    s1_valid_reg := false.B
  }.elsewhen(io.stall) {
    // All registers hold their values.  Do NOT overwrite fetch_pc_reg
    // with pc_reg here: by the time a stall takes effect, pc_reg has
    // already advanced past the stalled instruction.  Copying it would
    // corrupt s1_pc, breaking PC-relative branches after DELAY stalls.
  }.otherwise {
    s1_valid_reg := true.B
    fetch_pc_reg := pc_reg

    when(io.redirect && !delay_slot_pending) {
      // Branch evaluated. The in-flight fetch (pc_reg) is the only
      // architecturally visible delay slot, so start fetching the target now.
      delay_slot_pending := true.B
      pc_reg             := io.redirect_target
    }.elsewhen(delay_slot_pending) {
      // The delay-slot instruction is now in execute and the branch target is
      // in-flight, so resume sequential fetch from target+1.
      delay_slot_pending := false.B
      pc_reg             := pc_reg + 1.U
    }.otherwise {
      pc_reg := pc_reg + 1.U
    }
  }

  io.pc            := pc_reg
  io.s1_pc         := fetch_pc_reg
  io.s1_valid      := s1_valid_reg
  io.in_delay_slot := delay_slot_pending
}
