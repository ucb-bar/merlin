package atlas.sa

import chisel3._
import chisel3.util._

import sp26FPUnits._
import sp26FPUnits.hardfloat._

import atlas.common.PEArchitecture

/**
  * One processing element in systolic array. 
  * 
  * Computes a multiply-add by doing (act * weight) + addend. Different
  * architectures can be specified. See [[atlas.common.PEArchitecture]]
  * for more information.
  * 
  * @param peArch The desired hardware architecture of this processing element.
  * 
  */
class PE(peArch: PEArchitecture, fmaStages: Int = 0) extends Module {

  val mulW: Width = peArch.getMulWidth().W
  val addW: Width = peArch.getAddWidth().W

  val io = IO(new Bundle {

    /** An activation element. */
    val act = Input(UInt(mulW))

    /** A weight element from w0. */
    val weight0 = Input(UInt(mulW))

    /** A weight element from w1. */
    val weight1 = Input(UInt(mulW))

    /** A selector signal to use w0 or w1. */
    val weightReadSel = Input(Bool())

    /** An addend element. */
    val addend = Input(UInt(addW))

    /** An output activation element. */
    val actQ = Output(UInt(mulW))

    /** A combinational output MAC element. */
    val mac = Output(UInt(addW))

    /** An output selector signal to use w0 or w1. */
    val weightReadSelQ = Output(Bool())
	})

  val weight = Mux(io.weightReadSel, io.weight1, io.weight0)

  peArch match {
    case PEArchitecture.HardFloatFMA => 
      val BF16 = AtlasFPType.BF16

      val mac = Module(new MulAddRecFN(BF16.expWidth, BF16.sigWidth))
      mac.io.op := 0.U(2.W) // Bits to specify fused multiply-add operation.
      mac.io.a := io.act
      mac.io.b := weight
      mac.io.c := io.addend
      mac.io.roundingMode := consts.round_near_even
      mac.io.detectTininess := consts.tininess_afterRounding

      io.mac := mac.io.out
    
    case PEArchitecture.FP32Addition =>
      val BF16 = AtlasFPType.BF16
      val FP32 = AtlasFPType.FP32

      // Multiply in BF16
      val mul = Module(new MulRecFN(BF16.expWidth, BF16.sigWidth))
      mul.io.a := io.act
      mul.io.b := weight
      mul.io.roundingMode := consts.round_near_even
      mul.io.detectTininess := consts.tininess_afterRounding

      // Widen product BF16 -> FP32
      val widen = Module(new RecFNToRecFN(BF16.expWidth, BF16.sigWidth, FP32.expWidth, FP32.sigWidth))
      widen.io.in := mul.io.out
      widen.io.roundingMode := consts.round_near_even
      widen.io.detectTininess := consts.tininess_afterRounding
      widen.io.exceptionFlags := DontCare

      // Add in FP32
      val add = Module(new AddRecFN(FP32.expWidth, FP32.sigWidth))
      add.io.subOp := false.B
      add.io.a := widen.io.out
      add.io.b := io.addend
      add.io.roundingMode := consts.round_near_even
      add.io.detectTininess := consts.tininess_afterRounding

      io.mac := add.io.out

    case PEArchitecture.CustomFMA =>
      // `fmaStages` pipeline registers inside the FP8 mul->add->round datapath.
      // The mac output therefore has `fmaStages` cycles of latency; the systolic
      // reskew in SystolicArray accounts for this (bit-exact, extra latency only).
      val fma = Module(new E4M3FMA(fmaStages))
      fma.io.a := io.act(7, 0)
      fma.io.b := weight(7, 0)
      fma.io.addend16 := io.addend

      io.mac := fma.io.out16
  }

  io.actQ := RegNext(io.act)
  io.weightReadSelQ := RegNext(io.weightReadSel)
}
