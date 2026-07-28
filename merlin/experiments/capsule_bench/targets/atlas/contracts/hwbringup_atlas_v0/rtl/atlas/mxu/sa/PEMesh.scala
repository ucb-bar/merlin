package atlas.sa

import chisel3._
import chisel3.util._
import atlas.common._

/** Two-dimensional mesh of processing elements for the systolic-array MXU.
  *
  * Feeds activations across each row and partial sums down each column so the
  * PE grid performs one matrix multiply step per cycle once the wavefront is
  * full.
  *
  * @param rows    Number of PE rows in the mesh.
  * @param cols    Number of PE columns in the mesh.
  * @param peArch  Processing-element architecture to instantiate at each node.
  */
class PEMesh(rows: Int, cols: Int, peArch: PEArchitecture, fmaStages: Int = 0) extends Module {

  val mulW: Width = peArch.getMulWidth().W
  val addW: Width = peArch.getAddWidth().W

  val io = IO(new Bundle {
    val actVec = Input(Vec(rows, UInt(mulW)))
    val weights0 = Input(Vec(rows, Vec(cols, UInt(mulW))))
    val weights1 = Input(Vec(rows, Vec(cols, UInt(mulW))))
    val weightReadSelVec = Input(Vec(rows, Bool()))
    val addendVec = Input(Vec(cols, UInt(addW)))

    val outVec = Output(Vec(cols, UInt(addW)))
  })

  val pes = Seq.fill(rows, cols)(Module(new PE(peArch, fmaStages)))

  for (i <- 0 until rows) {
    pes(i)(0).io.act := io.actVec(i)
    pes(i)(0).io.weightReadSel := io.weightReadSelVec(i)
  }
  
  for (j <- 0 until cols) {
    pes(0)(j).io.addend := io.addendVec(j)
    io.outVec(j) := pes(rows - 1)(j).io.mac
  }

  for (i <- 0 until rows; j <- 0 until cols) {
    pes(i)(j).suggestName(s"pe_${i}_${j}")
    pes(i)(j).io.weight0 := io.weights0(i)(j)
    pes(i)(j).io.weight1 := io.weights1(i)(j)
  }

  // Connect PEs to each other
  for (i <- 0 until rows; j <- 0 until cols - 1) {
    pes(i)(j + 1).io.act := pes(i)(j).io.actQ
    pes(i)(j + 1).io.weightReadSel := pes(i)(j).io.weightReadSelQ
  }
  for (i <- 0 until rows - 1; j <- 0 until cols) {
    pes(i + 1)(j).io.addend := RegNext(pes(i)(j).io.mac)
  }
}