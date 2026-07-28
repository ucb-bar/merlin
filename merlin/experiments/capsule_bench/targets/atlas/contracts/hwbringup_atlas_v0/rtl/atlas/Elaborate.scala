// ============================================================================
// Elaborate.scala — Standalone elaboration entry point for atlas compute
// datapath modules (baseline RTL generation for modeling).
//
// This is NOT the (broken) chipyard/Elaborate.scala full-tile flow — it only
// elaborates plain Chisel Modules that the Mill `atlas` module actually
// compiles (src/main/scala/atlas), so it needs no rocketchip/Chipyard.
//
// Usage:
//   ./mill -i atlas.runMain atlas.Elaborate <module> --target-dir <dir> [firtool opts...]
//   <module> ∈ { systolic_array | ipt | vector_engine | all }
//
// Emits <Module>.sv (+ split files) via CIRCT firtool, and a .fir (CHIRRTL)
// alongside for each selected module.
// ============================================================================

package atlas

import chisel3._
import circt.stage.ChiselStage
import java.io.PrintWriter
import java.nio.file.{Files, Paths}

import atlas.common._
import atlas.sa.SystolicArray
import atlas.ipt.InnerProductTrees
import atlas.vector.VectorEngineTop

object Elaborate extends App {

  // ── Parse a --target-dir out of args; everything else is passed to CIRCT ──
  val targetDirIdx = args.indexOf("--target-dir")
  val targetDir =
    if (targetDirIdx >= 0 && targetDirIdx + 1 < args.length) args(targetDirIdx + 1)
    else "modeling/artifacts/atlas/rtl/gen"

  val which = args.headOption.getOrElse("all")

  // Args forwarded to the ChiselStage / firtool CLI (drop our custom flags).
  val passThrough: Array[String] =
    args
      .drop(1) // module selector
      .grouped(1)
      .flatten
      .toArray
      .filterNot(_ == "--target-dir")
      .filterNot(_ == which)

  Files.createDirectories(Paths.get(targetDir))
  val firrtlDir = targetDir.replace("/rtl/gen", "/firrtl")
  Files.createDirectories(Paths.get(firrtlDir))

  // firtool options: split modules into one file each, keep it readable.
  val firtoolOpts = Array(
    "-disable-all-randomization",
    "-strip-debug-info",
  )

  def emit(name: String, gen: => chisel3.RawModule, extraFirtoolOpts: Array[String] = Array()): Unit = {
    println(s"[atlas.Elaborate] elaborating $name -> $targetDir")
    // SystemVerilog (firtool-lowered).
    ChiselStage.emitSystemVerilogFile(
      gen,
      Array("--target-dir", targetDir),
      firtoolOpts ++ extraFirtoolOpts
    )
    // CHIRRTL (.fir) for the same design.
    val fir = ChiselStage.emitCHIRRTL(gen)
    val firPath = Paths.get(firrtlDir, s"$name.fir")
    val pw = new PrintWriter(firPath.toFile)
    try pw.write(fir)
    finally pw.close()
    println(s"[atlas.Elaborate] wrote $firPath (${fir.length} chars)")
  }

  // Default params are the arch defaults: 32x32 array, 16-lane VPU, 64 mregs.
  def doSystolic(): Unit =
    emit("SystolicArray", new SystolicArray(SystolicArrayParams()))

  // Pipelined FP8-FMA variants: `fmaStages` extra pipeline registers in the PE
  // mul->add->round datapath (bit-exact, extra latency). Same top module name so
  // it drops into the same ORFS config; emitted to a distinct target dir.
  def doSystolicPipe(stages: Int): Unit =
    // disallowLocalVariables: emit registered combinational cones as explicit
    // wires + simple `always` blocks instead of inlined `automatic` locals,
    // which the ORFS yosys read_verilog canonicalize step cannot parse.
    emit("SystolicArray", new SystolicArray(SystolicArrayParams(fmaStages = stages)),
      Array("--lowering-options=disallowLocalVariables"))

  def doIpt(): Unit =
    emit("InnerProductTrees", new InnerProductTrees(InnerProductTreeParams()))

  def doVector(): Unit =
    emit("VectorEngineTop", new VectorEngineTop(VpuParams(), MregParams()))

  which match {
    case "systolic_array"       => doSystolic()
    case "systolic_array_pipe1" => doSystolicPipe(1)
    case "systolic_array_pipe2" => doSystolicPipe(2)
    case "ipt"            => doIpt()
    case "vector_engine"  => doVector()
    case "all" =>
      doSystolic()
      doIpt()
      doVector()
    case other =>
      System.err.println(s"[atlas.Elaborate] unknown module '$other'; use " +
        "systolic_array | ipt | vector_engine | all")
      sys.exit(2)
  }

  println("[atlas.Elaborate] done.")
}
