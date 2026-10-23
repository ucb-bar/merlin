// This wrapper instantiates FireSim's existing timing core, never a port of its prices.
package merlin.reference

import chisel3._
import chisel3.stage.ChiselStage
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.diplomacy.AddressSet
import firesim.lib.nasti.NastiParameters
import midas.models.{AXI4EdgeSummary, BaseConfig, FasedAXI4Edge}
import junctions.NastiKey
import org.json4s._
import org.json4s.native.JsonMethods.parse

class FasedTimingStandalone(cfg: BaseConfig, output: String)(implicit p: Parameters) extends Module {
  private val model = cfg.elaborate()
  def leaves(name: String, data: Data): Seq[(String, Data)] = data match {
    case record: Record => record.elements.toSeq.flatMap { case (key, value) => leaves(s"${name}_${key}", value) }
    case vector: Vec[_] => vector.zipWithIndex.flatMap { case (value, index) => leaves(s"${name}_${index}", value) }.toSeq
    case leaf => Seq(name.stripPrefix("_") -> leaf)
  }
  private val leafPorts = leaves("", model.io)
  val io = IO(new Record {
    val elements = scala.collection.immutable.ListMap(leafPorts.map { case (name, data) => name -> (chisel3.reflect.DataMirror.directionOf(data) match {
      case chisel3.ActualDirection.Input => Input(chiselTypeOf(data))
      case chisel3.ActualDirection.Output => Output(chiselTypeOf(data))
      case other => throw new IllegalArgumentException(s"unsupported leaf direction: $other")
    }) }: _*)
  })
  leafPorts.foreach { case (name, data) => io.elements(name) <> data }
  val defaults = model.io.mmReg.getDefaults().map { case (name, value) => s"+mm_${name}_0=$value" }
  java.nio.file.Files.write(java.nio.file.Paths.get(output, "runtime-defaults.conf"),
    (defaults.mkString("\n") + "\n").getBytes(java.nio.charset.StandardCharsets.UTF_8))
}

object FasedTimingStandaloneMain {
  def main(args: Array[String]): Unit = {
    require(args.length == 2, "expected memory selection JSON and output directory")
    implicit val formats: Formats = DefaultFormats
    val input = scala.io.Source.fromFile(args(0))
    val doc = try parse(input.mkString) finally input.close()
    require((doc \ "schema").extract[String] == "merlin.memory_reference_selection.v1")
    val axi = doc \ "axi"
    def number(name: String): Int = (axi \ name).extract[Int]
    def optional(name: String): Option[Int] = (axi \ name).extract[Option[Int]]
    def big(name: String): BigInt = (axi \ name).extract[BigInt]
    val cls = Class.forName((doc \ "configuration").extract[String])
    val config = cls.getConstructor().newInstance().asInstanceOf[org.chipsalliance.cde.config.Config]
    implicit val p: Parameters = config.toInstance.alterPartial {
      case NastiKey => NastiParameters(number("data_bits"), number("address_bits"), number("id_bits"))
      case FasedAXI4Edge => Some(AXI4EdgeSummary(
        number("max_read_bytes"), number("max_write_bytes"), optional("id_reuse"),
        optional("max_flight"), AddressSet.misaligned(big("address_base"), big("address_size"))))
    }
    val cfg = p(firesim.configs.MemModelKey)
    (new ChiselStage).emitVerilog(new FasedTimingStandalone(cfg, args(1)), Array("--target-dir", args(1)))
  }
}
