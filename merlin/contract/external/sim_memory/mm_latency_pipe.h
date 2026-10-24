// A cycle-faithful copy of FireSim's FASED latency-bandwidth pipe, for chipyard-style simulator harnesses.
//
// WHY. An RTL simulator and a FireSim board can elaborate the same SoC and still disagree on every cycle
// count that reaches DRAM, because they bind the SoC's AXI memory port to different models. A chipyard
// Verilator or GSIM build answers through testchipip's mm_magic_t: each read the cycle after it is
// accepted, any number outstanding. A FireSim bitstream whose platform config carries FASED's default
// memory model (WithDefaultMemModel -> LatencyPipeConfig) answers each read a fixed number of target cycles
// after acceptance, with a bounded number outstanding. So every last-level-cache miss costs about one
// cycle on the first and about thirty on the second, and a schedule search run on the simulator ranks
// latency-exposed work as cheaper than the board does. This model removes that one difference: run under
// it, the simulator's memory port behaves at its AXI boundary as the FASED pipe does, cycle for cycle.
//
// WHAT IT REPRODUCES. firesim sim/midas/src/main/scala/midas/models/dram/ at b084672c -- TimingModel.scala
// (SplitTransactionModel), LatencyBandwidthPipe.scala, Util.scala (AXI4Releaser), EgressUnit.scala
// (egressUnitDelay):
//
//   * AR is accepted while fewer than max_reads reads are outstanding (acceptance until the last R beat)
//   * AW is accepted while fewer than max_writes writes are outstanding (acceptance until the B)
//   * W is accepted while fewer than max_writes COMPLETE W bursts await their B -- before, with or after
//     the burst's AW, as FASED accepts it
//   * a read's first R beat is valid exactly read_latency cycles after its AR was accepted; its beats
//     follow one per cycle and contiguously, and reads are answered in acceptance order across ids,
//     back to back
//   * a write's B is valid exactly write_latency cycles after the later of (its AW accepted, its last W
//     beat accepted), and writes are answered in that order
//
// Address, bank and row play no part in the timing, because they play none in the FASED pipe. This is a
// copy of that MODEL, not of DRAM: it makes a simulator agree with a board that runs the same model, and
// it says nothing about a chip with a real memory controller.
//
// Data follows mm_magic_t: read data is sampled when the AR is accepted, and a write lands when its W beat
// is paired with its AW.
//
// CYCLE ACCOUNTING. The harness calls tick() once per memory-port clock and registers the outputs it reads
// afterwards for the NEXT cycle, as testchipip's SimDRAM does. `cycle` is the index of the cycle whose
// outputs are being presented, so a request accepted in cycle t is answered in cycle t + latency. That is
// FASED's releaseCycle = latency + t - egressUnitDelay followed by the one-entry AXI4Releaser register.
//
// It is OFF unless asked for. Knobs, as plusargs or, when a plusarg is absent, as environment variables
// (MERLIN_MEM_LATENCY_PIPE, MERLIN_MEM_LATENCY_PIPE_<KNOB>):
//   +mem_latency_pipe=1                   enables the model (0 or absent: off)
//   +mem_latency_pipe_read_latency=<n>    cycles from AR acceptance to the first R beat     (n >= 1)
//   +mem_latency_pipe_write_latency=<n>   cycles from AW+W completion to the B              (n >= 1)
//   +mem_latency_pipe_max_reads=<n>       reads outstanding before AR backpressures         (n >= 1)
//   +mem_latency_pipe_max_writes=<n>      writes outstanding before AW/W backpressure       (n >= 1)
// The defaults (30, 30, 16, 16) are FASED's own: the RuntimeSetting defaults of LatencyBandwidthPipe.scala
// and WithDefaultMemModel's BaseParams. They exist for a hand run. A caller reproducing a particular
// bitstream passes all four from that bitstream's pinned memory_model and never relies on them.
#ifndef MERLIN_MM_LATENCY_PIPE_H
#define MERLIN_MM_LATENCY_PIPE_H

#include <cstdint>
#include <deque>
#include <vector>

#include "mm.h"

struct mm_latency_pipe_cfg_t {
  bool enabled = false;
  uint64_t read_latency = 30;
  uint64_t write_latency = 30;
  uint64_t max_reads = 16;
  uint64_t max_writes = 16;
};

// Parse the knobs out of an argv (plusargs, then environment). Unknown arguments are ignored; a malformed
// or out-of-range value is a hard error, since a timing run that silently fell back to another setting
// would report the wrong device's cycle counts.
mm_latency_pipe_cfg_t mm_latency_pipe_parse(int argc, const char *const *argv);

class mm_latency_pipe_t : public mm_t {
 public:
  mm_latency_pipe_t(const mm_latency_pipe_cfg_t &cfg, size_t mem_base, size_t mem_sz, size_t word_sz,
                    size_t line_sz, backing_data_t &dat);

  // Each output is a function of the state at the START of a cycle, as FASED's are of its registers.
  bool ar_ready() override { return pending_reads < cfg.max_reads; }
  bool aw_ready() override { return pending_aw < cfg.max_writes; }
  bool w_ready() override { return pending_w < cfg.max_writes; }
  bool b_valid() override { return !bq.empty() && bq.front().release <= cycle; }
  uint64_t b_resp() override { return 0; }
  uint64_t b_id() override { return b_valid() ? bq.front().id : 0; }
  bool r_valid() override { return !rq.empty() && rq.front().release <= cycle; }
  uint64_t r_resp() override { return 0; }
  uint64_t r_id() override { return r_valid() ? rq.front().id : 0; }
  void *r_data() override { return r_valid() ? (void *)&rq.front().beats[r_beat][0] : (void *)data; }
  bool r_last() override { return r_valid() && r_beat + 1 == rq.front().beats.size(); }

  void tick(bool reset, bool ar_valid, uint64_t ar_addr, uint64_t ar_id, uint64_t ar_size, uint64_t ar_len,
            bool aw_valid, uint64_t aw_addr, uint64_t aw_id, uint64_t aw_size, uint64_t aw_len, bool w_valid,
            uint64_t w_strb, void *w_data, bool w_last, bool r_ready, bool b_ready) override;

  // Counters a harness may print at exit: traffic the model answered, and cycles a request waited on a bound.
  uint64_t reads = 0, writes = 0, ar_stall_cycles = 0, aw_stall_cycles = 0, w_stall_cycles = 0;

 private:
  struct rburst_t { uint64_t id, release; std::vector<std::vector<char>> beats; };
  struct aw_t { uint64_t addr, id, size, beats_left; };
  struct wbeat_t { std::vector<char> data; uint64_t strb; bool last; };
  struct bresp_t { uint64_t id, release; };

  mm_latency_pipe_cfg_t cfg;
  uint64_t cycle = 0;
  // FASED's three SatUpDownCounters: reads to their last beat, AWs to their B, complete W bursts to their B.
  uint64_t pending_reads = 0, pending_aw = 0, pending_w = 0;
  std::deque<rburst_t> rq;     // accepted reads, in acceptance order (FASED's readPipe + currentRead)
  std::deque<aw_t> aw_wait;    // accepted AWs whose data has not all arrived
  std::deque<wbeat_t> w_early; // accepted W beats whose AW has not arrived
  std::deque<bresp_t> bq;      // complete writes, in completion order (FASED's writePipe + currentWrite)
  size_t r_beat = 0;
};

#endif
