#include "TestHarness.h"
#include <cerrno>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/resource.h>

extern uint64_t gsim_time_ps;
extern bool gsim_done;
extern int gsim_exit_code;
void harness_args(int argc, char** argv);
bool harness_dump_requested();
bool harness_dump_complete();
uint64_t harness_sample_step_ps();
uint64_t harness_reference_period_ps();

#include "refill_observer.hpp"

int main(int argc, char** argv) {
  try {
    uint64_t maximum_cycles = 20000000;
    uint64_t step_ps = 0, reference_period_ps = 0;
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg.rfind("+max-cycles=", 0) == 0) maximum_cycles = std::stoull(arg.substr(12));
      if (arg.rfind("+gsim-step-ps=", 0) == 0)
        step_ps = std::stoull(arg.substr(std::string("+gsim-step-ps=").size()));
      if (arg.rfind("+gsim-reference-period-ps=", 0) == 0)
        reference_period_ps = std::stoull(arg.substr(std::string("+gsim-reference-period-ps=").size()));
    }
    harness_args(argc, argv);
    RefillRecorder recorder(argc, argv);
    auto dut = std::make_unique<STestHarness>();
    dut->set_reset(1);
    dut->set_clock(0);
    // Sample time zero to initialize the clock-source adapters. Derive cadence
    // from their actual FIRRTL real parameters, not an accelerator/config name.
    dut->step();
    uint64_t declared_step = harness_sample_step_ps();
    if (!step_ps) step_ps = declared_step;
    if (!reference_period_ps) reference_period_ps = harness_reference_period_ps();
    if (!step_ps || declared_step % step_ps || reference_period_ps < 2 ||
        reference_period_ps % 2 || (reference_period_ps / 2) % step_ps ||
        maximum_cycles <= 100 || maximum_cycles > UINT64_MAX / reference_period_ps)
      throw std::runtime_error("invalid bounded simulation time cadence");
    fprintf(stderr, "GSIM time base: sample=%llu ps reference_period=%llu ps\n",
            static_cast<unsigned long long>(step_ps), static_cast<unsigned long long>(reference_period_ps));
    uint64_t reset_until_ps = 100 * reference_period_ps;
    uint64_t finish_ps = maximum_cycles * reference_period_ps;
    for (gsim_time_ps = step_ps; gsim_time_ps < finish_ps; gsim_time_ps += step_ps) {
      dut->set_clock((gsim_time_ps / (reference_period_ps / 2)) & 1);
      if (gsim_time_ps == reset_until_ps) dut->set_reset(0);
      dut->step();
      recorder.sample(*dut, gsim_time_ps, reference_period_ps);
      if (gsim_done || dut->get_io$$success()) {
        fprintf(stderr, "GSIM model finished execution.\nCycles: %llu\n",
                static_cast<unsigned long long>(gsim_time_ps / reference_period_ps));
        if (harness_dump_requested() && (!gsim_done || gsim_exit_code != 0 || !harness_dump_complete())) {
          fprintf(stderr, "[gsim-dump] incomplete: normal guest exit and full coherent dump required\n");
          return 5;
        }
        return gsim_exit_code;
      }
    }
    fprintf(stderr, "GSIM timeout: no RTL/TSI completion after %llu cycles\n",
            static_cast<unsigned long long>(maximum_cycles));
    return 124;
  } catch (const std::exception& error) {
    fprintf(stderr, "GSIM harness error: %s\n", error.what());
    return 2;
  }
}
