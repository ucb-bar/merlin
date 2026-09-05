// Blackbox (extmodule) implementations for the gemmini full-SoC DUT (ChipTop)
// under GSIM.  GSIM emits *calls* to every FIRRTL `extmodule` but leaves their
// bodies to the user (its shipped Rocket testbench links its own copies).  The
// pruned ChipTop.fir reaches exactly four blackboxes; their FIRRTL port lists
// (from the source .fir) fix the semantics:
//
//   extmodule GenericDigitalInIOCell  : input pad, output i, input ie
//   extmodule GenericDigitalOutIOCell : output pad, input o, input oe
//   extmodule plusarg_reader          : (DEFAULT, FORMAT, WIDTH) -> out   [rocket-chip PlusArg]
//   extmodule EICG_wrapper            : input in:Clock, test_en, en, output out:Clock
//
// GSIM abstracts Clock-typed ports away (step() advances all registers), so the
// EICG clock-gate reduces to a no-op in this model and the IO cells are plain
// passthroughs.  plusarg_reader returns its DEFAULT (arg0) since no +args are fed.
#include <cstdint>

// PlusArg: rocket-chip emits the call as (DEFAULT, "name=%d", WIDTH, out).
// With no runtime plusargs, the value is the compile-time default.
void plusarg_reader(int def, const char* /*fmt*/, int /*width*/, uint32_t& out) {
  out = (uint32_t)def;
}

// Input IO cell: pad drives internal i (ie = input-enable, irrelevant in sim).
void GenericDigitalInIOCell(uint8_t pad, uint8_t& i, uint8_t /*ie*/) {
  i = (uint8_t)(pad & 1u);
}

// Output IO cell: internal o drives pad (oe = output-enable, irrelevant in sim).
void GenericDigitalOutIOCell(uint8_t& pad, uint8_t o, uint8_t /*oe*/) {
  pad = (uint8_t)(o & 1u);
}

// Integrated clock gate: GSIM passes only the two data inputs (the Clock in/out
// are abstracted). No datapath effect in GSIM's clock model -> no-op.
void EICG_wrapper(uint8_t /*test_en*/, uint8_t /*en*/) {}
