#include "PrintfFilter.h"

int main() {
  SPrintfFilter dut;
  dut.set_reset(0);
  dut.step();
  return 0;
}
