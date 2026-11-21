#include <stdint.h>
#include <stdio.h>

/* The compilation manifest supplies this validated C symbol at build time. */
int MERLIN_EXAMPLE_SYMBOL(void *, const void *const *, void *const *);

int main(void) {
  int32_t input[6] = {1, 2, 3, 4, 5, 6}, output[6] = {0};
  const void *inputs[] = {input};
  void *outputs[] = {output};
  if (MERLIN_EXAMPLE_SYMBOL(0, inputs, outputs)) return 1;
  for (int i = 0; i < 6; ++i) {
    if (output[i] != input[i]) return 2;
    printf("%s%d", i ? " " : "", output[i]);
  }
  puts("");
  return 0;
}
