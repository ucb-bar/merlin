"""Portable complete-environment guards for a private exact-exp cache.

The snapshot is an opaque byte copy of the selected C implementation's fenv_t.
Equality is deliberately conservative: padding differences invalidate reuse.
Helpers preserve errno, never change the floating environment, and reject an
unsupported descriptor or environment larger than the private workspace.
"""

RUNTIME_C = r"""
#include <fenv.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
static int merlin_exp_environment(void *allocated, unsigned char *aligned,
    intptr_t offset, intptr_t size, intptr_t stride, int capture) {
  (void)allocated;
  if (sizeof(fenv_t) > 4096 || size < (intptr_t)sizeof(fenv_t) ||
      stride != 1 || offset != 0) return 0;
  fenv_t current;
  memset(&current, 0, sizeof(current));
  int previous_errno = errno;
  int status = fegetenv(&current);
  int result = 0;
  if (status == 0) {
    if (capture) { memcpy(aligned, &current, sizeof(current)); result = 1; }
    else {
      const volatile unsigned char *bytes = (const volatile unsigned char *)&current;
      result = 1;
      for (unsigned i = 0; i < sizeof(current); ++i)
        if (aligned[i] != bytes[i]) result = 0;
    }
  }
  errno = previous_errno;
  return result;
}
int __merlin_exp_environment_matches(void *allocated, unsigned char *aligned,
    intptr_t offset, intptr_t size, intptr_t stride) {
  return merlin_exp_environment(allocated, aligned, offset, size, stride, 0);
}
int __merlin_exp_environment_capture(void *allocated, unsigned char *aligned,
    intptr_t offset, intptr_t size, intptr_t stride) {
  return merlin_exp_environment(allocated, aligned, offset, size, stride, 1);
}
"""
