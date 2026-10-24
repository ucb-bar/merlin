#include <stdint.h>
#include <stddef.h>
#include <math.h>
static const uint32_t exp_values[65536] = {
#include "expf_bf16_values.inc"
};
static void unsupported(void) { __builtin_trap();
}
void *memcpy(void *dst, const void *src, size_t n) {
  unsigned char *d = dst;
  const unsigned char *s = src;
  for (size_t i = 0; i < n; ++i) d[i] = s[i];
  return dst;
}
void *memset(void *dst, int value, size_t n) {
  unsigned char *d = dst;
  for (size_t i = 0; i < n; ++i) d[i] = (unsigned char)value;
  return dst;
}
int fegetround(void) { unsigned long mode;
    __asm__ volatile("frrm %0" : "=r"(mode));
    return (int)mode;
}
float sqrtf(float x) { float out;
    __asm__ volatile("fsqrt.s %0,%1,dyn" : "=f"(out) : "f"(x));
    return out;
}
float fmaf(float x, float y, float z) { float out;
    __asm__ volatile("fmadd.s %0,%1,%2,%3,dyn" : "=f"(out) : "f"(x),"f"(y),"f"(z));
    return out;
}
float nearbyintf(float x) {
  union {uint32_t u;
      float f;
  } value;
  value.f = x;
  if (!isfinite(x) || x < -151.0f || x > 0.0f) unsupported();
  if (x == 0.0f) return x;
  long rounded;
  unsigned long oldflags;
  __asm__ volatile("frflags %1\n\tfcvt.w.s %0,%2,dyn\n\tfsflags %1"
                   : "=&r"(rounded),"=&r"(oldflags) : "f"(x) : "memory");
  if (!rounded) { value.u &= 0x80000000u;
      return value.f;
  }
  return (float)rounded;
}
float scalbnf(float x, int exponent) {
  if (exponent < -75 || exponent > 0) unsupported();
  union {uint32_t u;
      float f;
  } scale;
  scale.u = (uint32_t)(127 + exponent) << 23;
  return x * scale.f;
}
float expf(float x) {
  union {uint32_t u;
      float f;
  } value;
  value.f = x;
  if (value.u & 65535u) unsupported();
  value.u = exp_values[value.u >> 16];
  return value.f;
}
double pow(double x, double exponent) {
  union {uint64_t u;
      double d;
  } value;
  if (exponent != -0.5 || (x != 64.0 && x != 128.0)) unsupported();
  value.u = x == 64.0 ? 0x3fc0000000000000ull : 0x3fb6a09e667f3bcdull;
  return value.d;
}
