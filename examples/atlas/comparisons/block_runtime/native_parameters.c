typedef unsigned int u32;
typedef unsigned long long u64;
typedef float (*exp_fn)(float);
typedef double (*pow_fn)(double, double);
void exp_bits(exp_fn fn, u32 *out) {
  for (u32 i = 0; i < 65536; ++i) {
    union {u32 u;
        float f;
    } x;
    x.u = i << 16;
    x.f = fn(x.f);
    out[i] = x.u;
  }
}
void pow_bits(pow_fn fn, u64 *out, u32 *single) {
  for (u32 i = 0; i < 2; ++i) {
    union {u64 u;
        double d;
    } x;
    union {u32 u;
        float f;
    } f;
    x.d = fn(i ? 128.0 : 64.0, -0.5);
    f.f = (float)x.d;
    out[i] = x.u;
    single[i] = f.u;
  }
}
