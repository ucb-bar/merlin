// Receipt-bindable filter for high-volume FIRRTL printf streams in GSIM models.
//
// GSIM emits one global `gprintf` implementation into the first generated C++
// translation unit and calls it from all FIRRTL printf statements.  Build the
// first translation unit with `-Dgprintf=gsim_unfiltered_gprintf`, then compile
// this file with an explicitly declared GSIM_SUPPRESS_PRINTF_PREFIX.  The build
// receipt must pin this source and both compile commands.  No generated C++ is
// edited, and all nonmatching diagnostics retain GSIM's original formatting.

#include <cassert>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>

#ifndef GSIM_SUPPRESS_PRINTF_PREFIX
#error "GSIM_SUPPRESS_PRINTF_PREFIX must be provided by the recorded build command"
#endif

void gprintf(const char *fmt, ...) {
  constexpr const char *suppressed_prefix = GSIM_SUPPRESS_PRINTF_PREFIX;
  constexpr std::size_t suppressed_prefix_size =
      sizeof(GSIM_SUPPRESS_PRINTF_PREFIX) - 1;
  static_assert(suppressed_prefix_size > 0,
                "GSIM_SUPPRESS_PRINTF_PREFIX must not be empty");

  if (std::strncmp(fmt, suppressed_prefix, suppressed_prefix_size) == 0) {
    return;
  }

  std::FILE *fp = stderr;
  std::va_list args;
  va_start(args, fmt);
  std::size_t fmt_idx = 0;
  while (true) {
    const char c = fmt[fmt_idx++];
    switch (c) {
      case '%':
        break;
      case 0:
        va_end(args);
        return;
      default:
        std::fputc(c, fp);
        continue;
    }

    std::uint64_t value = 0;
    const int bits = va_arg(args, std::uint32_t);
    if (bits <= 32) {
      value = va_arg(args, std::uint32_t);
    } else if (bits <= 64) {
      value = va_arg(args, std::uint64_t);
    } else {
      assert(false && "GSIM printf argument wider than 64 bits");
    }

    switch (fmt[fmt_idx++]) {
      case 'd':
        std::fprintf(fp, "%ld", value);
        break;
      case 'c':
        std::fputc(value & 0xff, fp);
        break;
      case 'x':
        std::fprintf(fp, "%lx", value);
        break;
      default:
        assert(false && "unsupported GSIM printf format character");
    }
  }
}
