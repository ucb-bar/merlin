#ifndef MERLIN_IREE_BENCH_CLI_UTILS_H_
#define MERLIN_IREE_BENCH_CLI_UTILS_H_

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline int parse_int_or_default(const char *text, int dflt) {
	if (!text)
		return dflt;
	char *end = NULL;
	long v = strtol(text, &end, 10);
	return (end == text) ? dflt : (int)v;
}

static inline uint64_t parse_u64_hex_or_default(
	const char *text, uint64_t dflt) {
	if (!text)
		return dflt;
	char *end = NULL;
	unsigned long long v = strtoull(text, &end, 0);
	return (end == text) ? dflt : (uint64_t)v;
}

// Minimal flag parser: --key=value
static inline const char *get_flag_value(const char *arg, const char *key) {
	size_t klen = strlen(key);
	if (strncmp(arg, key, klen) != 0)
		return NULL;
	if (arg[klen] != '=')
		return NULL;
	return arg + klen + 1;
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MERLIN_IREE_BENCH_CLI_UTILS_H_
