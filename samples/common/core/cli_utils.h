/** @file cli_utils.h
 *  @brief C utility functions for CLI flag parsing.
 */
#ifndef MERLIN_CORE_CLI_UTILS_H_
#define MERLIN_CORE_CLI_UTILS_H_

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Parse a decimal integer from a string, returning a default on
 * failure.
 *  @param text  Null-terminated string to parse (may be NULL).
 *  @param dflt  Value returned when @p text is NULL or not a valid integer.
 *  @return Parsed integer, or @p dflt.
 */
static inline int parse_int_or_default(const char *text, int dflt) {
	if (!text)
		return dflt;
	char *end = NULL;
	long v = strtol(text, &end, 10);
	return (end == text) ? dflt : (int)v;
}

/** @brief Parse a uint64 from a string (auto-detecting hex via 0x prefix).
 *  @param text  Null-terminated string to parse (may be NULL).
 *  @param dflt  Value returned when @p text is NULL or not a valid integer.
 *  @return Parsed value, or @p dflt.
 */
static inline uint64_t parse_u64_hex_or_default(
	const char *text, uint64_t dflt) {
	if (!text)
		return dflt;
	char *end = NULL;
	unsigned long long v = strtoull(text, &end, 0);
	return (end == text) ? dflt : (uint64_t)v;
}

/** @brief Extract the value portion of a "--key=value" flag.
 *  @param arg  The full argument string (e.g. "--iterations=10").
 *  @param key  The key to match, including leading dashes (e.g.
 * "--iterations").
 *  @return Pointer to the value substring after '=', or NULL if @p arg does
 *          not match @p key.
 */
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

#endif // MERLIN_CORE_CLI_UTILS_H_
