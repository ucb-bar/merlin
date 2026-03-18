/** @file json_parser.h
 *  @brief Minimal recursive-descent JSON parser (header-only, C++).
 *
 *  Provides just enough functionality to parse configuration files used by
 *  the Merlin benchmark harness: strings, ints, doubles, and dependency
 *  arrays.  No dynamic allocation beyond std::string / std::vector.
 */
#ifndef MERLIN_CORE_JSON_PARSER_H_
#define MERLIN_CORE_JSON_PARSER_H_

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace merlin_bench {

/** @brief Lightweight, non-owning JSON pull-parser.
 *
 *  The caller points @c p and @c end at a UTF-8 buffer and then calls the
 *  various Parse / Skip helpers.  The parser advances @c p as it consumes
 *  tokens; on failure it returns @c false and leaves @c p in an
 *  unspecified position.
 */
struct JsonParser {
	const char *p = nullptr; /**< Current read position. */
	const char *end = nullptr; /**< One past the last valid byte. */

	/** @brief Advance past whitespace characters. */
	void SkipWs() {
		while (p < end) {
			const char c = *p;
			if (c == ' ' || c == '\n' || c == '\r' || c == '\t') {
				++p;
				continue;
			}
			break;
		}
	}

	/** @brief Consume a single expected character (after skipping whitespace).
	 *  @param expected  The character to match.
	 *  @return @c true if the character was found and consumed.
	 */
	bool Consume(char expected) {
		SkipWs();
		if (p >= end || *p != expected)
			return false;
		++p;
		return true;
	}

	/** @brief Parse a JSON string (including surrounding quotes).
	 *  @param[out] out  Receives the unescaped string content.
	 *  @return @c true on success.
	 */
	bool ParseString(std::string *out) {
		out->clear();
		SkipWs();
		if (p >= end || *p != '"')
			return false;
		++p;
		const char *start = p;
		bool has_escapes = false;
		while (p < end) {
			char c = *p++;
			if (c == '"')
				break;
			if (c == '\\') {
				has_escapes = true;
				if (p >= end)
					return false;
				++p;
			}
		}
		if (p > end)
			return false;
		const char *raw_end = p - 1;

		if (!has_escapes) {
			out->assign(start, raw_end - start);
			return true;
		}

		out->reserve(static_cast<size_t>(raw_end - start));
		const char *r = start;
		while (r < raw_end) {
			char c = *r++;
			if (c != '\\') {
				out->push_back(c);
				continue;
			}
			if (r >= raw_end)
				return false;
			char e = *r++;
			switch (e) {
				case '"':
					out->push_back('"');
					break;
				case '\\':
					out->push_back('\\');
					break;
				case 'n':
					out->push_back('\n');
					break;
				case 'r':
					out->push_back('\r');
					break;
				case 't':
					out->push_back('\t');
					break;
				default:
					out->push_back(e);
					break;
			}
		}
		return true;
	}

	/** @brief Parse a JSON integer.
	 *  @param[out] out  Receives the parsed value (set to 0 on failure).
	 *  @return @c true on success.
	 */
	bool ParseInt(int *out) {
		*out = 0;
		SkipWs();
		if (p >= end)
			return false;
		char *endptr = nullptr;
		long v = strtol(p, &endptr, 10);
		if (endptr == p)
			return false;
		p = endptr;
		*out = static_cast<int>(v);
		return true;
	}

	/** @brief Parse a JSON number as a double.
	 *  @param[out] out  Receives the parsed value (set to 0.0 on failure).
	 *  @return @c true on success.
	 */
	bool ParseDouble(double *out) {
		*out = 0.0;
		SkipWs();
		if (p >= end)
			return false;
		char *endptr = nullptr;
		double v = strtod(p, &endptr);
		if (endptr == p)
			return false;
		p = endptr;
		*out = v;
		return true;
	}

	/** @brief Skip over an arbitrary JSON value (string, number, object, array,
	 * bool, null). */
	bool SkipValue();
	/** @brief Skip over a JSON array and all of its elements. */
	bool SkipArray();
	/** @brief Skip over a JSON object and all of its key-value pairs. */
	bool SkipObject();
};

inline bool JsonParser::SkipArray() {
	if (!Consume('['))
		return false;
	SkipWs();
	if (Consume(']'))
		return true;
	while (true) {
		if (!SkipValue())
			return false;
		SkipWs();
		if (Consume(']'))
			return true;
		if (!Consume(','))
			return false;
	}
}

inline bool JsonParser::SkipObject() {
	if (!Consume('{'))
		return false;
	SkipWs();
	if (Consume('}'))
		return true;
	while (true) {
		std::string key;
		if (!ParseString(&key))
			return false;
		if (!Consume(':'))
			return false;
		if (!SkipValue())
			return false;
		SkipWs();
		if (Consume('}'))
			return true;
		if (!Consume(','))
			return false;
	}
}

inline bool JsonParser::SkipValue() {
	SkipWs();
	if (p >= end)
		return false;
	const char c = *p;
	if (c == '"') {
		std::string s;
		return ParseString(&s);
	} else if (c == '{') {
		return SkipObject();
	} else if (c == '[') {
		return SkipArray();
	} else if ((c >= '0' && c <= '9') || c == '-' || c == '+') {
		double dummy = 0.0;
		return ParseDouble(&dummy);
	} else if (!strncmp(p, "true", 4)) {
		p += 4;
		return true;
	} else if (!strncmp(p, "false", 5)) {
		p += 5;
		return true;
	} else if (!strncmp(p, "null", 4)) {
		p += 4;
		return true;
	}
	return false;
}

/** @brief Parse a JSON array of strings (e.g. a dependency list).
 *  @param jp   Parser positioned just before the opening '['.
 *  @param[out] out  Receives the parsed string elements.
 *  @return @c true on success.
 */
inline bool ParseDependenciesArray(
	JsonParser *jp, std::vector<std::string> *out) {
	out->clear();
	if (!jp->Consume('['))
		return false;
	jp->SkipWs();
	if (jp->Consume(']'))
		return true;
	while (true) {
		std::string dep;
		if (!jp->ParseString(&dep))
			return false;
		out->push_back(std::move(dep));
		jp->SkipWs();
		if (jp->Consume(']'))
			return true;
		if (!jp->Consume(','))
			return false;
	}
}

} // namespace merlin_bench

#endif // MERLIN_CORE_JSON_PARSER_H_
