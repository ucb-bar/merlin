#ifndef MERLIN_IREE_BENCH_JSON_PARSER_H_
#define MERLIN_IREE_BENCH_JSON_PARSER_H_

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace merlin_bench {

struct JsonParser {
	const char *p = nullptr;
	const char *end = nullptr;

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

	bool Consume(char expected) {
		SkipWs();
		if (p >= end || *p != expected)
			return false;
		++p;
		return true;
	}

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

	bool SkipValue();
	bool SkipArray();
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

#endif // MERLIN_IREE_BENCH_JSON_PARSER_H_
