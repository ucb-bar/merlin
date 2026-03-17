#ifndef MERLIN_IREE_BENCH_PATH_UTILS_H_
#define MERLIN_IREE_BENCH_PATH_UTILS_H_

#include <fstream>
#include <string>

namespace merlin_bench {

inline std::string PathDirname(const std::string &path) {
	const size_t pos = path.find_last_of('/');
	if (pos == std::string::npos)
		return ".";
	if (pos == 0)
		return "/";
	return path.substr(0, pos);
}

inline std::string PathJoin2(const std::string &a, const std::string &b) {
	if (a.empty())
		return b;
	if (b.empty())
		return a;
	if (b[0] == '/')
		return b;
	if (a.back() == '/')
		return a + b;
	return a + "/" + b;
}

inline bool FileReadable(const std::string &path) {
	std::ifstream f(path, std::ios::binary);
	return static_cast<bool>(f);
}

inline bool StartsWith(const std::string &s, const std::string &prefix) {
	return s.size() >= prefix.size() &&
		s.compare(0, prefix.size(), prefix) == 0;
}

inline bool EndsWith(const std::string &s, const std::string &suffix) {
	return s.size() >= suffix.size() &&
		s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

} // namespace merlin_bench

#endif // MERLIN_IREE_BENCH_PATH_UTILS_H_
