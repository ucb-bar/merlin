/** @file path_utils.h
 *  @brief Lightweight path manipulation and file-existence utilities (C++).
 */
#ifndef MERLIN_CORE_PATH_UTILS_H_
#define MERLIN_CORE_PATH_UTILS_H_

#include <fstream>
#include <string>

namespace merlin_bench {

/** @brief Return the directory component of @p path (like POSIX dirname).
 *  @param path  Filesystem path.
 *  @return Everything before the last '/', or "." if no separator is found.
 */
inline std::string PathDirname(const std::string &path) {
	const size_t pos = path.find_last_of('/');
	if (pos == std::string::npos)
		return ".";
	if (pos == 0)
		return "/";
	return path.substr(0, pos);
}

/** @brief Join two path segments with a single '/'.
 *  @param a  Left segment (may be empty).
 *  @param b  Right segment; if absolute, returned as-is.
 *  @return Concatenated path.
 */
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

/** @brief Check whether a file exists and is readable.
 *  @param path  Filesystem path to test.
 *  @return @c true if the file can be opened for reading.
 */
inline bool FileReadable(const std::string &path) {
	std::ifstream f(path, std::ios::binary);
	return static_cast<bool>(f);
}

/** @brief Test whether @p s begins with @p prefix. */
inline bool StartsWith(const std::string &s, const std::string &prefix) {
	return s.size() >= prefix.size() &&
		s.compare(0, prefix.size(), prefix) == 0;
}

/** @brief Test whether @p s ends with @p suffix. */
inline bool EndsWith(const std::string &s, const std::string &suffix) {
	return s.size() >= suffix.size() &&
		s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

} // namespace merlin_bench

#endif // MERLIN_CORE_PATH_UTILS_H_
