#!/usr/bin/env bash
set -euo pipefail

_merlin_scripts_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MERLIN_ROOT="$(cd "${_merlin_scripts_dir}/../.." && pwd)"
PATCH_STACK_ROOT="${MERLIN_ROOT}/build_tools/patches"

# shellcheck source=/dev/null
source "${PATCH_STACK_ROOT}/manifest.env"

series_entries() {
  local series_file="$1"
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%%#*}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "${line}" ]] && continue
    printf '%s\n' "${line}"
  done < "${series_file}"
}

repo_abs_path() {
  local repo_rel="$1"
  printf '%s/%s\n' "${MERLIN_ROOT}" "${repo_rel}"
}

patch_abs_path() {
  local patch_rel="$1"
  printf '%s/%s\n' "${MERLIN_ROOT}" "${patch_rel}"
}

patch_touched_files() {
  local patch_file="$1"
  awk '
    /^diff --git / {
      a=$3; b=$4;
      sub(/^a\//, "", a);
      sub(/^b\//, "", b);
      if (a == "/dev/null") {
        print b;
      } else if (b == "/dev/null") {
        print a;
      } else {
        print b;
      }
    }
  ' "${patch_file}" | sort -u
}

is_patch_applied() {
  local repo="$1"
  local patch_file="$2"
  git -C "${repo}" apply --reverse --check "${patch_file}" >/dev/null 2>&1
}

apply_patch_file() {
  local repo="$1"
  local patch_file="$2"

  if git -C "${repo}" apply --check "${patch_file}" >/dev/null 2>&1; then
    git -C "${repo}" apply "${patch_file}"
    echo "Applied: ${patch_file}"
    return 0
  fi

  if is_patch_applied "${repo}" "${patch_file}"; then
    echo "Already applied: ${patch_file}"
    return 0
  fi

  echo "ERROR: Cannot apply patch (and not already applied): ${patch_file}" >&2
  return 1
}

assert_base_commit() {
  local repo="$1"
  local expected_commit="$2"
  local actual_commit
  actual_commit="$(git -C "${repo}" rev-parse HEAD)"
  if [[ "${actual_commit}" != "${expected_commit}" ]]; then
    echo "ERROR: ${repo} HEAD ${actual_commit} != expected ${expected_commit}" >&2
    return 1
  fi
}
