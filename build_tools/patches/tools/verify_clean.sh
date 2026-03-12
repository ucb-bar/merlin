#!/usr/bin/env bash
set -euo pipefail

scripts_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${scripts_dir}/common.sh"

verify_repo() {
  local repo_rel="$1"
  local series_rel="$2"
  local expected_base="$3"

  local repo
  local series_file
  repo="$(repo_abs_path "${repo_rel}")"
  series_file="$(patch_abs_path "${series_rel}")"

  echo "== Verifying ${repo_rel}"
  assert_base_commit "${repo}" "${expected_base}"

  local -A allowed
  allowed=()

  while IFS= read -r patch_rel; do
    local patch_file
    patch_file="$(patch_abs_path "${patch_rel}")"

    if ! is_patch_applied "${repo}" "${patch_file}"; then
      echo "ERROR: expected patch not applied: ${patch_rel}" >&2
      return 1
    fi

    while IFS= read -r touched; do
      [[ -z "${touched}" ]] && continue
      allowed["${touched}"]=1
    done < <(patch_touched_files "${patch_file}")
  done < <(series_entries "${series_file}")

  local status_tmp
  status_tmp="$(mktemp)"
  git -C "${repo}" status --porcelain > "${status_tmp}"

  local failed=0
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue

    local path
    path="${line:3}"
    if [[ "${path}" == *" -> "* ]]; then
      path="${path##* -> }"
    fi

    if [[ -z "${allowed[${path}]+x}" ]]; then
      echo "ERROR: unexpected change in ${repo_rel}: ${path}" >&2
      failed=1
    fi
  done < "${status_tmp}"

  rm -f "${status_tmp}"

  if [[ "${failed}" -ne 0 ]]; then
    return 1
  fi

  echo "Verified ${repo_rel}: only expected patch-stack changes are present."
}

verify_repo "${IREE_REPO}" "build_tools/patches/series.iree" "${IREE_BASE_COMMIT}"
verify_repo "${LLVM_REPO}" "build_tools/patches/series.llvm" "${LLVM_BASE_COMMIT}"

echo "Patch cleanliness verification passed."
