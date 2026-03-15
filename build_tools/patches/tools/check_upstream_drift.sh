#!/usr/bin/env bash
set -euo pipefail

scripts_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${scripts_dir}/common.sh"

check_series_state() {
  local repo_rel="$1"
  local series_rel="$2"
  local expected_base="$3"
  local repo
  local series_file

  repo="$(repo_abs_path "${repo_rel}")"
  series_file="$(patch_abs_path "${series_rel}")"

  echo "== Drift check: ${repo_rel}"
  assert_base_commit "${repo}" "${expected_base}"

  while IFS= read -r patch_rel; do
    local patch_file
    patch_file="$(patch_abs_path "${patch_rel}")"
    if git -C "${repo}" apply --check "${patch_file}" >/dev/null 2>&1; then
      echo "Patch ${patch_rel} is not applied yet (clean apply possible)."
      continue
    fi

    if is_patch_applied "${repo}" "${patch_file}"; then
      echo "Patch ${patch_rel} is already applied."
      continue
    fi

    echo "ERROR: ${patch_rel} neither applies cleanly nor matches applied state." >&2
    return 1
  done < <(series_entries "${series_file}")
}

check_series_state "${IREE_REPO}" "build_tools/patches/series.iree" "${IREE_BASE_COMMIT}"
check_series_state "${LLVM_REPO}" "build_tools/patches/series.llvm" "${LLVM_BASE_COMMIT}"

echo "Upstream drift check passed."
