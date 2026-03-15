#!/usr/bin/env bash
set -euo pipefail

scripts_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${scripts_dir}/common.sh"

apply_series() {
  local repo_rel="$1"
  local series_rel="$2"
  local repo
  local series_file

  repo="$(repo_abs_path "${repo_rel}")"
  series_file="$(patch_abs_path "${series_rel}")"

  echo "== Applying series ${series_rel} to ${repo_rel}"
  while IFS= read -r patch_rel; do
    local patch_file
    patch_file="$(patch_abs_path "${patch_rel}")"
    apply_patch_file "${repo}" "${patch_file}"
  done < <(series_entries "${series_file}")
}

apply_series "${IREE_REPO}" "build_tools/patches/series.iree"
apply_series "${LLVM_REPO}" "build_tools/patches/series.llvm"

echo "All patch series processed successfully."
