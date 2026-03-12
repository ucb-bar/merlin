#!/usr/bin/env bash
set -euo pipefail

scripts_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${scripts_dir}/common.sh"

build_patch() {
  local repo_rel="$1"
  local patch_rel="$2"
  local include_re="$3"
  local exclude_re="$4"

  local repo
  local patch_file
  repo="$(repo_abs_path "${repo_rel}")"
  patch_file="$(patch_abs_path "${patch_rel}")"

  local tracked_list
  local untracked_list
  tracked_list="$(mktemp)"
  untracked_list="$(mktemp)"

  git -C "${repo}" diff --name-only \
    | awk -v i="${include_re}" -v e="${exclude_re}" '$0 ~ i && ($0 !~ e)' \
    > "${tracked_list}"

  git -C "${repo}" ls-files --others --exclude-standard \
    | awk -v i="${include_re}" -v e="${exclude_re}" '$0 ~ i && ($0 !~ e)' \
    > "${untracked_list}"

  : > "${patch_file}"

  if [[ -s "${tracked_list}" ]]; then
    mapfile -t tracked_files < "${tracked_list}"
    git -C "${repo}" diff --binary -- "${tracked_files[@]}" >> "${patch_file}"
  fi

  while IFS= read -r file; do
    [[ -z "${file}" ]] && continue
    git -C "${repo}" diff --binary --no-index -- /dev/null "${file}" >> "${patch_file}"
  done < "${untracked_list}"

  rm -f "${tracked_list}" "${untracked_list}"

  if [[ ! -s "${patch_file}" ]]; then
    echo "ERROR: produced empty patch file: ${patch_rel}" >&2
    return 1
  fi

  echo "Wrote patch: ${patch_rel}"
}

build_patch \
  "${IREE_REPO}" \
  "build_tools/patches/iree/0001-merlin-iree-riscv-and-stream.patch" \
  '^compiler/src/iree/compiler/(Codegen|Dialect/Stream/Transforms)/' \
  '(^|/)tmp/|\\.(vmfb|bc|ll|o|so|a|s)$|^profile\\.log$'

build_patch \
  "${LLVM_REPO}" \
  "build_tools/patches/llvm/0001-merlin-llvm-riscv-xsmtvdot.patch" \
  '^llvm/(include/llvm/IR/|lib/Target/RISCV/|test/CodeGen/RISCV/)' \
  '^$'

echo "Patch export completed."
