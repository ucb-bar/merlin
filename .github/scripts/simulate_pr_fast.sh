#!/usr/bin/env bash
# Mirror of .github/workflows/pr-fast.yml (lint-and-config-gate job) that
# you can run locally before pushing a PR. Uses a throwaway venv at
# /tmp/ci-sim so it does not depend on the merlin-dev conda env.
#
# Usage (from repo root):
#   ./.github/scripts/simulate_pr_fast.sh
#
# First run creates the venv (~5s). Subsequent runs are <10s total.
#
# If a step fails here, the real CI will fail too. Fix locally, re-run
# this, and only push when it ends with "All CI steps green".

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VENV=/tmp/ci-sim
if [[ ! -x "$VENV/bin/python" ]]; then
    echo "▶ Creating throwaway venv at $VENV"
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet pre-commit pyyaml jinja2 pytest
fi
export PATH="$VENV/bin:$PATH"

BASE_SHA="$(git merge-base origin/main HEAD)"
HEAD_SHA="$(git rev-parse HEAD)"

step() { echo; echo "▶ $1"; }

step "Pre-commit (changed files)"
SKIP=clang-format,cmake-format pre-commit run \
    --from-ref "$BASE_SHA" --to-ref "$HEAD_SHA" --show-diff-on-failure

step "Lint scripts and python entrypoints"
python3 tools/merlin.py ci lint

step "Check CLI docs drift"
python3 tools/merlin.py ci cli-docs-drift

step "Validate upstream tracking config"
python3 tools/merlin.py ci release-status --offline --json > /dev/null

step "Smoke test merlin CLI"
python3 tools/merlin.py --help > /dev/null
for sub in build compile setup ci patches benchmark chipyard ray targetgen; do
    python3 tools/merlin.py "$sub" --help > /dev/null
done

step "Validate conda env files"
python3 -c "import yaml; yaml.safe_load(open('env_linux.yml')); yaml.safe_load(open('env_macOS.yml'))"

step "Run Python tests"
pytest tests/ -v --tb=short

echo
echo "✅ All CI steps green"
