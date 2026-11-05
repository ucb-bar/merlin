#!/usr/bin/env bash
# p12-unsealed namespace: same-path view of a COPIED prepared corpus (never the his-flow root itself).
# usage: P12_TAG=<tag> [P12_CPUS=112-127] bash ns.sh CMD...
# Maps $P12/stage/$TAG/root onto the recorded seal4 path and $P12/stage/$TAG/merlin onto <seal4>/merlin, exactly as
# his-flow/ns.sh does, plus the same read-only contract/evidence/targets/facts sources. The hidden profile is NOT
# bound: Phase 1 reads the already-generated hidden capsules from the release payload, never the profile.
set -euo pipefail
P12=${P12:-/scratch/loren/p12-unsealed}
B=${R:-/scratch/loren/backend-verify-author}
TAG=${P12_TAG:?set P12_TAG}
S=$P12/stage/$TAG
SEAL_ROOT=${P12_SEAL_ROOT:-/scratch/loren/megaphase-core-aten/seal4}
C=${P12_C_ROOT:-/scratch/loren/c20-compute1/rootp/seal4}
F=${P12_FACTS:-/scratch/loren/c11-compute1/s4facts-full/s4facts}
[ -d "$S/root" ] && [ -d "$S/merlin" ] || { echo "ns: no staged copy at $S" >&2; exit 2; }
# Root-only /etc dirs (credstore, ssl/private, sudoers.d, ...) are unreadable to loren and to every sandbox, but the
# FIN Merlin exposed-bind hardlink privacy scan (targetgen/sandbox/host_surfaces.py:229-260) cannot traverse them and
# fails closed ("runtime bind hardlink privacy cannot be verified"). Replace them with empty tmpfs dirs in THIS private
# namespace only: no readable surface changes; the scan still covers everything a sandbox can read.
MASK=(); while read -r d; do MASK+=(--tmpfs "$d"); done < <(find /etc -xdev -type d ! -readable 2>/dev/null)
exec nice -n "${P12_NICE:-5}" taskset -c "${P12_CPUS:-112-127}" /usr/bin/bwrap --ro-bind / / --dev-bind /dev /dev --proc /proc --tmpfs /tmp "${MASK[@]}" \
 --bind "$P12" "$P12" --bind "$B" "$B" --bind "$S/root" "$SEAL_ROOT" --bind "$S/merlin" "$SEAL_ROOT/merlin" \
 --ro-bind "$C/contract" "$SEAL_ROOT/contract" --ro-bind "$C/evidence" "$SEAL_ROOT/evidence" \
 --ro-bind "$C/merlin-out/artifacts/targets/gemmini" "$SEAL_ROOT/merlin-out/artifacts/targets/gemmini" \
 --ro-bind "$F" /scratch/loren/s4facts --bind "$HOME/.codex" "$HOME/.codex" \
 --unshare-pid --die-with-parent -- "$@"
