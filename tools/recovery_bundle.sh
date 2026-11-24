#!/usr/bin/env bash
# Snapshot each Git branch and tag, including the budget leases and score journals.
set -euo pipefail
source_repo=${1:?repository URL or local repository required}
destination=${2:?new output directory required}
if [[ -d "$source_repo" ]]; then source_repo=$(cd "$source_repo" && pwd); fi
mkdir "$destination"
destination=$(cd "$destination" && pwd)
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
snapshot="$scratch/source.git"
git init -q --bare --initial-branch=main "$snapshot"
git -C "$snapshot" fetch --quiet --no-tags "$source_repo" '+refs/heads/*:refs/heads/*' '+refs/tags/*:refs/tags/*'
for branch in main research/champion-v2 research/champion-observations research/openai-promo-ledger-v2; do
  git -C "$snapshot" rev-parse --verify "refs/heads/$branch^{commit}" > /dev/null
done
git -C "$snapshot" for-each-ref --format='%(objectname) %(refname)' refs/heads refs/tags > "$destination/refs.txt"
git -C "$snapshot" bundle create "$destination/megalab.bundle" HEAD --branches --tags
git clone --quiet --mirror "$destination/megalab.bundle" "$scratch/restored.git"
git -C "$scratch/restored.git" fsck --full --no-dangling
git -C "$scratch/restored.git" for-each-ref --format='%(objectname) %(refname)' refs/heads refs/tags | diff -u "$destination/refs.txt" -
cat > "$destination/RESTORE.txt" <<'END'
Run sha256sum -c SHA256SUMS in this directory. Restore all Git history and
branches with git clone --mirror megalab.bundle megalab-restored.git.
For a working tree run git clone -b main megalab-restored.git megalab.
Preserve the existing ledger and observations branches, including unresolved
reservations. Compare live refs and active Actions runs before resuming operations.
Git bundles contain committed project data. Provider credentials, Actions secrets,
environment settings, timers and platform accounts live outside Git.
Recovery on another repository requires restoring those settings independently.
END
(cd "$destination" && sha256sum megalab.bundle refs.txt RESTORE.txt > SHA256SUMS)
