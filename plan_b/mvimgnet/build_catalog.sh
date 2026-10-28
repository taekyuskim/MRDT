#!/usr/bin/env bash
# Build a training catalog from the archives verified so far by extract_verify_archives.
#   bash plan_b/mvimgnet/build_catalog.sh <prepared dir> <new output dir> [workers]
# The output dir must be new for each build: index/audit identities include the archive set.
# Trainer --data-root is <output dir>/catalog.
set -euo pipefail
PREPARED=$1; OUT=$2; WORKERS=${3:-24}
if [ -e "$OUT" ]; then echo "output dir exists: $OUT (use a new one)" >&2; exit 1; fi
cd "$(dirname "$0")/../.."
python -m plan_b.mvimgnet.extract_verify_archives --raw "$PREPARED/../raw" --data "$PREPARED" --finalize
python -m plan_b.mvimgnet.index_verified_full_bin15 --data "$PREPARED" --output "$OUT/camera" --workers "$WORKERS"
python -m plan_b.mvimgnet.audit_full_masks --data "$PREPARED" --output "$OUT/audit" --workers "$WORKERS"
python -m plan_b.mvimgnet.build_masked_catalog --camera "$OUT/camera" --audit "$OUT/audit" --output "$OUT/catalog"
echo "catalog ready: $OUT/catalog"
