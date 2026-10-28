#!/usr/bin/env bash
# Build the fp32 SD-VAE latent cache: one shard per GPU in parallel, then finalize (marks the cache complete).
#   build_latents.sh <catalog dir> <view cache dir> <sd-vae-ft-mse dir> <out dir> <gpus e.g. 0,1,2,3> [workers per shard]
# Resumable: rerunning skips finished objects. Logs go to <out dir>.logs/.
set -euo pipefail
CAT=$1; VIEW=$2; VAE=$3; OUT=$4; GPUS=$5; WORKERS=${6:-8}
cd "$(dirname "$0")/.."
PY=${MRDT_PYTHON:-$(command -v python)}
IFS=, read -ra G <<< "$GPUS"; N=${#G[@]}
LOG=$OUT.logs; mkdir -p "$LOG"
ARGS=(--view-cache "$VIEW" --catalog "$CAT" --vae-id "$VAE" --out "$OUT" --num-shards "$N")
for i in "${!G[@]}"; do
  CUDA_VISIBLE_DEVICES=${G[$i]} "$PY" -m plan_b.mvimgnet.build_latent_cache "${ARGS[@]}" --shard "$i" --workers "$WORKERS" \
    > "$LOG/shard$i.log" 2>&1 &
  [ "$i" = 0 ] && sleep 20  # shard 0 writes the manifest first
done
wait
"$PY" -m plan_b.mvimgnet.build_latent_cache "${ARGS[@]}" --finalize > "$LOG/finalize.log" 2>&1
tail -1 "$LOG/finalize.log"
