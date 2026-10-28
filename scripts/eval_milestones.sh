#!/usr/bin/env bash
# Evaluate each new milestone checkpoint (step_*.pt) of a run on 1,000 val pairs (k=1,3,5).
#   eval_milestones.sh <run dir> <code snapshot dir> <python> <gpu id> <view cache> <catalog dir> [batch] [memory cap GiB]
# Runs next to training on the same GPU: small batch + allocator cap (defaults 8 / 7 GiB fit a 24 GB GPU).
set -u
RUN=$1; CODE=$2; PY=$3; GPU=$4; CACHE=$5; CAT=$6; BATCH=${7:-8}; CAP=${8:-7}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$CODE"
while true; do
  for ck in $(ls "$RUN"/checkpoints/step_*.pt 2>/dev/null | sort); do
    name=$(basename "$ck" .pt)
    [ "$name" = "step_000000" ] && continue
    out="$RUN/eval/$name"
    [ -f "$out/metrics.json" ] && continue
    echo "== $(date '+%F %T') evaluating $name"
    CUDA_VISIBLE_DEVICES=$GPU "$PY" -m plan_b.mvimgnet.eval_nvs --checkpoint "$ck" --catalog "$CAT" \
      --view-cache "$CACHE" --split val --max-objects 1000 --contexts 1 3 5 --batch $BATCH --workers 4 \
      --memory-cap-gib $CAP --out "$out" 2>&1 | grep '^{"k"'
  done
  # stop once training has ended and every milestone is evaluated
  if ! pgrep -f "mrdt_train\.py .*--output $RUN( |$)" >/dev/null; then
    pending=0
    for ck in $(ls "$RUN"/checkpoints/step_*.pt 2>/dev/null); do
      n=$(basename "$ck" .pt); [ "$n" != "step_000000" ] && [ ! -f "$RUN/eval/$n/metrics.json" ] && pending=1
    done
    [ $pending -eq 0 ] && { echo "== $(date '+%F %T') training finished; all milestones evaluated"; exit 0; }
  fi
  sleep 300
done
