#!/usr/bin/env bash
# Start (or resume) one MRDT run detached from the shell, plus its milestone-eval watcher.
#   launch_run.sh scratch <run dir> <gpus e.g. 0,1,2,3> <batch per gpu> <grad accum> <eval gpu> [run-specific trainer args...]
#   launch_run.sh resume  <run dir> <gpus> <batch per gpu> <grad accum> <eval gpu> [same run-specific args]
# Paths come from the environment (see README.md):
#   MRDT_CATALOG       catalog dir (catalog.sqlite, readiness.json, ...)
#   MRDT_VIEW_CACHE    canonical-view cache (build_view_cache)
#   MRDT_LATENT_CACHE  SD-VAE latent cache (build_latent_cache)
#   MRDT_MODEL_ROOT    dir holding siglip-so400m-patch14-384/ and sd-vae-ft-mse/ (default <repo>/pretrained)
#   MRDT_PYTHON        python of the conda env (default: python on PATH)
#   MRDT_MEM_CAP       per-GPU allocator cap in GiB (default 20; resume with a new value needs --reshard-resume)
# scratch: copies this repo to <run dir>.launch/code and runs from that snapshot, so later edits of the working
# tree never touch the running job or its resume. resume: reuses that snapshot. <run dir>.launch/ also holds
# train.log, eval_milestones.log and launch_history.txt (the trainer refuses a pre-existing <run dir> on --scratch).
set -euo pipefail
MODE=$1; RUN=$(realpath -m "$2"); GPUS=$3; BPG=$4; ACC=$5; EGPU=$6; shift 6
SRC=$(cd "$(dirname "$0")/.." && pwd)
: "${MRDT_CATALOG:?set MRDT_CATALOG}" "${MRDT_VIEW_CACHE:?set MRDT_VIEW_CACHE}" "${MRDT_LATENT_CACHE:?set MRDT_LATENT_CACHE}"
export MRDT_MODEL_ROOT=${MRDT_MODEL_ROOT:-$SRC/pretrained}
PY=${MRDT_PYTHON:-$(command -v python)}; CAP=${MRDT_MEM_CAP:-20}
export PATH=$(dirname "$PY"):$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=$RUN.launch
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
case $MODE in
  scratch)
    [ -e "$RUN" ] || [ -e "$L" ] && { echo "refusing: $RUN or $L already exists" >&2; exit 1; }
    mkdir -p "$L"
    rsync -a --exclude runs --exclude pretrained --exclude .git --exclude __pycache__ --exclude '*.pyc' "$SRC"/ "$L"/code/
    START=(--scratch) ;;
  resume)
    [ -f "$RUN"/checkpoints/latest.pt ] || { echo "no $RUN/checkpoints/latest.pt" >&2; exit 1; }
    rm -f "$RUN"/STOP
    START=(--resume "$RUN"/checkpoints/latest.pt) ;;
  *) echo "mode must be scratch or resume" >&2; exit 1 ;;
esac
cd "$L"/code
CMD=("$PY" -m torch.distributed.run --standalone --nproc_per_node=$NPROC mrdt_train.py --data-root "$MRDT_CATALOG"
  --view-cache "$MRDT_VIEW_CACHE" --latent-cache "$MRDT_LATENT_CACHE" --output "$RUN" --steps 200000 --batch-per-gpu $BPG
  --grad-accum $ACC --workers-per-gpu 8 --memory-cap-gib $CAP --precision bf16 --lr 1e-4 --warmup 1000 --lr-schedule constant
  "${START[@]}" "$@")
echo "$(date '+%F %T') $MODE CUDA_VISIBLE_DEVICES=$GPUS MRDT_MODEL_ROOT=$MRDT_MODEL_ROOT ${CMD[*]}" >> "$L"/launch_history.txt
CUDA_VISIBLE_DEVICES=$GPUS nohup setsid "${CMD[@]}" >> "$L"/train.log 2>&1 < /dev/null &
echo "train pid $!"
sleep 60  # let the trainer create <run dir> before the watcher starts polling
nohup setsid bash "$L"/code/scripts/eval_milestones.sh "$RUN" "$L"/code "$PY" "$EGPU" "$MRDT_VIEW_CACHE" "$MRDT_CATALOG" 8 7 \
  >> "$L"/eval_milestones.log 2>&1 < /dev/null &
echo "eval watcher pid $!"
