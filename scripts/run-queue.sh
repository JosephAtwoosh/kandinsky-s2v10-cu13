#!/usr/bin/env bash
# One Kandinsky S2V Lite queue on one GPU (runs ON the rented machine).
#   bash run_queue.sh <jobs-tsv> <gpu-index>
# The TSV holds whole picture runs in film order, so a chained shot always follows its previous
# shot in the same queue. Progress: /workspace/ks2v/run-gpu<N>.log, ends with "=== QUEUE <N> DONE".
set -uo pipefail
TSV="$1"; GPU="$2"
exec >> "/workspace/ks2v/run-gpu$GPU.log" 2>&1
export CUDA_VISIBLE_DEVICES="$GPU" PATH=/opt/ks2v/.venv/bin:$PATH HF_HOME=/models/.hf
export VAE_TILING=1 RESOLUTION="${RESOLUTION:-512x896}" OFFLOAD="${OFFLOAD:-module}"
OUT=/workspace/ks2v/out; mkdir -p "$OUT"
PREV=""
while IFS=$'\t' read -r -u 3 JOB CHAIN PREV_SLOT SEED KIND; do
  [ -z "$JOB" ] && continue
  D="/workspace/ks2v/jobs/$JOB"; IMG="$D/start.png"
  if [ -f "$OUT/$JOB-ks2v-lite.12fps.mp4" ]; then echo "=== lite $JOB already done"; PREV="$JOB"; continue; fi
  if [ "$CHAIN" = 1 ] && [ -f "$OUT/$PREV-ks2v-lite.mp4" ]; then
    T=$(python -c "print(max(0.0, $PREV_SLOT - 0.04))")
    ffmpeg -nostdin -v error -y -ss "$T" -i "$OUT/$PREV-ks2v-lite.mp4" -frames:v 1 "$D/chain_start.png" \
      && IMG="$D/chain_start.png" && echo "    chained from $PREV at ${T}s"
  fi
  echo "=== lite $JOB start $(date -u +%T) gpu $GPU"
  ks2v-run --model lite --image "$IMG" --audio "$D/audio.wav" --prompt-file "$D/prompt.txt" \
    --negative-file "$D/negative.txt" --seed "$SEED" --out "$OUT/$JOB-ks2v-lite.mp4" --fps 12 < /dev/null > /dev/null 2>&1 \
    && echo "=== lite $JOB done $(date -u +%T) gpu $GPU $(grep -o 'WALL_SECONDS=[0-9]*' "$OUT/$JOB-ks2v-lite.log")" \
    || { echo "=== lite $JOB FAILED $(date -u +%T) gpu $GPU"; tail -8 "$OUT/$JOB-ks2v-lite.log"; }
  PREV="$JOB"
done 3< "$TSV"
echo "=== QUEUE $GPU DONE $(date -u +%T)"
