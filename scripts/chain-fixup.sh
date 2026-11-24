#!/usr/bin/env bash
# Re-render chained shots whose start frame could not be taken (runs ON the machine after both queues).
#   bash fixup.sh <gpu> <job> [<job> ...]     jobs in film order; each chains from the shot before it
# Takes the previous clip's frame at the end of its narration slot, or the clip's own last frame when
# Kandinsky trimmed the clip shorter than the slot. Log: /workspace/ks2v/fixup-gpu<N>.log
set -uo pipefail
GPU="$1"; shift
exec >> "/workspace/ks2v/fixup-gpu$GPU.log" 2>&1
export CUDA_VISIBLE_DEVICES="$GPU" PATH=/opt/ks2v/.venv/bin:$PATH HF_HOME=/models/.hf
export VAE_TILING=1 RESOLUTION=512x896 OFFLOAD=none
cd /workspace/ks2v
for JOB in "$@"; do
  LINE=$(grep -hP "^$JOB\t" jobs.tsv); PREV=$(grep -hP -B1 "^$JOB\t" jobs.tsv | head -1 | cut -f1)
  SLOT=$(echo "$LINE" | cut -f3); SEED=$(echo "$LINE" | cut -f4)
  P="out/$PREV-ks2v-lite.mp4"; D="jobs/$JOB"
  DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$P")
  T=$(python -c "print(max(0.0, min($SLOT, $DUR) - 0.08))")
  rm -f "$D/chain_start.png"
  ffmpeg -nostdin -v error -y -ss "$T" -i "$P" -frames:v 1 "$D/chain_start.png"
  [ -s "$D/chain_start.png" ] || ffmpeg -nostdin -v error -y -sseof -0.2 -i "$P" -frames:v 1 "$D/chain_start.png"
  if [ ! -s "$D/chain_start.png" ]; then echo "=== FIX $JOB FAILED no frame from $PREV"; continue; fi
  rm -f "$D/.fallback"
  echo "=== FIX $JOB start $(date -u +%T) from $PREV at ${T}s (slot $SLOT, clip $DUR)"
  ks2v-run --model lite --image "$D/chain_start.png" --audio "$D/audio.wav" --prompt-file "$D/prompt.txt" \
    --negative-file "$D/negative.txt" --seed "$SEED" --out "out/$JOB-ks2v-lite.mp4" --fps 12 < /dev/null > /dev/null 2>&1 \
    && echo "=== FIX $JOB done $(date -u +%T) $(grep -o 'WALL_SECONDS=[0-9]*' "out/$JOB-ks2v-lite.log")" \
    || { echo "=== FIX $JOB FAILED $(date -u +%T)"; tail -5 "out/$JOB-ks2v-lite.log"; }
done
echo "=== FIXUP $GPU DONE $(date -u +%T)"
