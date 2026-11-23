#!/usr/bin/env bash
# ks2v-run: render one shot with Kandinsky S2V 1.0 (reference frame + speech -> video).
#
#   ks2v-run --image start.png --audio audio.wav --out shot.mp4 \
#            [--model pro|lite] [--base] [--hd] [--prompt-file prompt.txt] [--negative-file negative.txt] \
#            [--seed N] [--steps N] [--chunks N] [--fps 12]
#
# Defaults: Pro distilled, SD (512x768 / 768x512 / 512x512 bucket nearest the frame's aspect).
# The attention engine follows the GPU: FlashAttention 3 on Hopper (compute 9.x), PyTorch SDPA
# elsewhere (FA3 kernels do not run on an A100). VAE_TILING=1 cuts the decode peak to ~28 GiB.
# --fps N additionally writes <out>.<N>fps.mp4, resampled from the native 25 fps, keeping the audio.
# Also writes <out>.log, and <out>.gpu.csv (nvidia-smi once a second) and WALL_SECONDS in the log.
set -euo pipefail

MODEL=pro; VARIANT=_distill; HD=""; IMAGE=""; AUDIO=""; OUT=""; FPS=""
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --base) VARIANT=""; shift ;;
    --hd) HD=_hd; shift ;;
    --image) IMAGE="$2"; shift 2 ;;
    --audio) AUDIO="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --prompt-file) EXTRA+=(--prompt "$(cat "$2")"); shift 2 ;;
    --negative-file) EXTRA+=(--negative "$(cat "$2")"); shift 2 ;;
    --prompt|--negative|--seed|--steps|--chunks|--guidance) EXTRA+=("$1" "$2"); shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    -h|--help) sed -n 2,13p "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[ -n "$IMAGE" ] && [ -n "$AUDIO" ] && [ -n "$OUT" ] || { echo "need --image, --audio and --out" >&2; exit 2; }
[ "$MODEL" = lite ] && [ -n "$HD" ] && { echo "HD exists only for Pro" >&2; exit 2; }

CONFIG="/opt/ks2v/src/kandinsky_s2v/configs/s2v_${MODEL}${VARIANT}${HD}.yaml"
[ -f "$CONFIG" ] || { echo "no config $CONFIG" >&2; exit 2; }

# VAE tiling is a config-only switch, so write a patched copy next to the output.
if [ "${VAE_TILING:-0}" = "1" ]; then
  PATCHED="$(dirname "$OUT")/.$(basename "$OUT" .mp4).config.yaml"
  python - "$CONFIG" "$PATCHED" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
c.setdefault("offload", {})["vae_tiling"] = True
yaml.safe_dump(c, open(sys.argv[2], "w"), sort_keys=False)
PY
  CONFIG="$PATCHED"
fi

CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 || true)
case "${ATTENTION:-}" in
  "") case "$CC" in 9.*) ATTENTION=flash_attention_3 ;; *) ATTENTION=sdpa ;; esac ;;
esac

mkdir -p "$(dirname "$OUT")"
BASE="${OUT%.mp4}"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw --format=csv,noheader -l 1 > "$BASE.gpu.csv" 2>/dev/null &
SMI=$!
T0=$(date +%s)
set +e
kandy-s2v generate --config "$CONFIG" --image "$IMAGE" --audio "$AUDIO" --out "$OUT" \
  --attention-engine "$ATTENTION" --no-progress "${EXTRA[@]}" 2>&1 | tee "$BASE.log"
RC=${PIPESTATUS[0]}
set -e
T1=$(date +%s); kill $SMI 2>/dev/null || true
echo "MODEL=${MODEL}${VARIANT}${HD} ATTENTION=${ATTENTION} VAE_TILING=${VAE_TILING:-0}" | tee -a "$BASE.log"
echo "WALL_SECONDS=$((T1-T0))" | tee -a "$BASE.log"
[ "$RC" -eq 0 ] || exit "$RC"

if [ -n "$FPS" ]; then
  ffmpeg -v error -y -i "$OUT" -vf "fps=${FPS}" -c:v libx264 -crf 16 -pix_fmt yuv420p -c:a copy "$BASE.${FPS}fps.mp4"
  echo "wrote $BASE.${FPS}fps.mp4"
fi
