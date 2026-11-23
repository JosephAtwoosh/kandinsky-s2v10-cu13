#!/usr/bin/env bash
# Preflight for Kandinsky S2V inside the container. Read-only: never downloads weights.
# Exit codes: 0 = ready, 2 = at least one blocking problem, 3 = ready with warnings.
set -uo pipefail

FAILS=0; WARNS=0
ok()   { printf '  ok   %s\n' "$*"; }
warn() { printf '  !!   %s\n' "$*"; WARNS=$((WARNS+1)); }
fail() { printf '  XX   %s\n' "$*"; FAILS=$((FAILS+1)); }
info() { printf '  --   %s\n' "$*"; }

echo "Kandinsky S2V 1.0 container preflight"
echo "====================================="

echo; echo "GPU"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  fail "nvidia-smi not found: start the container with --gpus all"
else
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1)
  VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1)
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1)
  ok "GPU: ${GPU_NAME:-none} (compute ${CC:-?})"
  DRV_MAJOR=$(printf '%s' "${DRIVER:-0}" | cut -d. -f1)
  if [ "${DRV_MAJOR:-0}" -ge 580 ] 2>/dev/null; then ok "driver ${DRIVER} runs CUDA 13.0"
  else fail "driver ${DRIVER:-unknown} is older than 580 and cannot run this CUDA 13.0 image"; fi
  VRAM_GB=$(( ${VRAM_MIB:-0} / 1024 ))
  info "VRAM: ${VRAM_GB} GiB"
  # Upstream peaks (H100, 512x768, one 5 s chunk): Pro distilled 46 GiB, Lite distilled 57 GiB,
  # HD 55 GiB; offload.vae_tiling brings the peak to about 28 GiB with faint tile seams.
  if   [ "${VRAM_GB}" -ge 75 ]; then ok "VRAM fits every config, HD included"
  elif [ "${VRAM_GB}" -ge 46 ]; then warn "VRAM fits Pro at SD; Lite and HD need VAE_TILING=1"
  elif [ "${VRAM_GB}" -ge 30 ]; then warn "VRAM needs VAE_TILING=1 (about 28 GiB peak, faint tile seams)"
  else fail "VRAM ${VRAM_GB} GiB is too small even with VAE tiling"; fi
  case "${CC:-0}" in
    9.*) ok "Hopper: ks2v-run uses FlashAttention 3" ;;
    *)   info "not Hopper: ks2v-run uses PyTorch SDPA (FlashAttention 3 runs only on compute 9.x)" ;;
  esac
fi

echo; echo "Libraries"
if python - <<'PY'
import sys, torch, diffusers, transformers, kandinsky_s2v
print(f"  ok   python {sys.version.split()[0]} | torch {torch.__version__} (cu {torch.version.cuda}) | diffusers {diffusers.__version__} | transformers {transformers.__version__}")
print(f"  {'ok' if torch.cuda.is_available() else 'XX'}   torch.cuda.is_available() = {torch.cuda.is_available()}")
sys.exit(0 if torch.cuda.is_available() else 1)
PY
then :; else FAILS=$((FAILS+1)); fi
python -c "import flash_attn_interface" 2>/dev/null && ok "flash_attn_interface (FA3) importable" || info "FA3 not importable (SDPA will be used)"
command -v cc >/dev/null && ok "C compiler present (Triton)" || fail "no C compiler: Triton cannot compile its launcher"
/opt/vs2/venv/bin/python -c "import torch, transformers, qwen_vl_utils" 2>/dev/null && ok "VideoScore2 venv imports" || warn "VideoScore2 venv does not import"

echo; echo "Disk and weights"
FREE_GB=$(df -BG --output=avail /models 2>/dev/null | tail -n1 | tr -dc '0-9')
info "free under /models: ${FREE_GB:-?} GB (Pro needs about 65 GB, Lite 25 GB, VideoScore2 17 GB)"
[ -d /models/.hf/hub ] && info "HF cache: $(du -sh /models/.hf/hub 2>/dev/null | cut -f1)" || info "HF cache empty: run ks2v-download pro"
[ -f /models/videoscore2/config.json ] && ok "VideoScore2 weights present" || info "VideoScore2 weights absent (INCLUDE_VS2=1 ks2v-download)"

echo
if [ "$FAILS" -gt 0 ]; then echo "NOT READY: $FAILS problem(s)"; exit 2; fi
if [ "$WARNS" -gt 0 ]; then echo "READY with $WARNS warning(s)"; exit 3; fi
echo "READY"
