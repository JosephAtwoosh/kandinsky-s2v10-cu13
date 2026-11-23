#!/usr/bin/env bash
# Fill the Hugging Face cache (HF_HOME=/models/.hf) with every weight a config needs, using
# kandinsky-s2v's own resolver, so the first render does not stall on downloads.
#
#   ks2v-download pro     Pro distilled DiT, HunyuanVideo VAE, Qwen2.5-Omni-7B, CLIP  (~65 GB)
#   ks2v-download lite    Lite distilled DiT and the same shared pieces                    (~25 GB)
#   INCLUDE_VS2=1 ks2v-download pro    also fetch VideoScore2 into /models/videoscore2       (+17 GB)
#
#   DISTILLED_ONLY=0 ks2v-download pro also fetches the base (non-distilled) DiT
# HF_TOKEN is optional (the repos are public) but raises the rate limit.
set -euo pipefail
MODEL="${1:-pro}"
case "$MODEL" in
  pro)  CONFIGS="s2v_pro_distill.yaml s2v_pro.yaml" ;;
  lite) CONFIGS="s2v_lite_distill.yaml s2v_lite.yaml" ;;
  *) echo "usage: ks2v-download pro|lite" >&2; exit 2 ;;
esac
[ "${DISTILLED_ONLY:-1}" = "1" ] && CONFIGS="${CONFIGS%% *}"

export HF_HUB_ENABLE_HF_TRANSFER=0
cd /opt/ks2v
for c in $CONFIGS; do
  echo "== $c"
  python - "src/kandinsky_s2v/configs/$c" <<'PY'
import sys, time
from kandinsky_s2v.pipeline.config import load_config
from kandinsky_s2v.pipeline.weights import resolve
cfg = load_config(sys.argv[1])
for name, ref in cfg.paths.model_dump().items():
    t = time.time()
    print(f"{name}: {ref}", flush=True)
    print(f"  -> {resolve(ref)}  ({time.time() - t:.0f} s)", flush=True)
PY
done

if [ "${INCLUDE_VS2:-0}" = "1" ]; then
  echo "== VideoScore2"
  /opt/vs2/venv/bin/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('TIGER-Lab/VideoScore2', local_dir='/models/videoscore2'))"
fi
du -sh /models/.hf/hub 2>/dev/null || true
