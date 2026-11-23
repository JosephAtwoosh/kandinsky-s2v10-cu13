#!/usr/bin/env bash
# Entrypoint for the Kandinsky S2V 1.0 image.
#
#   <no args> | doctor   -> preflight report
#   shell                -> interactive bash
#   run [args...]        -> ks2v-run (one shot: reference frame + speech -> video)
#   download [pro|lite]  -> ks2v-download (fill the Hugging Face cache under /models)
#   <anything else>      -> executed as-is
#
# On vast.ai the SSH runtype replaces this entrypoint; call the ks2v-* tools directly there.
set -euo pipefail

cd /opt/ks2v

case "${1:-doctor}" in
  doctor)   exec /usr/local/bin/ks2v-doctor ;;
  shell|bash) shift || true; exec /bin/bash "$@" ;;
  run)      shift; exec /usr/local/bin/ks2v-run "$@" ;;
  download) shift; exec /usr/local/bin/ks2v-download "$@" ;;
  *)        exec "$@" ;;
esac
