# Running the image on vast.ai

vast.ai runs the image as the instance's container; it cannot build images (no Docker-in-Docker).
So: GitHub Actions builds and pushes `ghcr.io/josephatwoosh/kandinsky-s2v10:cu130`, and a vast.ai
template points at it.

## Template

| Field | Value |
|---|---|
| Name | `kandinsky-s2v10 cu130` |
| Image | `ghcr.io/josephatwoosh/kandinsky-s2v10:cu130` |
| Launch mode | SSH, direct connection (`ssh_direc ssh_proxy`) |
| Disk | 160 GB (Pro ≈ 65 GB, VideoScore2 17 GB, image ≈ 25 GB, outputs) |
| On-start | `mkdir -p /workspace/shots /workspace/out` |
| Env | `HF_HOME=/models/.hf` |
| Visibility | private |

The GHCR package must be public, or the template needs registry credentials.

## Picking a machine

- Driver ≥ 580 (CUDA 13.0). Check the offer's `cuda_max_good` ≥ 13.0 before renting.
- A100 80 GB or H100 for Pro at SD; a 40 GB A100 only with `VAE_TILING=1`.
- The studio's rental rules still apply (EPYC, RAM ≥ 128 GB, price limits, ask before renting).

## On the instance

```bash
ks2v-doctor
INCLUDE_VS2=1 ks2v-download pro
ks2v-run --image /workspace/shots/s001-01-v2/inputs/start.png \
         --audio /workspace/shots/s001-01-v2/inputs/audio.wav \
         --prompt-file /workspace/shots/s001-01-v2/inputs/prompt.txt \
         --negative-file /workspace/shots/s001-01-v2/inputs/negative.txt \
         --seed 835047 --out /workspace/out/s001-01-ks2v.mp4 --fps 12
```
