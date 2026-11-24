# kandinsky-s2v10-cu13

Docker image for **Kandinsky S2V 1.0** speech-to-video
([kandinskylab/kandinsky-s2v](https://github.com/kandinskylab/kandinsky-s2v), MIT), built for
rented vast.ai GPUs, the same way as our LTX-2.5 image (`ltx25-docker`).

`ghcr.io/josephatwoosh/kandinsky-s2v10:cu130`

One reference frame and one speech track produce a talking video at **25 fps**, in 5-second
chunks, at the 512×512, 768×512 or 512×768 bucket nearest the frame's shape (HD with `--hd`, Pro only).

## What is inside

| Path | What |
|---|---|
| `/opt/ks2v` | upstream kandinsky-s2v at a pinned commit, its `uv.lock` environment in `.venv` (Python 3.13, torch 2.13.0 cu130, flash-attn-3) |
| `/opt/vs2/venv` | VideoScore2's own environment (torch 2.6), the same scorer as the LTX image |
| `ks2v-doctor` | preflight: driver ≥ 580, VRAM, attention engine, libraries, weights |
| `ks2v-download pro\|lite` | fills the Hugging Face cache in `/models/.hf` (Pro ≈ 65 GB, Lite ≈ 25 GB; `INCLUDE_VS2=1` adds VideoScore2) |
| `ks2v-run` | renders one shot; logs, GPU trace and wall time next to the output |
| `ks2v-vs2-score` | VideoScore2 scoring of rendered clips |
| `ks2v-crop-run` | speaker-only lip-sync: renders a crop around the speaker and composites it back (below) |
| `ks2v-queue` | one render queue per GPU from a jobs TSV (chained shots start from the previous shot's cut frame) |
| `ks2v-chain-fixup` | re-renders chained shots whose start frame could not be taken |

Weights are not in the image.

`ks2v-run` also takes `RESOLUTION=HxW` (e.g. `512x896`, our LTX frame), `OFFLOAD=module` (Lite on 24 GB
cards) and `VAE_TILING=1` as environment switches.

## GPU

| Card | Works? |
|---|---|
| H100 / H200 (Hopper) | Yes, with FlashAttention 3: upstream's measured setup |
| A100 80 GB | Yes, with PyTorch SDPA (`ks2v-run` picks it automatically; FA3 kernels only run on Hopper) |
| A100 40 GB / 48 GB cards | Only with `VAE_TILING=1` (peak ≈ 28 GiB, faint tile seams) |

Upstream's figures on one H100 at 512×768: Pro distilled takes 41.5 s per second of video at a
46 GiB peak; Lite distilled 8.7 s at 57 GiB. Expect an A100 to be slower.

## Use

```bash
ks2v-doctor
ks2v-download pro
ks2v-run --image start.png --audio audio.wav --prompt-file prompt.txt --negative-file negative.txt \
         --seed 835047 --out /workspace/out/s001-01.mp4 --fps 12
```

`--fps 12` also writes `s001-01.12fps.mp4` for our 12 fps narration timeline.

## Only the speaker talks: `ks2v-crop-run`

Kandinsky S2V was trained on one talking subject per clip. Given a picture with several characters it
moves every likely mouth (in our first full run the dog talked along with the man). `ks2v-crop-run`
sends Kandinsky only a crop around the speaker and blends the talking crop back:

```bash
ks2v-crop-run --image start.png --audio voice.wav --bbox 0.36,0.10,0.62,0.62 \
              --base listen.mp4 --prompt-file prompt.txt --seed 773762 --out shot.mp4 --fps 12
```

- `--bbox x1,y1,x2,y2`: the speaker, in pixels or 0–1 fractions. Boxes are worked out before the
  render (laptop pre-pass), not inside the container, so they can be checked and kept.
- `--base still|<video>`: the rest of the frame. `still` freezes everyone else (no seam possible);
  a video of the whole shot with nobody talking (an LTX take, or a silent render) keeps the
  listeners alive, and the crop edge must blend into it.
- The crop is padded (`--pad 0.30`), grown to a trained shape (1:1, 3:2, 2:3) so nothing is
  distorted, rendered at 512×512 / 768×512 / 512×768, scaled back with Lanczos, colour-matched once
  per shot (Lab mean/std from the first frame) and blended with a Gaussian feather (`--feather 0.14`).
- A box that no trained shape can hold is trimmed with a warning, and `box_fits: false` is
  recorded in `<stem>.crop.json`.
- For a chained shot, pass the chain frame as `--image`.

Test without a GPU (fake render, real cropping, colour matching, blending and audio muxing):
`python tests/test_crop_run.py`.

## Build

Push to `main` (or run **Actions → Build Kandinsky S2V 1.0 image**). The workflow builds on a
GitHub runner, pushes to GHCR, pulls the image back and smoke-tests it without a GPU.
See [docs/VAST.md](docs/VAST.md) for the vast.ai template.
