#!/usr/bin/env python3
"""ks2v-crop-run: make only the speaker talk, by rendering a crop around them and compositing it back.

Kandinsky S2V was trained on one talking subject per clip and has no way to bind the voice to one
face: given a picture with several characters, it moves every likely mouth. This wrapper sends
Kandinsky only a crop around the speaker, then blends the talking crop back over a base video of
the whole frame in which nobody talks.

    ks2v-crop-run --image start.png --audio voice.wav --bbox 0.36,0.10,0.62,0.62 --out shot.mp4 \
                  [--base still | --base listen.mp4] [--prompt-file p.txt] [--negative-file n.txt] \
                  [--seed N] [--fps 12] [--pad 0.30] [--feather 0.14] [--model lite|pro] [--dry-run]

--bbox     speaker box x1,y1,x2,y2 in pixels, or 0..1 fractions of the image (all values <= 1).
--base     what the rest of the frame shows: "still" (the start picture, frozen: no seam, but
           nobody else moves) or a video of the whole shot with nobody talking (e.g. an LTX take
           or a silent Kandinsky render: everyone listens and reacts; the crop edge must blend).
--pad      grow the box by this fraction of its size on every side before cropping.
--feather  width of the soft blend edge, as a fraction of the crop's shorter side.
--dry-run  do not call ks2v-run: fake the talking crop (the crop with a colour shift and a moving
           patch) so cropping, colour matching, blending and audio muxing can be tested without a GPU.

The crop is grown to one of Kandinsky's trained shapes (1:1, 3:2 or 2:3) so nothing is distorted,
rendered at that bucket's size (512x512, 768x512 or 512x768), scaled back with Lanczos, colour-
matched once per shot (mean/std per channel in Lab, from the first frame) and blended through a
Gaussian-feathered mask. Output: 25 fps at the image's size with the --audio track, plus
<stem>.<N>fps.mp4 when --fps N is given, <stem>.crop.json (the geometry) and <stem>.log.
For a chained shot, pass the chain frame as --image: crop and base both start from it.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

FPS = 25
BUCKETS = {1.0: (512, 512), 1.5: (512, 768), 2 / 3: (768, 512)}  # aspect w/h -> (h, w)
SAFEGUARD = ("Locked-off tripod camera, completely static background, minimal head movement; "
             "only this one character talks.")


def ffmpeg_bin() -> str:
    exe = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg  # laptop fallback

    return imageio_ffmpeg.get_ffmpeg_exe()


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)


def parse_bbox(text: str, w: int, h: int) -> tuple[float, float, float, float]:
    v = [float(x) for x in text.split(",")]
    if len(v) != 4:
        raise SystemExit("--bbox needs four numbers: x1,y1,x2,y2")
    if all(0.0 <= x <= 1.0 for x in v):
        v = [v[0] * w, v[1] * h, v[2] * w, v[3] * h]
    x1, y1, x2, y2 = v
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise SystemExit(f"--bbox {text} is outside the {w}x{h} image or empty")
    return x1, y1, x2, y2


def bucket_crop(box, w: int, h: int, pad: float) -> tuple[tuple[int, int, int, int], float]:
    """Pad the box, then grow it to the nearest trained aspect, kept inside the image.
    Returns (x1, y1, x2, y2) in whole pixels and the aspect (w/h) chosen."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = x1 - pad * bw, x2 + pad * bw
    y1, y2 = y1 - pad * bh, y2 + pad * bh
    cw, ch = min(x2 - x1, w), min(y2 - y1, h)
    aspect = min(BUCKETS, key=lambda a: abs(np.log((cw / ch) / a)))
    if cw / ch < aspect:            # too narrow: widen, or if the image is too narrow, shorten
        cw = ch * aspect
        if cw > w:
            cw, ch = w, w / aspect
    else:                           # too wide: heighten, or if the image is too short, narrow
        ch = cw / aspect
        if ch > h:
            ch, cw = h, h * aspect
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    cx = min(max(cx, cw / 2), w - cw / 2)
    cy = min(max(cy, ch / 2), h - ch / 2)
    cw, ch = int(round(cw)) // 2 * 2, int(round(ch)) // 2 * 2
    ax1 = int(round(cx - cw / 2))
    ay1 = int(round(cy - ch / 2))
    ax1, ay1 = min(max(ax1, 0), w - cw), min(max(ay1, 0), h - ch)
    return (ax1, ay1, ax1 + cw, ay1 + ch), aspect


def read_frames(path: Path, size: tuple[int, int] | None = None, fps: int = FPS) -> np.ndarray:
    """Decode a video to RGB uint8 frames at `fps` (and `size` = (w, h) if given) via ffmpeg."""
    w, h = size if size else probe_size(path)
    vf = f"fps={fps}" + (f",scale={w}:{h}:flags=lanczos" if size else "")
    out = subprocess.run([ffmpeg_bin(), "-v", "error", "-i", str(path), "-vf", vf, "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], check=True, capture_output=True, stdin=subprocess.DEVNULL).stdout
    return np.frombuffer(out, np.uint8).reshape(-1, h, w, 3)


def probe_size(path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(path))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not w or not h:
        raise SystemExit(f"cannot read video size of {path}")
    return w, h


def audio_seconds(path: Path) -> float:
    r = subprocess.run([ffmpeg_bin(), "-i", str(path)], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    for line in r.stderr.splitlines():
        if "Duration:" in line:
            hh, mm, ss = line.split("Duration:")[1].split(",")[0].strip().split(":")
            return int(hh) * 3600 + int(mm) * 60 + float(ss)
    raise SystemExit(f"cannot read the length of {path}")


def colour_transfer(src_ref: np.ndarray, dst_ref: np.ndarray):
    """Per-channel mean/std match in Lab, from one reference frame each; returns a function."""
    s = cv2.cvtColor(src_ref, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    d = cv2.cvtColor(dst_ref, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    sm, ss, dm, ds = s.mean(0), s.std(0) + 1e-3, d.mean(0), d.std(0) + 1e-3
    gain, bias = ds / ss, dm - sm * (ds / ss)

    def apply(frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab = np.clip(lab * gain + bias, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return apply


def feather_mask(cw: int, ch: int, feather: float) -> np.ndarray:
    """1.0 in the middle, falling smoothly to 0.0 at the crop's border."""
    f = max(2, int(round(feather * min(cw, ch))))
    m = np.zeros((ch, cw), np.float32)
    m[f:ch - f, f:cw - f] = 1.0
    k = 2 * f + 1
    m = cv2.GaussianBlur(m, (k, k), f / 2)
    return m[..., None]


def fake_render(crop_png: Path, seconds: float, out: Path, size_wh: tuple[int, int]) -> None:
    """--dry-run stand-in for ks2v-run: the crop at the bucket size, shifted in colour (as the VAE
    does), with a small patch moving like a talking mouth, at 25 fps for the audio's length."""
    img = cv2.cvtColor(cv2.imread(str(crop_png)), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size_wh, interpolation=cv2.INTER_LANCZOS4)
    img = np.clip(img.astype(np.float32) * 0.9 + 18, 0, 255).astype(np.uint8)
    n = int(round(seconds * FPS))
    w, h = size_wh
    p = subprocess.Popen([ffmpeg_bin(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(out)],
                         stdin=subprocess.PIPE)
    for i in range(n):
        f = img.copy()
        o = int(0.08 * h * (1 + np.sin(i / 2.0)))
        cv2.rectangle(f, (w // 2 - w // 10, h // 2 + h // 8), (w // 2 + w // 10, h // 2 + h // 8 + o), (20, 10, 10), -1)
        p.stdin.write(f.tobytes())
    p.stdin.close()
    if p.wait():
        raise SystemExit("dry-run fake render failed")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ks2v-crop-run", description=__doc__.split("\n\n")[0])
    ap.add_argument("--image", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bbox", required=True)
    ap.add_argument("--base", default="still")
    ap.add_argument("--prompt-file")
    ap.add_argument("--negative-file")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--fps", type=int)
    ap.add_argument("--pad", type=float, default=0.30)
    ap.add_argument("--feather", type=float, default=0.14)
    ap.add_argument("--model", default="lite", choices=("lite", "pro"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-temp", action="store_true")
    a = ap.parse_args(argv)

    t0 = time.time()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.with_suffix("")
    image = cv2.cvtColor(cv2.imread(a.image), cv2.COLOR_BGR2RGB)
    if image is None:
        raise SystemExit(f"cannot read {a.image}")
    H, W = image.shape[:2]
    box = parse_bbox(a.bbox, W, H)
    (x1, y1, x2, y2), aspect = bucket_crop(box, W, H, a.pad)
    cw, ch = x2 - x1, y2 - y1
    bh, bw = BUCKETS[aspect]
    seconds = audio_seconds(Path(a.audio))
    # A box wider or taller than any trained shape allows (e.g. the whole 1.75:1 frame) cannot fit:
    # the crop then trims it. Never silently: it is recorded and warned about.
    box_fits = x1 <= box[0] + 1 and y1 <= box[1] + 1 and x2 >= box[2] - 1 and y2 >= box[3] - 1
    if not box_fits:
        print(f"WARNING: the speaker box {a.bbox} does not fit any trained crop shape; the crop trims it", file=sys.stderr)
    geo = {"image": [W, H], "bbox": [round(v, 1) for v in box], "box_fits": box_fits,
           "crop": [x1, y1, x2, y2], "crop_size": [cw, ch],
           "aspect": round(aspect, 4), "render": [bw, bh], "pad": a.pad, "feather": a.feather, "base": a.base,
           "audio_seconds": round(seconds, 3), "dry_run": a.dry_run}

    tmp = Path(tempfile.mkdtemp(prefix="ks2v-crop-", dir=str(out.parent)))
    try:
        crop_png = tmp / "crop.png"
        cv2.imwrite(str(crop_png), cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_RGB2BGR))
        prompt = (Path(a.prompt_file).read_text(encoding="utf-8").strip() if a.prompt_file else "")
        (tmp / "prompt.txt").write_text((prompt + " " + SAFEGUARD).strip() + "\n", encoding="utf-8")
        crop_mp4 = tmp / "crop.mp4"
        if a.dry_run:
            fake_render(crop_png, seconds, crop_mp4, (bw, bh))
        else:
            cmd = ["ks2v-run", "--model", a.model, "--image", str(crop_png), "--audio", a.audio,
                   "--prompt-file", str(tmp / "prompt.txt"), "--out", str(crop_mp4)]
            if a.negative_file:
                cmd += ["--negative-file", a.negative_file]
            if a.seed is not None:
                cmd += ["--seed", str(a.seed)]
            env = dict(os.environ, RESOLUTION=f"{bh}x{bw}")
            subprocess.run(cmd, check=True, env=env, stdin=subprocess.DEVNULL)

        crop_frames = read_frames(crop_mp4, (cw, ch))  # Lanczos back to the crop's own size
        n = len(crop_frames)
        if a.base == "still":
            base = np.repeat(image[None], n, axis=0)
        else:
            base = read_frames(Path(a.base), (W, H))
            if len(base) < n:  # hold the base's last frame if it is shorter than the voice
                base = np.concatenate([base, np.repeat(base[-1:], n - len(base), axis=0)])
            base = base[:n]
        match = colour_transfer(crop_frames[0], base[0, y1:y2, x1:x2])
        mask = feather_mask(cw, ch, a.feather)

        silent = tmp / "video.mp4"
        p = subprocess.Popen([ffmpeg_bin(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                              "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                              str(silent)], stdin=subprocess.PIPE)
        for i in range(n):
            frame = base[i].copy()
            region = frame[y1:y2, x1:x2].astype(np.float32)
            talk = match(crop_frames[i]).astype(np.float32)
            frame[y1:y2, x1:x2] = np.clip(region * (1 - mask) + talk * mask, 0, 255).astype(np.uint8)
            p.stdin.write(frame.tobytes())
        p.stdin.close()
        if p.wait():
            raise SystemExit("encoding the composite failed")
        run([ffmpeg_bin(), "-v", "error", "-y", "-i", str(silent), "-i", a.audio, "-map", "0:v", "-map", "1:a",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)])
        if a.fps and a.fps != FPS:
            run([ffmpeg_bin(), "-v", "error", "-y", "-i", str(out), "-vf", f"fps={a.fps}", "-c:v", "libx264",
                 "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "copy", f"{stem}.{a.fps}fps.mp4"])
        geo.update(frames=n, wall_seconds=round(time.time() - t0))
        Path(f"{stem}.crop.json").write_text(json.dumps(geo, indent=1) + "\n", encoding="utf-8")
        with open(f"{stem}.log", "a", encoding="utf-8") as log:
            log.write(f"CROP {json.dumps(geo)}\nWALL_SECONDS={geo['wall_seconds']}\n")
        print(f"wrote {out} ({n} frames, crop {cw}x{ch} at {x1},{y1}, rendered {bw}x{bh})")
    finally:
        if not a.keep_temp:
            shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
