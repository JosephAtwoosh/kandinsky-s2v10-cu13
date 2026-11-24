"""No-GPU checks for ks2v-crop-run: bucket snapping, colour matching, feathered blending, audio muxing.
Run:  python tests/test_crop_run.py      (needs numpy, opencv-python-headless and ffmpeg)
"""
import json
import math
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import crop_run as cr  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok' if ok else 'XX'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILS.append(name)


def test_bucket_snapping():
    print("bucket snapping")
    W, H = 896, 512
    cases = {"tall person": (380, 60, 470, 500), "wide group": (100, 300, 800, 420), "square head": (600, 100, 700, 200),
             "near left edge": (0, 50, 60, 200), "near bottom right": (820, 440, 896, 512), "whole frame": (0, 0, 896, 512)}
    for name, box in cases.items():
        (x1, y1, x2, y2), aspect = cr.bucket_crop(box, W, H, 0.30)
        cw, ch = x2 - x1, y2 - y1
        inside = 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H
        contains = x1 <= box[0] + 1 and y1 <= box[1] + 1 and x2 >= min(box[2], W) - 1 and y2 >= min(box[3], H) - 1
        err = abs((cw / ch) / aspect - 1)
        # The whole 1.75:1 frame cannot fit a trained shape: the crop must trim it (and the wrapper warns).
        want_contains = name != "whole frame"
        check(f"{name}: crop {cw}x{ch} aspect {cw / ch:.3f} -> bucket {aspect:.3f}",
              inside and contains == want_contains and err < 0.02 and cw % 2 == 0 and ch % 2 == 0,
              f"inside={inside} contains={contains} aspect_err={err:.3%}")


def test_bbox_parsing():
    print("bbox parsing")
    check("normalised", cr.parse_bbox("0.5,0.25,0.75,1.0", 800, 400) == (400.0, 100.0, 600.0, 400.0))
    check("pixels", cr.parse_bbox("10,20,110,220", 800, 400) == (10.0, 20.0, 110.0, 220.0))


def test_end_to_end():
    print("end to end (dry run)")
    W, H, secs = 896, 512, 2.0
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        img = np.zeros((H, W, 3), np.uint8)
        img[:] = np.linspace(40, 200, W, dtype=np.uint8)[None, :, None]          # gradient background
        cv2.circle(img, (450, 230), 70, (200, 120, 60), -1)                       # the "speaker"
        cv2.circle(img, (150, 300), 50, (60, 160, 200), -1)                       # a listener, outside the crop
        cv2.imwrite(str(d / "start.png"), img)
        with wave.open(str(d / "voice.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            t = np.arange(int(16000 * secs)) / 16000
            w.writeframes((np.sin(2 * math.pi * 220 * t) * 8000).astype(np.int16).tobytes())
        (d / "prompt.txt").write_text("A hen speaks.", encoding="utf-8")
        out = d / "shot.mp4"
        cr.main(["--image", str(d / "start.png"), "--audio", str(d / "voice.wav"), "--out", str(out),
                 "--bbox", "380,160,520,300", "--prompt-file", str(d / "prompt.txt"), "--dry-run", "--fps", "12"])
        geo = json.loads((d / "shot.crop.json").read_text())
        x1, y1, x2, y2 = geo["crop"]
        check("output and 12 fps copy written", out.is_file() and (d / "shot.12fps.mp4").is_file())
        frames = cr.read_frames(out)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        check("size and frame count", frames.shape[1:3] == (H, W) and abs(len(frames) - secs * 25) <= 1,
              f"{frames.shape}, {len(frames)} frames")
        outside = np.ones((H, W), bool)
        outside[y1:y2, x1:x2] = False
        diff_out = np.abs(frames[5].astype(int) - rgb.astype(int))[outside].mean()
        check("outside the crop unchanged (still base)", diff_out < 3.0, f"mean diff {diff_out:.2f}")
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        probe = (slice(y1 + 8, y1 + 30), slice(cx - 10, cx + 10))                  # inside, away from the fake mouth
        shift = np.abs(frames[0][probe].astype(int).mean((0, 1)) - rgb[probe].astype(int).mean((0, 1))).max()
        check("colour shift of the fake render corrected", shift < 12.0, f"max channel diff {shift:.1f} (fake shift ~18)")
        ring = np.abs(np.diff(frames[5][cy, x1 - 3:x1 + 25].astype(int), axis=0)).max()
        check("blend edge has no hard step", ring < 30, f"max step across the left edge {ring}")
        r = subprocess.run([cr.ffmpeg_bin(), "-i", str(out)], capture_output=True, text=True)
        check("audio track muxed", "Audio: aac" in r.stderr)
        check("length matches the voice", abs(cr.audio_seconds(out) - secs) < 0.15, f"{cr.audio_seconds(out):.2f} s")

        # Moving base at 12 fps and a different size: resampled to 25 fps and the image size, held at its end.
        base = d / "base.mp4"
        p = subprocess.Popen([cr.ffmpeg_bin(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "448x256",
                              "-r", "12", "-i", "-", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(base)],
                             stdin=subprocess.PIPE)
        small = cv2.resize(rgb, (448, 256))
        for i in range(18):  # 1.5 s, shorter than the 2 s voice
            f = small.copy()
            cv2.circle(f, (75 + 3 * i, 150), 25, (60, 160, 200), -1)  # the listener moves
            p.stdin.write(f.tobytes())
        p.stdin.close()
        p.wait()
        out2 = d / "moving.mp4"
        cr.main(["--image", str(d / "start.png"), "--audio", str(d / "voice.wav"), "--out", str(out2),
                 "--bbox", "380,160,520,300", "--base", str(base), "--dry-run"])
        mv = cr.read_frames(out2)
        moved = np.abs(mv[1].astype(int) - mv[30].astype(int))[outside].mean()
        check("moving base: full length, listener moves outside the crop",
              abs(len(mv) - secs * 25) <= 1 and moved > 0.5, f"{len(mv)} frames, outside change {moved:.2f}")


if __name__ == "__main__":
    test_bucket_snapping()
    test_bbox_parsing()
    test_end_to_end()
    print("\nALL PASSED" if not FAILS else f"\nFAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)
