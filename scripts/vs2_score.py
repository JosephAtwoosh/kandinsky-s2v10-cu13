#!/opt/vs2/venv/bin/python
"""ltx2-vs2-score: score rendered shots with VideoScore2 (runs in /opt/vs2/venv).

    ltx2-vs2-score                 score every $VS2_OUT/<shot>.mp4 that has no up-to-date score
    ltx2-vs2-score s02-01 s02-02   score these shots

Env: VS2_MODEL (default /models/videoscore2), VS2_OUT (/workspace/out: <shot>.mp4 in, <shot>.vs2.json
out), VS2_SHOTS (/workspace/shots: <shot>/prompt.txt is the prompt each clip is judged against).

Writes /workspace/out/<shot>.vs2.json:
  {"visual": soft 1-5, "alignment": soft 1-5, "physics": soft 1-5, "hard": [v, t, p], "rationale": text}
Method follows the VideoScore2 model card (Qwen2.5-VL chat template, 2 fps sampling, scores
1-5 plus log-likelihood "soft" scores), loaded in bfloat16 with a fixed seed.
"""
import glob
import json
import os
import re
import sys
import time
from string import Template

import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForVision2Seq, AutoProcessor

MODEL = os.environ.get("VS2_MODEL", "/models/videoscore2")
OUT = os.environ.get("VS2_OUT", "/workspace/out")
SHOTS = os.environ.get("VS2_SHOTS", "/workspace/shots")
QUERY = Template("""
You are an expert for evaluating AI-generated videos from three dimensions:
(1) visual quality – clarity, smoothness, artifacts;
(2) text-to-video alignment – fidelity to the prompt;
(3) physical/common-sense consistency – naturalness and physics plausibility.

Video prompt: $t2v_prompt

Please output in this format:
visual quality: <v_score>;
text-to-video alignment: <t_score>,
physical/common-sense consistency: <p_score>
""")


def soft(hard, idx, scores, tok):
    if hard is None or idx < 0:
        return None
    logp = torch.log_softmax(scores[idx][0].float(), dim=-1)
    probs = []
    for s in range(1, 6):
        ids = tok.encode(str(s), add_special_tokens=False)
        if len(ids) == 1:
            probs.append((s, float(np.exp(logp[ids[0]].item()))))
    if not probs:
        return None
    best, pmax = max(probs, key=lambda x: x[1])
    total = sum(p for _, p in probs)
    return round(best * pmax / total, 4) if total else None


def score_index(label, tok, gen_ids):
    text = tok.decode(gen_ids, skip_special_tokens=False)
    m = re.search(r"(?:\(\d+\)\s*|\n\s*)?" + re.escape(label), text, flags=re.I)
    if not m:
        return -1
    d = re.search(r"\d", text[m.end():])
    if not d:
        return -1
    target = text[:m.end() + d.start() + 1]
    for i in range(len(gen_ids)):
        if tok.decode(gen_ids[:i + 1], skip_special_tokens=False) == target:
            return i
    return -1


def todo(ids):
    if ids:
        return ids
    out = []
    for mp4 in sorted(glob.glob(f"{OUT}/s??-??.mp4")):
        sid = os.path.basename(mp4)[:-4]
        js = f"{OUT}/{sid}.vs2.json"
        if not os.path.exists(js) or os.path.getmtime(js) < os.path.getmtime(mp4):
            out.append(sid)
    return out


def main():
    ids = todo(sys.argv[1:])
    if not ids:
        print("nothing to score")
        return
    torch.manual_seed(1234)
    model = AutoModelForVision2Seq.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda").eval()
    proc = AutoProcessor.from_pretrained(MODEL)
    tok = proc.tokenizer
    for sid in ids:
        t0 = time.time()
        prompt = open(f"{SHOTS}/{sid}/prompt.txt", encoding="utf-8").read().strip()
        msgs = [{"role": "user", "content": [
            {"type": "video", "video": f"{OUT}/{sid}.mp4", "fps": 2.0},
            {"type": "text", "text": QUERY.substitute(t2v_prompt=prompt)}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msgs)
        inputs = proc(text=[text], images=imgs, videos=vids, fps=2.0, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            g = model.generate(**inputs, max_new_tokens=1024, output_scores=True, return_dict_in_generate=True,
                               do_sample=True, temperature=0.7)
        gen = g.sequences[0, inputs["input_ids"].shape[1]:].tolist()
        out_text = proc.batch_decode(g.sequences[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        m = re.search(r"visual quality:\s*(\d+).*?text-to-video alignment:\s*(\d+).*?physical/common-sense consistency:\s*(\d+)",
                      out_text, re.S | re.I)
        hard = [int(m.group(i)) for i in (1, 2, 3)] if m else [None, None, None]
        labels = ("visual quality:", "text-to-video alignment:", "physical/common-sense consistency:")
        softs = [soft(h, score_index(l, tok, gen), g.scores, tok) for h, l in zip(hard, labels)]
        res = {"shot": sid, "visual": softs[0], "alignment": softs[1], "physics": softs[2], "hard": hard,
               "rationale": out_text.strip(), "seconds": round(time.time() - t0, 1)}
        json.dump(res, open(f"{OUT}/{sid}.vs2.json", "w"), indent=1)
        print(f"{sid}: visual {softs[0]}  alignment {softs[1]}  physics {softs[2]}  ({res['seconds']} s)", flush=True)


if __name__ == "__main__":
    main()
