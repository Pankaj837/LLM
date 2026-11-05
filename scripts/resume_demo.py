#!/usr/bin/env python
"""Stop-and-resume demonstration on CPU (Raju's task 4): proves a run can be stopped or killed and continued with no change.

    python scripts/resume_demo.py --out runs/resume_demo

Trains a tiny model on synthetic tokens in four ways and compares the FINAL WEIGHTS bit for bit:
  A  uninterrupted run (reference)
  B  graceful stop (a STOP file is created mid-run), then ``--resume auto``
  C  graceful stop by signal (CTRL_BREAK on Windows, SIGTERM elsewhere), then ``--resume auto``
  D  hard kill (the process is killed outright, no chance to save), then ``--resume auto``
For B, C and D the final checkpoint must equal A exactly. D redoes the steps since the last periodic save, so it also shows
how many steps a hard kill costs. Exit status 0 only if every scenario matches. Writes ``<out>/demo_report.md``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from llm import data                                    # noqa: E402
from llm.checkpoint import read_checkpoint               # noqa: E402

VOCAB, STEPS, SAVE = 64, 120, 10


def make_bins(d: str):
    rng = np.random.RandomState(0)
    for name, n, seed in (("train.bin", 20000, 0), ("val.bin", 4000, 1)):
        rng = np.random.RandomState(seed)
        base = np.tile(np.arange(1, 17), n // 16 + 1)[:n]
        ids = np.where(rng.rand(n) < 0.02, rng.randint(1, VOCAB, n), base)
        data.write_bin(os.path.join(d, name), ids, vocab_size=VOCAB, eos_id=VOCAB - 1)


def cmd(d: str, out: str, resume: bool):
    c = [sys.executable, "-m", "llm.train", "--train-bin", os.path.join(d, "train.bin"), "--val-bin", os.path.join(d, "val.bin"),
         "--out", out, "--preset", "tiny", "--seq-len", "32", "--batch-size", "8", "--max-steps", str(STEPS),
         "--warmup-steps", "5", "--eval-interval", "20", "--eval-batches", "4", "--save-interval", str(SAVE),
         "--log-interval", "1", "--lr", "3e-3", "--device", "cpu", "--dtype", "fp32"]
    return c + (["--resume", "auto"] if resume else [])


def env():
    e = dict(os.environ)
    e["PYTHONPATH"] = os.path.join(HERE, "..", "src") + os.pathsep + e.get("PYTHONPATH", "")
    return e


def last_logged_step(out: str) -> int:
    p = os.path.join(out, "metrics.jsonl")
    if not os.path.exists(p):
        return 0
    steps = [json.loads(l)["step"] for l in open(p, encoding="utf-8") if l.strip()]
    return max(steps, default=0)


def ckpt_step(out: str) -> int:
    p = os.path.join(out, "ckpt_last.pt")
    return int(read_checkpoint(p)[2]["step"]) if os.path.exists(p) else 0


def run_until(proc, out: str, step: int, timeout=180):
    t0 = time.time()
    while proc.poll() is None and last_logged_step(out) < step:
        if time.time() - t0 > timeout:
            proc.kill()
            raise TimeoutError("training did not reach the stop point")
        time.sleep(0.05)


def same_weights(a: str, b: str) -> bool:
    sa, sb = read_checkpoint(a)[0], read_checkpoint(b)[0]
    return all(torch.equal(sa[k], sb[k]) for k in sa)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/resume_demo")
    args = ap.parse_args()
    root = os.path.abspath(args.out)
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    make_bins(root)
    e = env()
    rows = []

    ref = os.path.join(root, "A_uninterrupted")
    subprocess.run(cmd(root, ref, False), env=e, check=True, stdout=subprocess.DEVNULL)
    ref_ckpt = os.path.join(ref, "ckpt_last.pt")
    rows.append(("A", "uninterrupted run", STEPS, "-", "reference"))

    # B: STOP file
    out = os.path.join(root, "B_stop_file")
    p = subprocess.Popen(cmd(root, out, False), env=e, stdout=subprocess.DEVNULL)
    run_until(p, out, 47)
    open(os.path.join(out, "STOP"), "w").close()
    p.wait()
    stopped = ckpt_step(out)
    subprocess.run(cmd(root, out, True), env=e, check=True, stdout=subprocess.DEVNULL)
    rows.append(("B", "STOP file, then --resume auto", ckpt_step(out), f"stopped and saved at step {stopped}", same_weights(ref_ckpt, os.path.join(out, "ckpt_last.pt"))))

    # C: signal
    out = os.path.join(root, "C_signal")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    p = subprocess.Popen(cmd(root, out, False), env=e, stdout=subprocess.DEVNULL, creationflags=flags)
    run_until(p, out, 53)
    p.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGTERM)
    p.wait()
    stopped = ckpt_step(out)
    subprocess.run(cmd(root, out, True), env=e, check=True, stdout=subprocess.DEVNULL)
    rows.append(("C", "signal (CTRL_BREAK/SIGTERM), then --resume auto", ckpt_step(out), f"stopped and saved at step {stopped}", same_weights(ref_ckpt, os.path.join(out, "ckpt_last.pt"))))

    # D: hard kill
    out = os.path.join(root, "D_hard_kill")
    p = subprocess.Popen(cmd(root, out, False), env=e, stdout=subprocess.DEVNULL)
    run_until(p, out, 67)
    p.kill()
    p.wait()
    logged = last_logged_step(out)
    saved = ckpt_step(out)
    subprocess.run(cmd(root, out, True), env=e, check=True, stdout=subprocess.DEVNULL)
    rows.append(("D", "hard kill, then --resume auto", ckpt_step(out), f"killed after step {logged}; last save was step {saved}, {logged - saved} steps redone", same_weights(ref_ckpt, os.path.join(out, "ckpt_last.pt"))))

    lines = ["# Stop-and-resume demonstration (CPU, tiny model)", "",
             f"Synthetic data, {STEPS} steps, checkpoint every {SAVE} steps, seed fixed. Final weights compared bit for bit with the uninterrupted run A.", "",
             "| Scenario | What happens | Final step | Interruption | Final weights equal to A |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4] if r[0] != 'A' else 'reference'} |")
    ok = all(r[4] is True for r in rows if r[0] != "A")
    lines += ["", "**Result: " + ("PASS, every interrupted run ends with weights identical to the uninterrupted one." if ok else "FAIL") + "**"]
    text = "\n".join(lines)
    with open(os.path.join(root, "demo_report.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
