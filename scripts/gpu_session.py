#!/usr/bin/env python
"""The whole scarce-GPU session as one command: preflight -> benchmark -> 1.5B smoke test -> timed 100M run -> per-domain evaluation.

    python scripts/gpu_session.py preflight --bins bins --expect-tag scale-run-rc6        # free: run it on the GPU machine first
    python scripts/gpu_session.py all       --bins bins --expect-tag scale-run-rc6 --minutes 100

Why it exists: only ~3 H100-hours are available, and every decision a person makes at the keyboard is billed time. This script
makes the decisions from measurements, with stop rules, and writes one report to post in the group.

  preflight   (no GPU time) GPU visible, code == the expected tag and clean, tokenizer == the frozen hash, build manifests carry that
              hash, every token file passes ``check_bins``, one balanced validation file exists, disk and permissions are fine.
  bench       ``benchmark.py`` on synthetic tokens for the training preset and the smoke preset; picks the fastest micro-batch that
              leaves memory headroom and divides the global batch.
  smoke       the smoke preset (1p5b) for 60 steps -> checkpoint -> resume to 80 steps on the real mix. Finite losses, loss falling,
              resume continuing: otherwise the session STOPS before the expensive run. Skipped (with a note) if 1.5B does not fit.
  train       a 30-step calibration on the real mix measures the true tokens/s (data loading included), then
              ``max_steps = tokens_per_s x budget / tokens_per_step``; the run is killed if it overruns its budget by 10%.
              Re-running the same command resumes from ``ckpt_last.pt`` with the same step count.
  eval        ``eval_domains.py`` logic on the best and last checkpoints, one validation file per domain.
  report      ``session_report.md``: hardware, benchmark table, smoke verdict, training curve summary, per-domain table, cost.

Everything is rehearsed on CPU (``--cpu-ok --preset tiny --smoke-preset small --seq-len 64 --global-seqs 8``): see tests/test_gpu_session.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import benchmark as bm  # noqa: E402
import build_dataset as bd  # noqa: E402
import check_bins as cb  # noqa: E402
import eval_domains as ed  # noqa: E402
import merge_bins as mb  # noqa: E402
import torch  # noqa: E402
from llm.data import read_header  # noqa: E402
from llm.tokenizer import BPETokenizer  # noqa: E402
from llm.train import PRESETS, run_info  # noqa: E402

DEFAULT_VOCAB = os.path.join(ROOT, "tokenizers", "vocab32k_v2", "vocab32k.tok")
DEFAULT_CONFIG = os.path.join(ROOT, "configs", "run_100m.example.json")


# ── small helpers ───────────────────────────────────────────────────────────────────────────────

frozen_sha = bd.frozen_sha


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30).stdout.strip()


def read_metrics(d: str) -> list:
    p = os.path.join(d, "metrics.jsonl")
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()] if os.path.exists(p) else []


def run_logged(cmd: list, log_path: str, timeout_s: float | None = None) -> int:
    """Run a command, echo its output, keep a copy in ``log_path``; kill it when ``timeout_s`` is exceeded (returns 124)."""
    env = {**os.environ, "PYTHONPATH": os.path.join(ROOT, "src") + os.pathsep + os.environ.get("PYTHONPATH", ""), "PYTHONUNBUFFERED": "1"}
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    print("$ " + " ".join(cmd), flush=True)
    with open(log_path, "a", encoding="utf-8") as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", cwd=ROOT, env=env)

        def pump():
            for line in p.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)

        t = threading.Thread(target=pump, daemon=True)
        t.start()
        try:
            rc = p.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            p.terminate()
            try:
                p.wait(30)
            except subprocess.TimeoutExpired:
                p.kill()
            rc = 124
        t.join(10)
    return rc


class Session:
    """``session.json`` in the output folder: every decision and measurement, so a restart repeats nothing."""

    def __init__(self, out: str):
        self.dir = out
        self.path = os.path.join(out, "session.json")
        os.makedirs(out, exist_ok=True)
        self.data = json.load(open(self.path, encoding="utf-8")) if os.path.exists(self.path) else {}

    def save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, self.path)


# ── data selection ──────────────────────────────────────────────────────────────────────────────

def load_mix(a) -> tuple[list, list]:
    """([(name, weight)] for domains whose train bin exists, [names missing]) from the config, in config order."""
    cfg = json.load(open(a.config, encoding="utf-8"))
    present, missing = [], []
    for d in cfg["domains"]:
        (present if os.path.exists(os.path.join(a.bins, f"{d['name']}.train.bin")) else missing).append((d["name"], float(d["weight"])))
    return present, [n for n, _ in missing]


def val_paths(a, names) -> dict:
    return {n: os.path.join(a.bins, f"{n}.val.bin") for n in names if os.path.exists(os.path.join(a.bins, f"{n}.val.bin"))}


def ensure_val_mix(a, names) -> str:
    """One validation file with the same number of tokens from every domain (``--val-bin`` takes a single file)."""
    out = os.path.join(a.bins, "val_mix.bin")
    vals = list(val_paths(a, names).values())
    if not vals:
        raise SystemExit("no validation files (<domain>.val.bin) in " + a.bins)
    if not os.path.exists(out) or os.path.getmtime(out) < max(os.path.getmtime(p) for p in vals):
        taken = mb.capped_mix(vals, out, a.val_cap)
        print(f"val_mix.bin: {sum(taken.values()):,} tokens from {len(taken)} domains (<= {a.val_cap:,} each)")
    return out


def train_args(a, present, val_mix: str) -> list:
    mix = [f"{os.path.join(a.bins, n + '.train.bin')}={w}" for n, w in present]
    return ["--train-mix", *mix, "--val-bin", val_mix, "--vocab", a.vocab, "--seq-len", str(a.seq_len)]


def launcher(a) -> list:
    if a.gpus > 1:
        return [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={a.gpus}", "-m", "llm.train"]
    return [sys.executable, "-m", "llm.train"]


# ── stages ──────────────────────────────────────────────────────────────────────────────────────

def stage_preflight(a, st: Session) -> int:
    checks = []

    def add(name: str, ok: bool, detail: str = "", warn: bool = False):
        checks.append((name, ok, detail, warn))

    cuda = torch.cuda.is_available()
    if cuda:
        p = torch.cuda.get_device_properties(0)
        mem = p.total_memory / 1e9
        add("GPU visible", torch.cuda.device_count() >= a.gpus, f"{torch.cuda.device_count()} x {p.name}, {mem:.0f} GB, bf16={torch.cuda.is_bf16_supported()}")
        add("bf16 supported", torch.cuda.is_bf16_supported())
        add("GPU memory", mem >= 24, f"{mem:.0f} GB (the 1.5B smoke test needs about 24 GB of state)", warn=mem < 40)
    else:
        add("GPU visible", a.cpu_ok, "no CUDA device" + (" (rehearsal on CPU)" if a.cpu_ok else ""))
    info = run_info()
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    add("code", True, f"commit {str(info['git_commit'])[:10]}" + (" (modified files!)" if dirty else ""), warn=dirty)
    if a.expect_tag:
        tag_commit = git("rev-parse", a.expect_tag + "^{commit}")
        add(f"checked out = {a.expect_tag}", bool(tag_commit) and tag_commit == info["git_commit"] and not dirty,
            f"HEAD {str(info['git_commit'])[:10]} vs tag {tag_commit[:10] or 'not found'}" + ("; working tree modified" if dirty else ""))
    sha = BPETokenizer.file_sha256(a.vocab)
    add("tokenizer = frozen hash", sha == frozen_sha(), sha[:16] + "...")
    manifests = sorted(glob.glob(os.path.join(a.bins, "*manifest*.json")))
    bad = [m for m in manifests if json.load(open(m, encoding="utf-8")).get("tokenizer_sha256") != sha]
    add("build manifests carry the frozen hash", bool(manifests) and not bad,
        f"{len(manifests)} manifest(s)" + (f", WRONG: {[os.path.basename(m) for m in bad]}" if bad else ""), warn=not manifests)
    present, missing = load_mix(a)
    add("domains with token files", len(present) > 0, f"{len(present)} of {len(present) + len(missing)}"
        + (f"; missing: {', '.join(missing)}" if missing else ""), warn=bool(missing))
    names = [n for n, _ in present]
    if present:
        try:
            val_mix = ensure_val_mix(a, names)
            files = [os.path.join(a.bins, f"{n}.train.bin") for n in names] + list(val_paths(a, names).values()) + [val_mix]
            add("token files pass check_bins", cb.main([*files, "--vocab", a.vocab]) == 0, f"{len(files)} files")
        except (SystemExit, ValueError) as e:
            add("token files pass check_bins", False, str(e))
        no_val = [n for n in names if n not in val_paths(a, names)]
        add("every domain has a validation file", not no_val, ", ".join(no_val), warn=bool(no_val))
    free = shutil.disk_usage(st.dir).free / 1e9
    add("free disk", free >= a.min_free_gb, f"{free:.0f} GB (need {a.min_free_gb:.0f})")
    add("preset known", a.preset in PRESETS and a.smoke_preset in PRESETS, f"{a.preset}, {a.smoke_preset}")
    add("global batch divides", a.global_seqs % a.gpus == 0, f"{a.global_seqs} sequences over {a.gpus} GPU(s)")
    print(f"\n{'check':<42}{'':<6}detail")
    failed = 0
    for name, ok, detail, warn in checks:
        tag = "ok" if ok and not warn else "WARN" if ok else "FAIL"
        failed += not ok
        print(f"{name:<42}{tag:<6}{detail}")
    print("\nPREFLIGHT:", "PASS" if not failed else f"FAIL ({failed})")
    st.data["preflight"] = {"ok": not failed, "run_info": info, "domains": names, "missing": missing}
    st.save()
    return 1 if failed else 0


def best_row(rows: list, preset: str, global_seqs: int, gpus: int, mem_total_gb: float | None):
    ok = [r for r in rows if r["preset"] == preset and r["status"] == "ok" and (global_seqs // gpus) % r["batch"] == 0]
    if mem_total_gb:
        roomy = [r for r in ok if r["peak_mem_GB"] is None or r["peak_mem_GB"] <= 0.85 * mem_total_gb]
        ok = roomy
    return max(ok, key=lambda r: r["tokens_per_s"]) if ok else None


def stage_bench(a, st: Session) -> int:
    rows = []
    for preset, sizes in ((a.preset, a.bench_batches), (a.smoke_preset, a.smoke_batches)):
        if preset in [r["preset"] for r in rows]:
            continue
        out = os.path.join(st.dir, f"bench_{preset}.json")
        argv = ["--presets", preset, "--batch-sizes", *map(str, sizes), "--seq-len", str(a.seq_len), "--rupees-per-hour", str(a.rupees),
                "--steps", str(a.bench_steps), "--warmup", "2", "--out", out]
        bm.main(argv)
        rows += json.load(open(out, encoding="utf-8"))["rows"]
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else None
    main_row = best_row(rows, a.preset, a.global_seqs, a.gpus, mem)
    smoke_row = best_row(rows, a.smoke_preset, a.global_seqs, a.gpus, mem) if a.smoke_preset != a.preset else None
    if main_row is None:
        print(f"BENCH FAILED: no usable micro-batch for {a.preset} (all rows OOM or invalid)")
        return 1
    st.data["bench"] = {"rows": rows, "main": main_row, "smoke": smoke_row, "gpu_mem_gb": mem}
    st.save()
    print(f"\nchosen for {a.preset}: micro-batch {main_row['batch']} at {main_row['tokens_per_s']:,} tokens/s (benchmark, synthetic tokens)"
          + (f"; smoke micro-batch {smoke_row['batch']}" if smoke_row else f"; {a.smoke_preset}: no row fits, smoke test will be skipped"))
    return 0


def _accum(a, micro: int) -> int:
    return max(1, a.global_seqs // (micro * a.gpus))


def stage_smoke(a, st: Session) -> int:
    if a.skip_smoke:
        st.data["smoke"] = {"status": "skipped (--skip-smoke)"}
        st.save()
        return 0
    row = (st.data.get("bench") or {}).get("smoke")
    if a.smoke_preset == a.preset or row is None:
        st.data["smoke"] = {"status": f"skipped: {a.smoke_preset} does not fit (or equals the training preset)"}
        st.save()
        print("SMOKE skipped:", st.data["smoke"]["status"])
        return 0
    present, _ = load_mix(a)
    val_mix = ensure_val_mix(a, [n for n, _ in present])
    d = os.path.join(st.dir, "smoke")
    shutil.rmtree(d, ignore_errors=True)
    common = [*launcher(a), *train_args(a, present, val_mix), "--out", d, "--preset", a.smoke_preset, "--batch-size", str(row["batch"]),
              "--grad-accum", str(a.smoke_accum), "--max-steps", "80",
              "--warmup-steps", "10", "--lr", "3e-4", "--log-interval", "10", "--eval-interval", "30", "--eval-batches", "4",
              "--save-interval", "1000"]
    log = os.path.join(st.dir, "smoke.log")
    t0 = time.time()
    rc = run_logged([*common, "--stop-step", "60"], log, a.smoke_minutes * 60)
    if rc == 0:
        rc = run_logged([*common, "--resume", os.path.join(d, "ckpt_last.pt")], log, a.smoke_minutes * 60)
    recs = [r for r in read_metrics(d) if "loss" in r]
    problems = []
    if rc != 0:
        problems.append(f"training process exited with code {rc}")
    if not recs or not all(math.isfinite(r["loss"]) for r in recs):
        problems.append("missing or non-finite loss")
    steps = [r["step"] for r in recs]
    if recs and (max(steps) < 80 or 60 not in steps):
        problems.append(f"did not reach step 80 after resuming (steps seen: {steps})")
    if recs and 60 in steps and max(steps) > 60:
        after = next(r for r in recs if r["step"] > 60)
        before = next(r for r in recs if r["step"] == 60)
        if after["loss"] > before["loss"] + 1.0:
            problems.append(f"loss jumped after resume: {before['loss']:.3f} -> {after['loss']:.3f}")
    if recs and recs[-1]["loss"] > recs[0]["loss"]:
        problems.append(f"loss did not fall: {recs[0]['loss']:.3f} -> {recs[-1]['loss']:.3f}")
    last = recs[-1] if recs else {}
    st.data["smoke"] = {"status": "FAIL" if problems else "PASS", "problems": problems, "micro_batch": row["batch"], "minutes": round((time.time() - t0) / 60, 1),
                        "first_loss": recs[0]["loss"] if recs else None, "last_loss": last.get("loss"),
                        "tokens_per_s": last.get("tokens_per_s"), "peak_mem_gb": last.get("peak_mem_gb")}
    st.save()
    for f in glob.glob(os.path.join(d, "*.pt")):                  # a 1.5B checkpoint with optimizer state is ~18 GB
        os.remove(f)
    print("\nSMOKE:", st.data["smoke"]["status"], "; ".join(problems))
    return 1 if problems else 0


def stage_train(a, st: Session) -> int:
    bench = st.data.get("bench")
    if not bench:
        print("run `bench` first")
        return 1
    present, _ = load_mix(a)
    val_mix = ensure_val_mix(a, [n for n, _ in present])
    micro = bench["main"]["batch"]
    accum = _accum(a, micro)
    tokens_per_step = micro * accum * a.gpus * a.seq_len
    out = os.path.join(st.dir, "run")
    base = [*launcher(a), *train_args(a, present, val_mix), "--preset", a.preset, "--batch-size", str(micro), "--grad-accum", str(accum),
            "--lr", str(a.lr), "--eval-interval", str(a.eval_interval), "--save-interval", str(a.save_interval), "--log-interval", "10"]
    plan = st.data.get("train")
    if plan and os.path.exists(os.path.join(out, "ckpt_last.pt")):
        print(f"resuming the interrupted run: {plan['max_steps']} steps planned")
    else:
        shutil.rmtree(os.path.join(st.dir, "calib"), ignore_errors=True)
        rc = run_logged([*base, "--out", os.path.join(st.dir, "calib"), "--max-steps", "30", "--warmup-steps", "5", "--eval-interval", "1000",
                         "--save-interval", "1000"], os.path.join(st.dir, "calib.log"), 600)
        recs = [r for r in read_metrics(os.path.join(st.dir, "calib")) if "tokens_per_s" in r]
        if rc != 0 or not recs:
            print("CALIBRATION FAILED (the real mix does not train): do not start the run")
            return 1
        measured = recs[-1]["tokens_per_s"]
        budget_s = a.minutes * 60 * a.efficiency
        steps = int(measured * budget_s / tokens_per_step)
        if a.max_steps_cap:
            steps = min(steps, a.max_steps_cap)
        steps = max(steps, 50)
        warm = min(300, max(5, steps // 20))
        plan = {"micro_batch": micro, "grad_accum": accum, "tokens_per_step": tokens_per_step, "measured_tokens_per_s": measured,
                "bench_tokens_per_s": bench["main"]["tokens_per_s"], "max_steps": steps, "warmup_steps": warm,
                "planned_tokens": steps * tokens_per_step, "minutes_budget": a.minutes}
        st.data["train"] = plan
        st.save()
        shutil.rmtree(os.path.join(st.dir, "calib"), ignore_errors=True)
        # epochs each domain will see: a warning, never a stop
        total = sum(w for _, w in present)
        for n, w in present:
            ep = (w / total) * plan["planned_tokens"] / max(read_header(os.path.join(a.bins, f"{n}.train.bin")).total_tokens, 1)
            if ep > 4:
                print(f"WARNING: {n} repeats {ep:.1f}x in this run")
        print(f"\nplan: {steps} steps x {tokens_per_step:,} tokens = {plan['planned_tokens']:,} tokens; "
              f"measured {measured:,.0f} tokens/s (benchmark said {bench['main']['tokens_per_s']:,})")
    cmd = [*base, "--out", out, "--max-steps", str(plan["max_steps"]), "--warmup-steps", str(plan["warmup_steps"])]
    if os.path.exists(os.path.join(out, "ckpt_last.pt")):
        cmd += ["--resume", os.path.join(out, "ckpt_last.pt")]
    t0 = time.time()
    rc = run_logged(cmd, os.path.join(st.dir, "train.log"), a.minutes * 60 * 1.10 + 120)
    recs = read_metrics(out)
    st.data["train"]["wall_minutes"] = round(st.data["train"].get("wall_minutes", 0) + (time.time() - t0) / 60, 1)
    st.data["train"]["exit_code"] = rc
    st.save()
    if rc == 124:
        print("TRAINING STOPPED: exceeded its time budget. ckpt_last.pt is kept; evaluate it, or re-run this command to continue.")
        return 1
    if rc != 0:
        print(f"TRAINING FAILED (exit code {rc}); re-running the same command resumes from the last checkpoint")
        return 1
    return 0


def stage_eval(a, st: Session) -> int:
    present, _ = load_mix(a)
    names = [n for n, _ in present]
    vals = val_paths(a, names)
    out = os.path.join(st.dir, "run")
    dev = "auto"
    results = {}
    for tag in ("ckpt_last", "ckpt_best"):
        ck = os.path.join(out, tag + ".pt")
        if not os.path.exists(ck):
            continue
        results[tag] = ed.evaluate_domains(ck, vals, a.seq_len, a.eval_batch_size, a.eval_batches, dev, "auto", a.vocab)
    if not results:
        print("no checkpoint to evaluate")
        return 1
    with open(os.path.join(st.dir, "eval_domains.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    st.data["eval"] = {k: {"step": v["step"], "mean_val_loss": v["mean_val_loss"], "ok": v["ok"],
                           "domains": {d: x["val_loss"] for d, x in v["domains"].items()}} for k, v in results.items()}
    st.save()
    for tag, v in results.items():
        print(f"\n{tag} (step {v['step']}): mean val loss {v['mean_val_loss']:.4f}")
        for d, x in v["domains"].items():
            print(f"  {d:<14}" + (f"loss {x['val_loss']:.4f}  ppl {x['val_ppl']:.1f}  ({x['tokens']:,} tokens)" if x["val_loss"] is not None else "no full window"))
    return 0 if all(v["ok"] for v in results.values()) else 1


def stage_report(a, st: Session) -> int:
    d = st.data
    L = ["# GPU session report", ""]
    info = (d.get("preflight") or {}).get("run_info") or run_info()
    L += [f"- Hardware: {info.get('gpu') or 'CPU (rehearsal)'}; torch {info.get('torch')}, CUDA {info.get('cuda')}",
          f"- Code: commit `{str(info.get('git_commit'))[:10]}`; tokenizer SHA-256 `{BPETokenizer.file_sha256(a.vocab)[:16]}...`",
          f"- Domains in the mix: {', '.join((d.get('preflight') or {}).get('domains', []))}"
          + (f" (missing: {', '.join(d['preflight']['missing'])})" if (d.get('preflight') or {}).get('missing') else ""), ""]
    b = d.get("bench")
    if b:
        L += ["## Benchmark (synthetic tokens)", "", "| preset | micro-batch | tokens/s | step s | peak GB | MFU | Rs per 1B tokens | status |", "|---|---|---|---|---|---|---|---|"]
        for r in b["rows"]:
            if r["status"] == "ok":
                L.append(f"| {r['preset']} | {r['batch']} | {r['tokens_per_s']:,} | {r['step_s']} | {r['peak_mem_GB']} | {r['mfu']:.1%} | {r['rupees_per_1B_tokens']:,} | ok |")
            else:
                L.append(f"| {r['preset']} | {r['batch']} | | | | | | {r['status']} |")
        L.append("")
    s = d.get("smoke")
    if s:
        L += ["## Smoke test", "", f"- {s['status']}" + (f": {'; '.join(s['problems'])}" if s.get("problems") else ""),
              *([f"- loss {s['first_loss']:.3f} -> {s['last_loss']:.3f}, {s['tokens_per_s']:,.0f} tokens/s, peak {s['peak_mem_gb']} GB, {s['minutes']} min"] if s.get("first_loss") is not None else []), ""]
    t = d.get("train")
    if t:
        recs = read_metrics(os.path.join(st.dir, "run"))
        tr = [r for r in recs if "loss" in r]
        va = [r for r in recs if "val_loss" in r]
        L += ["## Training", "", f"- {t['max_steps']:,} steps x {t['tokens_per_step']:,} tokens = {t['planned_tokens']:,} tokens "
              f"(micro-batch {t['micro_batch']}, accumulation {t['grad_accum']}); measured {t['measured_tokens_per_s']:,.0f} tokens/s",
              f"- wall time {t.get('wall_minutes')} min; exit code {t.get('exit_code')}"]
        if tr:
            L.append(f"- training loss {tr[0]['loss']:.3f} (step {tr[0]['step']}) -> {tr[-1]['loss']:.3f} (step {tr[-1]['step']})")
        if va:
            best = min(va, key=lambda r: r["val_loss"])
            L.append(f"- validation loss {va[0]['val_loss']:.3f} -> {va[-1]['val_loss']:.3f} (best {best['val_loss']:.3f} at step {best['step']})")
        L.append("")
    e = d.get("eval")
    if e:
        for tag, v in e.items():
            L += [f"## Per-domain validation, {tag} (step {v['step']}), mean {v['mean_val_loss']:.4f}", "", "| domain | val loss | perplexity |", "|---|---|---|"]
            for n, x in v["domains"].items():
                L.append(f"| {n} | {x:.4f} | {math.exp(min(x, 50)):.1f} |" if x is not None else f"| {n} | n/a | n/a |")
            L.append("")
    minutes = d.get("clock_minutes")
    if minutes:
        L += [f"Total session clock: {minutes} min = about Rs {minutes / 60 * a.rupees:,.0f} at Rs {a.rupees:.0f}/h.", ""]
    path = os.path.join(st.dir, "session_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("report written:", path)
    return 0


# ── command line ────────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["preflight", "bench", "smoke", "train", "eval", "report", "all"])
    ap.add_argument("--bins", default="bins", help="folder with <domain>.train.bin / .val.bin and the build manifests")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="the full mix config (weights); domains without a token file are left out")
    ap.add_argument("--vocab", default=DEFAULT_VOCAB)
    ap.add_argument("--out", default="session")
    ap.add_argument("--expect-tag", help="git tag the checkout must equal exactly (e.g. scale-run-rc6)")
    ap.add_argument("--preset", default="100m")
    ap.add_argument("--smoke-preset", default="1p5b")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--global-seqs", type=int, default=128, help="sequences per optimizer step over all GPUs (128 x 2048 = 262,144 tokens)")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--minutes", type=float, default=100, help="time budget of the 100M run")
    ap.add_argument("--efficiency", type=float, default=0.95, help="safety factor on the budget when converting tokens/s to steps")
    ap.add_argument("--max-steps-cap", type=int, default=0)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--save-interval", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=50)
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--val-cap", type=int, default=100000, help="tokens per domain in the balanced validation file")
    ap.add_argument("--bench-batches", nargs="+", type=int, default=[4, 8, 16, 32])
    ap.add_argument("--smoke-batches", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--bench-steps", type=int, default=8)
    ap.add_argument("--smoke-minutes", type=float, default=15)
    ap.add_argument("--smoke-accum", type=int, default=2, help="gradient accumulation of the smoke test (a small step: it tests plumbing, not quality)")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--redo", action="store_true", help="`all`: repeat the benchmark and smoke test even if session.json already holds their results")
    ap.add_argument("--rupees", type=float, default=330.0, help="Rs per GPU hour")
    ap.add_argument("--min-free-gb", type=float, default=40.0)
    ap.add_argument("--cpu-ok", action="store_true", help="rehearsal without a GPU")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    st = Session(a.out)
    stages = {"preflight": stage_preflight, "bench": stage_bench, "smoke": stage_smoke, "train": stage_train,
              "eval": stage_eval, "report": stage_report}
    if a.stage != "all":
        return stages[a.stage](a, st)
    t0 = time.time()
    prior = st.data.get("clock_minutes", 0)
    for name in ("preflight", "bench", "smoke", "train", "eval"):
        done = {"bench": bool(st.data.get("bench")), "smoke": str((st.data.get("smoke") or {}).get("status", "")).startswith(("PASS", "skipped"))}
        if not a.redo and done.get(name):
            print(f"\n######## {name}: already done in {st.path}, not repeated (use --redo to repeat) ########", flush=True)
            continue
        print(f"\n######## {name}  (session clock {prior + (time.time() - t0) / 60:.1f} min) ########", flush=True)
        rc = stages[name](a, st)
        st.data["clock_minutes"] = round(prior + (time.time() - t0) / 60, 1)
        st.save()
        if rc != 0:
            print(f"\nSTOPPED at {name}. Nothing after it was started.")
            stage_report(a, st)
            return rc
    stage_report(a, st)
    print(f"\nDONE in {st.data['clock_minutes']} min (about Rs {st.data['clock_minutes'] / 60 * a.rupees:,.0f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
