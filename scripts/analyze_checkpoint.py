#!/usr/bin/env python
"""Checkpoint analysis: loss curves, per-domain next-token metrics, generation quality and consistency, checkpoint integrity.

    python scripts/analyze_checkpoint.py --ckpt run/ckpt_best.pt --compare-ckpt run/ckpt_last.pt \
        --metrics run/metrics.jsonl --config run/train_config.json --vocab tokenizers/vocab32k_v2/vocab32k.tok \
        --val math=bins/math.val.bin physics=bins/physics.val.bin ... --evals evals --out results/analysis_100m

Everything is written to ``<out>.json`` (and the loss curve to ``<out>_curve.csv``). Nothing is sampled at random except the
seeded sampling in part 3, so a re-run gives the same numbers.

1. ``loss``: training and validation loss from the run's ``metrics.jsonl``: start, best and final values, the train-validation
   gap, how fast validation loss was still falling at the end, gradient-norm spikes, tokens seen and throughput.
2. ``next_token``: for every domain's validation file, the same non-overlapping windows every time: cross-entropy, perplexity,
   top-1 and top-5 accuracy, and the calibration error of the top-1 prediction (do 60%-confident guesses come true 60% of the time?).
3. ``generation``: held-out continuations and the fixed probes of ``sample_outputs.py``. Quality: probe pass rate. Repetition:
   share of repeated 4-grams (against the real continuation of the same documents), distinct 2-grams, repetition loops, for
   greedy decoding and for one seeded sample (T=0.8) of the same prompt, which shows how much of the looping greedy decoding causes.
   Consistency: greedy decoding twice gives identical text; three seeded samples (T=0.8) agree with each other on each probe;
   a multiple-choice answer survives reversing the order of the options.
4. ``integrity``: SHA-256 of each file, all weights finite, parameter count, tokenizer hash, the step and validation loss stored in
   the checkpoint against ``metrics.jsonl``, the difference between the two checkpoints, identical logits on a repeated forward
   pass, and CPU against device logits.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm.checkpoint import load_checkpoint, read_checkpoint  # noqa: E402
from llm.data import TokenDataset  # noqa: E402
from llm.model import TransformerModel  # noqa: E402
from llm.tokenizer import BPETokenizer  # noqa: E402
from llm.train import resolve_device  # noqa: E402
from eval_science import load_items, score_mcq  # noqa: E402
from sample_outputs import PROBES, continue_greedy, flags, held_out_cases, repetition_loop  # noqa: E402


# ------------------------------------------------------------------ 1. loss
def loss_analysis(metrics_path: str, tokens_per_step: int, tail_steps: int = 1000) -> tuple:
    """Summary of a run's metrics.jsonl and the rows of the loss curve (step, train loss, smoothed, validation loss)."""
    with open(metrics_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    train = [(r["step"], r["loss"]) for r in rows if r.get("loss") is not None]
    val = [(r["step"], r["val_loss"]) for r in rows if r.get("val_loss") is not None]
    smooth, ema = [], None
    for _, x in train:
        ema = x if ema is None else 0.95 * ema + 0.05 * x
        smooth.append(ema)
    best = min(val, key=lambda v: v[1])
    last_step = val[-1][0]
    earlier = [v for v in val if v[0] <= last_step - tail_steps]
    slope = (val[-1][1] - earlier[-1][1]) / (val[-1][0] - earlier[-1][0]) * 1000 if earlier else None
    near = [x for s, x in train if val[-1][0] - 250 < s <= val[-1][0]]
    gnorm = [r["grad_norm"] for r in rows if r.get("grad_norm") is not None]
    med = float(np.median(gnorm)) if gnorm else None
    summary = {
        "logged_steps": len(train), "last_step": train[-1][0], "tokens_per_step": tokens_per_step,
        "tokens_seen": train[-1][0] * tokens_per_step,
        "train_loss_first": train[0][1], "train_loss_last_smoothed": smooth[-1],
        "val_points": len(val), "val_loss_first": val[0][1], "val_loss_best": best[1], "val_best_step": best[0],
        "val_ppl_best": math.exp(best[1]), "val_loss_last": val[-1][1], "val_last_step": val[-1][0],
        "train_minus_val_at_last": (sum(near) / len(near) - val[-1][1]) if near else None,
        "val_change_per_1k_steps_at_end": slope,
        "grad_norm_median": med, "grad_norm_max": max(gnorm) if gnorm else None,
        "grad_norm_spikes": sum(g > 3 * med for g in gnorm) if gnorm else None,
        "tokens_per_s_mean": float(np.mean([r["tokens_per_s"] for r in rows if r.get("tokens_per_s")])),
        "peak_mem_gb": max((r.get("peak_mem_gb") or 0) for r in rows),
        "all_finite": all(math.isfinite(x) for _, x in train + val),
        "val_curve": [{"step": s, "val_loss": x, "val_ppl": math.exp(x)} for s, x in val],
    }
    vd = dict(val)
    curve = [(s, x, sm, vd.get(s)) for (s, x), sm in zip(train, smooth)]
    return summary, curve


# ------------------------------------------------------------------ 2. next-token metrics
@torch.no_grad()
def next_token_metrics(model, path: str, seq_len: int, windows: int, device, bins: int = 10) -> dict:
    """Cross-entropy, perplexity, top-1 / top-5 accuracy and top-1 calibration error over the first ``windows`` windows."""
    ds = TokenDataset(path, seq_len, model.config.vocab_size if hasattr(model, "config") else None)
    ce = n = top1 = top5 = 0.0
    conf_sum, hit_sum, cnt = np.zeros(bins), np.zeros(bins), np.zeros(bins)
    for x, y in ds.sequential_batches(1, windows, device):
        logp = F.log_softmax(model(x).float(), dim=-1)
        ce += float(-logp.gather(-1, y.unsqueeze(-1)).sum())
        top = logp.topk(5, dim=-1).indices
        hit1 = top[..., 0] == y
        top1 += float(hit1.sum())
        top5 += float((top == y.unsqueeze(-1)).any(-1).sum())
        conf = logp.max(-1).values.exp().flatten().cpu().numpy()
        b = np.minimum((conf * bins).astype(int), bins - 1)
        np.add.at(conf_sum, b, conf)
        np.add.at(hit_sum, b, hit1.flatten().cpu().numpy())
        np.add.at(cnt, b, 1)
        n += y.numel()
    ece = float(np.sum(np.abs(conf_sum - hit_sum)) / max(n, 1))
    loss = ce / max(n, 1)
    return {"tokens": int(n), "windows": int(n // seq_len), "cross_entropy": loss, "perplexity": math.exp(min(loss, 50)),
            "top1": top1 / max(n, 1), "top5": top5 / max(n, 1), "calibration_error": ece}


# ------------------------------------------------------------------ 3. generation
def repeated_ngram_share(ids, n: int = 4) -> float:
    """Share of the n-grams of a text that already occurred earlier in it (0 = no repetition)."""
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)


def distinct(ids, n: int = 2) -> float:
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 0.0 if not grams else len(set(grams)) / len(grams)


@torch.no_grad()
def sample(model, prompt_ids, n, eos, device, seed, temperature=0.8) -> list:
    torch.manual_seed(seed)
    y = model.generate(torch.tensor([prompt_ids], device=device), max_new_tokens=n, temperature=temperature, eos_token_id=eos)
    out = y[0, len(prompt_ids):].tolist()
    return out[: out.index(eos)] if eos in out else out


@torch.no_grad()
def generation_analysis(model, tok, vals: dict, items: list, device, per_domain: int, prompt_tokens: int, new_tokens: int,
                        probe_tokens: int, seeds=(1, 2, 3), mcq_n: int = 400) -> dict:
    eos = tok.eos_id
    out = {"domains": {}, "probes": [], "examples": []}
    for name, path in vals.items():
        rows = []
        for prompt, ref in held_out_cases(path, tok, eos, prompt_tokens, new_tokens, per_domain):
            gen = continue_greedy(model, tok, prompt, new_tokens, eos, device)
            smp = sample(model, prompt, new_tokens, eos, device, seeds[0])
            text = tok.decode(gen)
            rows.append({"rep4": repeated_ngram_share(gen), "rep4_ref": repeated_ngram_share(ref), "distinct2": distinct(gen),
                         "loop": repetition_loop(gen), "flags": flags(prompt, gen, text),
                         "rep4_sampled": repeated_ngram_share(smp), "loop_sampled": repetition_loop(smp)})
            if len(out["examples"]) < 2 * len(vals) and len(rows) <= 2:
                out["examples"].append({"domain": name, "prompt": tok.decode(prompt)[-200:], "model": text[:300],
                                        "truth": tok.decode(ref)[:300]})
        out["domains"][name] = {"n": len(rows), "rep4": float(np.mean([r["rep4"] for r in rows])),
                                "rep4_reference": float(np.mean([r["rep4_ref"] for r in rows])),
                                "distinct2": float(np.mean([r["distinct2"] for r in rows])),
                                "loops": sum(r["loop"] for r in rows),
                                "rep4_sampled": float(np.mean([r["rep4_sampled"] for r in rows])),
                                "loops_sampled": sum(r["loop_sampled"] for r in rows),
                                "flagged": sum(bool(r["flags"]) for r in rows)}
    same_greedy = 0
    for domain, prompt, pattern in PROBES:
        ids = tok.encode(prompt, allowed_special=set())
        g1 = continue_greedy(model, tok, ids, probe_tokens, eos, device)
        g2 = continue_greedy(model, tok, ids, probe_tokens, eos, device)
        same_greedy += g1 == g2
        passes = [bool(re.search(pattern, tok.decode(s))) for s in
                  (sample(model, ids, probe_tokens, eos, device, seed) for seed in seeds)]
        greedy_text = tok.decode(g1)
        out["probes"].append({"domain": domain, "prompt": prompt, "greedy": greedy_text[:160],
                              "greedy_pass": bool(re.search(pattern, greedy_text)), "sample_passes": passes,
                              "samples_agree": len(set(passes)) == 1, "loop": repetition_loop(g1)})
    p = out["probes"]
    out["probe_summary"] = {"probes": len(p), "greedy_pass": sum(x["greedy_pass"] for x in p),
                            "sample_pass_rate": float(np.mean([np.mean(x["sample_passes"]) for x in p])),
                            "samples_agree": sum(x["samples_agree"] for x in p), "greedy_loops": sum(x["loop"] for x in p),
                            "greedy_identical_twice": same_greedy}
    mcq = [it for it in items if it["type"] == "mcq" and len(it["choices"]) == 4]
    mcq = mcq[:: max(1, len(mcq) // mcq_n)][:mcq_n]
    keep = acc_fwd = acc_rev = 0
    for it in mcq:
        a = score_mcq(model, tok, it, device, model.config.max_position_embeddings)
        rev = dict(it, choices=it["choices"][::-1], answer=len(it["choices"]) - 1 - it["answer"])
        rev.pop("distractor_types", None)
        b = score_mcq(model, tok, rev, device, model.config.max_position_embeddings)
        keep += it["choices"][a["pick_norm"]] == rev["choices"][b["pick_norm"]]
        acc_fwd += a["correct_norm"]
        acc_rev += b["correct_norm"]
    out["order_consistency"] = {"items": len(mcq), "same_answer": keep / len(mcq), "acc_original": acc_fwd / len(mcq),
                                "acc_reversed": acc_rev / len(mcq), "random_picker": 0.25}
    return out


# ------------------------------------------------------------------ 4. integrity
def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


@torch.no_grad()
def integrity(paths: list, vocab: str, metrics_path: str, device, sample_ids: list) -> dict:
    vsha = BPETokenizer.file_sha256(vocab)
    with open(metrics_path, encoding="utf-8") as f:
        val = {r["step"]: r.get("val_loss") for r in (json.loads(line) for line in f if line.strip())}
    out, states = {"checkpoints": {}}, []
    for p in paths:
        state, cfg, meta, raw = read_checkpoint(p)
        finite = all(bool(torch.isfinite(t).all()) for t in state.values() if t.is_floating_point())
        step = meta.get("step")
        logged = val.get(step)
        out["checkpoints"][os.path.basename(p)] = {
            "sha256": file_sha256(p), "bytes": os.path.getsize(p), "format_version": raw.get("format_version"),
            "keys": sorted(raw), "has_optimizer": raw.get("optimizer") is not None, "step": step,
            "params": int(sum(t.numel() for t in state.values())), "tensors": len(state), "all_finite": finite,
            "vocab_matches": meta.get("vocab_sha256") == vsha, "untrained": meta.get("untrained"),
            "val_loss_in_checkpoint": meta.get("val_loss"), "val_loss_in_metrics": logged,
            "val_loss_matches": None if logged is None or meta.get("val_loss") is None else abs(logged - meta["val_loss"]) < 1e-6,
            "torch_version": meta.get("torch_version"), "config": cfg.__dict__ if hasattr(cfg, "__dict__") else str(cfg)}
        states.append((state, cfg))
    if len(states) == 2:
        (a, ca), (b, cb) = states
        num = sum(float(((a[k].float() - b[k].float()) ** 2).sum()) for k in a)
        den = sum(float((a[k].float() ** 2).sum()) for k in a)
        out["pair"] = {"same_config": ca.__dict__ == cb.__dict__, "same_tensor_names": set(a) == set(b),
                       "relative_l2_difference": math.sqrt(num / den),
                       "max_abs_difference": max(float((a[k].float() - b[k].float()).abs().max()) for k in a)}
    state, cfg = states[0]
    model = TransformerModel(cfg)
    load_checkpoint(paths[0], model, expected_vocab_sha256=vsha)
    model.eval()
    x = torch.tensor([sample_ids])
    cpu = model(x).float()
    model.to(device)
    d1, d2 = model(x.to(device)).float().cpu(), model(x.to(device)).float().cpu()
    out["forward"] = {"tokens": len(sample_ids), "device": str(device),
                      "repeat_max_abs_diff": float((d1 - d2).abs().max()),
                      "cpu_vs_device_max_abs_diff": float((cpu - d1).abs().max()),
                      "cpu_vs_device_argmax_agreement": float((cpu.argmax(-1) == d1.argmax(-1)).float().mean())}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--compare-ckpt", help="a second checkpoint of the same run, for the integrity comparison")
    ap.add_argument("--metrics", required=True, help="metrics.jsonl of the training run")
    ap.add_argument("--config", required=True, help="train_config.json of the training run")
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--val", nargs="+", required=True, metavar="NAME=PATH")
    ap.add_argument("--evals", default="evals")
    ap.add_argument("--out", default="analysis")
    ap.add_argument("--windows", type=int, default=100, help="validation windows per domain")
    ap.add_argument("--per-domain", type=int, default=6, help="held-out continuations per domain")
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--new-tokens", type=int, default=96)
    ap.add_argument("--probe-tokens", type=int, default=40)
    ap.add_argument("--mcq", type=int, default=400, help="multiple-choice items for the option-order check")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--parts", default="loss,next_token,generation,integrity")
    a = ap.parse_args(argv)
    parts = set(a.parts.split(","))
    dev = resolve_device(a.device)
    with open(a.config, encoding="utf-8") as f:
        tcfg = json.load(f)["train"]
    seq_len = tcfg["seq_len"]
    vals = dict(v.split("=", 1) for v in a.val)
    tok = BPETokenizer.from_file(a.vocab)
    res = {"checkpoint": os.path.abspath(a.ckpt), "device": str(dev), "seq_len": seq_len}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    if "loss" in parts:
        res["loss"], curve = loss_analysis(a.metrics, seq_len * tcfg["batch_size"] * tcfg["grad_accum"])
        with open(a.out + "_curve.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "train_loss", "train_loss_smoothed", "val_loss"])
            w.writerows(curve)
        print(f"loss: best val {res['loss']['val_loss_best']:.4f} at step {res['loss']['val_best_step']}")
    _, cfg, meta, _ = read_checkpoint(a.ckpt)
    model = TransformerModel(cfg).to(dev)
    load_checkpoint(a.ckpt, model, expected_vocab_sha256=BPETokenizer.file_sha256(a.vocab))
    model.eval()
    res["step"] = meta.get("step")
    if "next_token" in parts:
        res["next_token"] = {}
        for name, path in vals.items():
            res["next_token"][name] = r = next_token_metrics(model, path, seq_len, a.windows, dev)
            print(f"next-token {name}: ce {r['cross_entropy']:.3f} ppl {r['perplexity']:.2f} top1 {r['top1']:.3f} top5 {r['top5']:.3f}")
    if "generation" in parts:
        res["generation"] = generation_analysis(model, tok, vals, load_items(a.evals), dev, a.per_domain, a.prompt_tokens,
                                                a.new_tokens, a.probe_tokens, mcq_n=a.mcq)
        print("generation:", res["generation"]["probe_summary"], res["generation"]["order_consistency"])
    if "integrity" in parts:
        ids = tok.encode("The finite element method divides a structure into small elements. " * 20, allowed_special=set())[:512]
        res["integrity"] = integrity([a.ckpt] + ([a.compare_ckpt] if a.compare_ckpt else []), a.vocab, a.metrics, dev, ids)
        print("integrity:", res["integrity"]["forward"], res["integrity"].get("pair"))
    with open(a.out + ".json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, default=str)
    print(f"wrote {a.out}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
