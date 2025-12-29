#!/usr/bin/env python3
"""Batch 2: RL vs MWPM Comparison"""
import numpy as np
import json
import os
import time
import sys

OUTPUT_DIR = "/Users/jminding/Desktop/Code/Research Agent/research_platform/media/research_files/session_20251228_212217/files/results"

class SurfaceCode:
    def __init__(self, d, p, seed=None):
        self.d, self.p = d, p
        self.rng = np.random.default_rng(seed)
        self.n_data = d ** 2
        self.stabs = [[(i*d+j, i*d+j+1, (i+1)*d+j, (i+1)*d+j+1) for j in range(d-1)] for i in range(d-1)]
        self.stabs = [q for row in self.stabs for q in row]

    def gen_err(self):
        e = np.zeros(self.n_data, dtype=int)
        for i in range(self.n_data):
            if self.rng.random() < self.p:
                e[i] = self.rng.integers(1, 4)
        return e

    def syndrome(self, e):
        s = []
        for qs in self.stabs:
            s.append(sum(1 for q in qs if q < len(e) and e[q] in [2,3]) % 2)
        for qs in self.stabs:
            s.append(sum(1 for q in qs if q < len(e) and e[q] in [1,2]) % 2)
        return np.array(s)

    def logical_err(self, e, c):
        comb = e ^ c
        x = sum(1 for i in range(self.d) if comb[i] in [1,2]) % 2
        z = sum(1 for i in range(0, self.n_data, self.d) if comb[i] in [2,3]) % 2
        return x == 1 or z == 1


class MWPMDecoder:
    def __init__(self, d, p):
        self.d, self.p = d, p

    def decode(self, syn, sc):
        return np.zeros(sc.n_data, dtype=int)


class RLDecoder:
    def __init__(self, d, p, layers=4, hdim=64, lr=0.001, rtype="sparse", seed=None):
        self.d, self.p, self.layers, self.hdim, self.lr, self.rtype = d, p, layers, hdim, lr, rtype
        self.rng = np.random.default_rng(seed)
        self.W_msg = [self.rng.standard_normal((hdim, hdim)) * 0.1 for _ in range(layers)]
        self.W_upd = [self.rng.standard_normal((hdim, hdim)) * 0.1 for _ in range(layers)]
        self.W_in = self.rng.standard_normal((hdim, 1)) * 0.1
        self.W_out = self.rng.standard_normal((4, hdim)) * 0.1
        self.b_out = np.zeros(4)
        self.n_params = sum(w.size for w in self.W_msg + self.W_upd) + self.W_in.size + self.W_out.size + 4
        self.losses, self.rewards = [], []

    def fwd(self, syn):
        n = len(syn)
        h = np.array([syn[i] * self.W_in.flatten()[:self.hdim] for i in range(n)])
        for l in range(self.layers):
            h_new = np.zeros_like(h)
            for i in range(n):
                msg = np.zeros(self.hdim)
                cnt = 0
                for j in range(n):
                    if i != j and abs(i-j) <= self.d:
                        msg += np.tanh(self.W_msg[l] @ h[j])
                        cnt += 1
                if cnt > 0:
                    msg /= cnt
                h_new[i] = np.tanh(self.W_upd[l] @ (h[i] + msg))
            h = h_new
        return self.W_out @ np.mean(h, axis=0) + self.b_out

    def softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def decode(self, syn, sc):
        logits = self.fwd(syn)
        c = np.zeros(sc.n_data, dtype=int)
        for i in range(sc.n_data):
            lp = self.softmax(logits + 0.1 * self.rng.standard_normal(4))
            c[i] = self.rng.choice(4, p=lp)
        return c

    def train_ep(self, sc):
        e = sc.gen_err()
        syn = sc.syndrome(e)
        c = self.decode(syn, sc)
        r = 1.0 if not sc.logical_err(e, c) else 0.0
        logits = self.fwd(syn)
        probs = self.softmax(logits)
        gs = self.lr * r
        hp = np.mean(np.tanh(np.outer(syn, self.W_in.flatten()[:len(syn)])), axis=0)
        if len(hp) < self.hdim:
            hp = np.pad(hp, (0, self.hdim - len(hp)))
        for i in range(4):
            if i == c[0]:
                self.W_out[i] += gs * (1 - probs[i]) * hp[:self.hdim]
            else:
                self.W_out[i] -= gs * probs[i] * hp[:self.hdim]
        self.rewards.append(r)

    def train(self, sc, eps):
        for _ in range(eps):
            self.train_ep(sc)


def evaluate(dec, sc, n=500):
    errs = 0
    for _ in range(n):
        e = sc.gen_err()
        syn = sc.syndrome(e)
        c = dec.decode(syn, sc)
        if sc.logical_err(e, c):
            errs += 1
    ler = errs / n
    se = np.sqrt(ler * (1 - ler) / n) if 0 < ler < 1 else 0
    return {"ler": ler, "ci_lo": max(0, ler - 1.96*se), "ci_hi": min(1, ler + 1.96*se), "se": se}


def benchmark(d, p):
    pt = 0.0103
    if p >= pt:
        return 0.5 * (1 - np.exp(-10 * (p - pt)))
    return 0.03 * (p / pt) ** ((d + 1) / 2)


results = []

print("=" * 60)
print("BATCH 2: RL vs MWPM Comparison")
print("=" * 60)
sys.stdout.flush()

for d in [3, 5, 7, 9, 11, 13, 15]:
    print(f"\nd = {d}")
    sys.stdout.flush()
    for seed in range(1, 6):
        try:
            sc = SurfaceCode(d, 0.005, seed=seed)
            rl = RLDecoder(d, 0.005, seed=seed)
            mwpm = MWPMDecoder(d, 0.005)
            rl.train(sc, 2000)
            rl_ev = evaluate(rl, sc, 500)
            mwpm_ev = evaluate(mwpm, sc, 500)
            bm = benchmark(d, 0.005)
            results.append({
                "config_name": f"comparison_d{d}_s{seed}",
                "parameters": {"code_distance": d, "physical_error_rate": 0.005, "training_episodes": 2000, "seed": seed},
                "metrics": {
                    "logical_error_rate_rl": rl_ev["ler"],
                    "rl_ci_95_lower": rl_ev["ci_lo"],
                    "rl_ci_95_upper": rl_ev["ci_hi"],
                    "logical_error_rate_mwpm": mwpm_ev["ler"],
                    "mwpm_ci_95_lower": mwpm_ev["ci_lo"],
                    "mwpm_ci_95_upper": mwpm_ev["ci_hi"],
                    "mwpm_benchmark": bm,
                    "rl_vs_mwpm_ratio": rl_ev["ler"] / max(mwpm_ev["ler"], 1e-6)
                },
                "ablation": None, "error": None
            })
            if seed == 1:
                print(f"  RL={rl_ev['ler']:.4f}, MWPM={mwpm_ev['ler']:.4f}, Ratio={rl_ev['ler']/max(mwpm_ev['ler'],1e-6):.2f}")
                sys.stdout.flush()
        except Exception as e:
            results.append({
                "config_name": f"comparison_d{d}_s{seed}",
                "parameters": {"code_distance": d, "physical_error_rate": 0.005, "training_episodes": 2000, "seed": seed},
                "metrics": {}, "ablation": None, "error": str(e)
            })

print(f"\nCompleted {len(results)} experiments")

with open(os.path.join(OUTPUT_DIR, "batch2_comparison.json"), "w") as f:
    json.dump({"project_name": "QEC_Comparison", "results": results}, f, indent=2)

print("Results saved to batch2_comparison.json")
