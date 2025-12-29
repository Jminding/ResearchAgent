#!/usr/bin/env python3
"""Batch 3: Reward Shaping and GNN Architecture Ablations"""
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

    def reward(self, c, e, syn, sc):
        if self.rtype == "sparse":
            return 1.0 if not sc.logical_err(e, c) else 0.0
        elif self.rtype == "dense_syndrome":
            cs = sc.syndrome(e ^ c)
            r = (np.sum(syn) - np.sum(cs)) / max(np.sum(syn), 1)
            return r + (2.0 if not sc.logical_err(e, c) else -1.0)
        elif self.rtype == "dense_distance":
            r = 1.0 - np.sum(e != c) / len(e)
            return r + (1.0 if not sc.logical_err(e, c) else -0.5)
        return 1.0 if not sc.logical_err(e, c) else 0.0

    def train_ep(self, sc):
        e = sc.gen_err()
        syn = sc.syndrome(e)
        c = self.decode(syn, sc)
        r = self.reward(c, e, syn, sc)
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
        t0 = time.time()
        for _ in range(eps):
            self.train_ep(sc)
        return {"time": time.time() - t0, "reward": float(np.mean(self.rewards[-100:])) if self.rewards else 0}


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


results = []

# Part A: Reward Shaping Ablation
print("=" * 60)
print("BATCH 3A: Reward Shaping Ablation")
print("=" * 60)
sys.stdout.flush()

for d in [7, 15]:
    for rtype in ["sparse", "dense_syndrome", "dense_distance"]:
        print(f"\nd={d}, reward={rtype}")
        sys.stdout.flush()
        for seed in range(1, 4):
            try:
                sc = SurfaceCode(d, 0.005, seed=seed)
                dec = RLDecoder(d, 0.005, rtype=rtype, seed=seed)
                ts = dec.train(sc, 2000)
                ev = evaluate(dec, sc, 500)
                results.append({
                    "config_name": f"reward_d{d}_{rtype}_s{seed}",
                    "parameters": {"code_distance": d, "physical_error_rate": 0.005, "training_episodes": 2000, "seed": seed, "reward_type": rtype},
                    "metrics": {"logical_error_rate": ev["ler"], "ci_95_lower": ev["ci_lo"], "ci_95_upper": ev["ci_hi"],
                               "final_reward": ts["reward"], "training_time_sec": ts["time"]},
                    "ablation": f"reward_{rtype}", "error": None
                })
                if seed == 1:
                    print(f"  LER={ev['ler']:.4f}")
                    sys.stdout.flush()
            except Exception as e:
                results.append({
                    "config_name": f"reward_d{d}_{rtype}_s{seed}",
                    "parameters": {"code_distance": d, "physical_error_rate": 0.005, "training_episodes": 2000, "seed": seed, "reward_type": rtype},
                    "metrics": {}, "ablation": f"reward_{rtype}", "error": str(e)
                })

# Part B: GNN Architecture Ablation
print("\n" + "=" * 60)
print("BATCH 3B: GNN Architecture Ablation")
print("=" * 60)
sys.stdout.flush()

for d in [7, 15]:
    for layers, hdim in [(2, 64), (4, 64), (6, 64), (4, 128)]:
        print(f"\nd={d}, L={layers}, H={hdim}")
        sys.stdout.flush()
        for seed in range(1, 4):
            try:
                sc = SurfaceCode(d, 0.005, seed=seed)
                dec = RLDecoder(d, 0.005, layers=layers, hdim=hdim, seed=seed)
                ts = dec.train(sc, 2000)
                ev = evaluate(dec, sc, 500)
                results.append({
                    "config_name": f"gnn_d{d}_L{layers}_H{hdim}_s{seed}",
                    "parameters": {"code_distance": d, "physical_error_rate": 0.005, "training_episodes": 2000, "seed": seed, "gnn_layers": layers, "hidden_dim": hdim},
                    "metrics": {"logical_error_rate": ev["ler"], "ci_95_lower": ev["ci_lo"], "ci_95_upper": ev["ci_hi"],
                               "model_params": dec.n_params, "training_time_sec": ts["time"]},
                    "ablation": f"gnn_L{layers}_H{hdim}", "error": None
                })
                if seed == 1:
                    print(f"  LER={ev['ler']:.4f}, params={dec.n_params}")
                    sys.stdout.flush()
            except Exception as e:
                results.append({
                    "config_name": f"gnn_d{d}_L{layers}_H{hdim}_s{seed}",
                    "parameters": {"code_distance": d, "physical_error_rate": 0.005, "training_episodes": 2000, "seed": seed, "gnn_layers": layers, "hidden_dim": hdim},
                    "metrics": {}, "ablation": f"gnn_L{layers}_H{hdim}", "error": str(e)
                })

print(f"\nCompleted {len(results)} experiments")

with open(os.path.join(OUTPUT_DIR, "batch3_ablations.json"), "w") as f:
    json.dump({"project_name": "QEC_Ablations", "results": results}, f, indent=2)

print("Results saved to batch3_ablations.json")
