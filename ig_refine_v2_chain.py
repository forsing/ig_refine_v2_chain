from __future__ import annotations

# IG = Information Geometry (informaciona geometrija)

"""
Refinement calculus na loto CSV — v2 (lanac)

Na koraku t: S_t = draws[t]
P0: uniforman 7-izbor van last (seed=39+t)
P1: top7 Hebbian masa
P2: Hebbian · Perez L · circ + lokalna pretraga sa kaznom zbijenosti
    (ne sirovi top7 → ne Perez-blok oko zenita)

Merilo μ = mean |pred ∩ S_t|
Empirijski lanac: μ(P0) ≤ μ(P1) ≤ μ(P2)

CSV: loto7_4652_k57.csv, seed=39.
Ime: ig_refine_v2_chain.py
"""

import csv
from itertools import combinations
from math import cos, exp, pi
from pathlib import Path

import numpy as np

SEED = 39
FRONT_N = 39
FRONT_SELECT = 7
LAMBDA_TEMP = 0.35
WARMUP = 500
STEP = 50
ZENITH = 20.0
PEREZ_A = 4.0
PEREZ_B = 0.6
PEREZ_C = 1.2
PEREZ_D = 2.5
CIRC_PERIOD = 39
CIRC_KAPPA = 0.25
CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "loto7_4652_k57.csv"

np.random.seed(SEED)


def load_draws(csv_path: Path = CSV_PATH) -> np.ndarray:
    draws = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < FRONT_SELECT:
                continue
            try:
                draw = sorted(int(x.strip()) for x in row[:FRONT_SELECT])
            except ValueError:
                continue
            if len(draw) == FRONT_SELECT and all(1 <= x <= FRONT_N for x in draw):
                if len(set(draw)) == FRONT_SELECT:
                    draws.append(draw)
    if not draws:
        raise ValueError(f"Nema validnih kola u {csv_path}")
    return np.array(draws, dtype=int)


def hebbian_weights(draws, lam=LAMBDA_TEMP):
    W = np.zeros((FRONT_N, FRONT_N), dtype=float)
    for d in draws:
        idx = [int(x) - 1 for x in d.tolist()]
        for a, b in combinations(idx, 2):
            W[a, b] += 1.0
            W[b, a] += 1.0
    for t in range(len(draws) - 1):
        a_idx = [int(x) - 1 for x in draws[t].tolist()]
        b_idx = [int(x) - 1 for x in draws[t + 1].tolist()]
        for a in a_idx:
            for b in b_idx:
                if a == b:
                    continue
                W[a, b] += lam
                W[b, a] += lam
    np.fill_diagonal(W, 0.0)
    return W


def hebbian_add_draw(W, prev, cur, lam=LAMBDA_TEMP):
    idx = [int(x) - 1 for x in cur.tolist()]
    for a, b in combinations(idx, 2):
        W[a, b] += 1.0
        W[b, a] += 1.0
    a_idx = [int(x) - 1 for x in prev.tolist()]
    for a in a_idx:
        for b in idx:
            if a == b:
                continue
            W[a, b] += lam
            W[b, a] += lam
    np.fill_diagonal(W, 0.0)


def energy_distribution(W):
    D = W.copy()
    row = D.sum(axis=1, keepdims=True)
    row = np.where(row < 1e-18, 1.0, row)
    return D / row


def hebbian_mass(D, last):
    idx = [int(x) - 1 for x in last.tolist()]
    return D[idx].mean(axis=0)


def perez_luminance(sun):
    L = np.zeros(FRONT_N)
    for i in range(FRONT_N):
        n = i + 1
        gamma = abs(n - sun) / float(FRONT_N)
        theta = abs(n - ZENITH) / float(FRONT_N)
        L[i] = (1.0 + PEREZ_C * exp(-PEREZ_A * gamma * gamma)) * (
            1.0 + PEREZ_B * exp(-PEREZ_D * theta * theta)
        )
    return L / L.sum()


def circadian_field(t_index):
    phi = 2.0 * pi * (t_index % CIRC_PERIOD) / float(CIRC_PERIOD)
    circ = np.zeros(FRONT_N)
    for i in range(FRONT_N):
        circ[i] = 1.0 + CIRC_KAPPA * cos(phi + 2.0 * pi * i / float(FRONT_N))
    return circ


def hits(pred, actual) -> int:
    return len(set(pred) & set(int(x) for x in actual.tolist()))


def top7_from_score(score, ban):
    ranked = sorted(
        (n for n in range(1, FRONT_N + 1) if n not in ban),
        key=lambda n: (-float(score[n - 1]), n),
    )
    return sorted(ranked[:FRONT_SELECT])


def predict_P0(last, t_index: int) -> list[int]:
    ban = set(int(x) for x in last.tolist())
    pool = [n for n in range(1, FRONT_N + 1) if n not in ban]
    rng = np.random.default_rng(SEED + t_index)
    pick = rng.choice(pool, size=FRONT_SELECT, replace=False)
    return sorted(int(x) for x in pick)


def predict_P1(W, last) -> list[int]:
    ban = set(int(x) for x in last.tolist())
    mass = hebbian_mass(energy_distribution(W), last)
    return top7_from_score(mass, ban)


def _combo_fit(combo, score, ban):
    """Energija skora + kazna zbijenosti (anti Perez-blok)."""
    nums = sorted(combo)
    if any(x in ban for x in nums):
        return -1e18
    s = sum(score[x] for x in nums)
    gaps = [nums[i + 1] - nums[i] for i in range(FRONT_SELECT - 1)]
    s -= 0.25 * sum(1.0 / g for g in gaps)
    s += 0.01 * (nums[-1] - nums[0])
    return s


def predict_from_score_spread(score_vec, ban) -> list[int]:
    score = {i + 1: float(score_vec[i]) for i in range(FRONT_N)}
    for n in ban:
        score[n] = -1e18
    ranked = sorted(
        (n for n in range(1, FRONT_N + 1) if n not in ban),
        key=lambda n: (-score[n], n),
    )
    candidates = [sorted(ranked[:FRONT_SELECT])]
    for start in range(0, min(20, len(ranked) - FRONT_SELECT + 1)):
        candidates.append(sorted(ranked[start : start + FRONT_SELECT]))
    best, best_fit = None, -1e18
    for base in candidates:
        fit = _combo_fit(base, score, ban)
        if fit > best_fit:
            best_fit, best = fit, list(base)
        for i in range(FRONT_SELECT):
            for repl in ranked[:30]:
                cand = sorted(set(base[:i] + base[i + 1 :] + [repl]))
                if len(cand) != FRONT_SELECT:
                    continue
                fit = _combo_fit(cand, score, ban)
                if fit > best_fit:
                    best_fit, best = fit, cand
    return best if best is not None else sorted(ranked[:FRONT_SELECT])


def predict_P2(W, last, t_index: int) -> list[int]:
    ban = set(int(x) for x in last.tolist())
    mass = hebbian_mass(energy_distribution(W), last)
    L = perez_luminance(float(np.mean(last)))
    circ = circadian_field(t_index)
    score = mass * L * circ
    return predict_from_score_spread(score, ban)


def walk_refine(draws: np.ndarray) -> dict:
    T = len(draws)
    W = hebbian_weights(draws[:WARMUP])
    h0, h1, h2 = [], [], []
    t = WARMUP
    while t < T:
        if (t - WARMUP) % STEP == 0:
            last = draws[t - 1]
            S = draws[t]
            h0.append(hits(predict_P0(last, t), S))
            h1.append(hits(predict_P1(W, last), S))
            h2.append(hits(predict_P2(W, last, t - 1), S))
        if t < T - 1:
            hebbian_add_draw(W, draws[t - 1], draws[t])
        t += 1

    mu0 = float(np.mean(h0)) if h0 else 0.0
    mu1 = float(np.mean(h1)) if h1 else 0.0
    mu2 = float(np.mean(h2)) if h2 else 0.0
    return {
        "n_eval": len(h0),
        "mu_P0": mu0,
        "mu_P1": mu1,
        "mu_P2": mu2,
        "P0_sqsubseteq_P1": mu1 >= mu0,
        "P1_sqsubseteq_P2": mu2 >= mu1,
        "chain": (mu1 >= mu0) and (mu2 >= mu1),
    }


def run_v2(csv_path: Path = CSV_PATH) -> None:
    draws = load_draws(csv_path)
    stats = walk_refine(draws)

    last = draws[-1]
    t_now = len(draws) - 1
    W_full = hebbian_weights(draws)
    next_p2 = predict_P2(W_full, last, t_now)

    print(f"CSV: {csv_path.name}")
    print(
        f"Kola: {len(draws)} | seed={SEED} | WARMUP={WARMUP} STEP={STEP} | ig_refine_v2"
    )
    print(f"last: {last.tolist()}")
    print()
    print("=== spec ===")
    print("S_t = draws[t]")
    print("P0 → P1 → P2  |  μ = mean |pred ∩ S_t|")
    print()
    print("=== empirijski lanac ===")
    print(
        {
            "n_eval": stats["n_eval"],
            "mu_P0": round(stats["mu_P0"], 4),
            "mu_P1": round(stats["mu_P1"], 4),
            "mu_P2": round(stats["mu_P2"], 4),
            "P0_sqsubseteq_P1": stats["P0_sqsubseteq_P1"],
            "P1_sqsubseteq_P2": stats["P1_sqsubseteq_P2"],
            "chain": stats["chain"],
        }
    )
    print()
    print("=== next (P2 na celom CSV) ===")
    print("next:", next_p2)


if __name__ == "__main__":
    run_v2()



"""
CSV: loto7_4652_k57.csv
Kola: 4652 | seed=39 | WARMUP=500 STEP=50 | ig_refine_v2
last: [7, 8, 14, 15, 17, 23, 32]

=== spec ===
S_t = draws[t]
P0 → P1 → P2  |  μ = mean |pred ∩ S_t|

=== empirijski lanac ===
{'n_eval': 84, 'mu_P0': 1.1429, 'mu_P1': 1.3929, 'mu_P2': 1.4405, 'P0_sqsubseteq_P1': True, 'P1_sqsubseteq_P2': True, 'chain': True}

=== next (P2 na celom CSV) ===
next: [1, 13, 16, 18, 20, 30, 34]
"""



"""
v2 — poslednji korak refine linije
P0 ⊑ P1 ⊑ P2; P2 = Hebbian·L·circ top7.
"""



"""
v2 — drugi korak: P1 ⊑ P2, gde je P2 jači (npr. Hebbian+Perez/circ ili mean_sK), isto S_t i μ. Ako μ(P2) ≥ μ(P1) → lanac P0 ⊑ P1 ⊑ P2.

lanac P0 ⊑ P1 ⊑ P2 (P2 = Hebbian·L·circ), next

nije Perez-blok. Lanac i dalje drži (1.14 → 1.39 → 1.44).
"""
 