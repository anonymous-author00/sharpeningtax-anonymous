#!/usr/bin/env python3
"""Sharpening Tax: pass@k / pass^k of a base model vs. its post-trained sibling (Sec. 2, 4).

With c_i successes out of n_i rollouts on task i, averaged over tasks:

    pass@k   = 1 - C(n - c, k) / C(n, k)                coverage
    pass^k   = C(c, k) / C(n, k)                        consistency
    A(K)     = sum_{k<K} [pass@K - pass@k]              Eq. (1)
    S(K)     = A(K) / ((K - 1)(1 - pass@1))            Eq. (2)
    Tax_X(K) = X_base(K) - X_RL(K),  X in {A, S}       Eq. (3)

A positive tax means post-training reduced the gain from more rollouts.

    python sharpening_tax_minimal.py --base base.jsonl --rl rl.jsonl [--K 128]
    python sharpening_tax_minimal.py --demo
"""
from __future__ import annotations

import argparse
import json

import numpy as np

Counts = dict  # task_id -> (c, n)


# Estimators

def _check(c, n, K):
    c, n = np.asarray(c, dtype=np.float64), np.asarray(n, dtype=np.float64)
    if K > n.min():
        raise ValueError(f"K={K} exceeds the smallest number of rollouts per task ({int(n.min())})")
    return c[:, None], n[:, None], np.arange(K)


def pass_at_k(c, n, K: int) -> np.ndarray:
    """[T, K] unbiased pass@k for k = 1..K (cumulative product, stable for n = 128)."""
    c, n, j = _check(c, n, K)
    return 1.0 - np.cumprod(np.clip(n - c - j, 0.0, None) / (n - j), axis=1)


def pass_hat_k(c, n, K: int) -> np.ndarray:
    """[T, K] unbiased pass^k for k = 1..K."""
    c, n, j = _check(c, n, K)
    return np.cumprod(np.clip(c - j, 0.0, None) / (n - j), axis=1)


def scalability(curve: np.ndarray, K: int) -> tuple[float, float]:
    """(A(K), S(K)) from a dataset-level curve with curve[k-1] = pass@k."""
    A = float(np.sum(curve[K - 1] - curve[: K - 1]))
    headroom = 1.0 - float(curve[0])
    S = A / headroom / (K - 1) if headroom > 0 else float("nan")
    return A, S


# Base vs. post-trained comparison

def _paired(base: Counts, rl: Counts):
    shared = sorted(set(base) & set(rl))
    if not shared:
        raise ValueError("the two policies share no task_id")
    arr = lambda d, i: np.array([d[t][i] for t in shared])  # noqa: E731
    return shared, arr(base, 0), arr(base, 1), arr(rl, 0), arr(rl, 1)


def sharpening_tax(base: Counts, rl: Counts, K: int, n_boot: int = 1000,
                   seed: int = 0) -> dict:
    """Tax_A(K), Tax_S(K) with 95% paired task-bootstrap intervals."""
    shared, cb, nb, cr, nr = _paired(base, rl)
    MB, MR = pass_at_k(cb, nb, K), pass_at_k(cr, nr, K)
    A_b, S_b = scalability(MB.mean(axis=0), K)
    A_r, S_r = scalability(MR.mean(axis=0), K)
    out = {"K": K, "n_tasks": len(shared), "A_base": A_b, "A_rl": A_r,
           "S_base": S_b, "S_rl": S_r, "tax_A": A_b - A_r, "tax_S": S_b - S_r}
    if n_boot:
        rng = np.random.default_rng(seed)
        T, reps = len(shared), []
        for _ in range(n_boot):
            idx = rng.integers(0, T, T)
            a_b, s_b = scalability(MB[idx].mean(axis=0), K)
            a_r, s_r = scalability(MR[idx].mean(axis=0), K)
            if np.isfinite(s_b) and np.isfinite(s_r):
                reps.append((a_b - a_r, s_b - s_r))
        reps = np.asarray(reps)
        out["ci95_tax_A"] = np.percentile(reps[:, 0], [2.5, 97.5]).tolist()
        out["ci95_tax_S"] = np.percentile(reps[:, 1], [2.5, 97.5]).tolist()
        out["P(tax_S > 0)"] = float((reps[:, 1] > 0).mean())
    return out


def task_categories(counts: Counts) -> dict:
    """Shares of tasks that always pass, pass given compute, and always fail."""
    c = np.array([v[0] for v in counts.values()])
    n = np.array([v[1] for v in counts.values()])
    return {"always_pass": float(np.mean(c == n)),
            "pass_given_compute": float(np.mean((c > 0) & (c < n))),
            "always_fail": float(np.mean(c == 0))}


def report(base: Counts, rl: Counts, K: int, n_boot: int = 1000) -> dict:
    """Print pass@k / pass^k on a doubling grid and the tax at K."""
    _, cb, nb, cr, nr = _paired(base, rl)
    ks = [k for k in (1, 2, 4, 8, 16, 32, 64, 128, 256) if k <= K]
    if ks[-1] != K:
        ks.append(K)
    cov_b, cov_r = pass_at_k(cb, nb, K).mean(0), pass_at_k(cr, nr, K).mean(0)
    con_b, con_r = pass_hat_k(cb, nb, K).mean(0), pass_hat_k(cr, nr, K).mean(0)
    print(f"{'k':>5} | {'pass@k base':>11} {'pass@k RL':>10} | {'pass^k base':>11} {'pass^k RL':>10}")
    for k in ks:
        print(f"{k:>5} | {cov_b[k-1]:>11.4f} {cov_r[k-1]:>10.4f} | {con_b[k-1]:>11.4f} {con_r[k-1]:>10.4f}")
    tax = sharpening_tax(base, rl, K, n_boot=n_boot)
    print(f"\nK={K}, {tax['n_tasks']} shared tasks")
    print(f"  A(K):  base {tax['A_base']:.3f}   RL {tax['A_rl']:.3f}   "
          f"Tax_A = {tax['tax_A']:+.3f}" + (f"  95% CI {np.round(tax['ci95_tax_A'], 3).tolist()}" if n_boot else ""))
    print(f"  S(K):  base {tax['S_base']:.4f}  RL {tax['S_rl']:.4f}  "
          f"Tax_S = {tax['tax_S']:+.4f}" + (f"  95% CI {np.round(tax['ci95_tax_S'], 4).tolist()}" if n_boot else ""))
    for name, counts in (("base", base), ("RL", rl)):
        cat = task_categories(counts)
        print(f"  {name:>4} tasks: always pass {cat['always_pass']:.1%}, pass given compute "
              f"{cat['pass_given_compute']:.1%}, always fail {cat['always_fail']:.1%}")
    return {"pass@k": {"base": cov_b.tolist(), "rl": cov_r.tolist()},
            "pass^k": {"base": con_b.tolist(), "rl": con_r.tolist()}, "tax": tax}


# IO + demo

def load_counts(path: str) -> Counts:
    """task_id -> (c, n) from an outcome JSONL."""
    out = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "pass_flags" in row:
                flags = row["pass_flags"]
                out[str(row["task_id"])] = (int(sum(map(bool, flags))), len(flags))
            else:
                out[str(row["task_id"])] = (int(row["c"]), int(row["n"]))
    return out


def demo(T: int = 500, N: int = 128, lam: float = 0.5, seed: int = 0) -> None:
    """Sharpen a random lam-fraction of tasks to p in {0, 1}; theory: Tax_A(K) = lam * A_base(K)."""
    rng = np.random.default_rng(seed)
    p_base = rng.beta(0.5, 1.5, T)
    sharpened = rng.random(T) < lam
    p_rl = np.where(sharpened, (rng.random(T) < p_base).astype(float), p_base)
    base = {f"t{i}": (int(c), N) for i, c in enumerate(rng.binomial(N, p_base))}
    rl = {f"t{i}": (int(c), N) for i, c in enumerate(rng.binomial(N, p_rl))}
    print(f"Synthetic pair: {T} tasks, N={N} rollouts, {sharpened.mean():.0%} of tasks sharpened\n")
    report(base, rl, K=N, n_boot=200)
    def area(p, K):  # A(K) from the true success rates
        q = 1.0 - p
        return float(np.mean(sum(q ** k for k in range(1, K)) - (K - 1) * q ** K))

    print("\nTheory check, Tax_A(K) = lam * A_base(K):")
    print(f"  {'K':>3} | {'true-p Tax_A':>12} {'lam * A_base':>12} | {'estimated Tax_A':>15}")
    for K in (8, 32, N):
        true_tax, predicted = area(p_base, K) - area(p_rl, K), sharpened.mean() * area(p_base, K)
        est = sharpening_tax(base, rl, K, n_boot=0)["tax_A"]
        print(f"  {K:>3} | {true_tax:>12.3f} {predicted:>12.3f} | {est:>15.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", help="outcome JSONL of the base policy")
    ap.add_argument("--rl", help="outcome JSONL of the post-trained policy")
    ap.add_argument("--K", type=int, default=None, help="budget (default: largest shared n)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--demo", action="store_true", help="run on synthetic data")
    args = ap.parse_args()
    if args.demo:
        demo()
        return
    if not (args.base and args.rl):
        ap.error("--base and --rl are required unless --demo is given")
    base, rl = load_counts(args.base), load_counts(args.rl)
    K = args.K or min(n for d in (base, rl) for _, n in d.values())
    report(base, rl, K, n_boot=args.n_boot)


if __name__ == "__main__":
    main()
