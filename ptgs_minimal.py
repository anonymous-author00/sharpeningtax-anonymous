"""PTGS (posterior-tempered group sampling, Sec. 5), the whole method in one file.

Each prompt keeps a discounted Beta posterior over its current success rate, estimated from its
past rollout groups. Before the next group, p_hat is drawn from it and the group is decoded at
T = tau^h(p_hat): hard prompts are heated (up to tau), mastered ones cooled (down to 1/tau).
The loss is unchanged; PTGS is off at evaluation.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


# The pivot (target success rate) ramps geometrically from 0.25 to 0.5; 0.5 from step 0 would
# heat every prompt of a policy that still fails almost everything.
def pivot_at(step: int, total_steps: int, pivot_start: float = 0.25, pivot_end: float = 0.5) -> float:
    u = min(max(step / max(total_steps, 1), 0.0), 1.0)
    return pivot_start * (pivot_end / pivot_start) ** u


# Eq. (4): T(0) = tau * T_ref, T(pivot) = T_ref, T(1) = T_ref / tau.
def temperature_from_p(p_hat: float, pivot: float, tau: float, t_ref: float = 1.0) -> float:
    p_hat = min(max(p_hat, 0.0), 1.0)
    h = (pivot - p_hat) / (pivot if p_hat <= pivot else 1.0 - pivot)
    return t_ref * tau**h


class PTGS:
    """Posteriors keyed by a stable task id (e.g. the environment seed), not the prompt text."""

    def __init__(self, tau=1.5, gamma=0.95, t_ref=1.0, pivot_start=0.25, pivot_end=0.5, seed=0):
        self.tau, self.gamma, self.t_ref = tau, gamma, t_ref
        self.pivot_start, self.pivot_end = pivot_start, pivot_end
        self.s = defaultdict(float)  # discounted successes
        self.f = defaultdict(float)  # discounted failures
        self.rng = np.random.default_rng(seed)
        self.pivot = pivot_start

    def set_progress(self, step: int, total_steps: int) -> None:
        self.pivot = pivot_at(step, total_steps, self.pivot_start, self.pivot_end)

    def temperature(self, key: str) -> float:
        # prior of mass 2 centred on the pivot, so an unseen prompt is neutral
        a = 2.0 * self.pivot + self.s[key]
        b = 2.0 * (1.0 - self.pivot) + self.f[key]
        p_hat = self.rng.beta(a, b)  # Thompson sampling
        return temperature_from_p(p_hat, self.pivot, self.tau, self.t_ref)

    def update(self, key: str, successes: float, n: float) -> None:
        self.s[key] = self.gamma * self.s[key] + successes
        self.f[key] = self.gamma * self.f[key] + (n - successes)


# Algorithm 1 in an RL loop. (C) matters: rollouts come from softmax(logits / T_x), so the
# policy-gradient ratio must use T_x too, or updates are off-policy (most on the hottest prompts).
def rl_loop_sketch(task_pool, policy, rl_update, total_steps=200, group_size=16, batch_tasks=8):
    ptgs = PTGS(tau=1.5, gamma=0.95, t_ref=1.0, pivot_start=0.25, pivot_end=0.5)

    for step in range(total_steps):
        ptgs.set_progress(step, total_steps)  # (D) advance the pivot ramp
        batch = task_pool.sample(batch_tasks)

        temps = {x: ptgs.temperature(x) for x in batch}  # (A) one T per prompt group
        groups = {x: policy.generate(x, n=group_size, temperature=temps[x]) for x in batch}

        for x, group in groups.items():  # (B) fold the outcome back into the posterior
            ptgs.update(x, successes=sum(g.success for g in group), n=group_size)

        rl_update(  # (C) score log-probs at the *sampling* temperature
            groups,
            log_probs=policy.log_probs(groups, temperature=temps),
        )


if __name__ == "__main__":
    tau, t_ref = 1.5, 1.0
    print(f"Tempering rule, tau={tau}, T_ref={t_ref}  (Eq. 4)\n")
    print(f"{'p_hat':>7} |" + "".join(f"{f'p~={pv}':>10}" for pv in (0.25, 0.5)))
    print("-" * 29)
    for p in (0.0, 0.1, 0.25, 0.4, 0.5, 0.75, 1.0):
        row = "".join(f"{temperature_from_p(p, pv, tau, t_ref):10.3f}" for pv in (0.25, 0.5))
        print(f"{p:7.2f} |{row}")

    print("\nPivot ramp over a 200-step run (geometric, 0.25 -> 0.50)")
    print("  " + "  ".join(f"step {s:>3}: {pivot_at(s, 200):.3f}" for s in (0, 50, 100, 150, 200)))

    print("\nOne prompt's trajectory: fails everything, then learns (n=16/group, gamma=0.95)")
    ptgs, key = PTGS(seed=0), "task_0017"
    for step, observed_successes in enumerate([0, 0, 1, 3, 8, 13, 15, 16]):
        ptgs.set_progress(step * 25, 200)
        a = 2 * ptgs.pivot + ptgs.s[key]
        b = 2 * (1 - ptgs.pivot) + ptgs.f[key]
        t = ptgs.temperature(key)
        print(
            f"  step {step * 25:>3}  pivot={ptgs.pivot:.3f}  "
            f"posterior mean={a / (a + b):.3f}  "
            f"-> T={t:.3f}   (observed {observed_successes:>2}/16 successes)"
        )
        ptgs.update(key, observed_successes, 16)
