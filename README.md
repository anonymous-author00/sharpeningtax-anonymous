# Sharpening Tax in Post-Training: code skeleton

Three self-contained files, one per contribution of the paper. Each can be read top to
bottom in a few minutes and runs on its own.

| File | Paper | What it contains |
| --- | --- | --- |
| `evaluate_minimal.py` | Sec. 2-3, appendix on the harness | The text harness that lets a base model act as a tool-calling agent (Python-stub tool catalog, JSON `tool_call` contract, plain-text transcript, stop sequences, lenient parser); the base policy (harness over `/v1/completions`) and the post-trained policy (native chat template over `/v1/chat/completions`); N rollouts per task through an `Environment` interface, with a small multi-step tool-use task |
| `sharpening_tax_minimal.py` | Sec. 2 and 4, Eq. (1)-(3) | Unbiased pass@k and pass^k, raw and calibrated scalability A(K), S(K), the Sharpening Tax Tax_A(K), Tax_S(K) with paired task-bootstrap CIs, and task categories; a synthetic check of the proportional-tax theorem |
| `ptgs_minimal.py` | Sec. 5, Eq. (4), Algorithm 1 | Posterior-tempered group sampling: the pivot schedule, the tempering rule, the discounted Beta posterior, and the four places it plugs into an RL loop |

## Metrics

With c_i successes out of n_i rollouts on task i, and dataset metrics averaged over tasks:

```
pass@k_i = 1 - C(n_i - c_i, k) / C(n_i, k)          coverage
pass^k_i = C(c_i, k) / C(n_i, k)                    consistency
A(K)     = sum_{k=1}^{K-1} [pass@K - pass@k]         raw area scalability           Eq. (1)
S(K)     = A(K) / ((K - 1)(1 - pass@1))             calibrated scalability, [0, 1]  Eq. (2)
Tax_X(K) = X_base(K) - X_RL(K),  X in {A, S}        Sharpening Tax                  Eq. (3)
```

A positive tax means post-training reduced how much the policy gains from additional rollouts.

## PTGS

Each prompt `x` keeps discounted success/failure counts from its past rollout groups. Before
its next group, PTGS draws `p_hat ~ Beta(2*pivot + s_x, 2*(1 - pivot) + f_x)` and decodes the
group at `T_x = tau ** h(p_hat)`, where `h` is +1 at `p_hat = 0`, 0 at the pivot and -1 at
`p_hat = 1`. Prompts the policy keeps failing are heated (up to `tau`), mastered ones cooled
(down to `1/tau`). The pivot ramps from 0.25 to 0.5 over training; the paper uses `tau = 1.5`
and `gamma = 0.95`. Nothing in the loss changes. The one trainer-side requirement is that
the policy-gradient log-probs are computed at each prompt's own sampling temperature.

## Usage

Python 3.10+ with `numpy`; `evaluate_minimal.py` also needs `openai` and two vLLM servers.

```bash
python sharpening_tax_minimal.py --demo    # metrics on synthetic data + the proportional-tax theorem
python ptgs_minimal.py                     # the tempering rule, the pivot ramp, one prompt's trajectory

# base vs. post-trained evaluation end to end, with any base / post-trained pair
vllm serve Qwen/Qwen2.5-7B --port 8000
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001 --enable-auto-tool-choice --tool-call-parser hermes
python evaluate_minimal.py --base-model Qwen/Qwen2.5-7B --base-url http://127.0.0.1:8000/v1 \
                           --rl-model Qwen/Qwen2.5-7B-Instruct --rl-url http://127.0.0.1:8001/v1 --n 16

# the tax for your own rollouts: JSONL, one row per task with "task_id" and "pass_flags"
python sharpening_tax_minimal.py --base base.jsonl --rl rl.jsonl --K 128
```

## Paper setting

Sections 3-4 evaluate 14 base / post-trained pairs (Gemma-4, Ministral-3, Qwen2.5, Qwen3.5)
with N = 128 rollouts per task on BFCL v4 `multi_turn_base` (200 tasks), WebShop (500 goals)
and ACEBench (770 tasks), serving every model with vLLM 0.19. Decoding: temperature 0.4
(BFCL) or 0.7 (WebShop, ACEBench), top-p 0.95; thinking mode is disabled for post-trained
models that have one, and the vLLM tool-call parser must match the model family (e.g.
`gemma4`, `mistral`, `hermes` for Qwen2.5, `qwen3_xml` for Qwen3.5).

Section 5 trains `Qwen2.5-7B-Instruct` with multi-turn PPO (StarPO-S) in RAGEN on FrozenLake
and Sokoban for 200 steps (8 tasks x 16 rollouts per step, five seeds), with and without PTGS,
and evaluates checkpoints with 128 rollouts per task at a fixed temperature of 0.5.