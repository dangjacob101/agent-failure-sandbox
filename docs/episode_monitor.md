# Earlier episode-level monitor

Predict **failure to complete a task within a fixed action budget**, before the
agent finishes. The pipeline keeps the paper's Monte Carlo continuation score
and conformal prediction idea, and replaces its stepwise threshold scripts with
finite-sample, episode-level calibration for repeated checks. It does not require
activations, probes, supervised fine-tuning, preference data, or steering.

This is a sandbox implementation, not evidence of reliable 100+ turn prediction.
MiniWoB tasks here are short. A synthetic 120-turn preset tests the software path;
it is not a long-horizon MiniWoB benchmark or a scientific performance result.

## Quick start

Python 3.10+; Python 3.12 is used for local validation. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[probes]'
python -m unittest discover -s tests -v
python -m early_failure --smoke --output-dir runs/synthetic_smoke
```

The synthetic smoke needs no downloads, GPU, or browser. It tests collection,
calibration, prediction, metrics, and output writing. `synthetic: true` is included
in its results. Low-sample runs may yield only uncertain predictions.

For real Qwen + MiniWoB++:

```bash
pip install -e '.[hf,miniwob]'
# Alternatively, reproduce dependency versions from the checked-in uv.lock:
# uv sync --frozen --extra hf --extra miniwob
# Install Google Chrome/Chromium; Selenium needs a compatible ChromeDriver.
# MiniWoB 1.0 bundles the task HTML; a separate web server is not required.
python -m early_failure --config configs/miniwob_tiny_smoke.json
# After the integration smoke, collect a larger independent calibration set:
python -m early_failure --config configs/miniwob_small.json
# Same larger model as the reference collection script:
python -m early_failure --config configs/miniwob_reference7b.json
```

The tiny smoke uses only two calibration episodes and one evaluation episode;
it **cannot establish detection quality or reject at alpha=0.1**. The small
preset uses 20 calibration / 10 evaluation seeds per environment, 8 turns, and
4 continuations per check. Even this may be statistically underpowered. Each
run refuses to overwrite a nonempty output directory. Use `--output-dir` for
another run. The first real run downloads model weights; 7B requires considerably
more memory. The adapter chooses CUDA, then Apple MPS, then CPU, or accepts
`--device cpu`, `mps`, or `cuda:0` explicitly.

The default is `Qwen/Qwen2.5-Coder-0.5B-Instruct`, matching the reference repo's
local test. Its collection script's `Qwen/Qwen2.5-Coder-7B-Instruct` is available
through the 7B preset. Model weights and inference stay local; there is no paid
model API in the default pipeline.

## One parent function

```python
from early_failure import ExperimentConfig, run_experiment

result = run_experiment(
    model_id="Qwen/Qwen2.5-Coder-0.5B-Instruct",
    env_ids=("miniwob/click-test-2-v1", "miniwob/enter-text-v1"),
    calibration_seeds=tuple(range(40)),
    evaluation_seeds=tuple(range(1000, 1020)),
    max_steps=8,
    mc_samples=8,
    check_every=1,
    temperature=0.7,
    alpha=0.1,
    output_dir="runs/my_experiment",
)
# Or pass ExperimentConfig / ExperimentConfig.from_json(...) and override fields.
```

`run_experiment` collects complete calibration episodes, calibrates separately
for each environment and final outcome, and then monitors held-out episodes.
Checks occur **before** the next action: `step=0` is before any action. Alerts do
not terminate or steer an episode. This preserves its eventual outcome for an
honest early-warning evaluation. Infrastructure errors abort the run and write
`error.json`; they are never relabeled as task failures.

Reusing a calibration artifact:

```python
config = ExperimentConfig.from_json("configs/miniwob_small.json")
result = run_experiment(
    config,
    calibration_path="runs/qwen05b_miniwob/calibration.json",
    evaluation_seeds=tuple(range(2000, 2010)),
    output_dir="runs/qwen05b_fresh_eval",
)
```

Keep the original calibration seed manifest. Model, task set, horizon,
generation, score, and environment settings are checked against a saved protocol
fingerprint. Resolved model revision, device/precision, browser/driver versions,
and Python dependency versions are checked too. Recalibrate after changes.
Prefer a pinned Hub `model_revision` for repeatable experiments; local checkpoint
contents and custom implementations must be versioned using the supplied tags.
Changing alpha is permitted, but selecting it after seeing evaluation outcomes
invalidates an untouched-test-set claim.

## Components and other evaluations

| File | Responsibility |
| --- | --- |
| `types.py` | Agent, environment, task, and observation interfaces |
| `models.py` | Hugging Face generation and bounded history |
| `miniwob.py` | DOM observations, actions, reward, and browser lifecycle |
| `history.py` | Shared conversation updates and independent random seeds |
| `scoring.py` | Default Monte Carlo scorer and checked prefix replay |
| `rollouts.py` | Real episode loop; calls a scorer before selected actions |
| `conformal.py` | Calibration, p-values, prediction sets, serialization |
| `metrics.py` | Early-warning metrics |
| `experiment.py` | Parent function, manifests, calibration reuse, outputs |

Read `types.py`, then `rollouts.py`, then `experiment.py` to follow the main flow.
Inline comments explain non-obvious choices; statistical details live below.

The model, environment, and scorer can each be replaced independently:

| Change | Parent function parameters |
| --- | --- |
| Another Hugging Face model | `model_id="..."` |
| Another model backend | `agent=my_agent, agent_tag="my-agent-v1"` |
| Another evaluation with the built-in model | `environment="custom", environment_factory=my_factory, environment_tag="my-env-v1"` |
| Another risk scoring method | `scorer=my_scorer, scorer_tag="my-scorer-v1"` |

Built-in `agent_backend` (`hf` or `synthetic`) and `environment` are separate
settings. The smoke option selects both synthetic backends. Use meaningful
`env_ids` for custom tasks and version tags for all injected implementations.
Adapters may expose `provenance()` to record runtime versions after reset.

The agent implements
`generate(messages, *, seed, temperature) -> str`. The factory must return a
fresh environment with `system_prompt`, `reset(Task) -> Observation`,
`step(str) -> Observation`, and `close()`. Observations are independent snapshots;
success must be known only from actual task completion, not guessed at reset.

The agent must be stateless apart from its weights: output depends on the supplied
history and seed, and hypothetical branches must not mutate the real agent's
memory. The built-in model isolates random generators for each main/branch call.
History trimming removes oldest complete exchanges while retaining the system,
initial task, and latest observation; irreducible oversized prompts fail clearly.
This is bounded-context behavior, not unlimited long-term memory.

The benchmark must support repeatable reset + prefix replay for the MC scorer.
Every replayed observation and adapter-provided state signature is checked. A
mismatch raises `ReplayMismatch`. Environments with external side effects,
hidden state, real-time events, or no reset mechanism need their own state
snapshot/branching mechanism or another score implementation. The conformal
module itself accepts any fixed success-oriented scalar score in [0,1], so a
future learned risk predictor can replace the expensive MC scorer and be
calibrated independently on completed episodes.

A custom scorer implements `score(prefix: EpisodePrefix) -> Score`. It receives
a copy of the observed history and remaining action budget. Return `Score(q)`
with a finite success-oriented number in [0,1]; optional diagnostics belong in
`Score(q, details={...})`. It must not learn from calibration/evaluation outcomes
or alter the deployed policy. The runner timestamps each score, and conformal
calibration and metrics work without Monte Carlo fields. Only the default MC
scorer requires branch environments and replay. Tests verify that a custom scorer
can run with an environment that refuses replay.

## What the conformal layer guarantees

At checkpoint t, sample N independent policy continuations. With the default
full remaining horizon, `q_t` is their fraction completing the task within the
remaining action budget. This score uses only the observed prefix and simulated
continuations, never the real trajectory's future.

For each **completed calibration episode** i, save one nonconformity maximum:

```text
A_i(success) = max_t (1 - q_it),  if the episode eventually succeeds
A_i(failure) = max_t q_it,        if the episode eventually fails

A_prefix(success) = max_{s <= t} (1 - q_s)
A_prefix(failure) = max_{s <= t} q_s
p_y = (1 + count(A_i(y) >= A_prefix(y))) / (n_y + 1)
prediction_set = {y: p_y > alpha}
```

Only the singleton `{failure}` triggers an alarm. `{success,failure}` is
uncertain; the empty set is logged as `out_of_distribution`, an operational
label rather than proof of distribution shift. p-values are compatibility
ranks, **not posterior failure probabilities**. Ties are conservative. A missing
calibration class receives p=1 and cannot be rejected. If no failed episodes were
calibrated, an alarm can still reflect rejection of success, but does not provide
positive evidence from a learned failure distribution.

Under exchangeability of calibration and new completed episodes **within each
environment and final outcome**, and with the policy, scoring rule, horizon, and
check schedule fixed, the probability of ever excluding the true outcome in a
new episode is at most alpha. This implies an episode-level false-alarm bound
among successful episodes. A prefix maximum cannot exceed its full-episode
maximum, so its p-value cannot be smaller than the full-episode p-value. The
usual finite-sample conformal rank bound therefore covers all the prefix checks;
it does not require independent turns or a separate alpha spend per turn.

This is an extension of the paper's per-step method, not a verbatim reproduction.
It is not a guarantee under distribution shift, adaptive policy changes, arbitrary
extensions beyond the calibrated horizon, or tuning on the evaluation set. It
does not guarantee failure recall or locate the first irreversible mistake.

The smallest possible p-value is `1/(n_y+1)`: at alpha=0.1, at least 9 calibration
episodes of a class are needed even to permit its rejection. Coarse MC scores and
ties may require substantially more data/samples. The saved diagnostics report
counts and rank resolution; no missing-class values are silently set to zero.

## MiniWoB protocol and scaling

The initial supported tasks are `click-test-2`, `click-button`,
`click-button-sequence`, `click-checkboxes`, `click-dialog`, `enter-text`, and
`login-user` (all `miniwob/<name>-v1`). These are a reviewed static DOM subset.
Dynamic, timed, pixel/canvas, and other unreviewed tasks are rejected explicitly.
Observable replay checks cannot establish equality of all hidden browser state.

MiniWoB's normal wall-clock episode timer can expire during model generation or
MC. The adapter uses a guarded JavaScript hook to disable that episode timer and
time-dependent reward decay, and evaluates full terminal success (`raw_reward ==
1.0`) under an **action-count budget**. This protocol differs from stock timed
MiniWoB and must be reported with results. Arbitrary task timers are not frozen;
the static-task restriction is deliberate. Headless mode is honored, seed zero
is valid, TYPE focuses its target, PRESS uses native key indices, and SCROLL
uses wheel events. Bad model action syntax consumes a no-op turn with feedback.
Bundled HTML paths are converted to escaped file URIs, including workspaces whose
paths contain spaces. Selenium's driver cache can be relocated with `SE_CACHE_PATH`
if your runtime restricts writing to the default home cache.

Full-horizon MC is expensive: roughly O(N * H^2 / check_every) generated actions,
plus replay and browser startup. `check_every` reduces monitoring frequency.
`mc_max_steps` optionally caps continuation length; then q measures completion
within the shorter lookahead, **not eventual completion probability**. Unfinished
short-cap branches are recorded as censored. Full-horizon step-budget exhaustion
is a bounded-horizon failure. Conformal calibration still uses the actual full
episode outcomes with the same capped score protocol.

For 100+ turn research, start by changing the synthetic horizon:

```bash
python -m early_failure --config configs/synthetic_long.json
```

Do not call an 8-turn task a long-horizon evaluation merely by setting
`max_steps=120`. A meaningful study needs genuinely long tasks, sufficient
successful and failed calibration episodes, a practical score/check budget, and
recalibration for that task distribution. A learned scorer calibrated with this
same episode-level method is a future route to lower-latency warnings; it is not
implemented or validated here.

## Outputs and verification

The fixed outcome pilot uses 100 calibration episodes and 50 separate evaluation
episodes of `click-test-2`, with the local 0.5B model, three actions, and four
continuations per check. Its protocol is in
`configs/miniwob_outcome_pilot_protocol.json`. It measures whether early warnings
match final outcomes on this one short task template; it does not test long tasks.

```bash
python -m early_failure --config configs/miniwob_outcome_pilot.json
# After every episode finishes:
python scripts/summarize_outcome_pilot.py runs/outcome_pilot_150
```

The report in `analysis/` includes final outcomes, warning recall and precision,
false alarms, warning lead time, uncertainty, and two fixed comparison rules.
It validates the saved predictions against calibration before reporting them.
Uncertain predictions never count as correct decisions. Missing successful or
failed episodes leave the corresponding rates undefined, and calibration score
ties are reported when they prevent any warning. Keep the model, prompt,
calibration, and reporting rules fixed while collecting an evaluation set.

Completed locally on 2026-10-05 in `runs/outcome_pilot_150/`: calibration had
3 successes and 97 failures; the separate test set had 3 successes and 47
failures. No failure received a warning. Of 148 checks, 147 were uncertain and
one correctly predicted success. There were no false warnings, but no warnings
at all. Three successful calibration episodes are too few to reject success at
alpha 0.1; tied worst-case scores further raise the smallest possible success
p-value to 0.75. This pilot validates the complete execution and reporting path,
but does not demonstrate useful early failure detection. The report includes
the fixed baselines and uncertainty intervals; its 10 validation tests pass.

Each run writes `config.json`, `metadata.json`, `calibration.json`,
`calibration_diagnostics.json`, `evaluation_episodes.jsonl`, and `summary.json`.
Fresh calibration also writes `calibration_episodes.jsonl`. Episodes include
actions, observations, pre-action MC scores, prediction sets, p-values, runtime,
and final outcome reasons. Metrics include successful/failed denominators,
episode false-alarm rate, failure recall, alarm precision, lead-time distribution
via per-alarm records, recall at several minimum lead times, uncertainty, and
simultaneous coverage by outcome class. Undefined denominators are `null`.

```bash
python -m unittest discover -s tests -v
# Optional actual Chrome integration, including a >10-second delayed click:
EARLY_FAILURE_BROWSER_TESTS=1 python -m unittest discover -s tests -p test_browser_integration.py -v
```

Validated locally on 2026-10-05: 59 unit tests passed, and all 3 opt-in Chrome
integration tests passed separately. A real Qwen 0.5B run on Apple MPS completed
two calibration episodes and one evaluation episode using the tiny preset. All
three episodes failed and all evaluation checks were uncertain; there were no
successful calibration examples. This verifies integration, not detection
quality. Its artifacts are in `runs/verified_qwen05b_browser/` in this working
copy (run data and downloaded weights are intentionally Git-ignored). The
refactored pipeline was rerun in `runs/verified_decoupled_browser/`. A
120-turn synthetic run and a 150-check conformal coverage test also completed;
neither establishes performance on genuine long tasks.

## Provenance

- [Paper](https://arxiv.org/abs/2604.19775): *From Actions to Understanding:
  Conformal Interpretability of Temporal Concepts in LLM Agents*.
- [Paper code](https://github.com/trilokpadhi/interpretability-of-llm-agents),
  cloned at `0b041ccdd2e5de66b6300cbb14d20f06f226d295`.
- [Reference sandbox](https://github.com/skarnati20/interp-sandbox), inspected at
  `af85541efd8fb8e9ca0e60ff1ec8b88e34470b10`; source of model choice and DOM/action setup.
- [MiniWoB API](https://miniwob.farama.org/content/basic_usage/) and MiniWoB 1.0
  source, used to verify environment/actions/reward/timer behavior.

The upstream threshold script uses reversed percentile tails relative to its
stated coverage and lacks the +1 rank correction. Its late-step pooling also
counts dependent steps as samples. Those code paths are retained for provenance
but are not imported here. The reference adapter's unconditional success return,
TYPE target handling, and SCROLL mapping are corrected in this implementation.
