# Agent Failure Sandbox

A small, local experiment in predicting whether an LLM agent will fail before
its task ends. The eventual research goal is prediction on **100+ turn tasks**;
the experiment here tests the pipeline on a **two-action MiniWoB++ task**.

## Failure prediction and linear probes

**Failure prediction is the goal; a linear probe is the small classifier used
to read a possible failure signal from the model's hidden activations.** Here,
we sample possible continuations and use conformal calibration to label
intermediate states. Those labels train the probes; uncertain states are excluded.

An alternative would train a probe directly on each run's eventual outcome;
that alternative has not been evaluated here. We use real outcomes to form
calibration groups and, separately, test whether our probe warnings precede
actual failures. The [earlier monitor](docs/episode_monitor.md) used conformal
predictions directly, without linear probes.

## Run the experiment

Install Python 3.12, [uv](https://docs.astral.sh/uv/), and Google Chrome/Chromium.
Then run:

```bash
git clone https://github.com/dangjacob101/agent-failure-sandbox.git
cd agent-failure-sandbox
uv sync --frozen --extra hf --extra miniwob --extra probes
export HF_HOME="$PWD/.cache/huggingface"
.venv/bin/python -m early_failure --pipeline paper \
  --config configs/miniwob_paper.json \
  --probe-config configs/miniwob_paper_probes.json \
  --output-dir runs/paper_rerun
```

This runs calibration, probe training, and held-out evaluation. The preset uses
an Apple GPU (MPS); for other hardware, edit `device` and `dtype` in a copied
config (`cuda` for NVIDIA, or `cpu` with `float32`). The first run downloads model
weights. Our main run took about 79 minutes locally. Choose a new output directory
for each rerun. To check the saved results without rerunning the experiment, see
[Reproduce or inspect locally](#reproduce-or-inspect-locally).

## Paper and reference implementation

This implements the continuation scoring, conformal state labeling, and linear
probing sequence from [*From Actions to Understanding*](https://arxiv.org/abs/2604.19775),
Sections 4.1–4.3. The Qwen model family and MiniWoB DOM/action interface were
informed by [interp-sandbox](https://github.com/skarnati20/interp-sandbox).
It is a small-model adaptation, not a reproduction of the paper's fine-tuning,
original benchmarks, or steering experiments.

## Main result

The primary probe flagged **17 of 25 eventual failures (68%)**, with **0 false
warnings among 15 successful runs**. Every warning arrived one action before
the end. Calibration, probe training, and final evaluation used separate
episodes: **80 / 40 / 40**, respectively.

| Layer | Failures warned about | False warnings on successful runs |
| --- | ---: | ---: |
| 7 | 17/25 | 2/15 |
| **14 — primary, selected before testing** | **17/25** | **0/15** |
| 21 | 18/25 | 0/15 |
| 28 | 18/25 | 0/15 |

**What the warnings detected matters:** all 17 primary warnings followed typing
into the form container (`ref=4`) instead of the input (`ref=5`), leaving the
input empty. With one action left, the agent could not both type correctly and
submit. A simple action/DOM rule could identify that same mistake. This is a
post-hoc observation; an advantage over that rule has not been established.

The result demonstrates a working labeling-and-probing pipeline and a narrow
predictive signal. It does **not** establish prediction of future mistakes,
causal failure directions, or long-horizon performance.

### State-label accuracy and final outcomes are different

The primary probe matched **19/20 conformal-labeled test states (95%)**. The
training-majority label baseline matched **18/20 (90%)**. Only two of those
twenty labels were success. Across all 78 test states, 58 remained uncertain.

Outcome evaluation also includes states with uncertain labels: the trained
probe predicted on 38/40 episodes. The two unsupported episodes failed on their
first action, before the trained checkpoint; they remain in the denominator
of 25 failures. Of the eight missed failures:

- Two clicked Submit immediately.
- Five typed correctly, then typed again instead of submitting.
- One clicked the input before typing, exhausting the two-action budget.

An always-warn baseline detects 25/25 failures but falsely warns on all 15
successes. A never-warn baseline detects none. The primary failure-recall
Wilson 95% interval is approximately **48–83%**; the false-warning interval is
**0–20%**. Zero observed false warnings is not a zero-risk guarantee.

Read the [full interpretation](docs/paper_pilot_result.md),
[machine-readable results](results/paper_pipeline_qwen15b/summary.json), and
[independent audit](results/paper_pipeline_qwen15b/artifact_audit.json).

## Methodology

### 1. Estimate each intermediate state's reward

Before each real action, replay the observed prefix in separate browser
instances and sample four continuations. Each continuation gets the full
remaining action budget. Its terminal success is 1 or 0, and `q` is the mean
of those four outcomes. These hypothetical branches never enter the real
episode's history. Main actions and branches use separate deterministic seeds.

The benchmark uses an action budget: MiniWoB's timer and reward decay are
disabled for the supported static DOM tasks. Unfinished tasks at the budget
are failures. Browser/model errors abort collection instead of becoming labels;
replayed observations must match the saved prefix.

### 2. Apply the paper's written conformal labeling rule

Group calibration scores by environment, timestep, and the **real episode's
final outcome**. Nonconformity is `1-q` for success and `q` for failure. For each
candidate class, calculate:

```text
p = (1 + number of same-class calibration scores >= the new score)
    / (number of same-class calibration scores + 1)

success target: p_success >= alpha and p_failure < alpha
failure target: p_failure >= alpha and p_success < alpha
otherwise: no binary training target
```

Here `alpha=0.1`. Ties count conservatively; rejection is strictly `p < alpha`.
Uncertain, empty, and unsupported states never inherit the real final outcome
as a substitute target. Calibration and probe-training episodes are separate.
There is no maximum over previous timesteps.

The primary `paper_equations` mode follows the written equations. The optional
`upstream_percentiles` compatibility mode reproduces the released repository's
different percentile/success-first rule and makes no conformal coverage claim.
It was **not** used for this run.

At this alpha, at least ten calibration examples of each relevant class at a
timestep are necessary to permit rejection; ties can still block labels. The
runner stops if calibration cannot produce both targets, or if training lacks
both classes. It does not fabricate a constant classifier.

### 3. Extract activations and fit linear probes

Reconstruct each real pre-action prefix using its saved system prompt and
observed history. Extract Hugging Face `hidden_states[layer]` at the last
nonpadding prompt token, using the same chat template and history limits as
generation. Layers are numbered 1 through N; zero is the embedding output.
The final layer's returned state may include final normalization.

Fit separate balanced logistic regressions on raw features for each environment,
timestep, and layer. The targets are conformal state labels, not final outcomes.
The four trained probes each used **17 states: 13 failure and 4 success**, with
1,536 features. Another 62 training states remained uncertain. No initial-state
probe could be trained; both labels were available only after the first action.

The [saved coefficients](results/paper_pipeline_qwen15b/probes.json) point toward
success; their negative is the failure-associated direction. Separability is
association, not evidence that intervening on a direction causes failure.

### 4. Test with the probes frozen

Save the probe bank before collecting test episodes. Report every selected
layer, with layer 14 fixed as the primary layer beforehand. The failure decision
threshold is `p_success < 0.5`. Features exclude later actions, terminal
observations, outcomes, and Monte Carlo branch text.

Evaluate agreement with conformal labels on labeled states, and separately
compare the first probe warning with the actual episode outcome. This uses
offline replay of pre-action prefixes: actions are never stopped or steered.
Pointwise conformal labeling statements do not automatically bound probe errors
or the chance of any warning across an entire episode.

## Fixed experiment settings

| Setting | Value |
| --- | --- |
| Model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| Model revision | `2e1fd397ee46e1388853d2af2c993145b0f1098a` |
| Environment | `miniwob/enter-text-v1` |
| Budget / continuations per state | 2 actions / 4 continuations |
| Calibration / training / test seeds | 50000–50079 / 60000–60039 / 70000–70039 |
| Sampling | temperature 0.7; at most 48 new tokens; seed 42 |
| Layers / primary layer | 7, 14, 21, 28 / 14 |
| Device / dtype | Apple MPS / float16 |
| Runtime | 78.8 minutes for the main run |

Full settings, the exact prompt, and the fixed protocol are in
[configs](configs/), with runtime versions in
[metadata.json](results/paper_pipeline_qwen15b/metadata.json).
The larger model and clearer prompt were selected using separate development
seeds 40000–40019. This is a fresh pilot, not a controlled comparison with the
earlier 0.5B experiment.

## Limits and next research questions

- Seventeen labeled training states and forty test episodes provide limited
  evidence; there were only four success targets for training.
- Disjoint episode seeds do not mean unseen text: **43/78 test prefixes exactly
  match training prompt text**. This is not an unseen-task evaluation.
- The detected mistake is visible in the action and DOM. The current result
  does not establish value beyond a simple task-specific rule.
- A two-action budget makes some errors unrecoverable that a longer run could
  repair. No 100+ turn browser experiment has been completed here.
- Four-sample continuation estimates are coarse and can leave many states
  uncertain. Later timesteps also lose calibration support as episodes finish.

The next useful experiments would increase labeled support, separate goal text
across splits, compare against simple observable-state baselines, and then
lengthen tasks. New model/layer/threshold choices need a fresh final test set.

## Reproduce or inspect locally

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). Start from the repository root.
Inspect the archived results without a model, GPU, or browser:

```bash
uv sync --frozen --extra probes
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python results/paper_pipeline_qwen15b/audit_paper_run.py \
  results/paper_pipeline_qwen15b --workspace snapshots/paper_pipeline_qwen15b
```

The suite runs 129 tests: 126 pass and three real-browser tests are opt-in.
The independent audit checks recorded ranks, targets, prefix identities, saved
weights, predictions, metrics, split separation, and the archived source hashes.
It does not regenerate activations. CI runs these checks without downloading
model weights. Omit `--workspace` to also require the current `src/` files to
match the historical run.

To collect another real experiment, use [Run the experiment](#run-the-experiment)
above. Runtime and sampled outcomes may differ across hardware and library
versions. No paid API or Colab is required. New outputs go in ignored `runs/`;
the curated `results/` directory is committed. Nonempty output directories
are refused.

For optional real-browser adapter tests after installing dependencies:

```bash
EARLY_FAILURE_BROWSER_TESTS=1 .venv/bin/python -m unittest discover \
  -s tests -p test_browser_integration.py -v
```

## One parent function, replaceable components

```python
from early_failure import ExperimentConfig, ProbeConfig, run_paper_experiment

result = run_paper_experiment(
    ExperimentConfig.from_json("configs/miniwob_paper.json"),
    probe_config=ProbeConfig(
        training_seeds=tuple(range(60000, 60040)),
        layers=(7, 14, 21, 28),
        primary_layer=14,
        labeling_mode="paper_equations",
    ),
    output_dir="runs/my_experiment",
)
```

`ExperimentConfig` controls the model, environment, scoring, horizon, and
calibration/test seeds. `ProbeConfig` controls training seeds, layers, and label
mode. When changing models through the function, also change `model_revision`
and choose valid layers. The CLI clears an inherited revision on `--model-id`
changes; `--model-revision` pins the new model.

Inject `agent`, `environment_factory`, `scorer`, or `feature_extractor`
independently, with their version tags. A custom feature extractor implements
`extract_features(messages, *, layers)`. A new benchmark may need its own replay
or scoring infrastructure; the interface does not remove that requirement.

| Module | Responsibility |
| --- | --- |
| `paper_pipeline.py` | Parent runner, split separation, support gates |
| `rollouts.py` / `scoring.py` | Real episodes / independent continuation scoring |
| `step_labeling.py` | Per-timestep conformal state labels |
| `models.py` / `features.py` | Model adapter / pre-action activation records |
| `probes.py` | Linear probe fitting, numeric weights, label evaluation |
| `probe_metrics.py` | Warnings compared with real final outcomes |
| `miniwob.py` | Browser tasks, actions, and observations |

`run_experiment` and the CLI's `episode` mode retain the earlier episode-maximum
monitor. Use **`run_paper_experiment` or `--pipeline paper`** for the method
described here. The two methods should not be conflated.

## Included results and provenance

The [result index](results/README.md) distinguishes the completed paper-pipeline
run, the earlier 150-episode monitor run, and development-only actor checks.
The main archive includes all raw episodes, scores, feature vectors, probe
weights, predictions, settings, source hashes, and audit scripts. The incomplete
H4/MC8 timing check is explicitly preserved as development work.

Model weights, virtual environments, caches, and the upstream repositories'
vendored projects are excluded. Only original-machine path strings in four
result files were normalized for publication; the
[archive manifest](results/archive_manifest.json) records original and published
hashes. Experimental values and source code were not changed during packaging.
The [source snapshot](snapshots/README.md) preserves the experiment's implementation
independently of future changes to the working code.

See [source attribution and licensing notes](THIRD_PARTY_NOTICES.md). Paper code
was inspected at `0b041ccdd2e5de66b6300cbb14d20f06f226d295`; the reference sandbox
at `af85541efd8fb8e9ca0e60ff1ec8b88e34470b10`.
