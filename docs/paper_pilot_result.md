# Corrected paper-pipeline pilot

The full pipeline ran locally: Monte Carlo continuation rewards → the paper's
written conformal state labels → hidden activations → linear probes → held-out
evaluation against both state labels and actual final outcomes.

The primary probe warned about **17 of 25 failures (68%)**, with **0 false
warnings among 15 successful runs**. Every warning arrived after the first
action, one action before the run ended. Eight failures were missed, including
two that ended before a supported probe could be used.

This is a successful execution of the corrected pipeline on a small task. It
does not reproduce the paper's original model training or establish long-horizon
performance. Inspection found that all 17 correct warnings followed typing into
the form container (`ref=4`) instead of its input (`ref=5`). The input remained
empty. A simple action/DOM rule could detect this same mistake; the result does
not establish an advantage over that rule. This is a post-hoc observation, not
a separately validated baseline.

## Fixed setup

- Model: Qwen2.5-Coder-1.5B-Instruct, revision
  `2e1fd397ee46e1388853d2af2c993145b0f1098a`, float16 on Apple MPS.
- Environment: MiniWoB++ `enter-text-v1`, two-action budget, timer disabled.
- Four complete continuations per state; written rank rule at `alpha=0.1`.
- Separate calibration/training/test episodes: 80 / 40 / 40.
- Layers 7, 14, 21, 28; **layer 14 chosen before testing**.
- Four balanced logistic probes trained on 17 labeled states: 13 failure,
  4 success. Another 62 training states remained uncertain.
- No initial-state probe could be trained. Both labels were available only
  after the first action. No labels were replaced with final task outcomes.
- Main run duration: 78.8 minutes. No paid API or Colab was used.

The larger model and clearer prompt were selected on separate development
seeds, with the user's permission. This is a fresh pilot, not a controlled
comparison with the earlier 0.5B episode-maximum experiment.

## Results

| Probe layer | Failures warned about | False warnings on successful runs |
| --- | ---: | ---: |
| 7 | 17/25 | 2/15 |
| **14 (primary)** | **17/25** | **0/15** |
| 21 | 18/25 | 0/15 |
| 28 | 18/25 | 0/15 |

Layer 14 matched **19/20 conformal-labeled test states (95%)**. Predicting the
training-majority label matched 18/20 (90%). Only 20 of 78 test states received
a binary conformal label; probe predictions were still evaluated against final
outcomes on every test episode, including unsupported episodes. The primary
probe produced a prediction on 38/40 episodes.

For the eight missed failures, two clicked Submit immediately, five typed into
the correct input but later typed again instead of submitting, and one clicked
the input before typing, exhausting the budget. The latter was the primary
probe's single error on a labeled test state. No actions were stopped or steered;
probe evaluation replayed the saved pre-action prefixes without future text.

An always-warn baseline catches 25/25 failures but falsely warns on all 15
successes. A never-warn baseline catches none. The primary failure-recall
Wilson 95% interval is approximately 48–83%; the false-warning interval is
0–20%. Zero observed false warnings is not a zero-risk guarantee.

The benchmark repeats goal text: 43/78 test prefixes exactly match a training
prefix, despite disjoint episode seeds. These results therefore do not
establish generalization to unseen text or tasks. Seventeen labeled training
states and forty test episodes are too few for a strong scientific conclusion.
The two-action budget also makes some mistakes unrecoverable that a longer
run might repair. Probe errors do not inherit the labeler's conformal guarantee.

## Verification and artifacts

The independent artifact audit passed: split separation, all 314 MC checks,
conformal ranks and targets, pre-action prefix identities, numeric predictions
from saved weights, all layer metrics, and 18 source hashes. Probe weights had
the same SHA-256 before and after testing. The code suite passed 126 tests;
three optional browser tests were skipped, while the actual experiment used
160 real browser episodes. Real-model activation checks also passed.

Artifacts are in [the run directory](../results/paper_pipeline_qwen15b/):
[readable report](../results/paper_pipeline_qwen15b/report.txt),
[full results](../results/paper_pipeline_qwen15b/summary.json),
[probe weights](../results/paper_pipeline_qwen15b/probes.json), and
[audit](../results/paper_pipeline_qwen15b/artifact_audit.json).

To recheck saved artifacts without model inference, run from the repository root:

```bash
.venv/bin/python results/paper_pipeline_qwen15b/audit_paper_run.py \
  results/paper_pipeline_qwen15b --workspace snapshots/paper_pipeline_qwen15b
```

This audit checks internal consistency; it does not regenerate activations or
prove predictive quality. The model, environment, scorer, feature extractor,
labeler, probe bank, and outcome metrics remain separate components behind
`run_paper_experiment`.
