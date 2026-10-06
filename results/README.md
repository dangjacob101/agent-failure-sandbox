# Saved experiments

Start with the **completed, corrected paper pipeline** in [paper_pipeline_qwen15b](paper_pipeline_qwen15b/). It uses per-timestep conformal labels to train linear probes, then compares warnings with actual final outcomes.

| Directory | Role | Contents |
| --- | --- | --- |
| [paper_pipeline_qwen15b](paper_pipeline_qwen15b/) | Main experiment, complete | Qwen 1.5B, MiniWoB enter-text, two-action budget, four continuations per checkpoint; **80 calibration + 40 training + 40 evaluation episodes**. |
| [outcome_pilot_150](outcome_pilot_150/) | Earlier outcome-monitor experiment | Qwen 0.5B, click-test-2, three-action budget; 100 calibration + 50 evaluation episodes. Uses episode-level calibration without activation probes. |
| [paper_actor_development](paper_actor_development/) | Development | Qwen 0.5B actor check: 20 enter-text and 20 click-button episodes, without continuation scoring. |
| [paper_actor_development_15b](paper_actor_development_15b/) | Development | Matching Qwen 1.5B actor check, plus separate continuation-cost and score checks. |

The main and earlier experiments differ in model, prompt, task, horizon, and method. Their results are **not a controlled comparison**. Development episodes are excluded from the main calibration, training, and evaluation splits.

## Main result and evidence

The fixed primary probe, layer 14, warned on **17/25 failed evaluation episodes**, one action before the end, with **0/15 false warnings on successful episodes**. Eight failures were missed. These small counts do not establish a zero false-alarm rate or performance on long tasks.

- [Report](paper_pipeline_qwen15b/report.txt), [full metrics](paper_pipeline_qwen15b/summary.json), and [frozen protocol](paper_pipeline_qwen15b/protocol.json).
- [Calibration trajectories](paper_pipeline_qwen15b/calibration_episodes.jsonl), [training trajectories](paper_pipeline_qwen15b/training_episodes.jsonl), and [evaluation trajectories](paper_pipeline_qwen15b/evaluation_episodes.jsonl).
- All saved activation vectors for layers 7, 14, 21, and 28: [training features](paper_pipeline_qwen15b/training_features.jsonl) and [evaluation features](paper_pipeline_qwen15b/evaluation_features.jsonl). These are the recorded prefix vectors, not every token's hidden states.
- [Fitted probes](paper_pipeline_qwen15b/probes.json), [probe predictions](paper_pipeline_qwen15b/probe_predictions.jsonl), [artifact audit](paper_pipeline_qwen15b/artifact_audit.json), and [methodology limitations](paper_pipeline_qwen15b/methodology_audit.md).

The main experiment completed all **160 episodes**. In the separate 1.5B development directory, the four-action/eight-continuation trial was intentionally stopped for cost after one complete episode and three checkpoints of the next. Its `development_mc8_error.json` records that interruption; it is **not a main-experiment error**. The [budget-stop record](paper_actor_development_15b/development_mc8_budget_stop.json) explains it. A subsequent [four-episode, two-action/four-continuation check](paper_actor_development_15b/development_mc4_h2_diagnostics.json) completed.

## Archive provenance

[archive_manifest.json](archive_manifest.json) compares every copied result file with its original run artifact using SHA-256 and byte counts. All 53 original files are included: 49 are byte-identical; four have only local-path substitutions. Three JSON files replace the original absolute run-directory prefix with `results`; the historical test log replaces its absolute workspace prefix with `<original-workspace>`. No scores, outcomes, vectors, or probe weights were changed. This index and the manifest were created for the public archive.
