# Methodology audit

This is a code and protocol review, written while the run is active. It contains no performance findings and does not confirm that calibration or probe training will succeed. The saved `config.json`, `probe_config.json`, and `metadata.json` define the reviewed run.

The sequence in [paper sections 4.1–4.3](https://arxiv.org/html/2604.19775v1#S4) is restored: estimate future reward with Monte Carlo continuations, assign conformal labels to intermediate states, extract representations, and train separate linear probes by task, timestep, and layer. The probes learn the conformal state labels; final task outcomes remain separate evaluation targets. There is no activation steering.

This run uses four continuations per pre-action state, each allowed the full remaining action budget. Calibration groups scores by real final outcome, environment, and timestep. Features come from the last nonpadding generation-prompt token at layers 7, 14, 21, and 28. Logistic regression uses raw features, balanced class weights, and a fixed 0.5 decision threshold. Layer 14 is the prespecified primary warning layer.

The chosen `paper_equations` rule implements the written plus-one ranks with nonconformity scores `1 - q` for success and `q` for failure. It retains a class when `p >= 0.1`; a binary label requires retaining exactly one class. Ties count conservatively. Ambiguous, empty-set, and unsupported states do not receive fallback training labels. The released scripts instead use interpolated reward percentiles and success-first threshold assignment. Those rules are not numerically equivalent; this run intentionally follows the written equations.

The adaptation uses Qwen2.5-Coder-1.5B-Instruct at pinned revision `2e1fd397ee46e1388853d2af2c993145b0f1098a`, MiniWoB++ `enter-text-v1`, a task-oriented action prompt, and a two-action horizon. It does not reproduce the paper's fine-tuned Llama agent or ScienceWorld/ALFWorld benchmarks. The HF feature convention includes the generation prompt; Qwen's final selected hidden state includes its final normalization. Only pre-action, nonterminal states are evaluated.

Leakage controls checked in the implementation:

- Calibration uses seeds 50000–50079; training uses 60000–60039; evaluation uses 70000–70039. Overlapping split seeds are rejected, and the probe evaluator independently rejects training-episode overlap.
- Reconstructed prompts use each episode's recorded system prompt and observed history. They exclude terminal observations, future actions, final-outcome labels, and hypothetical continuation text.
- Main-action and continuation random seeds are separate. Feature extraction preserves RNG state, and deterministic feature caching contains no labels.
- The probe bank is saved before evaluation trajectories are collected. Primary-layer selection and decision thresholds do not use evaluation outcomes.

If all stages complete, held-out probe accuracy measures agreement with conformal labels on the labeled subset. Separate final-outcome metrics measure episode failure recall, false warnings among successful episodes, precision, and first-warning lead. Unsupported episodes remain in outcome denominators. Always-warn and never-warn baselines retain the observed class imbalance. Wilson intervals use episode counts; pooled layer/check accuracy is descriptive because those observations are correlated.

Material limits remain: obtaining both binary targets requires at least ten calibration examples per outcome at a timestep, and four-sample MC ties can still prevent binary labels. The run stops when no timestep permits both targets or no probe has sufficient training examples. A minimum of two examples per class permits fitting but does not establish a stable probe. Forty evaluation episodes give imprecise performance estimates. These results cannot establish 100-turn performance, causal failure mechanisms, or a conformal bound on probe warnings across an episode. A labeler's fixed-timestep rank guarantee does not transfer automatically to a learned probe.
