#!/usr/bin/env python3
"""Audit finished paper-pipeline artifacts using only the Python standard library.

Usage: python audit_paper_run.py RUN_DIRECTORY [--write]
Without --write, print JSON only. A final summary is required; this never runs
models, browsers, training, or feature extraction.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import sys


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def equivalent(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(equivalent(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(equivalent(a, b) for a, b in zip(left, right))
    if type(left) is float or type(right) is float:
        return (type(left) in (int, float) and type(right) in (int, float)
                and math.isclose(left, right, rel_tol=1e-11, abs_tol=1e-12))
    return left == right


def binary_metrics(pairs):
    counts = Counter(pairs)
    tn, fp, fn, tp = (counts[0, 0], counts[0, 1], counts[1, 0], counts[1, 1])
    n = len(pairs)
    fs = ratio(2 * tp, 2 * tp + fp + fn) or 0.0
    ff = ratio(2 * tn, 2 * tn + fp + fn) or 0.0
    return {"n": n, "accuracy": ratio(tp + tn, n), "f1_success": fs if n else None,
            "f1_failure": ff if n else None, "macro_f1": (fs + ff) / 2 if n else None,
            "balanced_accuracy": ((tp / (tp + fn) + tn / (tn + fp)) / 2
                                  if tp + fn and tn + fp else None),
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}


def summarize_label_predictions(rows):
    labeled = [r for r in rows if r["target"] is not None]
    scored = [r for r in labeled if r["prediction"] is not None]
    baseline = [r for r in labeled if r["majority_prediction"] is not None]
    return {"n_records": len(rows), "n_labeled": len(labeled),
            "n_predicted": sum(r["prediction"] is not None for r in rows),
            "n_evaluated": len(scored), "n_unsupported": sum(r["status"] == "unsupported" for r in rows),
            "label_coverage": ratio(len(labeled), len(rows)),
            "evaluation_coverage": ratio(len(scored), len(labeled)),
            "metrics": binary_metrics([(r["target"], r["prediction"]) for r in scored]),
            "majority_baseline": binary_metrics([(r["target"], r["majority_prediction"]) for r in baseline])}


def wilson(numerator, denominator):
    if not denominator:
        return {"numerator": numerator, "denominator": 0, "lower": None, "upper": None}
    z = 1.959963984540054
    rate, scale = numerator / denominator, 1 + z * z / denominator
    center = (rate + z * z / (2 * denominator)) / scale
    radius = z * math.sqrt(rate * (1 - rate) / denominator + z * z / (4 * denominator**2)) / scale
    return {"numerator": numerator, "denominator": denominator,
            "lower": max(0.0, center - radius), "upper": min(1.0, center + radius)}


def outcome_metrics(episodes, rows):
    lookup = {e["episode_id"]: e for e in episodes}
    successes = sum(e["success"] for e in episodes)
    failures, total = len(episodes) - successes, len(episodes)
    supported = [r for r in rows if r["prediction"] is not None]
    supported_ids = {r["episode_id"] for r in supported}
    first = {}
    for r in supported:
        if r["prediction"] == 0:
            first[r["episode_id"]] = min(first.get(r["episode_id"], r["step"]), r["step"])
    alarms = [{"episode_id": key, "env_id": lookup[key]["env_id"], "success": lookup[key]["success"],
               "first_alarm_step": step, "lead_steps": lookup[key]["num_steps"] - step}
              for key, step in sorted(first.items())]
    detected = [r for r in alarms if not r["success"]]
    false = len(alarms) - len(detected)
    leads = [r["lead_steps"] for r in detected]
    lead_counts = {str(n): sum(x >= n for x in leads) for n in (1, 2, 3)}
    unsupported = sorted(set(lookup) - supported_ids)
    return {"episodes": total, "successful_episodes": successes, "failed_episodes": failures,
            "checks": len(rows), "supported_checks": len(supported),
            "unsupported_checks": len(rows) - len(supported),
            "supported_check_fraction": ratio(len(supported), len(rows)),
            "supported_episodes": len(supported_ids), "unsupported_episodes": len(unsupported),
            "episode_prediction_coverage": ratio(len(supported_ids), total),
            "has_supported_predictions": bool(supported), "unsupported_episode_ids": unsupported,
            "alarm_episodes": len(alarms), "detected_failure_episodes": len(detected),
            "missed_failure_episodes": failures - len(detected), "false_alarm_episodes": false,
            "failure_recall": ratio(len(detected), failures),
            "episode_false_alarm_rate": ratio(false, successes),
            "alarm_precision": ratio(len(detected), len(alarms)),
            "failure_recall_by_minimum_lead_steps": {k: ratio(v, failures) for k, v in lead_counts.items()},
            "detected_failures_by_minimum_lead_steps": lead_counts,
            "median_failure_lead_steps": median(leads) if leads else None,
            "confidence_intervals_95": {
                "failure_recall": wilson(len(detected), failures),
                "episode_false_alarm_rate": wilson(false, successes),
                "alarm_precision": wilson(len(detected), len(alarms)),
                "episode_prediction_coverage": wilson(len(supported_ids), total),
                "failure_recall_by_minimum_lead_steps": {k: wilson(v, failures) for k, v in lead_counts.items()}},
            "alarms": alarms}


def audit(run, workspace):
    errors, checks = [], Counter()

    def check(condition, description):
        checks[description.split(":", 1)[0]] += 1
        if not condition:
            errors.append(description)

    summary = read_json(run / "summary.json")
    config = read_json(run / "config.json")
    settings = read_json(run / "probe_config.json")
    protocol = read_json(run / "protocol.json")
    metadata = read_json(run / "metadata.json")
    snapshot = read_json(run / "source_snapshot.json")
    status = summary["status"]
    check(status in {"complete", "insufficient_calibration", "insufficient_training_labels"}, "final_status")
    check(not (run / "error.json").exists(), "no_error_artifact")
    check(config["max_steps"] == 2 and config["mc_samples"] == 4, "planned_H2_MC4")
    check(config["mc_max_steps"] is None or config["mc_max_steps"] >= config["max_steps"], "full_MC_horizon")
    check(settings["labeling_mode"] == "paper_equations", "paper_equation_mode")
    for field, expected in {
        "model": config["model_id"], "revision": config["model_revision"],
        "task": config["env_ids"][0], "horizon": config["max_steps"],
        "mc_samples": config["mc_samples"], "alpha": config["alpha"],
        "layers": settings["layers"], "primary_layer": settings["primary_layer"],
        "probe_decision_threshold": 0.5,
    }.items():
        check(equivalent(protocol.get(field), expected), f"protocol:{field}")
    check(summary["labeling_mode"] == settings["labeling_mode"] and summary["alpha"] == config["alpha"], "summary_settings")
    normalized = {k: v for k, v in config.items() if k not in {
        "calibration_seeds", "evaluation_seeds", "output_dir", "calibration_path", "alpha"}}
    check(metadata["protocol_fingerprint"] == digest({"version": 2, **normalized}), "protocol_fingerprint")
    check(metadata["probe_config"] == settings, "metadata_probe_config")
    actor = metadata["runtime"]["agent"]
    for field, expected in {"model_id": config["model_id"], "requested_revision": config["model_revision"],
                            "resolved_revision": config["model_revision"], "device": config["device"],
                            "dtype": config["dtype"]}.items():
        check(actor.get(field) == expected, f"runtime_actor:{field}")
    current_files = {p.name for p in (workspace / "src/early_failure").glob("*.py")}
    check(current_files == set(snapshot), "source_file_manifest")
    for name, expected in snapshot.items():
        path = workspace / "src/early_failure" / name
        check(path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"source_hash:{name}")

    manifests = {"calibration": config["calibration_seeds"], "training": settings["training_seeds"],
                 "evaluation": config["evaluation_seeds"]}
    check(len(set().union(*map(set, manifests.values()))) == sum(map(len, manifests.values())), "disjoint_split_seeds")
    for split, seeds in manifests.items():
        expected = [{"env_id": env, "seed": seed} for env in config["env_ids"] for seed in seeds]
        check(metadata["task_manifests"][split] == expected, f"metadata_manifest:{split}")
        count_key = "probe_training_episodes" if split == "training" else f"{split}_episodes"
        check(protocol[count_key] == len(expected), f"protocol_count:{split}")
        check(protocol["split_seeds"][split] == [min(seeds), max(seeds)], f"protocol_seed_bounds:{split}")
        check(seeds == list(range(min(seeds), max(seeds) + 1)), f"contiguous_planned_seeds:{split}")

    episodes, prefixes, outcome_counts, prompt_stats = {}, {}, {}, {}
    for split, seeds in manifests.items():
        path = run / f"{split}_episodes.jsonl"
        if not path.exists():
            check(split not in summary["outcomes"], f"absent_stage:{split}")
            continue
        rows = read_rows(path)
        episodes[split] = rows
        expected = [(env, seed) for env in config["env_ids"] for seed in seeds]
        check([(r["env_id"], r["seed"]) for r in rows] == expected, f"complete_split:{split}")
        check(len({r["episode_id"] for r in rows}) == len(rows), f"unique_episodes:{split}")
        prefix_map = {}
        for row in rows:
            key = row["episode_id"]
            turns, observed = row["turns"], row["initial_observation"]
            check(bool(turns) and len(turns) == row["num_steps"] <= config["max_steps"], f"turn_count:{key}")
            check(not observed["terminated"] and not observed["truncated"], f"initial_nonterminal:{key}")
            check(row["system_prompt"] == config["env_kwargs"]["system_prompt"], f"actual_system_prompt:{key}")
            messages = [{"role": "system", "content": row["system_prompt"]},
                        {"role": "user", "content": observed["text"]}]
            expected_steps = list(range(0, row["num_steps"], config["check_every"]))
            check([c["step"] for c in row["checks"]] == expected_steps, f"preterminal_checks:{key}")
            check([t["step"] for t in turns] == list(range(len(turns))), f"turn_steps:{key}")
            by_step = {c["step"]: c for c in row["checks"]}
            for step, turn in enumerate(turns):
                check(not observed["terminated"] and not observed["truncated"], f"no_actions_after_terminal:{key}:{step}")
                if step in by_step:
                    item, remaining = by_step[step], config["max_steps"] - step
                    d = item["details"]
                    check(d["mc_samples"] == config["mc_samples"] and d["mc_censored"] == 0,
                          f"MC_samples_and_censoring:{key}:{step}")
                    check(type(d["mc_successes"]) is int and 0 <= d["mc_successes"] <= d["mc_samples"],
                          f"MC_success_count:{key}:{step}")
                    check(item["q"] == d["mc_successes"] / d["mc_samples"], f"MC_q:{key}:{step}")
                    check(d["lookahead_steps"] == remaining, f"MC_full_remaining:{key}:{step}")
                    check(d["mc_samples"] <= d["generated_actions"] <= d["mc_samples"] * remaining,
                          f"MC_generated_action_bound:{key}:{step}")
                    prefix_map[key, step] = {"env_id": row["env_id"], "q": item["q"],
                                             "prompt_sha256": digest(messages)}
                observed = turn["observation"]
                messages += [{"role": "assistant", "content": turn["action"]},
                             {"role": "user", "content": observed["text"]}]
            actual = bool(observed["terminated"] and not observed["truncated"] and observed["info"].get("raw_reward") == 1)
            check(row["success"] == observed["success"] == actual, f"real_outcome:{key}")
            reason = ("success" if actual else "environment_timeout" if observed["truncated"] else
                      "environment_failure" if observed["terminated"] else "step_budget_exhausted")
            check(row["outcome_reason"] == reason, f"real_outcome_reason:{key}")
            check(observed["terminated"] or observed["truncated"] or len(turns) == config["max_steps"], f"no_early_stop:{key}")
        prefixes[split] = prefix_map
        successes = sum(row["success"] for row in rows)
        outcome_counts[split] = {"episodes": len(rows), "success": successes, "failure": len(rows) - successes}
        check(summary["outcomes"].get(split) == outcome_counts[split], f"summary_outcomes:{split}")
        prompt_stats[split] = {"checks": len(prefix_map),
                               "unique_prompt_hashes": len({v["prompt_sha256"] for v in prefix_map.values()}),
                               "unique_initial_prompt_hashes": len({v["prompt_sha256"] for (_, step), v in prefix_map.items() if step == 0})}
    check(set(episodes) == set(summary["outcomes"]), "completed_stage_manifest")
    for left, right in (("calibration", "training"), ("calibration", "evaluation"), ("training", "evaluation")):
        check(not ({r["episode_id"] for r in episodes.get(left, [])} & {r["episode_id"] for r in episodes.get(right, [])}),
              f"actual_episode_split_overlap:{left}:{right}")

    calibration = read_json(run / "step_calibration.json")
    rewards = {}
    for row in episodes["calibration"]:
        for item in row["checks"]:
            group = rewards.setdefault(row["env_id"], {}).setdefault(str(item["step"]), {"success": [], "failure": []})
            group["success" if row["success"] else "failure"].append(item["q"])
    for steps in rewards.values():
        for group in steps.values():
            for values in group.values():
                values.sort()
    support = {env: {step: {label: len(qs) for label, qs in group.items()} for step, group in steps.items()}
               for env, steps in rewards.items()}
    check(calibration["calibration_rewards"] == rewards and calibration["support"] == support, "recomputed_step_calibration")
    check(calibration["alpha"] == config["alpha"] and calibration["mode"] == settings["labeling_mode"], "calibration_settings")

    def label(env, step, q):
        group = rewards[env].get(str(step))
        if group is None:
            return {"target": None, "status": "unsupported"}
        ps = (1 + sum(1 - x >= 1 - q for x in group["success"])) / (len(group["success"]) + 1)
        pf = (1 + sum(x >= q for x in group["failure"])) / (len(group["failure"]) + 1)
        labels = [name for name, p in (("success", ps), ("failure", pf)) if p >= config["alpha"]]
        status = ("unsupported" if not all(group.values()) else labels[0] if len(labels) == 1 else
                  "uncertain" if labels else "empty")
        return {"target": 1 if status == "success" else 0 if status == "failure" else None,
                "status": status, "p_success": ps, "p_failure": pf, "prediction_set": labels,
                "support": {name: len(values) for name, values in group.items()}}

    diagnostics = read_json(run / "calibration_diagnostics.json")
    viable = []
    for env in config["env_ids"]:
        for step in range(0, config["max_steps"], config["check_every"]):
            group = rewards[env].get(str(step), {})
            qs = sorted({0.0, 1.0, *(q for values in group.values() for q in values)})
            candidates = qs + [(a + b) / 2 for a, b in zip(qs, qs[1:])]
            if {0, 1} <= {label(env, step, q)["target"] for q in candidates}:
                viable.append({"env_id": env, "step": step})
            if not group:
                continue
            for name, values in group.items():
                scores = [1 - q if name == "success" else q for q in values]
                floor = (1 + sum(q >= 1 for q in scores)) / (len(scores) + 1)
                expected = {"n": len(scores), "supported": bool(scores),
                            "min_rank_p_without_ties": 1 / (len(scores) + 1),
                            "min_achievable_p_at_nonconformity_1": floor,
                            "maximal_nonconformity_ties": sum(q >= 1 for q in scores),
                            "can_reject_at_alpha": bool(scores) and floor < config["alpha"]}
                check(equivalent(diagnostics["environments"][env][str(step)]["classes"][name], expected),
                      f"calibration_diagnostics:{env}:{step}:{name}")
    check(summary["viable_steps"] == viable, "viable_steps")

    features, vectors_by_hash, label_counts = {}, {}, {}
    for split in ("training", "evaluation"):
        if split not in episodes:
            continue
        rows = read_rows(run / f"{split}_features.jsonl")
        features[split] = rows
        identities = [(r["episode_id"], r["step"], r["layer"]) for r in rows]
        expected = {(key, step, layer) for key, step in prefixes[split] for layer in settings["layers"]}
        check(set(identities) == expected and len(identities) == len(expected), f"complete_feature_records:{split}")
        all_counts, per_step = Counter(), defaultdict(Counter)
        for (key, step), prefix in prefixes[split].items():
            calculated = label(prefix["env_id"], step, prefix["q"])
            all_counts[calculated["status"]] += 1
            per_step[f'{prefix["env_id"]}::step={step}'][calculated["status"]] += 1
        label_counts[split] = {"all_states": dict(all_counts),
                               "by_environment_step": {k: dict(v) for k, v in per_step.items()}}
        check(summary[f"{split}_labels"] == label_counts[split], f"summary_state_labels:{split}")
        for row in rows:
            key = f'{row["episode_id"]}:{row["step"]}:{row["layer"]}'
            prefix = prefixes[split][row["episode_id"], row["step"]]
            check(row["env_id"] == prefix["env_id"] and row["q"] == prefix["q"] and
                  row["prompt_sha256"] == prefix["prompt_sha256"], f"feature_prefix_identity:{key}")
            calculated = label(row["env_id"], row["step"], row["q"])
            check(row["target"] == calculated["target"] and row["state_label"]["target"] == calculated["target"] and
                  row["state_label"]["status"] == calculated["status"], f"conformal_target:{key}")
            if "p_success" in calculated:
                for field in ("p_success", "p_failure", "prediction_set", "support"):
                    check(equivalent(row["state_label"]["details"][field], calculated[field]), f"conformal_rank:{key}:{field}")
            vector = row["features"]
            check(bool(vector) and all(type(x) in (int, float) and math.isfinite(x) for x in vector), f"finite_features:{key}")
            cache_key = row["prompt_sha256"], row["layer"]
            if cache_key in vectors_by_hash:
                check(vectors_by_hash[cache_key] == vector, f"same_prompt_same_features:{key}")
            vectors_by_hash[cache_key] = vector

    bank, probe_map, predictions = None, {}, []
    if "training" in features:
        bank = read_json(run / "probes.json")
        check(bank["seed"] == config["seed"] and bank["min_per_class"] == settings["min_per_class"], "probe_settings")
        check(bank["provenance"]["target_source"] == "singleton_conformal_state_label" and
              bank["provenance"]["decision_threshold"] == 0.5, "probe_target_and_threshold")
        check(set(bank["train_episode_ids"]) == {r["episode_id"] for r in features["training"]}, "probe_training_manifest")
        groups = defaultdict(list)
        for row in features["training"]:
            groups[row["env_id"], row["step"], row["layer"]].append(row)
        probe_map = {(p["env_id"], p["step"], p["layer"]): p for p in bank["probes"]}
        check(set(probe_map) == set(groups) and len(probe_map) == len(bank["probes"]), "probe_group_manifest")
        for key, rows in groups.items():
            p = probe_map[key]
            counts = Counter(r["target"] for r in rows if r["target"] is not None)
            trained = min(counts[0], counts[1]) >= settings["min_per_class"]
            expected = {"n_records": len(rows), "n_labeled": counts[0] + counts[1], "n_unlabeled": len(rows) - counts[0] - counts[1],
                        "class_counts": {"failure": counts[0], "success": counts[1]},
                        "train_episode_ids": sorted({r["episode_id"] for r in rows}),
                        "majority_target": int(counts[1] > counts[0]) if counts else None,
                        "status": "trained" if trained else "unsupported"}
            for field, value in expected.items():
                check(p[field] == value, f"probe_training_group:{key}:{field}")
            check(all(len(r["features"]) == p["feature_width"] for r in rows), f"probe_feature_width:{key}")
            if trained:
                check(len(p["coefficients"]) == p["feature_width"] and
                      all(math.isfinite(x) for x in p["coefficients"] + [p["intercept"]]), f"finite_probe_weights:{key}")
        check(summary["trained_probes"] == sum(p["status"] == "trained" for p in probe_map.values()), "trained_probe_count")

    if "evaluation" in features:
        for row in features["evaluation"]:
            key = row["env_id"], row["step"], row["layer"]
            p = probe_map.get(key)
            value = {k: row[k] for k in ("episode_id", "env_id", "step", "layer", "target")}
            value.update(prediction=None, label=None, p_success=None,
                         majority_prediction=p["majority_target"] if p else None,
                         status="unsupported", reason=p["reason"] if p else "unknown_probe")
            if p and p["status"] == "trained":
                check(len(row["features"]) == len(p["coefficients"]), f"evaluation_feature_width:{key}")
                z = math.fsum(w * x for w, x in zip(p["coefficients"], row["features"])) + p["intercept"]
                probability = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
                prediction = int(probability >= 0.5)
                value.update(prediction=prediction, label="success" if prediction else "failure",
                             p_success=probability, status="trained", reason=None)
            predictions.append(value)
        check(equivalent(read_rows(run / "probe_predictions.jsonl"), predictions), "recomputed_numeric_weight_predictions")
        expected = summarize_label_predictions(predictions)
        expected["target_source"] = "singleton_conformal_state_label"
        groups = defaultdict(list)
        for row in predictions:
            groups[row["env_id"], row["step"], row["layer"]].append(row)
        expected["per_probe"] = [{"env_id": key[0], "step": key[1], "layer": key[2], **summarize_label_predictions(rows)}
                                 for key, rows in sorted(groups.items())]
        check(equivalent(summary["probe_evaluation"], expected), "recomputed_probe_label_summary")
        saved_outcomes = summary["outcome_evaluation"]
        check(saved_outcomes["primary_layer"] == settings["primary_layer"], "fixed_primary_layer")
        for layer in settings["layers"]:
            actual = outcome_metrics(episodes["evaluation"], [r for r in predictions if r["layer"] == layer])
            check(equivalent(saved_outcomes["by_layer"][str(layer)], actual), f"recomputed_outcomes_layer:{layer}")
            if layer == settings["primary_layer"]:
                check(equivalent(saved_outcomes["primary"], actual), "recomputed_primary_outcome_summary")
        for name, prediction in (("always_warn", 0), ("never_warn", 1)):
            baseline = [{"episode_id": e["episode_id"], "step": 0, "prediction": prediction} for e in episodes["evaluation"]]
            check(equivalent(saved_outcomes["baselines"][name], outcome_metrics(episodes["evaluation"], baseline)), f"outcome_baseline:{name}")

    if status == "insufficient_calibration":
        check(not viable and set(episodes) == {"calibration"} and not features and summary["trained_probes"] == 0,
              "honest_calibration_gate_stop")
    elif status == "insufficient_training_labels" and settings["stop_if_untrainable"]:
        check(set(episodes) == {"calibration", "training"} and summary["trained_probes"] == 0,
              "honest_training_gate_stop")
    elif status == "complete":
        check(set(episodes) == set(manifests) and summary["trained_probes"] > 0, "completed_all_stages")

    overlap = {}
    for left, right in (("calibration", "training"), ("calibration", "evaluation"), ("training", "evaluation")):
        if left in prefixes and right in prefixes:
            lhs = {v["prompt_sha256"] for v in prefixes[left].values()}
            rhs = {v["prompt_sha256"] for v in prefixes[right].values()}
            overlap[f"{left}_vs_{right}"] = {"shared_unique_prefix_hashes": len(lhs & rhs),
                                             "right_checks_sharing_left_prompt": sum(v["prompt_sha256"] in lhs for v in prefixes[right].values())}
    return {"passed": not errors, "run_status": status, "completed_stages": list(episodes),
            "outcomes": outcome_counts, "viable_steps": viable, "trained_probes": summary["trained_probes"],
            "checked_source_files": len(snapshot), "check_counts": dict(checks),
            "feature_records": {k: len(v) for k, v in features.items()}, "label_counts": label_counts,
            "prompt_diversity": prompt_stats, "cross_split_prompt_overlap": overlap,
            "primary_outcome_metrics": summary.get("outcome_evaluation", {}).get("primary"),
            "issues": errors,
            "limitations": ["Activations and optimizer fitting are not rerun; saved prefixes, numeric weights, and derived metrics are checked.",
                            "Distinct task seeds do not guarantee distinct prompt text; overlap is quantified separately.",
                            "Passing an artifact audit establishes internal consistency, not predictive quality or long-horizon validity."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--workspace", type=Path, help="Sandbox root; defaults to the run directory's grandparent")
    parser.add_argument("--write", action="store_true", help="Write artifact_audit.json after final artifacts are available")
    args = parser.parse_args()
    run = args.run_directory.resolve()
    if not (run / "summary.json").exists():
        parser.error("No final summary.json: refusing to audit or write into an active/incomplete run")
    try:
        result = audit(run, args.workspace.resolve() if args.workspace else run.parent.parent)
    except Exception as error:
        result = {"passed": False, "issues": [f"Audit could not complete: {type(error).__name__}: {error}"],
                  "run_directory": str(run)}
    result["run_directory"] = str(run)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.write:
        (run / "artifact_audit.json").write_text(text)
    print(text, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
