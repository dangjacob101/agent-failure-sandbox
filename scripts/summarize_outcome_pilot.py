#!/usr/bin/env python3
"""Check a completed run and compare early warnings with final outcomes."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist


LABELS = ("success", "failure")
STATUSES = ("success", "failure", "uncertain", "out_of_distribution")
BASELINE_NAMES = (
    "always predict failure before the first action",
    "alarm at first checkpoint with q < 0.5",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"{path.name} must contain an object")
    return value


def number(value, name, *, maximum=None):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            f"{name} must be a finite nonnegative number")
    require(maximum is None or value <= maximum, f"{name} exceeds {maximum}")
    return value


def count(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"{name} must be an integer >= {minimum}")
    return value


def proportion(numerator, denominator, *, interval=True):
    result = {"numerator": numerator, "denominator": denominator,
              "value": numerator / denominator if denominator else None}
    if interval:
        bounds = None
        if denominator:
            z = NormalDist().inv_cdf(0.975)
            p = numerator / denominator
            divisor = 1 + z * z / denominator
            center = (p + z * z / (2 * denominator)) / divisor
            radius = z * math.sqrt(p * (1 - p) / denominator + z * z / (4 * denominator**2)) / divisor
            bounds = [max(0.0, center - radius), min(1.0, center + radius)]
        result["wilson_95_percent_interval"] = bounds
    return result


def histogram(values):
    return {format(key, ".12g"): value for key, value in sorted(Counter(values).items())}


def expected_tasks(config, split):
    seeds = config[f"{split}_seeds"]
    require(isinstance(seeds, list) and seeds, f"{split} seeds must be a nonempty list")
    for seed in seeds:
        count(seed, f"{split} seed")
    require(len(seeds) == len(set(seeds)), f"duplicate {split} seeds")
    return {(env, seed) for env in config["env_ids"] for seed in seeds}


def read_episodes(path, expected, config, *, predictions):
    episodes, found = [], set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        require(bool(line.strip()), f"blank episode at {path.name}:{line_number}")
        episode = json.loads(line)
        require(isinstance(episode, dict), "each episode must be an object")
        env, seed = episode.get("env_id"), episode.get("seed")
        require(isinstance(env, str), "episode env_id must be a string")
        count(seed, "episode seed")
        key = (env, seed)
        require(key in expected, f"unexpected {path.name} task: {key}")
        require(key not in found, f"duplicate {path.name} task: {key}")
        found.add(key)
        require(episode.get("episode_id") == f"{env}::seed={seed}", "episode_id does not match task")
        require(type(episode.get("success")) is bool, "episode success must be a boolean")
        steps = count(episode.get("num_steps"), "num_steps", 1)
        require(steps <= config["max_steps"], "episode exceeds action budget")
        number(episode.get("elapsed_seconds"), "episode elapsed_seconds")
        turns = episode.get("turns")
        require(isinstance(turns, list) and len(turns) == steps, "turn count differs from num_steps")
        for step, turn in enumerate(turns):
            require(isinstance(turn, dict) and type(turn.get("step")) is int and turn["step"] == step,
                    "turn steps must be consecutive")
            require(isinstance(turn.get("action"), str), "action must be a string")
        checks = episode.get("checks")
        require(isinstance(checks, list) and checks, "episode must have checks")
        check_steps = []
        for check in checks:
            require(isinstance(check, dict), "check must be an object")
            step = count(check.get("step"), "check step")
            require(step < steps, "prediction must precede the terminal action")
            check_steps.append(step)
            number(check.get("q"), "q", maximum=1)
            number(check.get("elapsed_seconds"), "check elapsed_seconds")
            if predictions:
                prediction = check.get("prediction")
                require(isinstance(prediction, dict), "evaluation check is missing prediction")
                require(prediction.get("status") in STATUSES, "unknown prediction status")
                require(type(prediction.get("alarm")) is bool, "alarm must be a boolean")
                for label in LABELS:
                    number(prediction.get(f"p_{label}"), f"p_{label}", maximum=1)
                labels = prediction.get("prediction_set")
                require(isinstance(labels, list) and all(x in LABELS for x in labels),
                        "invalid prediction set")
                require(len(labels) == len(set(labels)), "duplicate prediction labels")
        require(check_steps == list(range(0, steps, config["check_every"])),
                "check schedule differs from config")
        episodes.append(episode)
    require(found == expected, f"incomplete {path.name}: expected {len(expected)}, found {len(found)} episodes")
    return episodes


def outcome_counts(episodes):
    successes = sum(episode["success"] for episode in episodes)
    return {"episodes": len(episodes), "success": successes, "failure": len(episodes) - successes}


def calibration_scores(episodes, env_ids):
    scores = {env: {label: [] for label in LABELS} for env in env_ids}
    for episode in episodes:
        label = "success" if episode["success"] else "failure"
        scores[episode["env_id"]][label].append(max(
            1 - check["q"] if episode["success"] else check["q"] for check in episode["checks"]))
    return {env: {label: sorted(values) for label, values in groups.items()} for env, groups in scores.items()}


def validate_predictions(episodes, scores, alpha):
    for episode in episodes:
        qs = []
        for check in episode["checks"]:
            qs.append(check["q"])
            prediction = check["prediction"]
            p_values = {}
            for label in LABELS:
                values = scores[episode["env_id"]][label]
                nonconformity = max(1 - q for q in qs) if label == "success" else max(qs)
                p_values[label] = (1 + sum(value >= nonconformity for value in values)) / (len(values) + 1)
                require(math.isclose(prediction[f"p_{label}"], p_values[label], abs_tol=1e-12),
                        "saved p-value differs from calibration")
            labels = [label for label in LABELS if p_values[label] > alpha]
            status = labels[0] if len(labels) == 1 else ("uncertain" if labels else "out_of_distribution")
            require(prediction["prediction_set"] == labels and prediction["status"] == status
                    and prediction["alarm"] == (status == "failure"),
                    "saved prediction differs from calibration")


def alarm_metrics(episodes, first_steps):
    outcomes = outcome_counts(episodes)
    warnings = [(episode, step) for episode, step in zip(episodes, first_steps) if step is not None]
    detected = [(episode, step) for episode, step in warnings if not episode["success"]]
    false_alarms = sum(episode["success"] for episode, _ in warnings)
    return {
        "failure_recall": proportion(len(detected), outcomes["failure"]),
        "false_alarm_rate": proportion(false_alarms, outcomes["success"]),
        "alarm_precision": proportion(len(detected), len(warnings)),
        "failure_recall_by_minimum_lead_actions": {
            str(lead): proportion(sum(episode["num_steps"] - step >= lead for episode, step in detected),
                                 outcomes["failure"]) for lead in (1, 2, 3)
        },
        "alarm_episodes": len(warnings),
        "missed_failure_episodes": outcomes["failure"] - len(detected),
    }


def first_alarm(episode, rule):
    return next((check["step"] for check in episode["checks"] if rule(check)), None)


def evaluate(episodes):
    checks = [check for episode in episodes for check in episode["checks"]]
    counts = Counter(check["prediction"]["status"] for check in checks)
    singleton = counts["success"] + counts["failure"]
    correct = sum(check["prediction"]["status"] == ("success" if episode["success"] else "failure")
                  for episode in episodes for check in episode["checks"])
    conformal_steps = [first_alarm(episode, lambda check: check["prediction"]["alarm"]) for episode in episodes]
    threshold_steps = [first_alarm(episode, lambda check: check["q"] < 0.5) for episode in episodes]
    return {
        "outcomes": outcome_counts(episodes),
        "check_counts": {"success_only": counts["success"], "failure_only": counts["failure"],
                         "uncertain": counts["uncertain"], "empty": counts["out_of_distribution"],
                         "total": len(checks)},
        "singleton_accuracy": proportion(correct, singleton, interval=False),
        "singleton_decision_fraction": proportion(singleton, len(checks), interval=False),
        "uncertain_fraction": proportion(counts["uncertain"], len(checks), interval=False),
        "empty_fraction": proportion(counts["out_of_distribution"], len(checks), interval=False),
        "conformal": alarm_metrics(episodes, conformal_steps),
        "baselines": {
            "always_failure_before_first_action": alarm_metrics(episodes, [0] * len(episodes)),
            "first_q_below_0_5": alarm_metrics(episodes, threshold_steps),
        },
    }


def score_diagnostics(calibration, evaluation, scores, alpha):
    result = {}
    for env, groups in scores.items():
        successful = groups["success"]
        minimum_rank = 1 / (len(successful) + 1)
        minimum_p = (1 + sum(value >= 1 for value in successful)) / (len(successful) + 1)
        reasons = []
        if not successful:
            reasons.append("No successful calibration episodes; success cannot be rejected.")
        elif minimum_rank > alpha:
            reasons.append("Too few successful calibration episodes to reject success at this alpha.")
        if minimum_p > alpha and minimum_p > minimum_rank:
            reasons.append("Ties at maximum success nonconformity prevent rejecting success at any score.")
        result[env] = {
            "nonconformity_histograms": {label: histogram(values) for label, values in groups.items()},
            "successful_calibration_episodes": len(successful),
            "minimum_p_success_without_ties": minimum_rank,
            "minimum_achievable_p_success_at_nonconformity_1": minimum_p,
            "success_can_be_rejected": minimum_p <= alpha,
            "failure_alarms_impossible_reasons": reasons,
            "q_histograms_by_split_and_outcome": {
                split: {
                    label: histogram(check["q"] for episode in episodes
                                     if episode["env_id"] == env and episode["success"] == (label == "success")
                                     for check in episode["checks"]) for label in LABELS
                } for split, episodes in (("calibration", calibration), ("evaluation", evaluation))
            },
        }
    return result


def episode_rows(episodes):
    rows = []
    for episode in episodes:
        checks = episode["checks"]
        first = first_alarm(episode, lambda check: check["prediction"]["alarm"])
        raw = first_alarm(episode, lambda check: check["q"] < 0.5)
        rows.append({
            "episode_id": episode["episode_id"], "env_id": episode["env_id"], "seed": episode["seed"],
            "actual_outcome": "success" if episode["success"] else "failure", "num_actions": episode["num_steps"],
            "outcome_reason": episode.get("outcome_reason", ""),
            "first_failure_alarm_before_action": first,
            "alarm_correct": not episode["success"] if first is not None else None,
            "alarm_lead_actions": episode["num_steps"] - first if first is not None else None,
            "threshold_baseline_first_alarm_before_action": raw,
            "check_steps": json.dumps([check["step"] for check in checks]),
            "q_scores": json.dumps([check["q"] for check in checks]),
            "prediction_statuses": json.dumps([check["prediction"]["status"] for check in checks]),
            "p_success": json.dumps([check["prediction"]["p_success"] for check in checks]),
            "p_failure": json.dumps([check["prediction"]["p_failure"] for check in checks]),
        })
    return rows


def analyze_run(run_dir):
    run_dir = Path(run_dir)
    require(not (run_dir / "error.json").exists(), "error.json exists; run is incomplete")
    artifacts = {name: read_json(run_dir / f"{name}.json")
                 for name in ("config", "metadata", "calibration", "summary")}
    config, metadata, saved, summary = (artifacts[name] for name in ("config", "metadata", "calibration", "summary"))
    env_ids = config.get("env_ids")
    require(isinstance(env_ids, list) and env_ids and all(isinstance(x, str) and x for x in env_ids),
            "env_ids must be nonempty strings")
    require(len(set(env_ids)) == len(env_ids), "duplicate env_ids")
    count(config.get("max_steps"), "max_steps", 1)
    count(config.get("check_every"), "check_every", 1)
    alpha = number(config.get("alpha"), "alpha", maximum=1)
    require(0 < alpha < 1, "alpha must be strictly between 0 and 1")
    expected_cal, expected_eval = (expected_tasks(config, split) for split in ("calibration", "evaluation"))
    require(not expected_cal & expected_eval, "calibration/evaluation task overlap")
    calibration = read_episodes(run_dir / "calibration_episodes.jsonl", expected_cal, config, predictions=False)
    evaluation = read_episodes(run_dir / "evaluation_episodes.jsonl", expected_eval, config, predictions=True)
    protocol = {key: value for key, value in config.items()
                if key not in ("calibration_seeds", "evaluation_seeds", "output_dir", "calibration_path", "alpha")}
    protocol = {"version": 2, **protocol}
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    require(type(saved.get("schema_version")) is int and saved["schema_version"] == 1
            and saved.get("protocol") == protocol, "saved protocol differs from config")
    require(saved.get("protocol_fingerprint") == metadata.get("protocol_fingerprint") == fingerprint,
            "protocol fingerprint mismatch")
    require(isinstance(metadata.get("runtime"), dict) and saved.get("runtime") == metadata["runtime"],
            "runtime provenance mismatch")
    require(type(metadata.get("synthetic")) is bool and type(summary.get("synthetic")) is bool
            and summary["synthetic"] == metadata["synthetic"],
            "synthetic flag mismatch")
    manifest = saved.get("calibration_tasks")
    require(isinstance(manifest, list) and all(isinstance(task, dict) for task in manifest),
            "invalid calibration task manifest")
    for task in manifest:
        require(isinstance(task.get("env_id"), str), "manifest env_id must be a string")
        count(task.get("seed"), "manifest seed")
    manifest_keys = [(task.get("env_id"), task.get("seed")) for task in manifest]
    require(len(manifest_keys) == len(expected_cal) and set(manifest_keys) == expected_cal,
            "calibration manifest mismatch")
    scores = calibration_scores(calibration, env_ids)
    monitor = saved.get("monitor")
    require(isinstance(monitor, dict) and type(monitor.get("schema_version")) is int
            and monitor["schema_version"] == 1
            and monitor.get("method") == "trajectory_max_class_conditional"
            and monitor.get("alpha") == alpha, "invalid saved calibrator")
    require(monitor.get("calibration_scores") == scores, "saved scores differ from calibration episodes")
    validate_predictions(evaluation, scores, alpha)
    results = evaluate(evaluation)
    metrics = summary.get("metrics")
    require(isinstance(metrics, dict), "summary is missing metrics")
    for key, value in (("episodes", len(evaluation)), ("successful_episodes", results["outcomes"]["success"]),
                       ("failed_episodes", results["outcomes"]["failure"]), ("checks", results["check_counts"]["total"]),
                       ("false_alarm_episodes", results["conformal"]["false_alarm_rate"]["numerator"])):
        require(type(metrics.get(key)) is int and metrics[key] == value, f"summary {key} disagrees with episodes")
    for source, target in (("failure_recall", "failure_recall"), ("episode_false_alarm_rate", "false_alarm_rate"),
                           ("alarm_precision", "alarm_precision")):
        require(metrics.get(source) == results["conformal"][target]["value"], f"summary {source} disagrees with episodes")
    declaration_path = run_dir / "protocol.json"
    if not declaration_path.exists():
        declaration_path = Path(__file__).resolve().parents[1] / "configs/miniwob_outcome_pilot_protocol.json"
    declaration_path = declaration_path.resolve()
    declaration = read_json(declaration_path)
    require(declaration.get("fixed_descriptive_baselines") == list(BASELINE_NAMES), "baseline declarations changed")
    report = {
        "run_dir": str(run_dir.resolve()), "synthetic": metadata["synthetic"],
        "protocol_fingerprint": fingerprint, "runtime": metadata["runtime"],
        "settings": {key: config[key] for key in ("model_id", "env_ids", "max_steps", "mc_samples", "alpha")},
        "comparison_rules_declared_in": str(declaration_path),
        "calibration_outcomes": outcome_counts(calibration), "evaluation": results,
        "by_environment": {env: {
            "calibration_outcomes": outcome_counts([episode for episode in calibration if episode["env_id"] == env]),
            "evaluation": evaluate([episode for episode in evaluation if episode["env_id"] == env]),
        } for env in env_ids},
        "score_diagnostics": score_diagnostics(calibration, evaluation, scores, alpha),
        "elapsed_seconds": {
            "calibration_episodes": sum(episode["elapsed_seconds"] for episode in calibration),
            "evaluation_episodes": sum(episode["elapsed_seconds"] for episode in evaluation),
            "all_episodes": sum(episode["elapsed_seconds"] for episode in calibration + evaluation),
            "all_scoring": sum(check["elapsed_seconds"] for episode in calibration + evaluation for check in episode["checks"]),
        },
        "interpretation_notes": [
            "Failure means no success within the fixed action budget.",
            "Uncertain and empty predictions are not counted as correct decisions.",
            "Check-level statistics are descriptive: repeated checks within an episode are correlated; no check-level confidence intervals are reported.",
            "Wilson intervals describe episode proportions; they do not establish conformal guarantees or long-horizon performance.",
            "Lead-action recall includes missed failures in its denominator. Step 0 is before the first action.",
            "Elapsed time sums recorded episode costs; it excludes startup and report generation.",
        ],
    }
    return report, episode_rows(evaluation)


def display_fraction(metric):
    value = metric["value"]
    text = f"{metric['numerator']}/{metric['denominator']}"
    return text + (f" ({value:.1%})" if value is not None else " (undefined)")


def report_text(report):
    cal, evaluation = report["calibration_outcomes"], report["evaluation"]
    outcomes, checks = evaluation["outcomes"], evaluation["check_counts"]
    lines = [
        "Early failure prediction pilot",
        f"Calibration: {cal['success']} successes, {cal['failure']} failures ({cal['episodes']} episodes).",
        f"Evaluation: {outcomes['success']} successes, {outcomes['failure']} failures ({outcomes['episodes']} episodes).",
        f"Conformal checks: {checks['success_only']} success-only, {checks['failure_only']} failure-only, {checks['uncertain']} uncertain, {checks['empty']} empty.",
        f"Accuracy when decisive: {display_fraction(evaluation['singleton_accuracy'])}; decisive checks: {display_fraction(evaluation['singleton_decision_fraction'])}.",
        "",
    ]
    for name, metrics in [("Conformal", evaluation["conformal"]),
                          ("Always warn before first action", evaluation["baselines"]["always_failure_before_first_action"]),
                          ("First score below 0.5", evaluation["baselines"]["first_q_below_0_5"])]:
        lines.append(f"{name}: caught failures {display_fraction(metrics['failure_recall'])}; false warnings {display_fraction(metrics['false_alarm_rate'])}; correct warnings {display_fraction(metrics['alarm_precision'])}.")
        leads = metrics["failure_recall_by_minimum_lead_actions"]
        lines.append("  Failures caught at least " + "; ".join(f"{lead} action(s) early: {display_fraction(leads[str(lead)])}" for lead in (1, 2, 3)) + ".")
    lines.append("")
    for env, diagnostic in report["score_diagnostics"].items():
        lines.append(f"{env}: smallest possible p_success = {diagnostic['minimum_achievable_p_success_at_nonconformity_1']:.4g} (alpha {report['settings']['alpha']}).")
        lines.extend(diagnostic["failure_alarms_impossible_reasons"])
    lines.extend([f"Recorded episode time: {report['elapsed_seconds']['all_episodes'] / 60:.1f} minutes.",
                  "Uncertainty is not a correct prediction. This is a short-task test; it does not validate long-horizon warnings.",
                  "Episode-level Wilson 95% intervals and full counts are in report.json. Check-level counts have no confidence intervals."])
    return "\n".join(lines) + "\n"


def write_report(run_dir, output_dir=None):
    report, rows = analyze_run(run_dir)
    output = Path(output_dir) if output_dir else Path(run_dir) / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (output / "episode_predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = report_text(report)
    (output / "report.txt").write_text(text)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        print(write_report(args.run_dir, args.output_dir), end="")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Cannot report this run: {error}\n")


if __name__ == "__main__":
    main()
