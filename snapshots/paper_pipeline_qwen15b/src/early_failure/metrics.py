"""Episode-level early-warning metrics; undefined denominators remain null."""
from __future__ import annotations

from statistics import median


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize(episodes: list[dict]) -> dict:
    success = [e for e in episodes if e["success"]]
    failure = [e for e in episodes if not e["success"]]
    alarms = []
    uncertain = 0
    checks_count = 0
    covered_episodes = 0
    covered_by_class = {"success": 0, "failure": 0}
    for episode in episodes:
        first_alarm = None
        covered = True
        true_class = "success" if episode["success"] else "failure"
        for check in episode["checks"]:
            prediction = check["prediction"]
            checks_count += 1
            uncertain += int(len(prediction["prediction_set"]) != 1)
            covered = covered and true_class in prediction["prediction_set"]
            if first_alarm is None and prediction["alarm"]:
                first_alarm = check["step"]
        covered_episodes += int(covered)
        covered_by_class[true_class] += int(covered)
        if first_alarm is not None:
            alarms.append({"episode_id": episode["episode_id"], "success": episode["success"],
                           "first_alarm_step": first_alarm,
                           "lead_steps": episode["num_steps"] - first_alarm})
    false_alarms = sum(a["success"] for a in alarms)
    detected = [a for a in alarms if not a["success"]]
    leads = sorted(a["lead_steps"] for a in detected)
    checks = [check for episode in episodes for check in episode["checks"]]
    # MC diagnostics are optional; other scorers only need q and elapsed time.
    details = [check.get("details", check) for check in checks]
    return {
        "episodes": len(episodes), "successful_episodes": len(success), "failed_episodes": len(failure),
        "success_rate": _ratio(len(success), len(episodes)),
        "false_alarm_episodes": false_alarms,
        "episode_false_alarm_rate": _ratio(false_alarms, len(success)),
        "failure_recall": _ratio(len(detected), len(failure)),
        "alarm_precision": _ratio(len(detected), len(alarms)),
        "failure_recall_by_minimum_lead_steps": {
            str(lead): _ratio(sum(a["lead_steps"] >= lead for a in detected), len(failure))
            for lead in (1, 5, 10, 25, 50, 100)
        },
        "median_failure_lead_steps": median(leads) if leads else None,
        "simultaneous_episode_coverage": _ratio(covered_episodes, len(episodes)),
        "simultaneous_coverage_by_class": {
            "success": _ratio(covered_by_class["success"], len(success)),
            "failure": _ratio(covered_by_class["failure"], len(failure)),
        },
        "uncertain_or_empty_check_rate": _ratio(uncertain, checks_count),
        "checks": checks_count,
        "scoring_elapsed_seconds": sum(c["elapsed_seconds"] for c in checks),
        "mc_generated_actions": sum(d.get("generated_actions", 0) for d in details),
        "mc_censored_continuations": sum(d.get("mc_censored", 0) for d in details),
        "mc_elapsed_seconds": sum(c["elapsed_seconds"] for c, d in zip(checks, details) if "mc_samples" in d),
        "alarms": alarms,
    }
