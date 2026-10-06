from __future__ import annotations

import argparse
import json

from .config import ExperimentConfig
from .experiment import run_experiment
from .paper_pipeline import ProbeConfig, run_paper_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo + conformal early failure sandbox")
    parser.add_argument("--config", help="Experiment JSON settings")
    parser.add_argument("--pipeline", choices=("paper", "episode"), default="episode")
    parser.add_argument("--probe-config", help="Probe settings JSON for the paper pipeline")
    parser.add_argument("--smoke", action="store_true", help="Synthetic test only; no model or browser")
    parser.add_argument("--model-id")
    parser.add_argument("--model-revision", help="Model commit or branch; use with --model-id when changing a pinned model")
    parser.add_argument("--agent-backend", choices=("hf", "synthetic"))
    parser.add_argument("--env-ids", nargs="+")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--mc-samples", type=int)
    parser.add_argument("--check-every", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    parser.add_argument("--calibration-path")
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config) if args.config else ExperimentConfig()
    overrides = {key: value for key, value in vars(args).items()
                 if key not in ("config", "smoke", "pipeline", "probe_config") and value is not None}
    if args.model_id and args.model_id != config.model_id and args.model_revision is None:
        overrides["model_revision"] = None
    if args.smoke:
        overrides.update(environment="synthetic", agent_backend="synthetic", env_ids=("synthetic/sequence",),
                         model_id="synthetic-policy", model_revision=None,
                         output_dir=args.output_dir or "runs/synthetic_smoke")
    if args.pipeline == "paper":
        if args.smoke:
            parser.error("The paper pipeline's synthetic feature test is provided by tests/test_paper_pipeline.py")
        probes = ProbeConfig(**json.loads(open(args.probe_config).read())) if args.probe_config else ProbeConfig()
        result = run_paper_experiment(config, probe_config=probes, **overrides)
    else:
        if args.probe_config:
            parser.error("--probe-config requires --pipeline paper")
        result = run_experiment(config, **overrides)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
