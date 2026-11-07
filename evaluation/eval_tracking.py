# evaluation/eval_tracking.py
"""SO-HOTA tracking evaluation wrapper.

Wraps TrackEval to evaluate MOT-format tracking results using the
SO-HOTA metric from the SMOT4SB challenge.

Requires TrackEval to be installed. The recommended source is the
MVA2025-SMOT4SB fork: https://github.com/IIM-TTIJ/MVA2025-SMOT4SB
"""

import argparse
import json
import sys
from pathlib import Path


def evaluate_tracking(
    gt_dir: str,
    tracker_dir: str,
    seqmap: str | None = None,
) -> dict:
    """Run SO-HOTA evaluation.

    Args:
        gt_dir: Directory containing GT in MOT format (seqname/gt/gt.txt).
        tracker_dir: Directory containing tracker output (seqname.txt or seqname/seqname.txt).
        seqmap: Path to seqmap file listing sequences to evaluate.

    Returns:
        Dict with SO-HOTA metrics.
    """
    try:
        import trackeval
    except ImportError:
        print(
            "TrackEval not installed. SO-HOTA lives on a branch, not master:\n"
            "  uv pip install "
            "'git+https://github.com/IIM-TTIJ/TrackEval.git@smot4sb_dev'\n"
            "master is a plain fork of upstream and has neither the SO_HOTA\n"
            "metric nor the SMOT4SBChallenge dataset class.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not hasattr(trackeval.datasets, "SMOT4SBChallenge"):
        print(
            "This TrackEval has no SMOT4SBChallenge dataset class, so it is\n"
            "the upstream package or the fork's master branch. Install the\n"
            "smot4sb_dev branch:\n"
            "  uv pip install "
            "'git+https://github.com/IIM-TTIJ/TrackEval.git@smot4sb_dev'",
            file=sys.stderr,
        )
        sys.exit(1)

    eval_config = {
        "USE_PARALLEL": False,
        "NUM_PARALLEL_CORES": 1,
        "PRINT_RESULTS": True,
        "PRINT_ONLY_COMBINED": True,
        "PRINT_CONFIG": False,
        "TIME_PROGRESS": False,
        "OUTPUT_SUMMARY": True,
        # The fork's summary writer reads back the detailed CSV, so turning
        # this off makes it fail with a missing-file error after the metrics
        # have already been computed.
        "OUTPUT_DETAILED": True,
        "PLOT_CURVES": False,
    }

    dataset_config = {
        "GT_FOLDER": gt_dir,
        "TRACKERS_FOLDER": tracker_dir,
        "TRACKERS_TO_EVAL": [""],
        "TRACKER_SUB_FOLDER": "",
        "OUTPUT_FOLDER": str(Path(tracker_dir) / "eval_output"),
        "SEQMAP_FILE": seqmap,
        "CLASSES_TO_EVAL": ["bird"],
        # DotD rather than IoU. The class computes the dataset-wide S from the
        # GT on construction; there is nothing to pass in.
        "USE_SO_HOTA": True,
    }

    evaluator = trackeval.Evaluator(eval_config)
    # SMOT4SBChallenge, not MotChallenge2DBox: the latter's similarity is IoU,
    # which collapses to zero the moment small boxes stop overlapping. Running
    # SO_HOTA against it would report DotD-named numbers computed from IoU.
    dataset_list = [trackeval.datasets.SMOT4SBChallenge(dataset_config)]
    metrics_list = [trackeval.metrics.SO_HOTA()]

    output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)

    return output_res


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tracking with SO-HOTA")
    parser.add_argument("--gt-dir", type=str, required=True, help="GT directory (MOT format)")
    parser.add_argument("--tracker-dir", type=str, required=True, help="Tracker output directory")
    parser.add_argument("--seqmap", type=str, default=None, help="Seqmap file")
    parser.add_argument("--output-json", type=str, default=None, help="Save results to JSON")
    args = parser.parse_args()

    results = evaluate_tracking(args.gt_dir, args.tracker_dir, args.seqmap)

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(str(results), f, indent=2)


if __name__ == "__main__":
    main()
