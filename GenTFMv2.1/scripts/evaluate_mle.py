#!/usr/bin/env python3
"""MLE-only entry point. Defaults: K=200, 200 synthetic rows, no calibration.

Shares all data-source and predictor options with evaluate_real.py. This script
does not retrain GenTFM or TabICL, but fits downstream estimators for each arm.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_real import main, parse_args


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not any(arg == "--context_sizes" or arg.startswith("--context_sizes=") for arg in argv):
        argv += ["--context_sizes", "200"]
    if not any(arg == "--calibration" or arg.startswith("--calibration=") for arg in argv):
        argv += ["--calibration", "none"]
    args = parse_args(argv)
    args.mle_only = True
    main(args)
