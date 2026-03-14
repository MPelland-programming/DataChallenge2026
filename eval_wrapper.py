"""
Evaluation Wrapper
==================
Orchestrates a full evaluation run:
  1. Runs main.py with the given scorer/selector/date range → generates opportunities.csv
  2. Runs evaluate.py and captures stdout
  3. Parses the stdout to extract aggregate and per-month metrics
  4. Writes results/{{scorer}}__{{selector}}__{{start}}_{{end}}.json

Usage:
  python eval_wrapper.py --scorer historical_profit_rate --selector default \\
      --start-month 2020-01 --end-month 2022-12

  python eval_wrapper.py --scorer activation_level --selector default \\
      --start-month 2023-01 --end-month 2023-12
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Wrapper — runs main.py then evaluate.py, saves JSON results.")
    parser.add_argument("--scorer", required=True, help="Scorer name (e.g. historical_profit_rate)")
    parser.add_argument("--selector", required=True, help="Selector name (e.g. default)")
    parser.add_argument("--start-month", required=True, metavar="YYYY-MM", help="First target month")
    parser.add_argument("--end-month", required=True, metavar="YYYY-MM", help="Last target month")
    parser.add_argument("--data-root", default=None, help="Override DATA_ROOT for main.py")
    parser.add_argument("--log-level", default=None, help="Log level for main.py")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip running main.py (use existing opportunities.csv)")
    return parser.parse_args()


def run_main(args):
    """Run main.py with the given arguments."""
    cmd = [
        sys.executable, "main.py",
        "--scorer", args.scorer,
        "--selector", args.selector,
        "--start-month", args.start_month,
        "--end-month", args.end_month,
    ]
    if args.data_root:
        cmd += ["--data-root", args.data_root]
    if args.log_level:
        cmd += ["--log-level", args.log_level]

    print(f"[eval_wrapper] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result.returncode


def run_evaluate(args):
    """Run evaluate.py and capture stdout."""
    cmd = [
        sys.executable, "evaluate.py",
        "opportunities.csv",
        "--start-month", args.start_month,
        "--end-month", args.end_month,
    ]
    print(f"[eval_wrapper] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.stdout


def parse_evaluate_output(stdout: str) -> dict:
    """
    Parse evaluate.py stdout and return a structured dict.

    Expected sections in stdout (see evaluate.py main()):
      - OFF-Peak and ON-Peak F1 blocks
      - Average F1-score
      - Total net profit, n_selected, n_profitable, n_losing
      - Monthly breakdown table
    """

    def _float(s: str) -> float:
        return float(s.replace(",", ""))

    def _int(s: str) -> int:
        return int(s.replace(",", ""))

    # ── Per-peak F1 metrics ──────────────────────────────────────────────────
    peak_metrics = {}
    for peak_name in ("OFF", "ON"):
        # Match the peak header, then TP/FP/FN line, then Precision/Recall/F1 line
        pattern = (
            rf"{peak_name}-Peak:\s+"
            rf"TP=([\d,]+)\s+FP=([\d,]+)\s+FN=([\d,]+)\s+"
            rf"Precision=([\d.]+)\s+Recall=([\d.]+)\s+F1=([\d.]+)"
        )
        m = re.search(pattern, stdout)
        if m:
            peak_metrics[peak_name] = {
                "tp": _int(m.group(1)),
                "fp": _int(m.group(2)),
                "fn": _int(m.group(3)),
                "precision": _float(m.group(4)),
                "recall": _float(m.group(5)),
                "f1": _float(m.group(6)),
            }
        else:
            peak_metrics[peak_name] = {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    # ── Average F1 ───────────────────────────────────────────────────────────
    m = re.search(r"Average F1-score:\s+([\d.]+)", stdout)
    avg_f1 = _float(m.group(1)) if m else 0.0

    # ── Profit section ───────────────────────────────────────────────────────
    m = re.search(r"Total selections:\s+([\d,]+)", stdout)
    n_selected = _int(m.group(1)) if m else 0

    m = re.search(r"Profitable:\s+([\d,]+)", stdout)
    n_profitable = _int(m.group(1)) if m else 0

    m = re.search(r"Losing:\s+([\d,]+)", stdout)
    n_losing = _int(m.group(1)) if m else 0

    m = re.search(r"Total net profit:\s+([-\d,. ]+)", stdout)
    net_profit = _float(m.group(1).strip()) if m else 0.0

    # ── Monthly breakdown ────────────────────────────────────────────────────
    # Lines look like:  "  2020-01     50     35     15      1,234.56"
    monthly = []
    for m in re.finditer(
        r"(\d{4}-\d{2})\s+(\d+)\s+(\d+)\s+(\d+)\s+([-\d,. ]+)", stdout
    ):
        monthly.append({
            "month": m.group(1),
            "n_selected": _int(m.group(2)),
            "tp": _int(m.group(3)),
            "fp": _int(m.group(4)),
            "profit": _float(m.group(5).strip()),
        })

    return {
        "aggregate": {
            "f1_off": peak_metrics["OFF"]["f1"],
            "f1_on": peak_metrics["ON"]["f1"],
            "f1_avg": avg_f1,
            "precision_off": peak_metrics["OFF"]["precision"],
            "precision_on": peak_metrics["ON"]["precision"],
            "recall_off": peak_metrics["OFF"]["recall"],
            "recall_on": peak_metrics["ON"]["recall"],
            "tp_off": peak_metrics["OFF"]["tp"],
            "fp_off": peak_metrics["OFF"]["fp"],
            "fn_off": peak_metrics["OFF"]["fn"],
            "tp_on": peak_metrics["ON"]["tp"],
            "fp_on": peak_metrics["ON"]["fp"],
            "fn_on": peak_metrics["ON"]["fn"],
            "net_profit": net_profit,
            "n_selected": n_selected,
            "n_profitable": n_profitable,
            "n_losing": n_losing,
        },
        "monthly": monthly,
    }


def write_results(scorer: str, selector: str, start: str, end: str, metrics: dict):
    """Write metrics to results/{scorer}__{selector}__{start}_{end}.json."""
    RESULTS_DIR.mkdir(exist_ok=True)
    start_slug = start.replace("-", "")  # 202001
    end_slug = end.replace("-", "")      # 202212
    filename = f"{scorer}__{selector}__{start_slug}_{end_slug}.json"
    path = RESULTS_DIR / filename

    payload = {
        "scorer": scorer,
        "selector": selector,
        "start_month": start,
        "end_month": end,
        **metrics,
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[eval_wrapper] Results saved to: {path}")
    return path


def main():
    args = parse_args()

    if not args.dry_run:
        run_main(args)

    stdout = run_evaluate(args)
    metrics = parse_evaluate_output(stdout)
    write_results(args.scorer, args.selector, args.start_month, args.end_month, metrics)

    # Print summary
    agg = metrics["aggregate"]
    print(f"\n[eval_wrapper] Summary:")
    print(f"  F1 avg:     {agg['f1_avg']:.4f}  (OFF={agg['f1_off']:.4f}, ON={agg['f1_on']:.4f})")
    print(f"  Net profit: {agg['net_profit']:,.2f}")
    print(f"  Selected:   {agg['n_selected']}  ({agg['n_profitable']} profitable, {agg['n_losing']} losing)")


if __name__ == "__main__":
    main()
