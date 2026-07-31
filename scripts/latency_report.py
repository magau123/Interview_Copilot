from __future__ import annotations

import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PATTERN = re.compile(r"METRIC\s+(\w+)\s+([\d.]+)")


def percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * ratio))
    return ordered[index]


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/latency_report.py <application.log>")
        return 2
    metrics: dict[str, list[float]] = defaultdict(list)
    for line in Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace").splitlines():
        if match := PATTERN.search(line):
            metrics[match.group(1)].append(float(match.group(2)))
    if not metrics:
        print("No latency metrics found.")
        return 1
    for name, values in metrics.items():
        print(
            f"{name}: n={len(values)} mean={statistics.mean(values):.0f}ms "
            f"p50={percentile(values, 0.50):.0f}ms p95={percentile(values, 0.95):.0f}ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
