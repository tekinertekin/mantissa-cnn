"""Benchmark plots — SKELETON. Consumes bench/results/{speed,accuracy}.json
(produced by bench.speed / bench.accuracy) and writes PNGs to assets/.

Planned figures (matplotlib, no seaborn, one chart per file):
  assets/fit_time.png   median fit seconds per (arch, dataset), grouped bars
                        per contender, log scale
  assets/accuracy.png   test accuracy per (arch, dataset) per contender,
                        sklearn bar hatched + annotated "non-CNN baseline"
  assets/peak_rss.png   peak RSS per contender (import + fit), grouped bars

Never invents data: exits with a message if a JSON input is missing.

Run from the repo root:  python -m bench.plots
"""
from __future__ import annotations


def main() -> int:
    raise NotImplementedError("bench skeleton — implementation lands in the "
                              "benchmark phase; the docstring is the spec")


if __name__ == "__main__":
    raise SystemExit(main())
