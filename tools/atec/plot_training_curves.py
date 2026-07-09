#!/usr/bin/env python3
"""Plot README-ready training curves from local TensorBoard event logs."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_RUNS = [
    (
        "Rough straight locomotion",
        "unitree_b2_rough_straight/2026-05-30_11-13-50_from_flat",
    ),
    (
        "B2W rough omni locomotion",
        "unitree_b2w_rough_omni/2026-06-21_19-02-24_from_straight",
    ),
    (
        "Flat push pretraining",
        "unitree_b2w_taskd_flat_pretrain/2026-06-28_19-11-54_from_rough_omni_2800",
    ),
    (
        "Short omni DR hardening",
        "unitree_b2w_taskf_short_walk/2026-06-29_17-53-22_from_robust",
    ),
]


@dataclass(frozen=True)
class ScalarSeries:
    label: str
    run_path: Path
    steps: list[int]
    values: list[float]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_plot_deps():
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing tensorboard. Run with: "
            "source scripts/env/activate-atec2026-sim.sh && "
            "MPLCONFIGDIR=/tmp/mpl python tools/atec/plot_training_curves.py"
        ) from exc

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return EventAccumulator, plt


def _ema(values: list[float], alpha: float) -> list[float]:
    if not values:
        return []
    smoothed = [values[0]]
    for value in values[1:]:
        smoothed.append(alpha * value + (1.0 - alpha) * smoothed[-1])
    return smoothed


def _load_scalar(EventAccumulator, run_dir: Path, label: str, scalar: str) -> ScalarSeries | None:
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        print(f"[WARN] no TensorBoard event file under {run_dir}", file=sys.stderr)
        return None

    event_file = event_files[-1]
    accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    if scalar not in tags:
        print(f"[WARN] {scalar!r} not found in {run_dir}", file=sys.stderr)
        return None

    events = accumulator.Scalars(scalar)
    if not events:
        print(f"[WARN] {scalar!r} has no samples in {run_dir}", file=sys.stderr)
        return None

    return ScalarSeries(
        label=label,
        run_path=run_dir,
        steps=[event.step for event in events],
        values=[float(event.value) for event in events],
    )


def _parse_run_specs(specs: list[str]) -> list[tuple[str, str]]:
    if not specs:
        return DEFAULT_RUNS

    parsed: list[tuple[str, str]] = []
    for spec in specs:
        if "=" in spec:
            label, path = spec.split("=", 1)
            parsed.append((label.strip(), path.strip()))
        else:
            path = spec.strip()
            parsed.append((Path(path).name, path))
    return parsed


def _plot(series: list[ScalarSeries], output: Path, scalar: str, ema_alpha: float) -> None:
    _, plt = _ensure_plot_deps()
    n = len(series)
    if n == 0:
        raise SystemExit("No scalar series loaded; nothing to plot.")

    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.2 * rows), squeeze=False)
    axes_flat = [axis for row in axes for axis in row]

    for axis, item in zip(axes_flat, series):
        smoothed = _ema(item.values, ema_alpha)
        axis.plot(item.steps, item.values, color="#8aa0b8", linewidth=1.0, alpha=0.35, label="raw")
        axis.plot(item.steps, smoothed, color="#1f5f99", linewidth=2.0, label="EMA")
        axis.set_title(f"{item.label}\n{item.run_path.name} | final step {item.steps[-1]}")
        axis.set_xlabel("Training iteration")
        axis.set_ylabel(scalar)
        axis.grid(True, color="#d9dee7", linewidth=0.8, alpha=0.8)
        axis.legend(loc="best", frameon=False)

    for axis in axes_flat[n:]:
        axis.axis("off")

    fig.suptitle(
        "TensorBoard Train/mean_reward from local RSL-RL logs",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/media/training_curves.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--scalar",
        default="Train/mean_reward",
        help="TensorBoard scalar tag to plot.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.08,
        help="EMA smoothing alpha in (0, 1].",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run path, or LABEL=run/path. May be repeated.",
    )
    args = parser.parse_args()

    if not 0.0 < args.ema_alpha <= 1.0:
        raise SystemExit("--ema-alpha must be in (0, 1].")

    EventAccumulator, _ = _ensure_plot_deps()
    root = _project_root()
    runs_root = root / "outputs" / "rsl_rl"
    series: list[ScalarSeries] = []
    for label, rel_path in _parse_run_specs(args.run):
        run_dir = runs_root / rel_path
        loaded = _load_scalar(EventAccumulator, run_dir, label, args.scalar)
        if loaded is not None:
            series.append(loaded)

    output = args.output
    if not output.is_absolute():
        output = root / output
    _plot(series, output, args.scalar, args.ema_alpha)
    print(f"Wrote {output.relative_to(root)}")


if __name__ == "__main__":
    main()
