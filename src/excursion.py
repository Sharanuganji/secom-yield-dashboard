"""Excursion detection: flag time windows where process drifts out of control."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def detect_excursions(times, values, window=20, threshold=2.0):
    """
    Detect excursions using rolling z-score.
    A point is an 'excursion' if its z-score exceeds threshold
    relative to the rolling window mean/std.
    """
    s = pd.Series(values)
    rolling_mean = s.rolling(window=window, min_periods=5).mean()
    rolling_std = s.rolling(window=window, min_periods=5).std()
    z_scores = (s - rolling_mean) / (rolling_std + 1e-10)

    excursion_mask = z_scores.abs() > threshold
    return z_scores, excursion_mask


def plot_excursions(times, values, z_scores, excursion_mask,
                    title="Excursion Detection", save_path=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(times, values, "b-", alpha=0.6, linewidth=0.8)
    ax1.plot(times[excursion_mask], values[excursion_mask], "r.",
             markersize=4, label=f"Excursions ({excursion_mask.sum()})")
    ax1.set_ylabel("Value")
    ax1.set_title(title)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(times, z_scores, "k-", alpha=0.5, linewidth=0.5)
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.axhline(2, color="red", linestyle="--", alpha=0.5)
    ax2.axhline(-2, color="red", linestyle="--", alpha=0.5)
    ax2.fill_between(times, -2, 2, alpha=0.05, color="green")
    ax2.set_ylabel("Z-score")
    ax2.set_xlabel("Time")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()   