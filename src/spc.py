"""SPC module for semiconductor yield analysis."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_i_mr(values):
    values = np.asarray(values, dtype=float)
    mr = np.abs(np.diff(values))
    mr_bar = mr.mean()
    sigma_i = mr_bar / 1.128
    ucl_i = values.mean() + 3 * sigma_i
    lcl_i = values.mean() - 3 * sigma_i
    ucl_mr = 3.267 * mr_bar
    return {"mean": values.mean(), "sigma": sigma_i,
            "ucl_i": ucl_i, "lcl_i": lcl_i,
            "mr_bar": mr_bar, "ucl_mr": ucl_mr, "lcl_mr": 0}


def compute_cusum(values, k=0.5, h=5.0):
    values = np.asarray(values, dtype=float)
    mu0, sigma = values.mean(), values.std()
    k_abs, h_abs = k * sigma, h * sigma
    cusum_pos = np.zeros(len(values))
    cusum_neg = np.zeros(len(values))
    for i in range(1, len(values)):
        cusum_pos[i] = max(0, cusum_pos[i-1] + (values[i] - mu0) - k_abs)
        cusum_neg[i] = max(0, cusum_neg[i-1] - (values[i] - mu0) - k_abs)
    signals = np.where((cusum_pos > h_abs) | (cusum_neg > h_abs))[0]
    return {"cusum_pos": cusum_pos, "cusum_neg": cusum_neg, "h": h_abs, "signals": signals}


def western_electric_rules(values):
    values = np.asarray(values, dtype=float)
    mean, std = values.mean(), values.std()
    ucl, lcl = mean + 3*std, mean - 3*std
    violations = []
    for i in range(len(values)):
        if values[i] > ucl or values[i] < lcl:
            violations.append((i, 1, "Beyond 3-sigma"))
        if i >= 2:
            w = values[i-2:i+1]
            if sum(1 for v in w if v > mean + 2*std) >= 2 or sum(1 for v in w if v < mean - 2*std) >= 2:
                violations.append((i, 2, "2 of 3 beyond 2-sigma"))
        if i >= 7:
            w = values[i-7:i+1]
            if all(v > mean for v in w) or all(v < mean for v in w):
                violations.append((i, 4, "8 consecutive same side"))
    return violations


def plot_i_mr(times, values, stats, title="I-MR Chart", save_path=None):
    mr = np.abs(np.diff(values))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(times, values, "b-", alpha=0.6, linewidth=0.8, label="Observation")
    ax1.axhline(stats["mean"], color="green", linestyle="--", label=f"Center ({stats['mean']:.2f})")
    ax1.axhline(stats["ucl_i"], color="red", linestyle="--", label=f"UCL ({stats['ucl_i']:.2f})")
    ax1.axhline(stats["lcl_i"], color="red", linestyle="--", label=f"LCL ({stats['lcl_i']:.2f})")
    ax1.set_ylabel("Individual"); ax1.set_title(title); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    ax2.plot(times[1:], mr, "b-", alpha=0.6, linewidth=0.8)
    ax2.axhline(stats["mr_bar"], color="green", linestyle="--", label=f"MR-bar ({stats['mr_bar']:.2f})")
    ax2.axhline(stats["ucl_mr"], color="red", linestyle="--", label=f"UCL ({stats['ucl_mr']:.2f})")
    ax2.set_ylabel("Moving Range"); ax2.set_xlabel("Time"); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()


def plot_cusum(times, values, cusum_stats, title="CUSUM Chart", save_path=None):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(times[1:], cusum_stats["cusum_pos"][1:], "b-", alpha=0.7, label="CUSUM+")
    ax.plot(times[1:], cusum_stats["cusum_neg"][1:], "r-", alpha=0.7, label="CUSUM-")
    ax.axhline(cusum_stats["h"], color="green", linestyle="--", label=f"h={cusum_stats['h']:.2f}")
    ax.axhline(-cusum_stats["h"], color="green", linestyle="--")
    if len(cusum_stats["signals"]) > 0:
        ax.axvline(times[cusum_stats["signals"][0]], color="orange", alpha=0.5)
        ax.annotate(f"{len(cusum_stats['signals'])} signals",
                    xy=(times[cusum_stats["signals"][0]], cusum_stats["h"]),
                    fontsize=9, color="orange")
    ax.set_title(title); ax.set_xlabel("Time"); ax.set_ylabel("CUSUM value")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()   