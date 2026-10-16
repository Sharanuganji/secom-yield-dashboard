"""Process capability indices for semiconductor SPC."""
import numpy as np
import matplotlib.pyplot as plt


def compute_cp_cpk(values, usl, lsl):
    """
    Compute Cp, Cpk, Pp, Ppk.
    values: array of observations
    usl: Upper Specification Limit
    lsl: Lower Specification Limit
    """
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    std = values.std(ddof=1)  # sample std

    # Cp uses within-subgroup variation (here we use overall std as proxy)
    cp = (usl - lsl) / (6 * std)
    cpu = (usl - mean) / (3 * std)
    cpl = (mean - lsl) / (3 * std)
    cpk = min(cpu, cpl)

    return {
        "mean": mean, "std": std,
        "cp": cp, "cpk": cpk, "cpu": cpu, "cpl": cpl,
        "usl": usl, "lsl": lsl,
    }


def interpret_cpk(cpk):
    if cpk >= 1.67:
        return "Excellent (≥1.67)"
    elif cpk >= 1.33:
        return "Adequate (≥1.33)"
    elif cpk >= 1.0:
        return "Marginal (≥1.0)"
    else:
        return "Inadequate (<1.0)"


def plot_capability(values, stats, title="Process Capability", save_path=None):
    """Histogram with spec limits and capability annotations."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(values, bins=50, density=True, alpha=0.6, edgecolor="black", color="steelblue")

    # Normal curve overlay
    x = np.linspace(values.min(), values.max(), 200)
    pdf = (1 / (stats["std"] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - stats["mean"]) / stats["std"])**2)
    ax.plot(x, pdf, "r-", linewidth=1.5, label="Normal fit")

    # Spec limits
    ax.axvline(stats["usl"], color="red", linestyle="--", linewidth=2, label=f"USL = {stats['usl']:.1f}")
    ax.axvline(stats["lsl"], color="red", linestyle="--", linewidth=2, label=f"LSL = {stats['lsl']:.1f}")
    ax.axvline(stats["mean"], color="green", linestyle="-", linewidth=1.5, label=f"Mean = {stats['mean']:.1f}")

    # Annotation
    txt = f"Cp = {stats['cp']:.2f}\nCpk = {stats['cpk']:.2f}\n({interpret_cpk(stats['cpk'])})"
    ax.text(0.95, 0.95, txt, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    ax.set_title(title)
    ax.set_xlabel("Measurement")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()   