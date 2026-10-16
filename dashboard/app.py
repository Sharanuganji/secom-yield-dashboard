"""
SPC + Yield Analysis Dashboard
Interactive Streamlit app for semiconductor process monitoring.
"""
import streamlit as st
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from spc import compute_i_mr, compute_cusum, western_electric_rules, plot_i_mr, plot_cusum
from capability import compute_cp_cpk, interpret_cpk, plot_capability
from excursion import detect_excursions, plot_excursions

# --- Page config ---
st.set_page_config(page_title="SECOM SPC Dashboard", layout="wide")
st.title("🏭 Semiconductor SPC & Yield Dashboard")
st.caption("UCI SECOM Dataset — Process Monitoring & Yield Analysis")

# --- Load data (cached) ---
@st.cache_data
def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    df = pd.read_csv(os.path.join(data_dir, "secom_cleaned.csv"), parse_dates=["Time"])
    df = df.sort_values("Time").reset_index(drop=True)
    return df

df = load_data()
sensor_cols = [c for c in df.columns if c.startswith("X")]
top_sensors = df[sensor_cols].var().nlargest(10).index.tolist()

# --- Sidebar ---
st.sidebar.header("Controls")
sensor = st.sidebar.selectbox("Select Sensor", top_sensors)
st.sidebar.markdown(f"**Mean:** {df[sensor].mean():.2f}")
st.sidebar.markdown(f"**Std:** {df[sensor].std():.2f}")
st.sidebar.markdown(f"**Fail rate:** {(df['Label'] == 1).mean():.2%}")   

tab = st.sidebar.radio("View", ["SPC Charts", "Process Capability", "Excursion Detection", "Yield Summary"])

values = df[sensor].values
times = df["Time"].values

# --- SPC Charts ---
if tab == "SPC Charts":
    st.subheader(f"I-MR Chart: {sensor}")
    stats = compute_i_mr(values)
    st.write(f"Center: {stats['mean']:.2f} | UCL: {stats['ucl_i']:.2f} | LCL: {stats['lcl_i']:.2f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    mr = np.abs(np.diff(values))
    ax1.plot(times, values, "b-", alpha=0.6, linewidth=0.8)
    ax1.axhline(stats["mean"], color="green", linestyle="--", label=f"CL ({stats['mean']:.1f})")
    ax1.axhline(stats["ucl_i"], color="red", linestyle="--", label=f"UCL ({stats['ucl_i']:.1f})")
    ax1.axhline(stats["lcl_i"], color="red", linestyle="--", label=f"LCL ({stats['lcl_i']:.1f})")
    ax1.set_ylabel("Individual"); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    ax2.plot(times[1:], mr, "b-", alpha=0.6, linewidth=0.8)
    ax2.axhline(stats["mr_bar"], color="green", linestyle="--")
    ax2.axhline(stats["ucl_mr"], color="red", linestyle="--")
    ax2.set_ylabel("MR"); ax2.set_xlabel("Time"); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader(f"CUSUM: {sensor}")
    cusum_stats = compute_cusum(values)
    st.write(f"Signals detected: **{len(cusum_stats['signals'])}**")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times[1:], cusum_stats["cusum_pos"][1:], "b-", alpha=0.7, label="CUSUM+")
    ax.plot(times[1:], cusum_stats["cusum_neg"][1:], "r-", alpha=0.7, label="CUSUM-")
    ax.axhline(cusum_stats["h"], color="green", linestyle="--", label=f"h={cusum_stats['h']:.1f}")
    ax.axhline(-cusum_stats["h"], color="green", linestyle="--")
    ax.set_title(f"CUSUM ({len(cusum_stats['signals'])} signals)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    violations = western_electric_rules(values)
    st.write(f"**Western Electric violations:** {len(violations)}")

# --- Process Capability ---
elif tab == "Process Capability":
    st.subheader(f"Process Capability: {sensor}")
    mean, std = values.mean(), values.std(ddof=1)
    usl, lsl = mean + 3*std, mean - 3*std
    stats = compute_cp_cpk(values, usl, lsl)

    col1, col2, col3 = st.columns(3)
    col1.metric("Cp", f"{stats['cp']:.2f}")
    col2.metric("Cpk", f"{stats['cpk']:.2f}")
    col3.metric("Rating", interpret_cpk(stats['cpk']))

    st.info("Specs assumed as mean ± 3σ (no real product specs in SECOM). In practice, specs come from the product definition.")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(values, bins=50, density=True, alpha=0.6, edgecolor="black", color="steelblue")
    x = np.linspace(values.min(), values.max(), 200)
    pdf = (1/(std*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-mean)/std)**2)
    ax.plot(x, pdf, "r-", linewidth=1.5)
    ax.axvline(usl, color="red", linestyle="--", label=f"USL={usl:.1f}")
    ax.axvline(lsl, color="red", linestyle="--", label=f"LSL={lsl:.1f}")
    ax.axvline(mean, color="green", label=f"Mean={mean:.1f}")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

# --- Excursion Detection ---
elif tab == "Excursion Detection":
    st.subheader(f"Excursion Detection: {sensor}")
    window = st.slider("Rolling window", 10, 100, 20)
    threshold = st.slider("Z-score threshold", 1.0, 4.0, 2.0, 0.1)

    z_scores, mask = detect_excursions(times, values, window=window, threshold=threshold)
    st.write(f"**Excursions:** {mask.sum()} ({mask.mean()*100:.1f}% of observations)")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(times, values, "b-", alpha=0.6, linewidth=0.8)
    ax1.plot(times[mask], values[mask], "r.", markersize=4, label=f"Excursions ({mask.sum()})")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    ax2.plot(times, z_scores, "k-", alpha=0.5, linewidth=0.5)
    ax2.axhline(threshold, color="red", linestyle="--")
    ax2.axhline(-threshold, color="red", linestyle="--")
    ax2.set_ylabel("Z-score"); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

# --- Yield Summary ---
elif tab == "Yield Summary":
    st.subheader("Dataset Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Wafers", f"{len(df)}")
    col2.metric("Fail Rate", f"{df['Label'].mean():.2%}")
    col3.metric("Sensors", f"{len(sensor_cols)}")

    st.subheader("Daily Fail Count")
    df["fail_flag"] = (df["Label"] == 1).astype(int)
    daily = df.set_index("Time")["fail_flag"].resample("D").sum()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(daily.index, daily.values, color="#e74c3c", alpha=0.7)
    ax.set_title("Daily Fail Count"); ax.set_ylabel("Fails"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Model Performance")
    st.table(pd.DataFrame({
        "Model": ["Random Forest (SMOTE)", "Logistic Regression", "Baseline (all-pass)"],
        "PR-AUC": [0.236, 0.177, 0.066],
    }))   
