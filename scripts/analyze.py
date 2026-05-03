#!/usr/bin/env python3
import os
import glob
import math
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DATA_DIR = os.path.join("data")
EVOL_HISTORY_DIR = os.path.join(BASE_DATA_DIR, "evolution_histories")
TRAIT_STATS_DIR = os.path.join(BASE_DATA_DIR, "trait_stats")
SIMILARITY_DIR = os.path.join(BASE_DATA_DIR, "similarity_logs")
ANALYSIS_DIR = os.path.join(BASE_DATA_DIR, "analysis")


def find_latest_run_tag():
    """
    Finds the latest run_tag based on the timestamp in the filename.
    Handles both old format (YYYYMMDD_HHMM) and new format (YYYYMMDD_HHMM_AGENT_ON/OFF).
    Returns the full run_tag string (including suffix) so all file paths resolve correctly.
    """
    candidates = []

    patterns = [
        (os.path.join(EVOL_HISTORY_DIR, "run_summary_*.txt"),   "run_summary_",   ".txt"),
        (os.path.join(EVOL_HISTORY_DIR, "evolution_run_*.csv"), "evolution_run_", ".csv"),
    ]

    for glob_pattern, prefix, suffix in patterns:
        for path in glob.glob(glob_pattern):
            base = os.path.basename(path)
            full_tag = base.replace(prefix, "").replace(suffix, "")

            # Strip optional _AGENT_ON / _AGENT_OFF to parse the timestamp
            timestamp_part = full_tag
            for label in ("_AGENT_ON", "_AGENT_OFF"):
                if timestamp_part.endswith(label):
                    timestamp_part = timestamp_part[: -len(label)]
                    break

            try:
                dt = datetime.strptime(timestamp_part, "%Y%m%d_%H%M")
            except ValueError:
                continue

            # Keep the full_tag (with suffix) so downstream file lookups work
            candidates.append((dt, full_tag))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_run_data(run_tag):
    """
    Load the main per-run CSV and related per-run stats.
    Returns dict of DataFrames (some may be None if missing).
    """
    data = {}

    evol_path = os.path.join(EVOL_HISTORY_DIR, f"evolution_run_{run_tag}.csv")
    if os.path.exists(evol_path):
        data["evolution"] = pd.read_csv(evol_path)
    else:
        print(f"⚠️ Could not find evolution history: {evol_path}")
        data["evolution"] = None

    trait_path = os.path.join(TRAIT_STATS_DIR, f"trait_stats_{run_tag}.csv")
    if os.path.exists(trait_path):
        data["traits"] = pd.read_csv(trait_path)
    else:
        print(f"⚠️ Could not find trait stats: {trait_path}")
        data["traits"] = None

    sim_path = os.path.join(SIMILARITY_DIR, f"similarity_log_{run_tag}.csv")
    if os.path.exists(sim_path):
        data["similarity"] = pd.read_csv(sim_path)
    else:
        print(f"⚠️ Could not find similarity log: {sim_path}")
        data["similarity"] = None

    summary_path = os.path.join(EVOL_HISTORY_DIR, f"run_summary_{run_tag}.txt")
    data["summary_path"] = summary_path if os.path.exists(summary_path) else None

    return data


def summarize_generations(evol_df):
    if evol_df is None or evol_df.empty:
        return None

    required_cols = [
        "Generation", "Fitness_Score", "AMP_Score",
        "Toxicity_Score", "Stability_Score",
        "Net_Charge", "Hydrophobicity", "Aggregation_Risk",
        "Realism_Score", "Hydrophobic_Moment",
    ]

    missing = [c for c in required_cols if c not in evol_df.columns]
    if missing:
        print(f"⚠️ Missing columns {missing} — generation summary will be limited.")

    group = evol_df.groupby("Generation")

    summary = group.agg(
        Avg_Fitness=("Fitness_Score", "mean"),
        Max_Fitness=("Fitness_Score", "max"),
        Avg_AMP=("AMP_Score", "mean"),
        Avg_Tox=("Toxicity_Score", "mean"),
        Avg_Stab=("Stability_Score", "mean"),
        Avg_Charge=("Net_Charge", "mean"),
        Avg_Hydrophobicity=("Hydrophobicity", "mean"),
        Avg_Aggregation_Risk=("Aggregation_Risk", "mean"),
        Avg_Realism=("Realism_Score", "mean"),
        Avg_HydroMoment=("Hydrophobic_Moment", "mean"),
        Count=("Peptide", "count"),
    ).reset_index()

    return summary


def extract_top_peptides(evol_df, out_dir, run_tag, top_n=100):
    if evol_df is None or evol_df.empty:
        return

    evol_df_sorted = evol_df.sort_values(
        by="Fitness_Score", ascending=False
    ).reset_index(drop=True)

    top_unique = evol_df_sorted.drop_duplicates(subset=["Peptide"]).head(top_n)
    top_path = os.path.join(out_dir, f"top_peptides_run_{run_tag}.csv")
    top_unique.to_csv(top_path, index=False)
    print(f"💾 Saved global top {len(top_unique)} peptides to {top_path}")

    best_by_gen = evol_df_sorted.groupby("Generation").head(1).reset_index(drop=True)
    best_gen_path = os.path.join(out_dir, f"best_by_generation_run_{run_tag}.csv")
    best_by_gen.to_csv(best_gen_path, index=False)
    print(f"💾 Saved best-by-generation peptides to {best_gen_path}")


def plot_hydromoment(summary_df, out_dir, run_tag):
    if summary_df is None or summary_df.empty or "Avg_HydroMoment" not in summary_df.columns:
        return

    fig, ax = plt.subplots()
    ax.plot(summary_df["Generation"], summary_df["Avg_HydroMoment"])
    ax.set_xlabel("Generation")
    ax.set_ylabel("Avg Hydrophobic Moment")
    ax.set_title(f"Amphipathic Structure Drift (run {run_tag})")
    fig.tight_layout()

    out_path = os.path.join(out_dir, "hydrophobic_moment_over_time.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"🌀 Saved {out_path}")


def plot_fitness(summary_df, out_dir, run_tag):
    if summary_df is None or summary_df.empty:
        return

    fig, ax = plt.subplots()
    ax.plot(summary_df["Generation"], summary_df["Avg_Fitness"], label="Avg Fitness")
    ax.plot(summary_df["Generation"], summary_df["Max_Fitness"], label="Max Fitness")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title(f"Fitness over Generations (run {run_tag})")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(out_dir, "fitness_over_time.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"📈 Saved {out_path}")


def plot_traits(traits_df, out_dir, run_tag):
    if traits_df is None or traits_df.empty:
        return

    fig, ax = plt.subplots()
    g = traits_df["Generation"]

    if "NetCharge" in traits_df.columns:
        ax.plot(g, traits_df["NetCharge"], label="Net Charge")
    if "Hydrophobicity" in traits_df.columns:
        ax.plot(g, traits_df["Hydrophobicity"], label="Hydrophobicity")
    if "AggregationRisk" in traits_df.columns:
        ax.plot(g, traits_df["AggregationRisk"], label="Aggregation Risk")
    if "Realism" in traits_df.columns:
        ax.plot(g, traits_df["Realism"], label="Realism")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Value")
    ax.set_title(f"Trait Drift over Generations (run {run_tag})")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(out_dir, "traits_over_time.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"📊 Saved {out_path}")


def plot_amp_tox_stab(summary_df, out_dir, run_tag):
    if summary_df is None or summary_df.empty:
        return

    fig, ax = plt.subplots()
    g = summary_df["Generation"]

    if "Avg_AMP" in summary_df.columns:
        ax.plot(g, summary_df["Avg_AMP"], label="Avg AMP")
    if "Avg_Tox" in summary_df.columns:
        ax.plot(g, 1.0 - summary_df["Avg_Tox"], label="Avg (1 - Tox)")
    if "Avg_Stab" in summary_df.columns:
        ax.plot(g, summary_df["Avg_Stab"], label="Avg Stability")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Score")
    ax.set_title(f"AMP / Toxicity / Stability over Generations (run {run_tag})")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(out_dir, "amp_tox_stab_over_time.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"🧪 Saved {out_path}")


def plot_similarity(sim_df, out_dir, run_tag):
    if sim_df is None or sim_df.empty:
        return

    fig, ax = plt.subplots()
    ax.plot(sim_df["Generation"], sim_df["Similarity"])
    ax.set_xlabel("Generation")
    ax.set_ylabel("Avg Pairwise Similarity")
    ax.set_title(f"Population Similarity over Generations (run {run_tag})")
    fig.tight_layout()

    out_path = os.path.join(out_dir, "similarity_over_time.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"🧬 Saved {out_path}")


def copy_run_summary(summary_path, out_dir):
    if summary_path is None or not os.path.exists(summary_path):
        return
    with open(summary_path, "r") as f:
        text = f.read()
    out = os.path.join(out_dir, "run_summary_copy.txt")
    with open(out, "w") as f:
        f.write(text)
    print(f"📝 Copied run summary to {out}")


def main():
    print("🔍 Analyzing latest AMP evolution run...")

    run_tag = find_latest_run_tag()
    if run_tag is None:
        print("❌ No evolution runs found in data/evolution_histories.")
        print("   Make sure you've run evolveAGENT at least once.")
        return

    print(f"📂 Latest run detected: {run_tag}")

    out_dir = os.path.join(ANALYSIS_DIR, f"run_{run_tag}")
    ensure_dir(out_dir)

    data = load_run_data(run_tag)
    evol_df = data["evolution"]
    traits_df = data["traits"]
    sim_df = data["similarity"]

    gen_summary = summarize_generations(evol_df)
    if gen_summary is not None:
        gen_summary_path = os.path.join(out_dir, f"generation_summary_run_{run_tag}.csv")
        gen_summary.to_csv(gen_summary_path, index=False)
        print(f"💾 Saved generation summary to {gen_summary_path}")

    extract_top_peptides(evol_df, out_dir, run_tag, top_n=100)

    plot_fitness(gen_summary, out_dir, run_tag)
    plot_traits(traits_df, out_dir, run_tag)
    plot_amp_tox_stab(gen_summary, out_dir, run_tag)
    plot_hydromoment(gen_summary, out_dir, run_tag)
    plot_similarity(sim_df, out_dir, run_tag)

    copy_run_summary(data["summary_path"], out_dir)

    print("\n✅ Analysis complete.")
    print(f"   Check: {out_dir}")


if __name__ == "__main__":
    main()