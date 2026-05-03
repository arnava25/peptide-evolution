#!/usr/bin/env python3
import os
import glob
import math
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DATA_DIR = "data"
EVOL_HISTORY_DIR = os.path.join(BASE_DATA_DIR, "evolution_histories")
ANALYSIS_DIR = os.path.join(BASE_DATA_DIR, "analysis")


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def find_latest_run_tag():
    """
    Try to infer the latest run_tag.

    Priority:
      1) run_summary_*.txt in data/evolution_histories
      2) evolution_run_*.csv in data/evolution_histories

    Returns the run_tag string (YYYYMMDD_HHMM or similar) or None if not found.
    """
    candidates = []

    # 1) From run_summary_*.txt
    pattern_summary = os.path.join(EVOL_HISTORY_DIR, "run_summary_*.txt")
    for path in glob.glob(pattern_summary):
        tag = os.path.basename(path).replace("run_summary_", "").replace(".txt", "")
        mtime = os.path.getmtime(path)
        candidates.append((mtime, tag))

    # 2) From evolution_run_*.csv (fallback when summary is missing)
    pattern_csv = os.path.join(EVOL_HISTORY_DIR, "evolution_run_*.csv")
    for path in glob.glob(pattern_csv):
        tag = os.path.basename(path).replace("evolution_run_", "").replace(".csv", "")
        mtime = os.path.getmtime(path)
        candidates.append((mtime, tag))

    if not candidates:
        return None

    # most recently modified wins
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def load_csv_if_exists(path: str, **read_kwargs):
    if not os.path.exists(path):
        print(f"⚠️  Missing file: {path}")
        return None
    try:
        return pd.read_csv(path, **read_kwargs)
    except Exception as e:
        print(f"❌ Error reading {path}: {e}")
        return None



def load_agent_state(path: str):
    """
    Expected columns in agent_state_<RUN_TAG>.csv:
      Generation,amp,safety,stability,realism,novelty,parsimony,curiosity,z,r,...
    """
    df = load_csv_if_exists(path)
    if df is None or df.empty:
        return None
    return df


def load_action_log(path: str):
    df = load_csv_if_exists(path)
    if df is None or df.empty:
        return None
    return df


def load_pwm_error(path: str):
    df = load_csv_if_exists(path)
    if df is None or df.empty:
        return None
    return df


def load_novelty_entropy(nov_path: str, ent_path: str):
    nov = load_csv_if_exists(nov_path)
    ent = load_csv_if_exists(ent_path)
    return nov, ent


def load_mutation_history(path: str):
    """
    mutation_rate_history_<RUN_TAG>.txt
    same parsing, but per-run file.
    """
    if not os.path.exists(path):
        print(f"⚠️  Missing mutation history file: {path}")
        return None

    gens, muts = [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("---") or line.startswith("Population"):
                continue
            if line.startswith("Generation"):
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                g = int(parts[0])
                m = float(parts[1])
            except ValueError:
                continue
            gens.append(g)
            muts.append(m)

    if not gens:
        print(f"⚠️  No parsed mutation entries found in {path}")
        return None

    return pd.DataFrame({"Generation": gens, "MutationRate": muts})


def load_fitness_stats(path: str):
    df = load_csv_if_exists(path)
    if df is None or df.empty:
        return None
    return df




# ===========================
# Plotting helpers
# ===========================
def plot_agent_weights(agent_df, out_dir, run_tag):
    """
    Plot each motive's weight over generations.
    """
    if agent_df is None or agent_df.empty:
        return

    if "Generation" not in agent_df.columns:
        print("⚠️  agent_state file missing 'Generation' column; skipping agent weights plot.")
        return

    # Only the canonical 7 motives
    canonical_motives = ["amp","safety","stability","realism","novelty","parsimony","curiosity"]
    motives = [m for m in canonical_motives if m in agent_df.columns]
    if not motives:
        print("⚠️  No motive columns found in agent_state; skipping motive weight plot.")
        return

    fig, ax = plt.subplots()
    for m in motives:
        ax.plot(agent_df["Generation"], agent_df[m], label=m)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Weight")
    ax.set_title(f"Agent Motive Weights over Generations (run {run_tag})")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(out_dir, "agent_motive_weights.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"🧠 Saved {out_path}")


def plot_agent_z(agent_df, out_dir, run_tag):
    """
    Plot chaotic latent z over generations.
    """
    if agent_df is None or agent_df.empty:
        return

    if "Generation" not in agent_df.columns or "z" not in agent_df.columns:
        print("⚠️  agent_state.csv missing 'Generation' or 'z'; skipping z plot.")
        return

    fig, ax = plt.subplots()
    ax.plot(agent_df["Generation"], agent_df["z"])

    ax.set_xlabel("Generation")
    ax.set_ylabel("z (logistic latent)")
    ax.set_title(f"Agent Chaotic Latent z over Generations (run {run_tag})")
    fig.tight_layout()

    out_path = os.path.join(out_dir, "agent_z_over_time.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"🌀 Saved {out_path}")


def plot_action_mix(action_df, out_dir, run_tag):
    """
    Plot counts of crossover vs mutate per generation.
    """
    if action_df is None or action_df.empty:
        return

    required = {"Generation", "ActionType"}
    if not required.issubset(action_df.columns):
        print("⚠️  action_log.csv missing required columns; skipping action mix plot.")
        return

    counts = (
        action_df.groupby(["Generation", "ActionType"])
        .size()
        .reset_index(name="Count")
    )

    pivot = counts.pivot(index="Generation", columns="ActionType", values="Count").fillna(0.0)
    fig, ax = plt.subplots()

    for col in pivot.columns:
        ax.plot(pivot.index, pivot[col], label=col)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Count per Generation")
    ax.set_title(f"Agent Actions per Generation (run {run_tag})")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(out_dir, "agent_actions_over_time.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"🧬 Saved {out_path}")


def plot_pwm_error(pwm_df, out_dir, run_tag):
    """
    Plot total and per-output world-model error.
    """
    if pwm_df is None or pwm_df.empty:
        return

    if "Generation" not in pwm_df.columns or "TotalError" not in pwm_df.columns:
        print("⚠️  world_model_error.csv missing required columns; skipping PWM error plot.")
        return

    fig, ax = plt.subplots()
    ax.plot(pwm_df["Generation"], pwm_df["TotalError"], label="Total Error")

    for col in ["AMP_Error", "Tox_Error", "Stab_Error", "Realism_Error"]:
        if col in pwm_df.columns:
            ax.plot(pwm_df["Generation"], pwm_df[col], label=col)

    ax.set_xlabel("Generation")
    ax.set_ylabel("MAE")
    ax.set_title(f"Peptide World Model Error over Generations (run {run_tag})")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(out_dir, "pwm_error_over_time.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"🔮 Saved {out_path}")


def plot_novelty_entropy(nov_df, ent_df, out_dir, run_tag):
    """
    Combine novelty (avg similarity) and entropy into one plot
    (two y-axes) for intuition.
    """
    if nov_df is None and ent_df is None:
        return

    fig, ax1 = plt.subplots()

    if nov_df is not None and not nov_df.empty and "Generation" in nov_df.columns:
        ax1.plot(nov_df["Generation"], nov_df["AvgSimilarity"], label="Avg Similarity")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Avg Similarity", color="tab:blue")
    else:
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Avg Similarity")

    ax2 = ax1.twinx()
    if ent_df is not None and not ent_df.empty and "Generation" in ent_df.columns:
        ax2.plot(ent_df["Generation"], ent_df["MeanEntropy"], label="Mean Entropy", linestyle="--")
        ax2.set_ylabel("Mean Entropy", color="tab:orange")
    else:
        ax2.set_ylabel("Mean Entropy")

    fig.suptitle(f"Novelty & Entropy over Generations (run {run_tag})")
    fig.tight_layout()

    out_path = os.path.join(out_dir, "novelty_entropy_over_time.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"🌌 Saved {out_path}")


def plot_mutation_rate(mut_df, fit_df, out_dir, run_tag):
    """
    Plot mutation rate over generation, optionally overlay AvgFitness
    (to see when mutation is high vs low).
    """
    if mut_df is None or mut_df.empty:
        return

    fig, ax1 = plt.subplots()

    ax1.plot(mut_df["Generation"], mut_df["MutationRate"], label="Mutation Rate")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Mutation Rate", color="tab:blue")

    if fit_df is not None and not fit_df.empty and "AvgFitness" in fit_df.columns:
        ax2 = ax1.twinx()
        ax2.plot(fit_df["Generation"], fit_df["AvgFitness"], label="Avg Fitness", linestyle="--")
        ax2.set_ylabel("Avg Fitness", color="tab:orange")

    fig.suptitle(f"Mutation Rate (and Fitness) over Generations (run {run_tag})")
    fig.tight_layout()

    out_path = os.path.join(out_dir, "mutation_rate_over_time.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"🧬 Saved {out_path}")


def main():
    print("🧠 Analyzing cognitive agent + PWM behavior...")

    run_tag = find_latest_run_tag()
    if run_tag is None:
        # Fallback: if we at least have agent logs, still run with a generic tag
        agent_files = [
            AGENT_STATE_FILE,
            ACTION_LOG_FILE,
            PWM_ERROR_FILE,
            NOVELTY_FILE,
            ENTROPY_FILE,
            MUT_HISTORY_FILE,
            FITNESS_STATS_FILE,
        ]
        if any(os.path.exists(p) for p in agent_files):
            run_tag = "untagged"
            print("⚠️ No run_summary_*.txt or evolution_run_*.csv found.")
            print("   Proceeding with tag 'untagged' based on existing agent logs.")
        else:
            print("❌ No run_summary_*.txt or evolution_run_*.csv found, and no agent logs present.")
            print("   Run evolveAGENT.py at least once before using this.")
            return


    print(f"📂 Latest run detected: {run_tag}")
    out_dir = os.path.join(ANALYSIS_DIR, f"run_{run_tag}", "agent")
    ensure_dir(out_dir)

    # 🔗 Build run-specific log paths
    agent_state_file = os.path.join(BASE_DATA_DIR, f"agent_state_{run_tag}.csv")
    action_log_file  = os.path.join(BASE_DATA_DIR, f"action_log_{run_tag}.csv")
    pwm_error_file   = os.path.join(BASE_DATA_DIR, f"world_model_error_{run_tag}.csv")
    novelty_file     = os.path.join(BASE_DATA_DIR, f"novelty_stats_{run_tag}.csv")
    entropy_file     = os.path.join(BASE_DATA_DIR, f"entropy_stats_{run_tag}.csv")
    mut_history_file = os.path.join(BASE_DATA_DIR, f"mutation_rate_history_{run_tag}.txt")
    fitness_file     = os.path.join(BASE_DATA_DIR, f"fitness_stats_{run_tag}.csv")
    reason_file      = os.path.join(BASE_DATA_DIR, f"agent_reasoning_{run_tag}.csv")  # NEW

    # Load everything
    agent_df = load_agent_state(agent_state_file)
    action_df = load_action_log(action_log_file)
    pwm_df    = load_pwm_error(pwm_error_file)
    nov_df, ent_df = load_novelty_entropy(novelty_file, entropy_file)
    mut_df   = load_mutation_history(mut_history_file)
    fit_df   = load_fitness_stats(fitness_file)
    reason_df = load_csv_if_exists(reason_file)  # NEW (optional)


    # Save raw agent + PWM CSV excerpts for inspection
    if agent_df is not None:
        agent_copy_path = os.path.join(out_dir, f"agent_state_run_{run_tag}.csv")
        agent_df.to_csv(agent_copy_path, index=False)
        print(f"💾 Copied agent_state.csv to {agent_copy_path}")
    if pwm_df is not None:
        pwm_copy_path = os.path.join(out_dir, f"pwm_error_run_{run_tag}.csv")
        pwm_df.to_csv(pwm_copy_path, index=False)
        print(f"💾 Copied world_model_error.csv to {pwm_copy_path}")
    if reason_df is not None:
        reason_copy = os.path.join(out_dir, f"agent_reasoning_run_{run_tag}.csv")
        reason_df.to_csv(reason_copy, index=False)
        print(f"💾 Copied {os.path.basename(reason_file)} to {reason_copy}")

    # Plots
    plot_agent_weights(agent_df, out_dir, run_tag)
    plot_agent_z(agent_df, out_dir, run_tag)
    plot_action_mix(action_df, out_dir, run_tag)
    plot_pwm_error(pwm_df, out_dir, run_tag)
    plot_novelty_entropy(nov_df, ent_df, out_dir, run_tag)
    plot_mutation_rate(mut_df, fit_df, out_dir, run_tag)

    print("\n✅ Agent / cognition analysis complete.")
    print(f"   Check: {out_dir}")
    print("   You should see agent motive weights, actions, PWM error, "
          "novelty/entropy, and mutation rate plots.")


if __name__ == "__main__":
    main()

