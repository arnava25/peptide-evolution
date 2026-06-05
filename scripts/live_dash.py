"""
evolveAGENT Live Dashboard
"""

import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DEFAULT_INTERVAL = 30

ABANDONMENT_TRIGGER = 150
RESTART_TRIGGER = 400
EARLY_STOP = 550

def latest(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def find_files():
    files = {}
    files["fitness"]     = latest(os.path.join(DATA_DIR, "fitness_stats_*.csv"))
    files["similarity"]  = latest(os.path.join(DATA_DIR, "similarity_logs", "similarity_log_*.csv"))
    files["traits"]      = latest(os.path.join(DATA_DIR, "trait_stats", "trait_stats_*.csv"))
    files["novelty"]     = latest(os.path.join(DATA_DIR, "novelty_stats_*.csv"))
    files["mutation"]    = latest(os.path.join(DATA_DIR, "mutation_rate_history_*.txt"))
    files["action"]      = latest(os.path.join(DATA_DIR, "action_log_*.csv"))
    files["pwm_error"]   = latest(os.path.join(DATA_DIR, "world_model_error_*.csv"))
    files["peak_events"] = latest(os.path.join(DATA_DIR, "peak_events_*.csv"))
    files["top_peps"]    = latest(os.path.join(DATA_DIR, "top_peptides_gen_*.csv"))
    files["grid"] = latest(os.path.join(DATA_DIR, "map_elites_grid_gen_*.csv"))
    if files["grid"] is None:
        files["grid"] = latest(os.path.join(DATA_DIR, "map_elites_grid_isl0_gen_*.csv"))
    return files

def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def load_mutation(path):
    gens, rates = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("---") or line.startswith("Pop") or line.startswith("Generation"):
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    gens.append(int(parts[0]))
                    rates.append(float(parts[1]))
                except ValueError:
                    continue
    return pd.DataFrame({"Generation": gens, "MutationRate": rates})

def load_action_log(path):
    df = load_csv(path)
    return df.groupby(["Generation", "ActionType"]).size().reset_index(name="Count")

def count_map_elites_cells(data_dir):
    shared = glob.glob(os.path.join(data_dir, "map_elites_grid_gen_*.csv"))
    if shared:
        gen_cells = {}
        for f in shared:
            base = os.path.basename(f)
            try:
                gen_str = base.split("_gen_")[1].split("_")[0]
                gen = int(gen_str)
                df = pd.read_csv(f)
                gen_cells[gen] = df[["charge_bin","hydro_bin"]].drop_duplicates().shape[0]
            except Exception:
                continue
        if gen_cells:
            return pd.DataFrame(sorted(gen_cells.items()), columns=["Generation","CellsFilled"])
    island = glob.glob(os.path.join(data_dir, "map_elites_grid_isl*_gen_*.csv"))
    if not island:
        return pd.DataFrame(columns=["Generation","CellsFilled"])
    gen_cells = {}
    for f in island:
        base = os.path.basename(f)
        try:
            gen_str = base.split("_gen_")[1].split("_")[0]
            gen = int(gen_str)
            df = pd.read_csv(f)
            n = df[["charge_bin","hydro_bin"]].drop_duplicates().shape[0]
            gen_cells[gen] = gen_cells.get(gen, 0) + n
        except Exception:
            continue
    if not gen_cells:
        return pd.DataFrame(columns=["Generation","CellsFilled"])
    return pd.DataFrame(sorted(gen_cells.items()), columns=["Generation","CellsFilled"])

def compute_stagnation(fitness_df, threshold=0.005):
    if "AvgFitness" not in fitness_df.columns or "MaxFitness" not in fitness_df.columns:
        return None, None
    avg = fitness_df["AvgFitness"].values
    mx  = fitness_df["MaxFitness"].values
    gens = fitness_df["Generation"].values
    stagnant = np.zeros(len(avg), dtype=int)
    best_avg = best_max = count = 0
    for i in range(len(avg)):
        if (mx[i] > best_max + threshold) or (avg[i] > best_avg + threshold):
            best_max = max(best_max, mx[i])
            best_avg = max(best_avg, avg[i])
            count = 0
        else:
            count += 1
        stagnant[i] = count
    return stagnant, gens

def find_niche_events(fitness_df):
    return []

DARK_BG   = "#0f1117"
PANEL_BG  = "#1a1d27"
GRID_CLR  = "#2a2d3a"
TEXT_CLR  = "#c8ccd8"
ACC1      = "#5b8dee"
ACC2      = "#f0a500"
ACC3      = "#50c878"
ACC4      = "#e05c5c"
ACC5      = "#bb86fc"
ACC6      = "#38b2ac"
NICHE_CLR = "#ff6b6b"
STAG_CLR  = "#ff9f43"

def style_ax(ax, title):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color=TEXT_CLR, fontsize=9, pad=6, loc="left")
    ax.tick_params(colors=TEXT_CLR, labelsize=7)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
    ax.grid(color=GRID_CLR, linewidth=0.5, linestyle="--", alpha=0.6)
    ax.set_xlabel("Generation", color=TEXT_CLR, fontsize=7)

def autoscale_y(ax, pad=0.05):
    ax.relim(); ax.autoscale_view()
    ymin, ymax = ax.get_ylim()
    span = max(ymax - ymin, 1e-6)
    ax.set_ylim(ymin - pad*span, ymax + pad*span)

def add_niche_markers(ax, events):
    for g in events:
        ax.axvline(g, color=NICHE_CLR, linewidth=0.8, linestyle=":", alpha=0.7)

def build_figure(files, window=0):
    def trim(df, gen_col="Generation"):
        if window > 0 and gen_col in df.columns:
            cutoff = df[gen_col].max() - window
            return df[df[gen_col] >= cutoff].copy()
        return df

    fig = plt.figure(figsize=(18, 17), facecolor=DARK_BG)
    fig.suptitle(
        f"evolveAGENT  ·  Live Dashboard  ·  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        color=TEXT_CLR, fontsize=11, y=0.995
    )

    gs = gridspec.GridSpec(6, 3, figure=fig, hspace=0.65, wspace=0.35,
                           left=0.05, right=0.97, top=0.94, bottom=0.03)

    ax_fit    = fig.add_subplot(gs[0, :2])
    ax_mut    = fig.add_subplot(gs[0, 2])
    ax_sim    = fig.add_subplot(gs[1, 0])
    ax_traits = fig.add_subplot(gs[1, 1])
    ax_map    = fig.add_subplot(gs[1, 2])
    ax_stag   = fig.add_subplot(gs[2, :2])
    ax_pwm    = fig.add_subplot(gs[2, 2])
    ax_action = fig.add_subplot(gs[3, :])
    ax_peaks  = fig.add_subplot(gs[4, :])
    ax_top    = fig.add_subplot(gs[5, :])

    niche_events = []
    fitness_df = None
    stag_current = 0
    best_fitness_ever = 0.0
    current_gen = 0
    me_filled = 0
    avg_sim_current = 0.0
    peak_events_df = None

    if files.get("peak_events"):
        try:
            _pe = load_csv(files["peak_events"])
            if not _pe.empty and "EventType" in _pe.columns:
                peak_events_df = _pe
        except Exception:
            pass

    # ── Fitness ───────────────────────────────────────────────────────────────
    if files["fitness"]:
        try:
            fitness_df = trim(load_csv(files["fitness"]))
            niche_events = find_niche_events(fitness_df)
            if "AvgFitness" in fitness_df.columns:
                ax_fit.plot(fitness_df["Generation"], fitness_df["AvgFitness"],
                            color=ACC1, linewidth=1.4, label="Avg Fitness")
                current_gen = int(fitness_df["Generation"].max())
            if "MaxFitness" in fitness_df.columns:
                ax_fit.plot(fitness_df["Generation"], fitness_df["MaxFitness"],
                            color=ACC2, linewidth=1.2, linestyle="--", label="Max Fitness")
                best_fitness_ever = float(fitness_df["MaxFitness"].max())
            if "StdFitness" in fitness_df.columns and "AvgFitness" in fitness_df.columns:
                ax_fit.fill_between(fitness_df["Generation"],
                    fitness_df["AvgFitness"] - fitness_df["StdFitness"],
                    fitness_df["AvgFitness"] + fitness_df["StdFitness"],
                    alpha=0.12, color=ACC1)
            add_niche_markers(ax_fit, niche_events)
            if peak_events_df is not None:
                ymax_fit = fitness_df["MaxFitness"].max() if "MaxFitness" in fitness_df.columns else 1.0
                for _, row in peak_events_df.iterrows():
                    g = row["Generation"]
                    if row["EventType"] == "abandonment":
                        ax_fit.axvline(g, color=ACC5, linewidth=1.5, linestyle="-.", alpha=0.9)
                        ax_fit.annotate("abandon", xy=(g, ymax_fit*0.96),
                                       color=ACC5, fontsize=6, ha="center", rotation=90)
                    elif row["EventType"] == "restart":
                        ax_fit.axvline(g, color=ACC3, linewidth=1.5, linestyle="-.", alpha=0.9)
                        ax_fit.annotate("restart", xy=(g, ymax_fit*0.96),
                                       color=ACC3, fontsize=6, ha="center", rotation=90)
            ax_fit.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=TEXT_CLR,
                          edgecolor=GRID_CLR, loc="lower right")
            if "AvgFitness" in fitness_df.columns and "MaxFitness" in fitness_df.columns:
                ymin = fitness_df["AvgFitness"].min() * 0.98
                ymax = fitness_df["MaxFitness"].max() * 1.02
                ax_fit.set_ylim(ymin, ymax)
        except Exception as e:
            ax_fit.text(0.5, 0.5, f"Error: {e}", transform=ax_fit.transAxes, color=TEXT_CLR, ha="center")
    else:
        ax_fit.text(0.5, 0.5, "fitness_stats not found", transform=ax_fit.transAxes, color=TEXT_CLR, ha="center")
    style_ax(ax_fit, "Fitness over Generations")

    # ── Mutation Rate ─────────────────────────────────────────────────────────
    if files["mutation"]:
        try:
            df = trim(load_mutation(files["mutation"]), gen_col="Generation")
            if not df.empty:
                ax_mut.plot(df["Generation"], df["MutationRate"], color=ACC4, linewidth=1.2)
                add_niche_markers(ax_mut, niche_events)
                autoscale_y(ax_mut)
        except Exception as e:
            ax_mut.text(0.5, 0.5, f"Error: {e}", transform=ax_mut.transAxes, color=TEXT_CLR, ha="center")
    else:
        ax_mut.text(0.5, 0.5, "not found", transform=ax_mut.transAxes, color=TEXT_CLR, ha="center")
    style_ax(ax_mut, "Mutation Rate")

    # ── Similarity ────────────────────────────────────────────────────────────
    if files["similarity"]:
        try:
            df = trim(load_csv(files["similarity"]))
            col = "Similarity" if "Similarity" in df.columns else "AvgSimilarity"
            if col in df.columns:
                avg_sim_current = float(df[col].iloc[-1])
                ax_sim.plot(df["Generation"], df[col], color=ACC3, linewidth=1.2)
                ax_sim.axhline(0.70, color=NICHE_CLR, linewidth=0.7, linestyle=":", alpha=0.6, label="Niche trigger")
                ax_sim.legend(fontsize=6, facecolor=PANEL_BG, labelcolor=TEXT_CLR, edgecolor=GRID_CLR)
                ax_sim.set_ylim(0, min(1.0, df[col].max() * 1.3))
                add_niche_markers(ax_sim, niche_events)
        except Exception as e:
            ax_sim.text(0.5, 0.5, f"Error: {e}", transform=ax_sim.transAxes, color=TEXT_CLR, ha="center")
    else:
        ax_sim.text(0.5, 0.5, "not found", transform=ax_sim.transAxes, color=TEXT_CLR, ha="center")
    style_ax(ax_sim, "Avg Pairwise Similarity")

    # ── Traits ────────────────────────────────────────────────────────────────
    if files["traits"]:
        try:
            df = trim(load_csv(files["traits"]))
            if "Realism" in df.columns:
                ax_traits.plot(df["Generation"], df["Realism"], color=ACC6, linewidth=1.1, label="Realism")
            if "Hydrophobicity" in df.columns:
                ax_traits.plot(df["Generation"], df["Hydrophobicity"], color=ACC2, linewidth=1.1, label="Hydrophobicity")
            if "AggregationRisk" in df.columns:
                ax_traits.plot(df["Generation"], df["AggregationRisk"], color=ACC4, linewidth=1.1, linestyle="--", label="Agg Risk")
            ax_traits.legend(fontsize=6, facecolor=PANEL_BG, labelcolor=TEXT_CLR, edgecolor=GRID_CLR)
            autoscale_y(ax_traits)
            add_niche_markers(ax_traits, niche_events)
        except Exception as e:
            ax_traits.text(0.5, 0.5, f"Error: {e}", transform=ax_traits.transAxes, color=TEXT_CLR, ha="center")
    else:
        ax_traits.text(0.5, 0.5, "not found", transform=ax_traits.transAxes, color=TEXT_CLR, ha="center")
    style_ax(ax_traits, "Biochemical Traits")

    # ── MAP-Elites Cells ──────────────────────────────────────────────────────
    try:
        df_map = count_map_elites_cells(DATA_DIR)
        if not df_map.empty:
            me_filled = int(df_map["CellsFilled"].iloc[-1])
            ax_map.plot(df_map["Generation"], df_map["CellsFilled"],
                        color=ACC5, linewidth=1.3, marker="o", markersize=4)
            ax_map.axhline(420, color=TEXT_CLR, linewidth=0.6, linestyle=":", alpha=0.5, label="Max (420)")
            ax_map.legend(fontsize=6, facecolor=PANEL_BG, labelcolor=TEXT_CLR, edgecolor=GRID_CLR)
            ax_map.set_ylim(0, 450)
            last_gen = df_map["Generation"].iloc[-1]
            ax_map.annotate(f"{me_filled}", xy=(last_gen, me_filled),
                           xytext=(5, 5), textcoords="offset points", color=ACC5, fontsize=8)
        else:
            ax_map.text(0.5, 0.5, "No grid files yet\n(saved every 100 gens)",
                       transform=ax_map.transAxes, color=TEXT_CLR, ha="center", fontsize=8)
    except Exception as e:
        ax_map.text(0.5, 0.5, f"Error: {e}", transform=ax_map.transAxes, color=TEXT_CLR, ha="center")
    style_ax(ax_map, "MAP-Elites Cells Filled / 420")

    # ── Stagnation Counter ────────────────────────────────────────────────────
    if fitness_df is not None:
        try:
            stag_vals, gens = compute_stagnation(fitness_df)
            if stag_vals is not None:
                stag_current = int(stag_vals[-1])
                ax_stag.fill_between(gens, 0, stag_vals, alpha=0.25, color=STAG_CLR)
                ax_stag.plot(gens, stag_vals, color=STAG_CLR, linewidth=1.3, label="Stagnant gens")
                ax_stag.axhline(ABANDONMENT_TRIGGER, color=ACC4, linewidth=0.8, linestyle="--", alpha=0.8,
                                label=f"Abandonment ({ABANDONMENT_TRIGGER})")
                ax_stag.axhline(RESTART_TRIGGER, color=ACC5, linewidth=0.8, linestyle="--", alpha=0.8,
                                label=f"Restart ({RESTART_TRIGGER})")
                ax_stag.axhline(EARLY_STOP, color=NICHE_CLR, linewidth=0.8, linestyle="--", alpha=0.8,
                                label=f"Early stop ({EARLY_STOP})")

                gens_to_abandon = ABANDONMENT_TRIGGER - stag_current
                gens_to_restart = RESTART_TRIGGER - stag_current

                if stag_current >= RESTART_TRIGGER:
                    countdown_text = f"Current: {stag_current}  |  RESTART ZONE"
                    countdown_color = ACC5
                elif stag_current >= ABANDONMENT_TRIGGER:
                    countdown_text = f"Current: {stag_current}  |  ABANDONMENT ZONE  ({gens_to_restart} to restart)"
                    countdown_color = ACC4
                elif gens_to_abandon <= 30:
                    countdown_text = f"Current: {stag_current}  |  {gens_to_abandon} gens to ABANDONMENT"
                    countdown_color = ACC4
                elif gens_to_abandon <= 60:
                    countdown_text = f"Current: {stag_current}  |  {gens_to_abandon} gens to abandonment"
                    countdown_color = ACC2
                else:
                    countdown_text = f"Current: {stag_current} stagnant gens"
                    countdown_color = STAG_CLR

                ax_stag.text(0.99, 0.92, countdown_text,
                             transform=ax_stag.transAxes, color=countdown_color, fontsize=8,
                             ha="right", va="top",
                             bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL_BG,
                                       edgecolor=countdown_color, alpha=0.8))

                ax_stag.legend(fontsize=6, facecolor=PANEL_BG, labelcolor=TEXT_CLR,
                               edgecolor=GRID_CLR, ncol=4, loc="upper left")
                ax_stag.set_ylim(0, max(600, stag_vals.max() * 1.1))
                add_niche_markers(ax_stag, niche_events)

                if peak_events_df is not None:
                    for _, row in peak_events_df.iterrows():
                        g = row["Generation"]
                        clr = ACC5 if row["EventType"] == "abandonment" else ACC3
                        ax_stag.axvline(g, color=clr, linewidth=1.0, linestyle="-.", alpha=0.7)

        except Exception as e:
            ax_stag.text(0.5, 0.5, f"Error: {e}", transform=ax_stag.transAxes, color=TEXT_CLR, ha="center")
    else:
        ax_stag.text(0.5, 0.5, "Waiting for fitness data...", transform=ax_stag.transAxes, color=TEXT_CLR, ha="center")
    style_ax(ax_stag, "Stagnation Counter  (abandonment @ 150 | restart @ 400 | stop @ 550)")

    # ── World Model Error ─────────────────────────────────────────────────────
    if files.get("pwm_error"):
        try:
            df = trim(load_csv(files["pwm_error"]))
            if "TotalError" in df.columns:
                ax_pwm.plot(df["Generation"], df["TotalError"], color=ACC5, linewidth=1.2, label="Total")
            if "AMP_Error" in df.columns:
                ax_pwm.plot(df["Generation"], df["AMP_Error"], color=ACC2, linewidth=0.8, linestyle="--", label="AMP")
            if "Tox_Error" in df.columns:
                ax_pwm.plot(df["Generation"], df["Tox_Error"], color=ACC4, linewidth=0.8, linestyle="--", label="Tox")
            ax_pwm.axhline(0.10, color=TEXT_CLR, linewidth=0.6, linestyle=":", alpha=0.5, label="Trust threshold")
            ax_pwm.legend(fontsize=6, facecolor=PANEL_BG, labelcolor=TEXT_CLR, edgecolor=GRID_CLR, ncol=2)
            autoscale_y(ax_pwm)
            add_niche_markers(ax_pwm, niche_events)
        except Exception as e:
            ax_pwm.text(0.5, 0.5, f"Error: {e}", transform=ax_pwm.transAxes, color=TEXT_CLR, ha="center")
    else:
        ax_pwm.text(0.5, 0.5, "PWM error not found", transform=ax_pwm.transAxes, color=TEXT_CLR, ha="center")
    style_ax(ax_pwm, "World Model Error")

    # ── Action Breakdown ──────────────────────────────────────────────────────
    if files["action"]:
        try:
            df = trim(load_action_log(files["action"]))
            colors_list = [ACC1, ACC2, ACC3, ACC4, ACC5, ACC6, "#f687b3", "#fbd38d"]
            gens = sorted(df["Generation"].unique())
            gen_df = df.pivot_table(index="Generation", columns="ActionType", values="Count", fill_value=0)
            gen_df = gen_df.reindex(gens).fillna(0)
            bottom = None
            for idx, col in enumerate(gen_df.columns):
                clr = colors_list[idx % len(colors_list)]
                vals = gen_df[col].values
                if bottom is None:
                    ax_action.fill_between(gens, 0, vals, alpha=0.7, color=clr, label=col)
                    bottom = vals.copy()
                else:
                    ax_action.fill_between(gens, bottom, bottom + vals, alpha=0.7, color=clr, label=col)
                    bottom = bottom + vals
            ax_action.legend(fontsize=6, facecolor=PANEL_BG, labelcolor=TEXT_CLR,
                             edgecolor=GRID_CLR, ncol=4, loc="upper left")
            add_niche_markers(ax_action, niche_events)
        except Exception as e:
            ax_action.text(0.5, 0.5, f"Error: {e}", transform=ax_action.transAxes, color=TEXT_CLR, ha="center")
    else:
        ax_action.text(0.5, 0.5, "action log not found", transform=ax_action.transAxes, color=TEXT_CLR, ha="center")
    style_ax(ax_action, "Action Type Breakdown per Generation")

    # ── Peak Events Table ─────────────────────────────────────────────────────
    ax_peaks.set_facecolor(PANEL_BG)
    ax_peaks.axis("off")
    for spine in ax_peaks.spines.values():
        spine.set_edgecolor(GRID_CLR)
    ax_peaks.set_title("Peak Events Log  (abandonment / restart history)",
                       color=TEXT_CLR, fontsize=9, pad=6, loc="left")

    if peak_events_df is not None and not peak_events_df.empty:
        try:
            cols_have = [c for c in ["Generation", "EventType", "PeakFitness", "PeakNumber", "Note"]
                         if c in peak_events_df.columns]
            pe_display = peak_events_df[cols_have].copy()

            if "Note" in pe_display.columns:
                pe_display["Note"] = pe_display["Note"].apply(
                    lambda x: str(x)[:50] + "..." if len(str(x)) > 50 else str(x))
            if "PeakFitness" in pe_display.columns:
                pe_display["PeakFitness"] = pe_display["PeakFitness"].apply(
                    lambda x: f"{float(x):.4f}" if str(x) not in ("nan", "") else "-")

            col_labels = ["Gen", "Event", "Fitness", "Peak #", "Note"][:len(cols_have)]

            table = ax_peaks.table(
                cellText=pe_display.values,
                colLabels=col_labels,
                cellLoc="center",
                loc="center",
                bbox=[0.0, 0.05, 1.0, 0.85]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            for j in range(len(cols_have)):
                cell = table[0, j]
                cell.set_facecolor("#252839")
                cell.set_text_props(color=ACC2, fontweight="bold")
                cell.set_edgecolor(GRID_CLR)
            for i in range(1, len(pe_display) + 1):
                event_type = str(pe_display.iloc[i-1]["EventType"]) if "EventType" in pe_display.columns else ""
                row_color = ACC5 if event_type == "abandonment" else (ACC3 if event_type == "restart" else TEXT_CLR)
                for j in range(len(cols_have)):
                    cell = table[i, j]
                    cell.set_facecolor("#1a1d27" if i % 2 == 0 else "#1f2233")
                    cell.set_text_props(color=row_color if j == 1 else TEXT_CLR)
                    cell.set_edgecolor(GRID_CLR)
        except Exception as e:
            ax_peaks.text(0.5, 0.5, f"Peak events error: {e}",
                         transform=ax_peaks.transAxes, color=TEXT_CLR, ha="center")
    else:
        ax_peaks.text(0.5, 0.65, "No peak events yet",
                     transform=ax_peaks.transAxes, color=TEXT_CLR, ha="center", fontsize=10)
        ax_peaks.text(0.5, 0.50,
                     "Abandonment fires at 150 stagnant gens  |  Restart fires at 400 stagnant gens",
                     transform=ax_peaks.transAxes, color=TEXT_CLR, ha="center", fontsize=8, alpha=0.6)
        if stag_current > 0:
            pct = min(stag_current / ABANDONMENT_TRIGGER * 100, 100)
            bar_color = (ACC4 if stag_current >= ABANDONMENT_TRIGGER - 30
                         else ACC2 if stag_current >= ABANDONMENT_TRIGGER - 60
                         else STAG_CLR)
            ax_peaks.text(0.5, 0.32,
                         f"Progress to abandonment:  {stag_current} / {ABANDONMENT_TRIGGER}  ({pct:.0f}%)",
                         transform=ax_peaks.transAxes, color=bar_color,
                         ha="center", fontsize=9, fontweight="bold")

    # ── Top Peptides Table ────────────────────────────────────────────────────
    ax_top.set_facecolor(PANEL_BG)
    ax_top.axis("off")
    for spine in ax_top.spines.values():
        spine.set_edgecolor(GRID_CLR)

    if files.get("top_peps"):
        try:
            df = load_csv(files["top_peps"])
            if not df.empty:
                cols_want = ["Peptide", "Fitness_Score", "AMP_Score",
                             "Toxicity_Score", "Stability_Score",
                             "Net_Charge", "Hydrophobicity", "MIC_Score"]
                cols_have = [c for c in cols_want if c in df.columns]
                display = df.sort_values("Fitness_Score", ascending=False).head(5)[cols_have].copy()
                for c in cols_have[1:]:
                    display[c] = display[c].apply(lambda x: f"{float(x):.3f}")
                col_labels = [c.replace("_Score","").replace("_"," ") for c in cols_have]
                table = ax_top.table(
                    cellText=display.values,
                    colLabels=col_labels,
                    cellLoc="center",
                    loc="center",
                    bbox=[0.0, 0.0, 1.0, 1.0]
                )
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                for j in range(len(cols_have)):
                    cell = table[0, j]
                    cell.set_facecolor("#252839")
                    cell.set_text_props(color=ACC2, fontweight="bold")
                    cell.set_edgecolor(GRID_CLR)
                for i in range(1, len(display) + 1):
                    for j in range(len(cols_have)):
                        cell = table[i, j]
                        cell.set_facecolor("#1a1d27" if i % 2 == 0 else "#1f2233")
                        cell.set_text_props(color=ACC3 if j == 1 else TEXT_CLR)
                        cell.set_edgecolor(GRID_CLR)
                fname = os.path.basename(files["top_peps"])
                try:
                    gen_label = fname.split("gen_")[1].split("_")[0].lstrip("0") or "0"
                except Exception:
                    gen_label = "?"
                ax_top.set_title(f"Top Peptides — gen {gen_label} snapshot  (updates every 50 gens)",
                                color=TEXT_CLR, fontsize=9, pad=6, loc="left")
        except Exception as e:
            ax_top.text(0.5, 0.5, f"Top peptides error: {e}",
                       transform=ax_top.transAxes, color=TEXT_CLR, ha="center")
    else:
        ax_top.text(0.5, 0.5, "No top peptides snapshot yet  (saved every 50 gens)",
                   transform=ax_top.transAxes, color=TEXT_CLR, ha="center", fontsize=9)

    # ── Summary Stats Bar ─────────────────────────────────────────────────────
    run_tag = ""
    if files["fitness"]:
        try:
            run_tag = os.path.basename(files["fitness"]).replace("fitness_stats_","").replace(".csv","")
        except Exception:
            pass

    n_peaks = len(peak_events_df) if peak_events_df is not None and not peak_events_df.empty else 0
    peak_summary = f"{n_peaks} fired" if n_peaks > 0 else "none yet"

    stats = [
        ("Generation", str(current_gen)),
        ("Best Fitness", f"{best_fitness_ever:.4f}"),
        ("Stagnant Gens", f"{stag_current}  /  150"),
        ("MAP Cells", f"{me_filled}  /  420"),
        ("Peak Events", peak_summary),
        ("Avg Similarity", f"{avg_sim_current:.4f}"),
        ("Run", run_tag),
    ]

    x_positions = np.linspace(0.03, 0.97, len(stats))
    for (label, value), x in zip(stats, x_positions):
        fig.text(x, 0.963, label, color=TEXT_CLR, fontsize=7, ha="center", va="bottom", alpha=0.55)
        val_color = ACC4 if (label == "Stagnant Gens" and stag_current >= ABANDONMENT_TRIGGER - 30) else ACC2
        fig.text(x, 0.954, value, color=val_color, fontsize=9, ha="center", va="bottom", fontweight="bold")

    fig.add_artist(plt.Line2D([0.03, 0.97], [0.952, 0.952],
                               transform=fig.transFigure,
                               color=GRID_CLR, linewidth=0.8))
    return fig


def main():
    files = find_files()
    total_gens = 0
    if files["fitness"]:
        try:
            df = pd.read_csv(files["fitness"])
            total_gens = int(df["Generation"].max())
        except Exception:
            pass

    print("\nevolveAGENT Live Dashboard")
    print("──────────────────────────")
    print("Time window:")
    if total_gens > 0:
        w5  = max(10, total_gens // 20)
        w10 = max(20, total_gens // 10)
        w25 = max(50, total_gens // 4)
        window_map = {"1": 0, "2": w5, "3": w10, "4": w25}
        print(f"  1. All {total_gens} generations")
        print(f"  2. Last {w5} gens  (5%)")
        print(f"  3. Last {w10} gens (10%)")
        print(f"  4. Last {w25} gens (25%)")
    else:
        window_map = {"1": 0, "2": 50, "3": 100, "4": 200}
        print("  1. All generations")
        print("  2. Last 50 gens")
        print("  3. Last 100 gens")
        print("  4. Last 200 gens")

    choice = input("\nSelect [1-4] (default 1): ").strip() or "1"
    window = window_map.get(choice, 0)
    print()

    files = find_files()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Refreshing...")
    for k, v in files.items():
        if v:
            print(f"  {k:12s} → {os.path.basename(v)}")
        else:
            print(f"  {k:12s} → NOT FOUND")

    fig = build_figure(files, window=window)
    plt.show()
    print("Done. Close the window to exit.")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()