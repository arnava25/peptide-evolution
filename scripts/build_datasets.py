#!/usr/bin/env python3
"""
build_datasets.py
-----------------
Builds versioned training datasets for AMP, toxicity, and stability models.

Sources:
  AMP:       Original (LAMP, DBAASP, APD3, ToxinPred, Hemolytik, Synthetic) — 56,906 sequences
  Toxicity:  Original ToxinPred (3,610) + new DRAMP hemolysis labels (~400–600)
  Stability: Original DRAMP-Stability (106) + new DRAMP parsed half-life labels

Run from project root:
    python3 scripts/build_datasets.py
"""

import os
import re
import pandas as pd
import numpy as np

# Paths
ORIG_AMP  = "data/model_trainers/amp_dataset.csv"
ORIG_TOX  = "data/model_trainers/toxicity_dataset_balanced.csv"
ORIG_STAB = "data/model_trainers/stability_dataset_balanced.csv"
DRAMP_GEN = "data/model_trainers/dramp_general.txt"
DRAMP_STAB = "data/model_trainers/dramp_stability.txt"

OUT_AMP   = "data/model_trainers/amp_dataset_v2.csv"
OUT_TOX   = "data/model_trainers/toxicity_dataset_v2.csv"
OUT_STAB  = "data/model_trainers/stability_dataset_v2.csv"

VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")

def is_valid(seq, min_len=5, max_len=60):
    s = str(seq).upper().strip()
    return len(s) >= min_len and len(s) <= max_len and all(c in VALID_AAS for c in s)

def clean_seq(seq):
    return str(seq).upper().strip()

# ============================================================
# 1.  AMP dataset — unchanged (already includes DRAMP/DBAASP)
# ============================================================
def build_amp():
    df = pd.read_csv(ORIG_AMP)
    df = df[df["Sequence"].apply(is_valid)].copy()
    df["Sequence"] = df["Sequence"].apply(clean_seq)
    df = df.drop_duplicates("Sequence")
    print(f"AMP dataset: {len(df)} sequences "
          f"({df['Label'].sum()} positive, {(df['Label']==0).sum()} negative)")
    print(f"  Sources: {df['Source'].value_counts().to_dict()}")
    df.to_csv(OUT_AMP, index=False)
    print(f"  Saved to {OUT_AMP}\n")
    return df

# ============================================================
# 2.  Toxicity dataset — expand with DRAMP hemolysis data
# ============================================================
def parse_dramp_hemolysis(dramp_path):
    """
    Extract binary hemolysis labels from DRAMP general dataset.

    Label 1 (hemolytic / toxic proxy):
      Entries with confirmed hemolysis values (HC50, IC50 with actual numbers,
      or explicit "induced hemolysis of X%").

    Label 0 (non-hemolytic):
      Entries with explicit statements of no hemolytic activity (not just
      "no information found").
    """
    df = pd.read_csv(dramp_path, sep="\t", on_bad_lines="skip")
    df = df.dropna(subset=["Sequence"])
    df = df[df["Sequence"].apply(is_valid)].copy()
    df["Sequence"] = df["Sequence"].apply(clean_seq)

    hem = df["Hemolytic_activity"].fillna("")

    # Confirmed non-hemolytic: explicit biological statement
    non_hemo = (
        hem.str.contains(r"no hemolytic activity", case=False) |
        hem.str.contains(r"not hemolyti", case=False) |
        hem.str.contains(r"non-hemolyti", case=False) |
        hem.str.contains(r"none of the peptides are hemolytic", case=False) |
        hem.str.contains(r"did not show hemolysis", case=False) |
        hem.str.contains(r"no.*hemolysis.*observed", case=False) |
        hem.str.contains(r"no significant.*hemolysis", case=False)
    ) & ~hem.str.contains(
        r"information.*not found|not found|not available|no comments", case=False
    )

    # Confirmed hemolytic: actual quantitative data
    yes_hemo = (
        hem.str.contains(r"HC50\s*[<=]\s*\d", case=False) |
        hem.str.contains(r"IC50\s*[<=]\s*\d+\s*µM", case=False) |
        hem.str.contains(r"induced hemolysis of \d", case=False) |
        hem.str.contains(r"hemolysis of \d+\s*%", case=False)
    ) & ~hem.str.contains(r"no hemolytic|non-hemo|not hemo|no.*hemolysis.*observ", case=False)

    neg = df[non_hemo][["Sequence"]].copy()
    neg["Label"]  = 0
    neg["Name"]   = "NonHemolytic_DRAMP"
    neg["Source"] = "DRAMP_Hemolysis"

    pos = df[yes_hemo][["Sequence"]].copy()
    pos["Label"]  = 1
    pos["Name"]   = "Hemolytic_DRAMP"
    pos["Source"] = "DRAMP_Hemolysis"

    combined = pd.concat([neg, pos]).drop_duplicates("Sequence")
    print(f"  DRAMP hemolysis: {(combined['Label']==0).sum()} non-hemolytic, "
          f"{(combined['Label']==1).sum()} hemolytic")
    return combined


def build_toxicity():
    orig = pd.read_csv(ORIG_TOX)
    orig = orig[orig["Sequence"].apply(is_valid)].copy()
    orig["Sequence"] = orig["Sequence"].apply(clean_seq)

    dramp_hem = parse_dramp_hemolysis(DRAMP_GEN)

    # Remove sequences already in original dataset
    existing = set(orig["Sequence"])
    dramp_hem = dramp_hem[~dramp_hem["Sequence"].isin(existing)]

    combined = pd.concat([orig, dramp_hem], ignore_index=True)
    combined = combined.drop_duplicates("Sequence")

    # Rebalance: match minority class
    pos = combined[combined["Label"] == 1]
    neg = combined[combined["Label"] == 0]
    n = min(len(pos), len(neg))
    balanced = pd.concat([
        pos.sample(n, random_state=42),
        neg.sample(n, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nToxicity dataset: {len(balanced)} sequences "
          f"({balanced['Label'].sum()} toxic, {(balanced['Label']==0).sum()} non-toxic)")
    print(f"  Sources: {balanced['Source'].value_counts().to_dict()}")
    balanced.to_csv(OUT_TOX, index=False)
    print(f"  Saved to {OUT_TOX}\n")
    return balanced


# ============================================================
# 3.  Stability dataset — expand with parsed DRAMP half-life
# ============================================================
def parse_dramp_stability_halflife(dramp_stab_path):
    """
    Parse DRAMP stability file's half_life column into binary labels.
    Stable (1): long half-life or explicit resistance statements.
    Unstable (0): short half-life or rapid degradation.
    """
    df = pd.read_csv(dramp_stab_path, sep="\t", on_bad_lines="skip")
    df = df.dropna(subset=["Sequence"])
    df = df[df["Sequence"].apply(is_valid)].copy()
    df["Sequence"] = df["Sequence"].apply(clean_seq)

    hl = df["half_life"].fillna("")

    stable_mask = (
        hl.str.contains(r"[>≥]\s*[4-9]\d*\s*h", case=False) |
        hl.str.contains(r"[>≥]\s*[23]\d{2,}\s*min", case=False) |
        hl.str.contains(r"stability increased", case=False) |
        hl.str.contains(r"resistant", case=False) |
        hl.str.contains(r"[6-9]\d\s*%.*intact|intact.*[6-9]\d\s*%", case=False) |
        hl.str.contains(r"remain.*[6-9]\d\s*%|[6-9]\d\s*%.*remain", case=False) |
        hl.str.contains(r">480", case=False)
    )

    unstable_mask = (
        hl.str.contains(r"degraded quickly", case=False) |
        hl.str.contains(r"half.live.*<\s*[01]\.[5-9]|half.live.*<\s*[01]\s*h", case=False) |
        hl.str.contains(r"completely gone", case=False) |
        hl.str.contains(r"half.lives.*about 1 h", case=False)
    )

    stable_df = df[stable_mask][["Sequence"]].copy()
    stable_df["Label"]  = 1
    stable_df["Name"]   = "Stable_DRAMP"
    stable_df["Source"] = "DRAMP-Stability-HalfLife"

    unstable_df = df[unstable_mask][["Sequence"]].copy()
    unstable_df["Label"]  = 0
    unstable_df["Name"]   = "Unstable_DRAMP"
    unstable_df["Source"] = "DRAMP-Stability-HalfLife"

    combined = pd.concat([stable_df, unstable_df]).drop_duplicates("Sequence")
    print(f"  DRAMP half-life parsed: {(combined['Label']==1).sum()} stable, "
          f"{(combined['Label']==0).sum()} unstable")
    return combined


def build_stability():
    orig = pd.read_csv(ORIG_STAB)
    orig = orig[orig["Sequence"].apply(is_valid)].copy()
    orig["Sequence"] = orig["Sequence"].apply(clean_seq)

    dramp_new = parse_dramp_stability_halflife(DRAMP_STAB)

    existing = set(orig["Sequence"])
    dramp_new = dramp_new[~dramp_new["Sequence"].isin(existing)]

    combined = pd.concat([orig, dramp_new], ignore_index=True)
    combined = combined.drop_duplicates("Sequence")

    # Rebalance
    pos = combined[combined["Label"] == 1]
    neg = combined[combined["Label"] == 0]
    n = min(len(pos), len(neg))
    balanced = pd.concat([
        pos.sample(n, random_state=42),
        neg.sample(n, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nStability dataset: {len(balanced)} sequences "
          f"({balanced['Label'].sum()} stable, {(balanced['Label']==0).sum()} unstable)")
    print(f"  Sources: {balanced['Source'].value_counts().to_dict()}")
    print(f"  NOTE: Dataset remains small. Train script uses 5-fold CV + regularization.")
    balanced.to_csv(OUT_STAB, index=False)
    print(f"  Saved to {OUT_STAB}\n")
    return balanced


if __name__ == "__main__":
    os.makedirs("data/model_trainers", exist_ok=True)
    print("=" * 60)
    print("Building training datasets...")
    print("=" * 60)
    print()
    build_amp()
    build_toxicity()
    build_stability()
    print("=" * 60)
    print("All datasets built.")
