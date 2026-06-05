import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'          # suppress TF C++ info/warnings
os.environ['TF_METAL_DEVICE_DEBUG'] = '0'          # suppress Metal device logs

import random
import string
import math
import numpy as np
import pandas as pd
import hashlib
from datetime import datetime
from typing import Optional
from collections import deque
import sys
import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from tensorflow import keras

# Allow importing from project root (../external/pyampa_integration.py)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)



def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def _sigmoid(x):
    # stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def _max_run_len(seq: str, charset: set) -> int:
    best = cur = 0
    for ch in seq:
        if ch in charset:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best

ARCHIVE_K = 3
ARCHIVE_MAX = 2000
ARCHIVE_SAMPLE_N = 400



archive_kmers = deque(maxlen=ARCHIVE_MAX)   # deque[set]
archive_seen  = set()                       # exact dedupe
archive_queue = deque(maxlen=ARCHIVE_MAX)   # deque[str]


niche_archive = []   # list of (kmer_set, fitness) for cleared niches
NICHE_PENALTY_GENS = 150  # how long to penalize similarity to cleared niches
niche_penalty_active = 0  # countdown timer

# Persistent cell crowding — tracks how many high-fitness sequences
# have been found per MAP-Elites cell. Never resets.
_cell_visit_counts: dict = {}   # (ci, hi) -> int

_CROWDING_PENALTY_START = 50
_CROWDING_PENALTY_MAX   = 0.92

_REASON_LOG_PATH = None
_stagnant_gens_global = 0  # mirrors stagnant_generations for use in cognitive_mutate
stability_model_global = None
world_model = None
naturalness_model = None 
prophet_model = None
mic_model = None
pwm_buffer = []
PWM_BUFFER_MAX = 30000   # max training samples to keep
PWM_BATCH_SIZE = 256
PWM_STEPS_PER_GEN = 3    # small # of gradient steps per generation


def sequence_identity(seq1, seq2):
    """Calculates % identity between two sequences (simple)."""
    matches = sum(a == b for a, b in zip(seq1, seq2))
    return matches / min(len(seq1), len(seq2))

CURRENT_POP_FILE = 'data/current_population.txt'
MUTATION_HISTORY_FILE = 'data/mutation_rate_history.txt'


# These are set per-run inside run_simulation() with RUN_TAG suffix
WORLD_MODEL_ERROR_FILE = None
NOVELTY_STATS_FILE = None
ENTROPY_STATS_FILE = None


os.makedirs('data', exist_ok=True)


EARLY_STOP_FILE = 'stop.txt'

# 👁️ Attentional motif salience memory
salient_motifs = {}  # Stores k-mers and their evolving salience values

# Positional entropy state — updated each generation, used in score_peptide
_position_entropy: list[float] = []   # entropy per position, populated by update_position_entropy()
_ENTROPY_BONUS_WEIGHT = 0.04          # how much positional rarity boosts fitness

from tensorflow.keras import layers, models, optimizers


def build_world_model(seq_len, vocab_size):
    """
    Tiny internal forward model:
    int sequence -> (amp_pred, tox_pred, stab_pred, realism_pred)
    """
    inp = layers.Input(shape=(seq_len,), dtype='int32')
    x = layers.Embedding(input_dim=vocab_size, output_dim=16, mask_zero=False)(inp)
    x = layers.Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(4, activation='linear')(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='mse'
    )
    return model






def pwm_add_samples(gen_df):
    """
    Add (X, Y) pairs from this generation to the replay buffer.
    Y = [AMP, Tox, Stability, Realism]
    Uses max_len for sequence encoding.
    """
    global pwm_buffer

    for _, row in gen_df.iterrows():
        seq = row['Peptide']
        
        # Encode to max_len = 50
        x = encode_int(seq, max_len=max_len)
        
        if len(x) != max_len:
            continue  # safety

        y = [
            float(row['AMP_Score']),
            float(row['Toxicity_Score']),
            float(row['Stability_Score']),
            float(row['Realism_Score'])
        ]
        pwm_buffer.append((x, y))

    # keep buffer bounded
    if len(pwm_buffer) > PWM_BUFFER_MAX:
        pwm_buffer = pwm_buffer[-PWM_BUFFER_MAX:]



def pwm_train_one_epoch():
    """
    Train the world model a tiny bit each generation from buffered samples.
    Uses global world_model and pwm_buffer.
    """
    global world_model, pwm_buffer
    if world_model is None or len(pwm_buffer) < PWM_BATCH_SIZE:
        return  # not enough data yet

    import numpy as np
    
    # random mini-batch
    batch = random.sample(pwm_buffer, PWM_BATCH_SIZE)
    X = np.array([b[0] for b in batch], dtype='int32')   # shape (batch, 50)
    Y = np.array([b[1] for b in batch], dtype='float32') # shape (batch, 4)

    world_model.fit(
        X, Y,
        batch_size=PWM_BATCH_SIZE,
        epochs=1,
        verbose=0
    )






def check_early_stop():
    return os.path.exists(EARLY_STOP_FILE)

def calculate_net_charge(peptide):
    """Calculates net charge at pH ~7."""
    positive = set('KRH')  # Lysine, Arginine, Histidine
    negative = set('DE')   # Aspartic acid, Glutamic acid

    pos_count = sum(1 for aa in peptide if aa in positive)
    neg_count = sum(1 for aa in peptide if aa in negative)

    return pos_count - neg_count

def calculate_hydrophobicity(peptide):
    """Estimates hydrophobic content (fraction)."""
    hydrophobic = set('AILMVFWY')  # Typical hydrophobic residues
    hydro_count = sum(1 for aa in peptide if aa in hydrophobic)

    return hydro_count / len(peptide)

def calculate_polar_content(peptide):
    """Returns the fraction of polar residues."""
    polar = set('STNQ')  # Ser, Thr, Asn, Gln
    return sum(1 for aa in peptide if aa in polar) / len(peptide)

def estimate_solubility(peptide):
    """Returns a solubility score between 0 and 1 based on heuristics."""
    charge = calculate_net_charge(peptide)
    hydrophobicity = calculate_hydrophobicity(peptide)
    polar_content = calculate_polar_content(peptide)

    score = 1.0

    # Penalty for high hydrophobicity
    if hydrophobicity > 0.6:
        score *= 0.5
    elif hydrophobicity > 0.5:
        score *= 0.8

    # Bonus for good charge
    if 2 <= charge <= 8:
        score *= 1.1
    elif charge < 0 or charge > 10:
        score *= 0.8

    # Bonus for polar content
    if polar_content > 0.2:
        score *= 1.1
    elif polar_content < 0.1:
        score *= 0.8

    return min(max(score, 0), 1)


def calculate_hydrophobic_moment(peptide, angle=100):
    """
    Calculates the hydrophobic moment (3D amphipathicity) assuming an alpha-helix.
    Returns a value roughly 0.0 to ~0.8. High (>0.3) is better for AMPs.
    """
    if not peptide: return 0.0
    
    # Eisenberg hydrophobicity scale (normalized)
    hydrophobicity_scale = {
        'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
        'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
        'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
        'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08
    }
    
    # Map sequence to hydrophobicity values
    h_vals = [hydrophobicity_scale.get(aa, 0) for aa in peptide]
    
    # Calculate vector components (H_i * cos(delta * i))
    x_comp = sum(h * math.cos(math.radians(i * angle)) for i, h in enumerate(h_vals))
    y_comp = sum(h * math.sin(math.radians(i * angle)) for i, h in enumerate(h_vals))
    
    moment = math.sqrt(x_comp**2 + y_comp**2)
    
    # Normalize by length to keep it fair across lengths
    return moment / len(peptide)


# Isoelectric point: simplified lookup based on pKa values
aa_pKa = {
    'C': 8.5, 'D': 3.9, 'E': 4.1, 'H': 6.0,
    'K': 10.5, 'R': 12.5, 'Y': 10.1
}

def calculate_isoelectric_point(peptide):
    """Estimate pI based on charged residue counts (very rough)."""
    pos_charge = sum(peptide.count(aa) for aa in 'KRH')
    neg_charge = sum(peptide.count(aa) for aa in 'DE')
    if pos_charge == neg_charge:
        return 7.0
    elif pos_charge > neg_charge:
        return 9.0 - 0.2 * (pos_charge - neg_charge)
    else:
        return 5.0 + 0.2 * (neg_charge - pos_charge)

boman_values = {
    'A': 0.17, 'C': 0.41, 'D': -0.07, 'E': -0.07, 'F': 0.61,
    'G': 0.01, 'H': 0.17, 'I': 0.37, 'K': 0.03, 'L': 0.56,
    'M': 0.40, 'N': 0.06, 'P': -0.07, 'Q': 0.00, 'R': 0.21,
    'S': 0.13, 'T': 0.14, 'V': 0.31, 'W': 0.99, 'Y': 0.61
}

def calculate_boman_index(peptide):
    """Returns the average protein-binding propensity (Boman index)."""
    return sum(boman_values.get(aa, 0) for aa in peptide) / len(peptide)

def estimate_aggregation_risk(peptide):
    """Returns a risk score from 0 (safe) to 1 (likely to aggregate)."""
    hydrophobic = set('AILMVFWY')
    charge = abs(calculate_net_charge(peptide))

    # Risk increases with more hydrophobicity
    hydro_fraction = calculate_hydrophobicity(peptide)

    # Check for repeats of hydrophobic AAs
    hydro_runs = sum(peptide[i] == peptide[i+1] and peptide[i] in hydrophobic
                     for i in range(len(peptide)-1))

    # Motif penalty
    motifs = ['AAAA', 'LLLL', 'VVVV', 'FFFF', 'IIII']
    motif_penalty = any(motif in peptide for motif in motifs)

    # Risk formula (simple heuristic)
    risk = 0.0
    if hydro_fraction > 0.55:
        risk += 0.4
    if hydro_runs > 2:
        risk += 0.3
    if charge < 2:
        risk += 0.2
    if motif_penalty:
        risk += 0.2

    return min(risk, 1.0)


def heuristic_amp_score(peptide):
    """Estimate AMP potential based on biophysical heuristics."""
    charge = calculate_net_charge(peptide)
    hydrophobicity = calculate_hydrophobicity(peptide)
    solubility = estimate_solubility(peptide)
    aggregation_risk = estimate_aggregation_risk(peptide)

    score = 1.0

    if 4 <= charge <= 8:
        score *= 1.2
    elif charge < 2 or charge > 10:
        score *= 0.8

    if 0.35 <= hydrophobicity <= 0.55:
        score *= 1.2
    elif hydrophobicity < 0.2 or hydrophobicity > 0.7:
        score *= 0.8

    if aggregation_risk > 0.5:
        score *= 0.8

    if solubility > 0.6:
        score *= 1.1

    return min(score, 1.0)

def save_population(population):
    with open(CURRENT_POP_FILE, 'w') as f:
        for pep in population:
            f.write(f"{pep}\n")

def load_population():
    if os.path.exists(CURRENT_POP_FILE):
        print("🔄 Loading saved population...")
        with open(CURRENT_POP_FILE, 'r') as f:
            return [line.strip() for line in f.readlines()]
    return None

def delete_saved_population():
    if os.path.exists(CURRENT_POP_FILE):
        os.remove(CURRENT_POP_FILE)
        print("🧹 Deleted saved population file after successful run.")

# Define the master evolution file path
MASTER_EVOLUTION_FILE = 'data/master_evolution_history.csv'

# ============================================================
# 🔧 RUN CONFIGURATION — edit here before each run
# ============================================================

USE_AGENT = True          # True = agent ON (cognitive evolution)
                          # False = agent OFF (standard GA baseline)

# Settings
amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
population_size = 400
generations = 2000
peptide_length = 25
max_len = 50

STAGNANT_LIMIT = 550

# How often the agent performs meta-cognitive introspection
META_INTROSPECTION_INTERVAL = 50  # e.g. every 25 generations


# How long to wait with no improvement before entering exploration mode
EXPLORATION_STAGNANT_TRIGGER = 150   # you can tune this
MAX_EXPLORATION_BOOST = 1.4         # cap on how hard we shift to curiosity/novelty


# ============================================================
# 🏝️ ISLAND MODEL CONFIGURATION
# ============================================================
USE_ISLANDS = True
N_ISLANDS = 4                    # number of independent subpopulations
MIGRATION_INTERVAL = 50          # migrate every N generations
MIGRATION_RATE = 0.1            # fraction of each island that migrates


# One-hot encoding function (you might not be using this directly in the scoring)
aa_to_idx = {aa: idx for idx, aa in enumerate(amino_acids)}
# Scoring function

def encode_int(seq: str, max_len: int) -> np.ndarray:
    """Encode peptide sequence to integer array. max_len must always be passed explicitly."""
    int_seq = np.zeros(max_len, dtype=np.int32)
    for i, aa in enumerate(seq[:max_len]):
        idx = aa_to_idx.get(aa)
        if idx is not None:
            int_seq[i] = idx + 1  # 1-indexed
    return int_seq

    
def predict_single_value(model, int_encoded):
    """Helper to predict a single float output cleanly."""
    return float(model.predict(int_encoded, verbose=0)[0][0])


def encode_seq_for_prophet(seq: str, max_len: int) -> np.ndarray:
    """
    One-hot encode a peptide sequence into shape (max_len, 20),
    padded with zeros. Uses the same amino_acids order as training.
    """
    x = np.zeros((max_len, len(amino_acids)), dtype="float32")
    aa_index = {aa: i for i, aa in enumerate(amino_acids)}
    for i, aa in enumerate(seq[:max_len]):
        idx = aa_index.get(aa)
        if idx is not None:
            x[i, idx] = 1.0
    return x


def _chaotic_choice_from_probs(probs: np.ndarray, agent) -> int:
    """
    Choose an index from a probability vector using the agent's logistic map
    instead of RNG, to stay deterministic but sensitive.
    """
    probs = np.asarray(probs, dtype="float32")
    probs = np.maximum(probs, 1e-8)
    probs = probs / probs.sum()

    # advance logistic map once
    EPS = 1e-15
    agent.z = agent.r * agent.z * (1.0 - agent.z)
    agent.z = max(min(agent.z, 1.0 - EPS), EPS)

    cdf = np.cumsum(probs)
    idx = int(np.searchsorted(cdf, agent.z, side="right"))
    if idx >= len(probs):
        idx = len(probs) - 1
    return idx

def predict_mic_score(peptide: str) -> float:
    """
    Returns a 0-1 score where higher = lower predicted MIC = more potent.
    Uses log10 MIC prediction, sigmoid-transformed.
    """
    global mic_model
    if mic_model is None:
        return 0.5
    try:
        x = encode_int(peptide, max_len=max_len).reshape(1, -1)
        log_mic = float(mic_model.predict(x, verbose=0)[0][0])
        # sigmoid centered at log10(10) = 1.0
        # score ~0.9 at MIC=0.1, ~0.5 at MIC=10, ~0.1 at MIC=100
        score = 1.0 / (1.0 + math.exp(log_mic - 1.0))
        return float(_clamp(score, 0.0, 1.0))
    except Exception:
        return 0.5

def score_peptide(
    peptide,
    amp_score,
    tox_score,
    stab_score,
    previous_peptides=None,
    agent=None,
    archive_kmers=None,   # FIXED name
):
    # --- Biophysics / heuristics ---
    solubility_score = estimate_solubility(peptide)
    aggregation_risk = estimate_aggregation_risk(peptide)
    net_charge = calculate_net_charge(peptide)
    hydrophobicity = calculate_hydrophobicity(peptide)
    polar_content = calculate_polar_content(peptide)
    pI = calculate_isoelectric_point(peptide)
    boman = calculate_boman_index(peptide)
    realism = realism_penalty_score(peptide)  # ~0.4–1.0
    quality_score = compute_quality_score(peptide)  # 0–1
    
    # 🧬 3D Structure Check (Hydrophobic Moment)
    hydro_moment = calculate_hydrophobic_moment(peptide)
    
    hm = hydro_moment
    good = _sigmoid((hm - 0.26) / 0.05) * _sigmoid((0.60 - hm) / 0.06)
    bad = _sigmoid((0.18 - hm) / 0.05)

    structure_bonus = 0.86 + 0.34 * good - 0.18 * bad
    structure_bonus = float(_clamp(structure_bonus, 0.80, 1.18))


    # 👽 Turing Test (Naturalness Check)
    # Uses global naturalness_model if available
    turing_bonus = 1.0
    global naturalness_model
    if naturalness_model is not None:
        try:
            nat_pred = float(naturalness_model.predict(encode_int(peptide, max_len=max_len).reshape(1, -1), verbose=0)[0][0])
            if nat_pred < 0.3:
                turing_bonus = 0.5
            elif nat_pred > 0.8:
                turing_bonus = 1.1
        except Exception:
            pass

    disagreement_penalty = 0.0

    # --- Stability heuristic (replaces stability CNN) ---
    # Combines aggregation safety, hydrophobic balance, and realism
    # Range roughly 0.3-0.9 for realistic sequences
    
    

    # use stability CNN if available, otherwise fall back to heuristic
    global stability_model_global
    if stability_model_global is not None:
        try:
            stab_score = float(stability_model_global.predict(
                encode_int(peptide, max_len=max_len).reshape(1, -1), verbose=0)[0][0])
        except Exception:
            agg_safe = 1.0 - aggregation_risk
            hydro_ok = float(_clamp(1.0 - abs(hydrophobicity - 0.40) / 0.40, 0.0, 1.0))
            stab_score = float(_clamp(0.4 * agg_safe + 0.3 * hydro_ok + 0.3 * realism, 0.0, 1.0))
    else:
        agg_safe = 1.0 - aggregation_risk
        hydro_ok = float(_clamp(1.0 - abs(hydrophobicity - 0.40) / 0.40, 0.0, 1.0))
        stab_score = float(_clamp(0.4 * agg_safe + 0.3 * hydro_ok + 0.3 * realism, 0.0, 1.0))



    # --- Novelty term ---
    if archive_kmers is None:
        novelty = 0.5
    else:
        novelty = novelty_vs_archive(peptide, archive_kmers, k=ARCHIVE_K, sample_n=ARCHIVE_SAMPLE_N)


    # --- MIC score
    mic_score = predict_mic_score(peptide)

    # --- Weights ---
    if agent is not None:
        m = agent.motives
        w_amp    = agent.w[m.index("amp")]
        w_safety = agent.w[m.index("safety")]
        w_stab   = agent.w[m.index("stability")]
        w_real   = agent.w[m.index("realism")]
        w_novel  = agent.w[m.index("novelty")]
        w_quality = 0.5 * (w_amp + w_safety)
        w_mic    = 0.10  # mixed-length experimental dataset (DBAASP), length-agnostic
    else:
        w_amp, w_safety, w_stab, w_quality, w_novel, w_real = 0.35, 0.25, 0.27, 0.10, 0.10, 0.0
        w_mic = 0.10  # mixed-length experimental dataset, length-agnostic

    # Historical novelty weight — scales up after peaks are archived
    n_peaks = len(getattr(run_simulation, '_historical_peaks', []))
    w_hist_novel = min(0.08, n_peaks * 0.02)  # 0 before any peaks, up to 0.08 after 4+

    weights = np.array([w_amp, w_safety, w_stab, w_quality, w_novel, w_real, w_mic], dtype=float)
    weights = weights / weights.sum()
    w_amp, w_safety, w_stab, w_quality, w_novel, w_real, w_mic = weights

    # --- Core Value ---
    hist_novel = historical_novelty_score(peptide)

    value = (
        w_amp     * amp_score +
        w_safety  * (1.0 - tox_score) +
        w_stab    * stab_score +
        w_quality * quality_score +
        w_novel   * novelty +
        w_real    * realism +
        w_mic     * mic_score +
        w_hist_novel * hist_novel
    )

    # Positional entropy bonus: reward amino acids that are rare at their position
    entropy_bonus = 1.0
    global _position_entropy, _ENTROPY_BONUS_WEIGHT
    if _position_entropy and len(_position_entropy) == len(peptide):
        max_H = math.log2(20)  # maximum possible per-position entropy
        # rarity at each position: low entropy = converged = rare aa gets bonus
        rarity_scores = []
        for pos, aa in enumerate(peptide):
            pos_H = _position_entropy[pos]
            convergence = max(0.0, 1.0 - pos_H / max_H)  # 1 = fully converged, 0 = diverse
            rarity_scores.append(convergence)
        avg_rarity = float(np.mean(rarity_scores)) if rarity_scores else 0.0
        entropy_bonus = 1.0 + _ENTROPY_BONUS_WEIGHT * avg_rarity

    nov_bonus  = archive_novelty_bonus(peptide, archive_kmers)
    crowd_pen  = crowding_penalty(peptide)
    cell_bonus = cell_diversity_bonus(peptide)
    # Cap combined multiplicative bonus to prevent saturation
    combined_bonus = structure_bonus * turing_bonus * nov_bonus * entropy_bonus * cell_bonus
    combined_bonus = float(_clamp(combined_bonus, 0.70, 1.08))
    gated = value * combined_bonus * niche_penalty_score(peptide) * crowd_pen * historical_peak_penalty(peptide)

    gated = max(0.0, gated - disagreement_penalty)
    gated = float(_clamp(gated, 0.0, 1.0))  # hard clamp before compression

    # --- Softplus compression ---
    scale = 3.5          # even looser
    shifted = gated - 0.68  # raise center further
    sp = math.log(1.0 + math.exp(_clamp(scale * shifted, -60.0, 60.0)))
    sp_max = math.log(1.0 + math.exp(scale * 0.32))
    fitness = float(_clamp(sp / sp_max, 0.0, 1.0))



    sol_tag = "✅ Soluble" if solubility_score >= 0.5 else "🔴 Low Solubility"
    agg_tag = "✅ Safe" if aggregation_risk <= 0.5 else "🔴 Risky"

    return (
        amp_score, tox_score, stab_score, fitness,
        solubility_score, aggregation_risk, pI, boman,
        net_charge, hydrophobicity, sol_tag, agg_tag,
        realism, hydro_moment
    )


# Mutation function

# Smarter mutation: conservative amino acid substitutions
similar_groups = {
    'hydrophobic': list('AILMVFWY'),
    'polar': list('STNQ'),
    'positive': list('KRH'),
    'negative': list('DE'),
    'special': list('CGP')
}

# Mapping from amino acid to its group
aa_to_group = {}
for group, aas in similar_groups.items():
    for aa in aas:
        aa_to_group[aa] = group

def smart_mutate(peptide, mutation_rate):
    peptide = list(peptide)

    # Define groups
    hydrophobic = set('AILMVFWY')
    polar = set('STNQ')
    positive = set('KRH')
    negative = set('DE')
    special = set('CGP')  # Cysteine and Proline treated as "special"

    for i in range(len(peptide)):
        if random.random() < mutation_rate:
            aa = peptide[i]
            # 5% chance to wild mutate to anything
            if random.random() < 0.05:
                peptide[i] = random.choice(amino_acids)
                continue

            if aa in hydrophobic:
                group = hydrophobic
            elif aa in polar:
                group = polar
            elif aa in positive:
                group = positive
            elif aa in negative:
                group = negative
            elif aa in special:
                # For C or P, less likely to mutate (protect structure)
                if random.random() < 0.7:
                    continue
                group = hydrophobic | polar  # broader swap
            else:
                group = set(amino_acids)

            group = list(group - {aa})  # exclude self
            if group:
                peptide[i] = random.choice(group)




    # Attention-based mutation (optional, 5% of the time)
    if random.random() < 0.05 and len(peptide) > 3:
        idx = random.randint(0, len(peptide) - 3)
        motif = ''.join(peptide[idx:idx+3])
        _sal_threshold = float(np.percentile(list(salient_motifs.values()), 75)) if len(salient_motifs) > 10 else 0.02
        if salient_motifs.get(motif, 0) > _sal_threshold:
            for j in range(3):
                aa = peptide[idx + j]
                gname = aa_to_group.get(aa, None)
                if gname is None:
                    choices = amino_acids
                else:
                    choices = similar_groups[gname]  # ✅ list of AAs
                peptide[idx + j] = random.choice(choices)

    return ''.join(peptide)



def realism_penalty_score(peptide):
    """
    Smooth realism score in [0, 1].
    Designed to spread values (avoid saturation at ~1.0)
    while still penalizing pathological sequences.
    """
    if not peptide:
        return 0.0

    uniq = len(set(peptide))
    hydro = calculate_hydrophobicity(peptide)
    polar = calculate_polar_content(peptide)
    charge = calculate_net_charge(peptide)
    hm = calculate_hydrophobic_moment(peptide)


    if peptide_length <= 15:
        div_term = _sigmoid((uniq - 5.0) / 1.2) * _sigmoid((13.0 - uniq) / 2.0)
    else:
        div_term = _sigmoid((uniq - 8.5) / 1.5) * _sigmoid((15.5 - uniq) / 2.0)

    # hydrophobic fraction
    hydro_term = _sigmoid((hydro - 0.22) / 0.06) * _sigmoid((0.66 - hydro) / 0.06)

    # polar fraction
    polar_term = _sigmoid((polar - 0.08) / 0.06) * _sigmoid((0.70 - polar) / 0.08)

    # charge window
    charge_upper = 8.0 if peptide_length <= 15 else 10.5
    charge_term = _sigmoid((charge - 1.0) / 1.2) * _sigmoid((charge_upper - charge) / 1.5)


    # cysteine / proline softness

    c_max = 2.0 if peptide_length <= 15 else 3.2
    p_max = 3.0 if peptide_length <= 15 else 4.8
    c_term = _sigmoid((c_max - peptide.count("C")) / 0.7)
    p_term = _sigmoid((p_max - peptide.count("P")) / 0.9)

    # hydrophobic run penalty
    hydrophobic_set = set("AILMVFWY")
    max_hydro_run = _max_run_len(peptide, hydrophobic_set)

    run_cutoff = 3.0 if peptide_length <= 15 else 4.2
    run_term = _sigmoid((run_cutoff - max_hydro_run) / 0.9)


    if "PPPP" in peptide or "CCCC" in peptide:
        run_term *= 0.6

    # amphipathic plausibility
    hm_term = _sigmoid((hm - 0.18) / 0.06)

    score = (
        div_term *
        hydro_term *
        polar_term *
        charge_term *
        c_term *
        p_term *
        run_term *
        hm_term
    )

    return float(_clamp(score, 0.0, 1.0))


def assess_peptide_quality(peptide):
    """Returns a tag if peptide has good properties."""
    net_charge = calculate_net_charge(peptide)
    hydrophobicity = calculate_hydrophobicity(peptide)
    realism_score = realism_penalty_score(peptide)

    if (2 <= net_charge <= 8) and (0.25 <= hydrophobicity <= 0.55) and realism_score >= 0.75:
        return "✅ Good"
    else:
        return "❌ Needs Review"

def compute_quality_score(peptide):
    """Returns a float between 0 (bad) and 1 (excellent) based on key bio traits."""

    realism = realism_penalty_score(peptide)
    realism_gate = 0.60 + 0.40 * realism




    score = 1.0

    net_charge = calculate_net_charge(peptide)
    hydrophobicity = calculate_hydrophobicity(peptide)
    polar_content = calculate_polar_content(peptide)
    unique_aas = len(set(peptide))

    # Charge range bonus zone

    charge_max = 6 if peptide_length <= 15 else 8
    if 2 <= net_charge <= charge_max:
        score *= 1.05
    else:
        score *= 0.95

    # Ideal hydrophobicity
    if 0.25 <= hydrophobicity <= 0.55:
        score *= 1.05
    elif hydrophobicity < 0.15 or hydrophobicity > 0.65:
        score *= 0.9

    # Polar content balance
    if 0.2 <= polar_content <= 0.5:
        score *= 1.05
    elif polar_content < 0.1 or polar_content > 0.7:
        score *= 0.95

    # Amino acid variety (diversity)
    if unique_aas >= 12:
        score *= 1.05
    elif unique_aas < 6:
        score *= 0.85
    elif unique_aas < 8:
        score *= 0.95

    return min(max(score, 0.0), 1.0)

def append_generation_to_master(generation_number, generation_data):
    """Appends the current generation's data to the master evolution file."""
    df = pd.DataFrame(generation_data)

    expected_cols = [
        'Peptide',
        'AMP_Score', 'Toxicity_Score', 'Stability_Score',
        'Solubility_Score', 'Aggregation_Risk', 'Isoelectric_Point',
        'Boman_Index', 'Net_Charge', 'Hydrophobicity',
        'Solubility_Tag', 'Aggregation_Tag',
        'Fitness_Score', 'Realism_Score', 'Quality_Tag',        
        'Hydrophobic_Moment',
        'MIC_Score',
    ]



    # 🛡️ This ensures both presence and ORDER of columns
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"❌ Missing expected column: {col}")

    df = df[expected_cols]  # Force correct order

    df['Generation'] = generation_number
    df['Timestamp'] = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    file_exists = os.path.isfile(MASTER_EVOLUTION_FILE)

    with open(MASTER_EVOLUTION_FILE, 'a') as f:
        if not file_exists or f.tell() == 0:
            df.to_csv(f, header=True, index=False)
        else:
            df.to_csv(f, header=False, index=False)


def update_salient_motifs(peptide, fitness, decay=0.9):
    """Update motif salience with decay based on fitness."""
    k = 3  # motif length
    for i in range(len(peptide) - k + 1):
        motif = peptide[i:i+k]
        prev = salient_motifs.get(motif, 0.0)
        # Update: decay previous + reward (fitness-0.5 centered)
        salient_motifs[motif] = prev * decay + 0.1 * (fitness - 0.5)

def prune_salient_motifs(max_motifs=5000):
    global salient_motifs
    if len(salient_motifs) > max_motifs:
        salient_motifs = dict(
            sorted(salient_motifs.items(), key=lambda x: x[1], reverse=True)[:max_motifs]
        )

SALIENCE_FILE = 'data/salient_motifs.json'

def save_salient_motifs():
    import json
    prune_salient_motifs()
    with open(SALIENCE_FILE, 'w') as f:
        json.dump(salient_motifs, f)

def load_salient_motifs():
    import json
    global salient_motifs
    if os.path.exists(SALIENCE_FILE):
        with open(SALIENCE_FILE, 'r') as f:
            loaded = json.load(f)
        # decay all loaded values to account for time between runs
        salient_motifs = {k: v * 0.5 for k, v in loaded.items() if v * 0.5 > 0.001}
        print(f"🧠 Loaded {len(salient_motifs)} salient motifs from previous runs")
    else:
        print("🧠 No prior salience memory found — starting fresh")

def salience_entropy():
    """
    Normalized entropy (0–1) over motif salience values.
    0 = one or few motifs dominate (very peaked)
    1 = salience spread evenly over many motifs.
    """
    if not salient_motifs:
        return 0.0

    vals = np.array([max(v, 0.0) for v in salient_motifs.values()], dtype=float)
    total = vals.sum()
    if total <= 0:
        return 0.0

    p = vals / total
    # Shannon entropy
    H = 0.0
    for pi in p:
        if pi > 0:
            H -= float(pi) * math.log2(float(pi))

    # Normalize by max entropy log2(N)
    H_max = math.log2(len(p))
    if H_max <= 0:
        return 0.0

    return float(H / H_max)

def update_position_entropy(population: list[str]):
    """
    Compute per-position Shannon entropy across the current population.
    Stored globally so score_peptide can apply a rarity bonus.
    """
    global _position_entropy
    if not population:
        _position_entropy = []
        return
    L = len(population[0])
    n = float(len(population))
    entropies = []
    for pos in range(L):
        counts: dict = {}
        for seq in population:
            if pos < len(seq):
                aa = seq[pos]
                counts[aa] = counts.get(aa, 0) + 1
        H = 0.0
        for c in counts.values():
            p = c / n
            if p > 0.0:
                H -= p * math.log2(p)
        entropies.append(H)
    _position_entropy = entropies

def append_mutation_rate(generation_number, mutation_rate, stagnant_gen=None):
    with open(MUTATION_HISTORY_FILE, 'a') as f:
        if stagnant_gen is not None:
            f.write(f"{generation_number},{mutation_rate},{stagnant_gen}\n")
        else:
            f.write(f"{generation_number},{mutation_rate}\n")




def tournament_pick(survivors, generation_df, k=3):
    """
    Pick one parent via k-way tournament selection.
    Higher k = stronger selection. Uses Fitness_Score from generation_df.
    Falls back to random choice if sampled peptides are not in generation_df
    (e.g. freshly injected randoms).
    """
    if len(survivors) <= 1:
        return survivors[0]

    k = min(k, len(survivors))
    sample = random.sample(survivors, k)
    sub = generation_df[generation_df['Peptide'].isin(sample)]

    if sub.empty:
        # Fallback: no sampled peptides found in generation_df (e.g. injected randoms)
        return random.choice(sample)

    sub = sub.sort_values(by="Fitness_Score", ascending=False)
    return str(sub.iloc[0]['Peptide'])


def archive_add_sequences(seqs: list[str], k: int = ARCHIVE_K):
    global archive_kmers, archive_seen, archive_queue

    for s in seqs:
        if not s:
            continue

        # exact dedupe
        if s in archive_seen:
            continue

        # If we're at capacity, evict ONE item ourselves so we know what left.
        if len(archive_queue) == ARCHIVE_MAX:
            old_seq = archive_queue.popleft()
            archive_seen.discard(old_seq)
            if len(archive_kmers) > 0:
                archive_kmers.popleft()  # keep kmers aligned with queue

        # insert newest
        archive_queue.append(s)
        archive_seen.add(s)
        archive_kmers.append(kmer_set(s, k))


# Diversity helper
def average_similarity(population, k: int = 3):
    if len(population) < 2:
        return 0.0
    kmers = [kmer_set(p, k) for p in population]
    sims = []
    for i in range(len(kmers)):
        A = kmers[i]
        for j in range(i + 1, len(kmers)):
            B = kmers[j]
            if not A and not B:
                sim = 1.0
            elif not A or not B:
                sim = 0.0
            else:
                sim = len(A & B) / len(A | B)
            sims.append(sim)
    return float(np.mean(sims)) if sims else 0.0


def historical_peak_penalty(peptide: str) -> float:
    """
    Penalizes sequences similar to historical peaks.
    Penalty decays over time so old peaks can be revisited if no better
    basin is found. Each peak's penalty fades over ~500 gens.
    """
    peaks = getattr(run_simulation, '_historical_peaks', [])
    if not peaks:
        return 1.0
    S = kmer_set(peptide, ARCHIVE_K)

    # Peak ages — tracked as gen number when peak was archived
    peak_ages = getattr(run_simulation, '_historical_peak_ages', [])

    worst_penalty = 1.0
    for i, (peak_kmers, peak_fitness) in enumerate(peaks):
        if not peak_kmers or not S:
            continue
        inter = len(S & peak_kmers)
        union = len(S | peak_kmers)
        sim = inter / union if union else 0.0
        if sim <= 0.25:
            continue

        # Decay penalty based on peak age
        if i < len(peak_ages):
            age = _stagnant_gens_global - peak_ages[i]  # gens since archived
            decay = max(0.0, 1.0 - age / 500.0)  # full penalty at age 0, gone at age 500
        else:
            decay = 1.0  # no age info, full penalty

        if sim > 0.40:
            penalty = 0.5 + 0.5 * (1.0 - decay)  # decays from 0.5 → 1.0
        else:
            penalty = 0.75 + 0.25 * (1.0 - decay)  # decays from 0.75 → 1.0

        worst_penalty = min(worst_penalty, penalty)

    return worst_penalty

def niche_penalty_score(peptide: str) -> float:
    """
    Returns a penalty multiplier (0.7-1.0) if peptide is similar to a cleared niche.
    1.0 = no penalty, 0.7 = heavy penalty.
    """
    global niche_archive, niche_penalty_active
    if not niche_archive or niche_penalty_active <= 0:
        return 1.0
    S = kmer_set(peptide, ARCHIVE_K)
    for archived_kmers, _ in niche_archive:
        if not archived_kmers or not S:
            continue
        inter = len(S & archived_kmers)
        union = len(S | archived_kmers)
        sim = inter / union if union else 0.0
        if sim > 0.6:
            return 0.7  # penalize strongly
    return 1.0

def archive_novelty_bonus(peptide: str, archive_kmers, k: int = ARCHIVE_K) -> float:
    """
    Returns a multiplier > 1.0 for novel sequences, < 1.0 for redundant ones.
    Based on sequence-space distance to the archive.
    - novelty = 1.0 (completely novel) → bonus = 1.25
    - novelty = 0.5 (moderately similar) → bonus = 1.0  
    - novelty = 0.0 (identical to archive) → bonus = 0.75
    This creates persistent pressure to find new regions regardless of fitness.
    """
    if not archive_kmers or len(archive_kmers) < 50:
        return 1.0  # early in run, no penalty
    nov = novelty_vs_archive(peptide, archive_kmers, k=k, sample_n=200)
    # linear mapping: nov 0→0.75, nov 0.5→1.0, nov 1.0→1.25

    bonus = 0.85 + 0.25 * nov
    return float(_clamp(bonus, 0.85, 1.10))


def historical_novelty_score(peptide: str) -> float:
    """
    Returns a novelty score in [0, 1] based on distance from all historical peaks.
    1.0 = completely unlike any past peak (genuinely new basin)
    0.0 = identical to a past peak
    Used as a fitness bonus to reward exploration of new sequence space.
    """
    peaks = getattr(run_simulation, '_historical_peaks', [])
    if not peaks:
        return 0.5  # neutral when no history yet
    S = kmer_set(peptide, ARCHIVE_K)
    if not S:
        return 0.5
    max_sim = 0.0
    for peak_kmers, _ in peaks:
        if not peak_kmers:
            continue
        inter = len(S & peak_kmers)
        union = len(S | peak_kmers)
        sim = inter / union if union else 0.0
        if sim > max_sim:
            max_sim = sim
    return float(1.0 - max_sim)

def similarity_penalty(pop, k: int = 3):
    if len(pop) < 2:
        return 0.0
    kmers = [kmer_set(p, k) for p in pop]
    sims = []
    for i in range(len(kmers)):
        A = kmers[i]
        for j in range(i + 1, len(kmers)):
            B = kmers[j]
            sims.append(len(A & B) / len(A | B) if A and B else 0.0)
    return float(np.mean(sims)) if sims else 0.0


def diverse_subset(peptides, threshold=0.85):
    unique = []
    for p in peptides:
        if all(sequence_identity(p, u) < threshold for u in unique):
            unique.append(p)
        if len(unique) >= int(0.1 * population_size):
            break
    return unique

def similarity_stats(population, k: int = 3):
    """
    Returns (avg_sim, min_sim, max_sim) over pairwise k-mer Jaccard similarity.
    """
    if len(population) < 2:
        return 0.0, 0.0, 0.0

    kmers = [kmer_set(p, k) for p in population]
    sims = []
    for i in range(len(kmers)):
        A = kmers[i]
        for j in range(i + 1, len(kmers)):
            B = kmers[j]
            if not A and not B:
                sim = 1.0
            elif not A or not B:
                sim = 0.0
            else:
                sim = len(A & B) / len(A | B)
            sims.append(sim)

    sims = np.array(sims, dtype=float)
    return float(sims.mean()), float(sims.min()), float(sims.max())


def _wm_predict_batch(seqs: list[str]) -> np.ndarray:
    """Return world_model preds shaped (N,4): [amp,tox,stab,real]."""
    global world_model
    X = np.array([encode_int(s, max_len=max_len) for s in seqs], dtype="int32")
    return world_model.predict(X, verbose=0)

def _dreamer_actions(parent: str, agent, n_actions: int = 6) -> list[str]:
    """Generate deterministic-ish candidate children from parent."""
    out = []
    seen = set([parent])
    L = len(parent)
    # point muts
    for _ in range(n_actions):
        agent.z, pos = _chaotic_index(agent.z, agent.r, L)
        agent.z, aa  = _chaotic_index(agent.z, agent.r, len(amino_acids))
        c = _deterministic_point_mutation(parent, pos, aa)
        if c not in seen:
            out.append(c); seen.add(c)
    return out

def dreamer_plan(parent: str, agent, archive_kmers: list[set], target_len: int,
                 horizon: int = 3, beam: int = 8, actions_per: int = 6) -> Optional[str]:
    """
    Dreamer-ish planning:
      - roll out mutation sequences in imagination using world_model
      - keep top beam states by agent-weighted predicted features
      - return best imagined final sequence
    """
    global world_model
    if world_model is None or len(archive_kmers) < 50:
        return None

    # Each state: (score, seq)
    states = [(0.0, parent)]

    for _t in range(horizon):
        # expand
        cand = []
        for _score, s in states:
            for c in _dreamer_actions(s, agent, n_actions=actions_per):
                cand.append(c)

        if not cand:
            return None

        # predict in batch
        preds = _wm_predict_batch(cand)  # amp,tox,stab,real
        amp_p = preds[:, 0]
        tox_p = preds[:, 1]
        stab_p = preds[:, 2]
        real_p = preds[:, 3]

        # cheap novelty + parsimony
        nov = np.array([novelty_vs_archive(c, archive_kmers, k=ARCHIVE_K, sample_n=ARCHIVE_SAMPLE_N)
                        for c in cand], dtype=float)
        pars = -np.abs(np.array([len(c) for c in cand], dtype=float) - float(target_len))

        # IMPORTANT: realism is NOT part of the score anymore (constraint-only),
        # but we can still *prune* obviously non-realistic imagined states.
        # Use the predicted realism as a soft filter only:

        keep = real_p >= 0.10
        if keep.sum() < 2:
            return None


        # build predicted motive dicts and score
        scored = []
        for i, c in enumerate(cand):
            if not keep[i]:
                continue
            feats = {
                "amp": float(amp_p[i]),
                "safety": float(1.0 - tox_p[i]),
                "stability": float(stab_p[i]),
                "realism": float(real_p[i]),   # exists but won't matter if you removed realism motive later
                "novelty": float(nov[i]),
                "parsimony": float(pars[i]),
                "curiosity": 0.0,              # we don't have uncertainty; keep neutral
            }
            scored.append((agent.reason_score(feats), c))

        if not scored:
            return None

        scored.sort(reverse=True, key=lambda x: x[0])
        # keep top beam as new states
        states = scored[:beam]

    # best imagined endpoint
    return states[0][1] if states else None




def compute_population_entropy(population):
    """
    Shannon entropy across positions.
    Returns (mean_entropy_per_position, max_entropy_over_positions).
    """
    if not population:
        return 0.0, 0.0

    L = len(population[0])
    n = float(len(population))
    entropies = []

    for pos in range(L):
        counts = {}
        for seq in population:
            aa = seq[pos]
            counts[aa] = counts.get(aa, 0) + 1

        H = 0.0
        for c in counts.values():
            p = c / n
            if p > 0.0:
                H -= p * math.log2(p)
        entropies.append(H)

    entropies = np.array(entropies, dtype=float)
    return float(entropies.mean()), float(entropies.max())


def log_world_model_error(generation_df, gen):
    """
    Log world-model prediction error for AMP, Tox, Stability, Realism.
    Uses the current generation as supervised data.
    """
    global world_model
    if world_model is None:
        return

    seqs = generation_df['Peptide'].tolist()
    if not seqs:
        return

    X = np.array([encode_int(s, max_len=max_len) for s in seqs], dtype='int32')

    # True targets from the current generation
    Y_true = generation_df[
        ['AMP_Score', 'Toxicity_Score', 'Stability_Score', 'Realism_Score']
    ].values.astype('float32')

    Y_pred = world_model.predict(X, verbose=0)
    errors = np.abs(Y_true - Y_pred)

    mean_err = errors.mean(axis=0)  # per-dimension
    total_err = errors.mean()       # overall scalar

    file_exists = os.path.exists(WORLD_MODEL_ERROR_FILE)
    with open(WORLD_MODEL_ERROR_FILE, 'a') as f:
        if not file_exists:
            f.write("Generation,TotalError,AMP_Error,Tox_Error,Stab_Error,Realism_Error\n")
        f.write(
            f"{gen},{total_err:.6f},"
            f"{mean_err[0]:.6f},{mean_err[1]:.6f},"
            f"{mean_err[2]:.6f},{mean_err[3]:.6f}\n"
        )

    return float(total_err)


def log_novelty_stats(avg_sim, min_sim, max_sim, gen):
    """
    Log similarity-based novelty measures per generation.
    """
    file_exists = os.path.exists(NOVELTY_STATS_FILE)
    with open(NOVELTY_STATS_FILE, 'a') as f:
        if not file_exists:
            f.write("Generation,AvgSimilarity,MinSimilarity,MaxSimilarity\n")
        f.write(f"{gen},{avg_sim:.4f},{min_sim:.4f},{max_sim:.4f}\n")


def log_entropy_stats(population, gen):
    """
    Log sequence entropy per generation.
    """
    mean_H, max_H = compute_population_entropy(population)
    file_exists = os.path.exists(ENTROPY_STATS_FILE)
    with open(ENTROPY_STATS_FILE, 'a') as f:
        if not file_exists:
            f.write("Generation,MeanEntropy,MaxEntropy\n")
        f.write(f"{gen},{mean_H:.4f},{max_H:.4f}\n")


# global_best_peptide and global_best_score are initialized inside run_simulation()

def crossover(parent1, parent2):
    """Two-point crossover (swap a segment between parents)."""
    if len(parent1) != len(parent2):
        raise ValueError("Parents must be same length!")
    length = len(parent1)
    # Pick two random crossover points
    pt1 = random.randint(0, length - 2)
    pt2 = random.randint(pt1 + 1, length - 1)
    # Swap the middle segment
    child = parent1[:pt1] + parent2[pt1:pt2] + parent1[pt2:]
    return child



def is_realistic(peptide, gen=1):
    return realism_penalty_score(peptide) >= 0.10


def _chaotic_index(z, r, upper):
    """Map logistic step to [0, upper). Returns new_z, idx."""
    z = r * z * (1.0 - z)
    # clamp to prevent stagnation
    EPS = 1e-15
    z = max(min(z, 1.0 - EPS), EPS)
    idx = int(z * upper)
    if idx >= upper: 
        idx = upper - 1
    return z, idx




# === SAFE SURVIVOR + WEIGHT PREP FUNCTION ===
def get_survivor_weights(
    survivors: list[str],
    generation_df: pd.DataFrame,
    min_realism_gate: float = 0.0,
    realism_power: float = 2.0,
    floor: float = 1e-6,
):
    """
    Returns weights aligned with survivors.
    - If survivor not in generation_df (e.g., injected), weight = 1.0
    - Otherwise: weight = Fitness * realism_factor
    - realism_factor softly downweights low-realism survivors without hard dropping
    """
    # fast lookup — deduplicate keeping highest fitness row per peptide
    deduped = (
        generation_df
        .sort_values("Fitness_Score", ascending=False)
        .drop_duplicates(subset="Peptide", keep="first")
    )
    sub = deduped.set_index("Peptide")[["Fitness_Score", "Realism_Score"]].to_dict("index")

    weights = []
    for s in survivors:
        row = sub.get(s)
        if row is None:
            weights.append(1.0)
            continue

        fit = float(row["Fitness_Score"])
        real = float(row["Realism_Score"])

        # soft realism gating into [0,1]
        if min_realism_gate > 0.0:
            real_factor = max(0.0, (real - min_realism_gate) / max(1e-9, (1.0 - min_realism_gate)))
        else:
            real_factor = real

        # emphasize realism if you want
        w = max(floor, fit * (real_factor ** realism_power))
        weights.append(w)

    return weights


def _deterministic_point_mutation(seq, pos, aa_idx):
    seq = list(seq)
    seq[pos] = amino_acids[aa_idx]
    return ''.join(seq)



def prophet_guided_mutation(parent: str, agent) -> Optional[str]:
    """
    Use the trained Prophet model to propose a single point mutation:
      - predicts a position distribution
      - predicts an amino-acid distribution
      - picks indices via chaotic sampler
    Returns the mutated child or None if Prophet is unavailable.
    """
    global prophet_model
    if prophet_model is None:
        return None

    # Encode parent for Prophet
    X = encode_seq_for_prophet(parent, max_len=max_len)  # (50, 20)
    try:
        pos_probs, aa_probs = prophet_model.predict(
            X[np.newaxis, ...], verbose=0
        )
    except Exception:
        return None

    pos_probs = pos_probs[0]   # (50,)
    aa_probs  = aa_probs[0]    # (20,)
    # Temperature > 1 flattens distribution = more exploratory mutations
    temperature = 1.5
    pos_probs = pos_probs ** (1.0 / temperature)
    aa_probs  = aa_probs  ** (1.0 / temperature)

    # Choose indices with chaotic sampling
    pos_idx = _chaotic_choice_from_probs(pos_probs, agent)

    # Clip position to actual sequence length (avoid mutating padded tail)
    if pos_idx >= len(parent):
        pos_idx = len(parent) - 1

    aa_idx = _chaotic_choice_from_probs(aa_probs, agent)

    # Build child
    child = _deterministic_point_mutation(parent, pos_idx, aa_idx)
    return child


# =========================
# 🧠 Cognitive Agent Module
# =========================
class AgentController:
    def __init__(self):
        self.motives = ["amp","safety","stability","realism","novelty","parsimony","curiosity"]
        self.w = np.ones(len(self.motives)) / len(self.motives)

        self.ema_improve = 0.0
        self.ema_best = 0.0
        self.ema_novelty = 0.5
        self.ema_sim = 0.6
        self.ema_explore_mass = 0.0
        self.explore_payoff = 0.0

        self.boredom = 0.0

        self.goal_mode = "normal"
        self.goal_mode_timer = 0

        self.z = 0.347
        self.r = 3.9
        self.delta_tie = 0.02

        # Slightly lower curiosity cap + lower floors so motives can actually move
        self.MAX_CUR = 0.22          # was 0.25
        self.MIN_W   = 0.03          # was 0.04

    def adapt_chaos(self, stagnation_count, avg_sim, total_err, avg_realism):
        """
        Self-tuned logistic parameter r (edge-of-chaos controller).

        - More stagnation + high similarity  -> increase r (more chaos)
        - Very high world-model error or low realism -> decrease r (cool down)
        - Slight baseline cooling over time to favor exploitation.
        """
        # --- Baseline slow cooling (drift toward exploitation) ---
        self.r -= 0.0005

        # --- Pressure to explore when stuck & converged ---
        # normalize stagnation into [0, 1]
        plateau = min(stagnation_count / 300.0, 1.0)
        # novelty collapse if avg_sim is high
        convergence = max(0.0, avg_sim - 0.6)  # 0 when <0.6, up to ~0.4

        explore_push = plateau * convergence
        if explore_push > 0:
            self.r += 0.015 * explore_push   # gently push toward chaos

        # --- If the world-model is very wrong or realism is sagging, cool down ---
        if total_err > 0.20:
            # PWM doesn't understand the regime → turn chaos down
            self.r -= 0.01

        if avg_realism < 0.65:
            # sequences are getting weird → stabilize
            self.r -= 0.01

        # --- Final clamp into the interesting range ---
        self.r = max(3.5, min(4.0, self.r))
        return self.r

    def reason_score(self, deltas: dict) -> float:
        """Weighted sum of expected improvements per motive."""
        return float(sum(self.w[i] * deltas[m] for i, m in enumerate(self.motives)))

    def chaotic_pick(self, scored):
        """Deterministic chaos tiebreaker among near-tied options."""
        if len(scored) < 2:
            return scored[0][1]
        top, second = scored[0][0], scored[1][0]
        if abs(top - second) >= self.delta_tie:
            return scored[0][1]
        # advance logistic map once, map to a near-top index (cap to 8 to keep decisions reason-led)
        self.z = self.r * self.z * (1.0 - self.z)
        idx = int(self.z * min(len(scored), 8))
        return scored[idx][1]

    def update_weights(self, pop_df: pd.DataFrame):
        """
        Gradual, bounded, homeostatic motive updates.
        Now includes a HARD CAP on Curiosity to prevent biological collapse.
        """

        # --- read signals ---
        avg_tox = float(pop_df['Toxicity_Score'].mean())
        avg_realism = float(pop_df['Realism_Score'].mean())
        avg_novelty = 1.0 - average_similarity(pop_df['Peptide'].tolist())

        if "Surprise_AMP" in pop_df.columns:
            avg_surprise = float(
                (pop_df["Surprise_AMP"].mean() + pop_df["Surprise_Toxicity"].mean()) / 2.0
            )
        else:
            avg_surprise = 0.0

        # === baseline personality (soft exploitation)
        baseline = np.array([
            0.24,  # amp          ← slight bump since novelty now handled structurally
            0.18,  # safety
            0.19,  # stability
            0.09,  # realism
            0.07,  # novelty      ← reduced: archive bonus handles this now
            0.08,  # parsimony
            0.15,  # curiosity
        ])

        # === build target vector ("what the agent wants NOW")
        target = baseline.copy()

        # toxicity too high → emphasize safety
        if avg_tox > 0.50:
            target[self.motives.index("safety")] += 0.10

        # realism too low → push realism up
        if avg_realism < 0.60:
            target[self.motives.index("realism")] += 0.10

        # novelty stagnation → more curiosity & novelty
        if avg_novelty < 0.40:
            target[self.motives.index("curiosity")] += 0.12
            target[self.motives.index("novelty")]   += 0.05

        # PyAMPA disagreement → curiosity up
        if avg_surprise > 0.15:
            target[self.motives.index("curiosity")] += 0.10

        # chaotic rhythm
        if self.z > 0.7:
            target[self.motives.index("novelty")]   += 0.05
            target[self.motives.index("curiosity")] += 0.05
        elif self.z < 0.3:
            target[self.motives.index("amp")]     += 0.05
            target[self.motives.index("safety")] += 0.05

        # --- normalize target ---
        target = np.maximum(target, 1e-6)
        target /= target.sum()

        eta = 0.03
        kappa = 0.04

        # Use the class variable we are tracking!
        boredom = self.boredom

        t = target.copy()
        i_n = self.motives.index("novelty")
        i_c = self.motives.index("curiosity")
        t[i_n] *= (1.0 - 0.35 * boredom)
        t[i_c] *= (1.0 - 0.50 * boredom)
        t = np.maximum(t, 1e-6)
        t /= t.sum()

        self.w = self.w + eta * (t - self.w) + kappa * (baseline - self.w)

        self.w = np.maximum(self.w, self.MIN_W)
        self.w[i_c] = min(self.w[i_c], self.MAX_CUR)
        # Floor on stability — never let agent zero it out
        i_stab = self.motives.index("stability")
        self.w[i_stab] = max(self.w[i_stab], 0.12)

        i_amp = self.motives.index("amp")
        i_saf = self.motives.index("safety")
        if self.w[i_amp] + self.w[i_saf] < 0.30:
            boost = (0.30 - (self.w[i_amp] + self.w[i_saf])) / 2
            self.w[i_amp] += boost
            self.w[i_saf] += boost

        self.w /= self.w.sum()

    def update_progress(self, top5_mean, best_so_far, avg_sim, avg_novelty):
        improved = top5_mean > (self.ema_best + 0.002)

        # track best
        self.ema_best = max(self.ema_best, top5_mean)

        # boredom grows if no improvement, decays if improvement
        if improved:
            self.boredom *= 0.7
        else:
            self.boredom = min(0.95, self.boredom + 0.04)

        # track exploration mass
        i_n = self.motives.index("novelty")
        i_c = self.motives.index("curiosity")
        explore_mass = self.w[i_n] + self.w[i_c]

        payoff = (1.0 if improved else 0.0) / max(explore_mass, 1e-6)
        self.explore_payoff = float(_clamp(payoff, 0.0, 2.0))


    def introspect_and_remap(self, signals: dict):
        """
        Meta-cognitive step: use slow-timescale signals to reshape motive weights.
        `signals` contains:
          - plateau: 0–1  (how long we've been stuck)
          - novelty: 0–1  (1 = very novel, 0 = identical)
          - diversity: 0–1
          - salience_entropy: 0–1
          - chaos: 0–1
        """
        m_idx = self.motives.index

        plateau   = float(signals.get("plateau", 0.0))
        novelty   = float(signals.get("novelty", 0.5))
        diversity = float(signals.get("diversity", 0.5))
        sal_ent   = float(signals.get("salience_entropy", 0.5))
        chaos     = float(signals.get("chaos", 0.5))

        target = self.w.copy()

        # === 1) Plateau logic: distinguish two cases ===========================
        #
        # Case A: plateau + LOW novelty  -> we're exploiting a basin, not exploring.
        #         => push curiosity/novelty (what you already do).
        #
        # Case B: plateau + HIGH novelty -> we're wandering around but not improving.
        #         => dial DOWN curiosity/novelty and UP AMP/safety/stability/realism
        #            so the agent gets "bored of exploration" and tries a different
        #            motive mix.
        #
        if plateau > 0.2:  # only care once we've been stuck for a bit
            if novelty < 0.35:
                # Case A: stuck & redundant -> explore more
                strength = plateau * (0.35 - novelty) / 0.35  # 0–1
                target[m_idx("curiosity")] += 0.15 * strength
                target[m_idx("novelty")]   += 0.10 * strength
            else:
                # Case B: stuck but already quite novel -> explore *motive space*,
                # not sequence space: move back toward exploitation & realism.
                strength = plateau * (novelty - 0.35)  # grows with both plateau & high novelty
                target[m_idx("curiosity")] -= 0.20 * strength
                target[m_idx("novelty")]   -= 0.12 * strength
                target[m_idx("amp")]       += 0.12 * strength
                target[m_idx("safety")]    += 0.06 * strength
                target[m_idx("stability")] += 0.05 * strength
                target[m_idx("realism")]   += 0.05 * strength

        # === 2) Diversity collapse + motif lock-in =============================
        collapse = (1.0 - diversity) * (1.0 - sal_ent)
        if collapse > 0:
            target[m_idx("amp")]      -= 0.08 * collapse
            target[m_idx("safety")]   -= 0.04 * collapse
            target[m_idx("curiosity")] += 0.08 * collapse

        # === 3) Chaos rhythm ===================================================
        if chaos > 0.8:
            target[m_idx("curiosity")] -= 0.08
            target[m_idx("novelty")]   -= 0.05
            target[m_idx("realism")]   += 0.06
            target[m_idx("parsimony")] += 0.04
        elif chaos < 0.2:
            target[m_idx("curiosity")] += 0.08
            target[m_idx("novelty")]   += 0.04

        # === 4) Smooth toward target with a bit stronger meta learning =========
        target = np.maximum(target, 1e-6)

        eta_meta = 0.05   # was 0.02 — more plastic when we introspect
        self.w = (1.0 - eta_meta) * self.w + eta_meta * target

        # === 5) Clamp & renormalize with curiosity cap =========================
        cur_idx = m_idx("curiosity")
        self.w = np.maximum(self.w, self.MIN_W)
        self.w[cur_idx] = min(self.w[cur_idx], self.MAX_CUR)
        self.w[m_idx("stability")] = max(self.w[m_idx("stability")], 0.12)
        self.w /= self.w.sum()


    def maybe_switch_goal_mode(self, stagnant_generations, avg_sim, avg_realism):
        """
        If stuck, switch modes for a while:
          - explore: if converged + stuck
          - exploit: if wandering but not improving
        """
        if self.goal_mode_timer > 0:
            self.goal_mode_timer -= 1
            return self.goal_mode

        # decide new mode only when timer expired
        if stagnant_generations > 180:
            if avg_sim > 0.70:
                self.goal_mode = "explore"
                self.goal_mode_timer = 80
            else:
                self.goal_mode = "exploit"
                self.goal_mode_timer = 80
        elif avg_realism < 0.60:
            self.goal_mode = "stabilize"
            self.goal_mode_timer = 60
        else:
            self.goal_mode = "normal"
            self.goal_mode_timer = 0

        return self.goal_mode

def get_island_agent(island_idx: int) -> AgentController:
    """
    Each island gets a slightly different personality.
    """
    agent = AgentController()
    if island_idx == 0:
        # Island 0: AMP-focused
        agent.w = np.array([0.35, 0.20, 0.15, 0.08, 0.10, 0.05, 0.07])
    elif island_idx == 1:
        # Island 1: Safety/stability focused
        agent.w = np.array([0.20, 0.30, 0.25, 0.08, 0.08, 0.05, 0.04])
    elif island_idx == 2:
        # Island 2: Novelty focused
        agent.w = np.array([0.18, 0.15, 0.12, 0.07, 0.28, 0.05, 0.15])
    elif island_idx == 3:
        # Island 3: MIC/realism focused
        agent.w = np.array([0.25, 0.20, 0.18, 0.15, 0.10, 0.07, 0.05])
    agent.w = agent.w / agent.w.sum()
    return agent


def kmer_set(seq: str, k: int = 3) -> set:
    if len(seq) < k:
        return {seq} if seq else set()
    return {seq[i:i+k] for i in range(len(seq) - k + 1)}

def jaccard_kmer_similarity(a: str, b: str, k: int = 3) -> float:
    A = kmer_set(a, k)
    B = kmer_set(b, k)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0

def novelty_floor_check(peptide: str, archive_kmers, threshold: float = 0.40) -> bool:
    """
    Returns True if peptide is novel enough to be considered.
    Rejects sequences too similar to anything in the archive.
    Post-restart: threshold lowered temporarily to allow fresh randoms through.
    """
    if not archive_kmers or len(archive_kmers) < 100:
        return True  # early in run, don't filter
    # Lower threshold for 100 gens after restart to let fresh randoms through
    _restart_gen = getattr(run_simulation, '_restart_gen', 0)
    _exp_until = getattr(run_simulation, '_expansion_active_until', 0)
    if _exp_until > 0 and _stagnant_gens_global < 50:
        threshold = 0.20  # much more permissive post-restart
    nov = novelty_vs_archive(peptide, archive_kmers, k=ARCHIVE_K, sample_n=200)
    return nov >= threshold



def novelty_vs_archive(seq: str, archive_kmers: list[set], k: int = 3, sample_n: int = 400) -> float:
    """
    Returns novelty in [0,1] as 1 - max_jaccard_sim to a sample of archive kmers.
    archive_kmers: list of precomputed kmer sets (reservoir).
    """
    if not archive_kmers:
        return 0.5
    S = kmer_set(seq, k)
    if not S:
        return 0.5

    n = len(archive_kmers)
    if n <= sample_n:
        sample = archive_kmers
    else:
        # deterministic-ish sampling: stride through reservoir based on hash
        start = int(hashlib.md5(seq.encode("utf-8")).hexdigest(), 16) % n 
        step = max(1, n // sample_n)
        sample = [archive_kmers[(start + i*step) % n] for i in range(sample_n)]

    best_sim = 0.0
    for T in sample:
        if not T:
            continue
        inter = len(S & T)
        union = len(S | T)
        sim = inter / union if union else 0.0
        if sim > best_sim:
            best_sim = sim
            if best_sim >= 0.98:
                break
    return float(1.0 - best_sim)


def _estimate_deltas_batch(
    candidates: list[str],
    amp_model, tox_model, stab_model,
    archive_kmers: list[set],
    target_len: int,
    k: int = 3
) -> list[dict]:
    """
    Batch compute motive-feature dicts for candidates.
    Uses CNNs in batch + realism in python + PWM in batch + novelty vs archive kmers.
    """
    X = np.array([encode_int(c, max_len=max_len) for c in candidates], dtype="int32")

    # Use world model for fast scoring if trained, fall back to real CNNs
    global world_model
    if world_model is not None and len(pwm_buffer) >= PWM_BATCH_SIZE * 40:    
        pwm_preds = world_model.predict(X, verbose=0)
        amp  = np.clip(pwm_preds[:, 0], 0.0, 1.0)
        tox  = np.clip(pwm_preds[:, 1], 0.0, 1.0)
        stab = np.clip(pwm_preds[:, 2], 0.0, 1.0)
    else:
        amp = amp_model.predict(X, verbose=0).reshape(-1)
        tox = tox_model.predict(X, verbose=0).reshape(-1)
        stab = np.zeros(len(candidates), dtype=float)

    realism = np.array([realism_penalty_score(c) for c in candidates], dtype=float)

    if world_model is not None:
        pwm = world_model.predict(X, verbose=0)  # (N,4)
        # compare to [amp,tox,stab,realism]
        truth = np.stack([amp, tox, stab, realism], axis=1)
        pred_err = np.abs(truth - pwm).sum(axis=1)
    else:
        pred_err = np.zeros(len(candidates), dtype=float)

    novelty = np.array([
        novelty_vs_archive(c, archive_kmers, k=k, sample_n=ARCHIVE_SAMPLE_N)
        for c in candidates
    ], dtype=float)


    parsimony = -np.abs(np.array([len(c) for c in candidates], dtype=float) - float(target_len))

    # salience bonus: reward candidates that preserve high-salience motifs
    def _salience_score(seq):
        if not salient_motifs:
            return 0.0
        scores = []
        for i in range(len(seq) - 2):
            m = seq[i:i+3]
            s = salient_motifs.get(m, 0.0)
            if s > 0:
                scores.append(s)
        return float(np.mean(scores)) if scores else 0.0

    sal_threshold = float(np.percentile(list(salient_motifs.values()), 75)) if len(salient_motifs) > 10 else 0.0
    salience = np.array([
        _salience_score(c) / max(sal_threshold, 1e-6)
        for c in candidates
    ], dtype=float)
    salience = np.clip(salience, 0.0, 2.0)

    out = []
    for i in range(len(candidates)):
        out.append({
            "amp": float(amp[i]),
            "safety": float(1.0 - tox[i]),
            "stability": float(stab[i]),
            "realism": float(realism[i]),
            "novelty": float(novelty[i]),
            "parsimony": float(parsimony[i] + 0.05 * salience[i]),  # fold into parsimony
            "curiosity": float(pred_err[i]),
        })
    return out



def _estimate_deltas(candidate: str,
                     amp_model, tox_model, stab_model,
                     population: list, target_len: int) -> dict:
    """
    Deterministic 'reasons' for/against a candidate mutation.
    All terms are quick-to-compute proxies you already use elsewhere.
    """

    int_enc = np.array([encode_int(candidate, max_len=max_len)])
    amp = float(amp_model.predict(int_enc, verbose=0)[0][0])
    tox = float(tox_model.predict(int_enc, verbose=0)[0][0])
    stab = 0.0  # stability is heuristic now, computed inside score_peptide

    realism = float(realism_penalty_score(candidate))

    # 🔮 Peptide World Model prediction
    global world_model
    if world_model is not None:
        pwm_pred = world_model.predict(int_enc, verbose=0)[0]
        pwm_amp, pwm_tox, pwm_stab, pwm_real = map(float, pwm_pred)

        pe_amp  = abs(amp  - pwm_amp)
        pe_tox  = abs(tox  - pwm_tox)
        pe_stab = abs(stab - pwm_stab)
        pe_real = abs(realism - pwm_real)

        prediction_error = pe_amp + pe_tox + pe_stab + pe_real
    else:
        prediction_error = 0.0  # should only happen at the very start




    # novelty as inverse similarity to closest existing sequence (deterministic)
    if population:
        max_sim = max(sequence_identity(candidate, p) for p in population)
        novelty = 1.0 - max_sim
    else:
        novelty = 0.5

    # parsimony: prefer sequences near target_len; negative distance is better
    parsimony = -abs(len(candidate) - target_len)

    # map to the motive dictionary
    return {
        "amp": amp,
        "safety": 1.0 - tox,
        "stability": stab,
        "realism": realism,
        "novelty": novelty,
        "parsimony": parsimony,
        "curiosity": prediction_error  # ⬅️ NEW curiosity
    }

def cognitive_mutate(parent: str, mutation_rate: float, agent: AgentController,
                     amp_model, tox_model, stab_model,
                     archive_kmers: list[set], target_len: int):
    """
    Generate a small set of candidate children in a deterministic way:
      - 1 Prophet-guided mutation (if available)
      - 1–2 chaos-based point mutations
    Then evaluate each via the agent's motive weights and pick the best.
    Returns (child_sequence, reasons_dict, source_str).
    """
    global prophet_model

    candidates = []
    seen = set()
    length = len(parent)

    # 1) Prophet-guided candidate (if we have a model)
    # Reduce prophet influence when stagnant — it's replaying known solutions
    prophet_prob = max(0.3, 1.0 - _stagnant_gens_global / 400.0)
    if prophet_model is not None and random.random() < prophet_prob:
        child = prophet_guided_mutation(parent, agent)
        if child is not None and child not in seen:
            candidates.append(("prophet", child)) 
            seen.add(child)

    # 1b) Dreamer-planned candidate (imagination rollouts using world_model)
    # Reduce dreamer influence faster than prophet during stagnation
    dreamer_prob = max(0.3, 1.0 - _stagnant_gens_global / 300.0)
    dream = None
    if random.random() < dreamer_prob:
        dream = dreamer_plan(
            parent, agent,
            archive_kmers=archive_kmers,
            target_len=target_len,
            horizon=5, beam=16, actions_per=6
        )
    if dream is not None and dream not in seen:
        candidates.append(("dreamer", dream))
        seen.add(dream)
    
    # 2) Chaos-based candidates (fallback + diversity)
    k_chaos = 2 if not candidates else 1  # if Prophet gave one, add 1 more chaos child
    for _ in range(k_chaos):
        # choose a position via chaos
        agent.z, pos_idx = _chaotic_index(agent.z, agent.r, length)
        # choose an amino acid via chaos
        agent.z, aa_idx = _chaotic_index(agent.z, agent.r, len(amino_acids))
        cand = _deterministic_point_mutation(parent, pos_idx, aa_idx)

        # occasional (deterministic) swap to mimic your smart_mutate flavor
        agent.z, swap_a = _chaotic_index(agent.z, agent.r, length)
        agent.z, swap_b = _chaotic_index(agent.z, agent.r, length)
        if swap_a != swap_b and mutation_rate > 0.2:
            c = list(cand)
            c[swap_a], c[swap_b] = c[swap_b], c[swap_a]
            cand = ''.join(c)

        if cand not in seen:
            candidates.append(("chaos", cand))
            seen.add(cand)


    # Safety: if for some reason we ended up with no candidates, just return parent
    if not candidates:
        feats = _estimate_deltas_batch(
            [parent],
            amp_model, tox_model, stab_model,
            archive_kmers=archive_kmers,
            target_len=target_len,
            k=3
        )
        reasons = feats[0]
        return parent, reasons, "fallback", []


    # 3) Batch-score candidates by agent motives
    cand_seqs = [c for _, c in candidates]
    feats = _estimate_deltas_batch(
        cand_seqs,
        amp_model, tox_model, stab_model,
        archive_kmers=archive_kmers,
        target_len=target_len,
        k=3
    )

    scored = []
    for (source, cand), features in zip(candidates, feats):
        score = agent.reason_score(features)
        scored.append((score, source, cand, features))

    scored.sort(reverse=True, key=lambda x: x[0])

    # Gradually reduce prophet/dreamer influence as stagnation deepens
    # At 0 stagnant: full prophet/dreamer, at 400 stagnant: 20% chance only
    _guided_prob = max(0.20, 1.0 - _stagnant_gens_global / 400.0)
    converging = (random.random() > _guided_prob)
    if converging:
        chaos_candidates = [(s, src, c, f) for s, src, c, f in scored if src == "chaos"]
        if chaos_candidates:
            best_score, best_source, best_cand, best_reasons = chaos_candidates[0]
            return best_cand, best_reasons, best_source, scored

    # Force chaos to win more often — dreamer dominates early before world model is calibrated
    _chaos_override = 0.50 if scored[0][1] == "dreamer" else 0.35
    if scored[0][1] in ("prophet", "dreamer") and random.random() < _chaos_override:
        chaos_candidates = [(s, src, c, f) for s, src, c, f in scored if src == "chaos"]
        if chaos_candidates:
            best_score, best_source, best_cand, best_reasons = chaos_candidates[0]
            return best_cand, best_reasons, best_source, scored

    best_score, best_source, best_cand, best_reasons = scored[0]
    return best_cand, best_reasons, best_source, scored

def pareto_dominates_4obj(a, b, keys):
    better_on_one = False
    for key in keys:
        if a[key] < b[key]:
            return False
        if a[key] > b[key]:
            better_on_one = True
    return better_on_one


# ============================================================
# 🗺️ MAP-Elites Grid
# ============================================================

# Grid axes: net charge (-2 to 12, step 1) x hydrophobicity (0.1 to 0.75, step 0.05)
CHARGE_BINS = list(range(-2, 13))                              # -2,-1,0,...,12  → 15 bins
HYDRO_BINS  = [round(x * 0.025, 3) for x in range(4, 32)]     # 0.10,0.125,...,0.775 → 28 bins

def get_cell(peptide: str):
    """Return (charge_bin, hydro_bin) grid indices for a peptide."""
    charge = calculate_net_charge(peptide)
    hydro  = calculate_hydrophobicity(peptide)

    # clamp to grid bounds
    charge = max(CHARGE_BINS[0], min(CHARGE_BINS[-1], charge))
    hydro  = max(HYDRO_BINS[0],  min(HYDRO_BINS[-1],  hydro))

    # nearest bin
    ci = min(range(len(CHARGE_BINS)), key=lambda i: abs(CHARGE_BINS[i] - charge))
    hi = min(range(len(HYDRO_BINS)),  key=lambda i: abs(HYDRO_BINS[i]  - hydro))
    return ci, hi


PARETO_CELL_SIZE = 5  # max non-dominated sequences per cell
PARETO_KEYS = ["amp", "safety", "stability", "realism"]

def map_elites_update(grid: dict, peptide: str, fitness: float,
                      amp: float = 0.5, tox: float = 0.5,
                      stab: float = 0.5, real: float = 0.5) -> bool:
    """
    Insert peptide into pareto front for its MAP-Elites cell.
    Each cell stores up to PARETO_CELL_SIZE non-dominated sequences.
    grid keys: (ci, hi), values: list of (fitness, peptide, {objectives})
    """
    ci, hi = get_cell(peptide)
    key = (ci, hi)
    entry = {
        "fitness": fitness,
        "peptide": peptide,
        "amp": amp,
        "safety": 1.0 - tox,
        "stability": stab,
        "realism": real,
    }

    if key not in grid:
        grid[key] = [entry]
        return True

    front = grid[key]

    # Check if new entry is dominated by any existing entry
    for existing in front:
        if pareto_dominates_4obj(existing, entry, PARETO_KEYS):
            return False  # dominated, skip

    # Remove entries dominated by the new one
    front = [e for e in front if not pareto_dominates_4obj(entry, e, PARETO_KEYS)]
    front.append(entry)

    # Cap cell size — keep highest fitness if over limit
    if len(front) > PARETO_CELL_SIZE:
        front = sorted(front, key=lambda e: e["fitness"], reverse=True)[:PARETO_CELL_SIZE]

    grid[key] = front
    return True


def map_elites_sample(grid: dict, n: int) -> list[str]:
    """
    Sample n peptides from pareto fronts across cells.
    Bias toward lower-fitness cells to encourage exploration.
    """
    if not grid:
        return []
    keys = list(grid.keys())
    # cell fitness = mean of front fitnesses
    cell_fits = np.array([
        float(np.mean([e["fitness"] for e in grid[k]]))
        for k in keys
    ])
    # inverse-fitness weighting: lower fitness cells sampled more
    inv = 1.0 / (cell_fits + 1e-6)
    weights = inv / inv.sum()
    
    n_sample = min(n, len(keys))
    chosen = np.random.choice(len(keys), size=n_sample,
                            replace=False, p=weights)
    result = []
    seen_peptides = set()
    for i in chosen:
        front = grid[keys[i]]
        pep = random.choice(front)["peptide"]
        if pep not in seen_peptides:
            result.append(pep)
            seen_peptides.add(pep)
    return result




def map_elites_stats(grid: dict) -> tuple:
    """Returns (n_filled, total_cells, mean_fitness, max_fitness)."""
    total = len(CHARGE_BINS) * len(HYDRO_BINS)
    filled = len(grid)
    if not grid:
        return filled, total, 0.0, 0.0
    all_fits = [e["fitness"] for front in grid.values() for e in front]
    return filled, total, float(np.mean(all_fits)), float(np.max(all_fits))

def crowding_penalty(peptide: str) -> float:
    """
    Returns a penalty multiplier based on how many times this MAP-Elites
    cell has already produced high-fitness sequences.
    1.0 = unexplored cell, down to _CROWDING_PENALTY_MAX for saturated cells.
    """
    ci, hi = get_cell(peptide)
    visits = _cell_visit_counts.get((ci, hi), 0)
    if visits < _CROWDING_PENALTY_START:
        return 1.0
    # log decay: lots of visits → strong penalty, but never zero
    penalty = 1.0 - 0.04 * math.log(1 + visits - _CROWDING_PENALTY_START)
    return float(_clamp(penalty, _CROWDING_PENALTY_MAX, 1.0))


def crowding_update(peptide: str, fitness: float, threshold: float = 0.70):
    """
    Record a visit to this cell if the sequence is high-fitness.
    Only counts sequences above threshold to avoid polluting with junk.
    """
    if fitness >= threshold:
        ci, hi = get_cell(peptide)
        key = (ci, hi)
        _cell_visit_counts[key] = _cell_visit_counts.get(key, 0) + 1

def cell_diversity_bonus(peptide: str) -> float:
    """
    Returns a fitness bonus multiplier for sequences in underrepresented
    MAP-Elites cells. Complements crowding_penalty — instead of only
    penalizing overcrowded cells, we also reward sequences that fill
    sparse or empty cells.
    1.05 = empty cell (never visited)
    1.0  = moderately visited cell
    0.95 = heavily visited (handled by crowding_penalty separately)
    """
    ci, hi = get_cell(peptide)
    visits = _cell_visit_counts.get((ci, hi), 0)

    if visits == 0:
        return 1.08  # empty cell — strong reward for genuinely new niche
    elif visits < 10:
        return 1.04  # sparse cell — moderate reward
    elif visits < _CROWDING_PENALTY_START:
        return 1.0   # normal cell — no bonus or penalty
    else:
        return 1.0   # crowding_penalty handles the penalty side


def migrate_islands(islands: list[list[str]], migration_rate: float) -> list[list[str]]:
    """
    Ring migration: each island sends its top migrants to the next island.
    Replaces the weakest sequences in the receiving island.
    """
    n_islands = len(islands)
    # compute per island so size drift doesn't corrupt smaller islands
    island_sizes = [len(isl) for isl in islands]
    
    # collect migrants from each island (top by fitness proxy = just take first n,
    # since islands are already sorted by the end of each generation)
    
    migrants = []
    for isl, size in zip(islands, island_sizes):
        n = max(1, int(size * migration_rate))
        migrants.append(isl[:n])
        
    # ring: island 0 -> island 1 -> ... -> island N-1 -> island 0
    new_islands = []
    for i, island in enumerate(islands):
        donor = migrants[(i - 1) % n_islands]
        n = len(donor)
        new_island = island[:-n] if n < len(island) else []
        new_island = new_island + donor
        random.shuffle(new_island)
        new_islands.append(new_island)
    
    return new_islands


def run_simulation():

    # Load models
    print("📦 Loading models...")

    global world_model, naturalness_model, prophet_model
    global global_best_peptide, global_best_score
    global _stagnant_gens_global
    global niche_archive, niche_penalty_active
    global population_size

    amp_model = keras.models.load_model('models/amp_model.keras', compile=False)
    toxicity_model = keras.models.load_model('models/toxicity_cnn_model.keras', compile=False)
    stability_model = keras.models.load_model('models/stability_cnn_model.keras', compile=False)
    global stability_model_global
    stability_model_global = stability_model
    print("✅ Stability CNN loaded")

    naturalness_model = keras.models.load_model('models/naturalness_discriminator.keras')
    global mic_model
    try:
        mic_model = keras.models.load_model('models/mic_ecoli_model.keras', compile=False)
        print("✅ Models loaded: AMP, Toxicity, Stability, Naturalness Discriminator, MIC E.coli")
    except Exception as e:
        mic_model = None
        print(f"⚠️ MIC model not found ({e}). Running without MIC objective.")


    # 🔮 Prophet mutation model (learned from past runs)
    if USE_AGENT:
        try:
            prophet_model = keras.models.load_model('models/prophet_model.keras', compile=False)
            print("✅ Prophet model loaded (AI-guided mutation online)")
        except Exception as e:
            prophet_model = None
            print(f"⚠️ Prophet model not available ({e}). Falling back to purely cognitive mutations.")

        # 🧠 Build internal Peptide World Model (PWM)
        global world_model
        vocab_size = len(amino_acids) + 1
        world_model = build_world_model(max_len, vocab_size)
        print("✅ Internal world model initialized")
    else:
        prophet_model = None
        world_model = None
        print("🔕 Agent OFF — world model and prophet model skipped")

    # 🧠 Initialize cognitive controller
    if USE_AGENT:
        if USE_ISLANDS:
            agents = [get_island_agent(i) for i in range(N_ISLANDS)]
            agent = agents[0]  # default for non-island code paths
            print(f"🧠 {N_ISLANDS} island agents initialized with distinct personalities")
        else:
            agents = None
            agent = AgentController()
            print("🧠 Cognitive agent initialized (USE_AGENT=True)")
    else:
        agents = None
        agent = None

        print("⚙️  Running as standard GA baseline (USE_AGENT=False)")

    load_salient_motifs()
    run_start_time = datetime.now()

    # Similarity log path (unique per run)
    os.makedirs('data/similarity_logs', exist_ok=True)
    _agent_label = "AGENT_ON" if USE_AGENT else "AGENT_OFF"
    RUN_TAG = run_start_time.strftime('%Y%m%d_%H%M') + f"_MAPE_{peptide_length}mer"
    SIMILARITY_LOG_PATH = f"data/similarity_logs/similarity_log_{RUN_TAG}.csv"
    print(f"🏷️  Run tag: {RUN_TAG}")


    # 🔗 Make the other logs run-specific as well
    global WORLD_MODEL_ERROR_FILE, NOVELTY_STATS_FILE, ENTROPY_STATS_FILE, MUTATION_HISTORY_FILE
    WORLD_MODEL_ERROR_FILE = f"data/world_model_error_{RUN_TAG}.csv"
    NOVELTY_STATS_FILE     = f"data/novelty_stats_{RUN_TAG}.csv"
    ENTROPY_STATS_FILE     = f"data/entropy_stats_{RUN_TAG}.csv"
    MUTATION_HISTORY_FILE  = f"data/mutation_rate_history_{RUN_TAG}.txt"
    FITNESS_STATS_FILE     = f"data/fitness_stats_{RUN_TAG}.csv"

    PEAK_EVENTS_FILE = f"data/peak_events_{RUN_TAG}.csv"
    with open(PEAK_EVENTS_FILE, 'w') as f:
        f.write("Generation,EventType,PeakFitness,PeakNumber,Note\n")

    mode = "normal"

    # Per-run logs
    AGENT_LOG  = f"data/agent_state_{RUN_TAG}.csv"
    ACTION_LOG = f"data/action_log_{RUN_TAG}.csv"

    # create headers if new
    if USE_AGENT:
        if not os.path.exists(AGENT_LOG):
            with open(AGENT_LOG, 'w') as f:
                f.write(
                    "Generation," +
                    ",".join(AgentController().motives) +
                    ",z,r,stagnant,avg_sim,mean_entropy,total_err,avg_realism\n"
                )

    if not os.path.exists(ACTION_LOG):
        with open(ACTION_LOG, 'w') as f:
            f.write("Generation,ActionType\n")

    REASON_LOG = f"data/agent_reasoning_{RUN_TAG}.csv"
    global _REASON_LOG_PATH
    _REASON_LOG_PATH = REASON_LOG

    if USE_AGENT:
        with open(REASON_LOG, 'w') as f:
            f.write("Generation,Parent,Child,Source,Cand_AMP,Cand_Safety,Cand_Stability,Cand_Realism,Cand_Novelty,Reason_Score\n")
    else:
        # Stub so downstream references don't crash
        with open(REASON_LOG, 'w') as f:
            f.write("Generation,Parent,Child,Source,Cand_AMP,Cand_Safety,Cand_Stability,Cand_Realism,Cand_Novelty,Reason_Score\n")

    # Create trait stats output folder if needed
    os.makedirs('data/trait_stats', exist_ok=True)
    os.makedirs('data/evolution_histories', exist_ok=True)

    trait_log_path = f"data/trait_stats/trait_stats_{RUN_TAG}.csv"
    THIS_RUN_FILE = f"data/evolution_histories/evolution_run_{RUN_TAG}.csv"


    # Initialize islands

    saved_pop = load_population()

    if saved_pop:
        population = saved_pop
        print(f"🔄 Resuming from saved population: {len(population)} sequences.")
        import json
        state_path = 'data/run_state.json'



        if os.path.exists(state_path):
            with open(state_path) as f:
                state = json.load(f)

            best_fitness_so_far   = state.get('best_fitness_so_far', 0)
            best_avg_so_far       = state.get('best_avg_so_far', 0)
            stagnant_generations  = state.get('stagnant_generations', 0)
            _stagnant_gens_global = stagnant_generations
            global_best_score     = state.get('global_best_score', -np.inf)
            global_best_peptide   = state.get('global_best_peptide', None)
            # Restore historical peaks
            _saved_peaks = state.get('historical_peaks', [])
            if _saved_peaks:
                run_simulation._historical_peaks = [
                    (set(s for kmer_list in pk for s in kmer_list), fit)
                    for pk, fit in _saved_peaks
                ]
                print(f"🔄 Restored {len(run_simulation._historical_peaks)} historical peaks.")
            # Restore restart gen anchor
            run_simulation._restart_gen = state.get('restart_gen', 0)
            run_simulation._historical_peak_ages = state.get('historical_peak_ages', [])
            print(f"🔄 Restored state: gen {state.get('gen')}, stagnant={stagnant_generations}, best_avg={best_avg_so_far:.4f}")

        else:
            print("⚠️ No run state found — starting stagnation tracking fresh.")


        # Prepopulate archive from saved population to prevent novelty inflation on resume
        print(f"🔄 Prepopulating novelty archive from saved population ({len(population)} seqs)...")
        archive_add_sequences(population, k=ARCHIVE_K)
        print(f"🔄 Archive prepopulated: {len(archive_kmers)} entries")

    else:

        population = [''.join(random.choices(amino_acids, k=peptide_length)) for _ in range(population_size)]
        print(f"🌱 Random population: {population_size} sequences.")


    # Split population into islands if using island model
    if USE_ISLANDS:
        chunk = len(population) // N_ISLANDS
        islands = [population[i*chunk:(i+1)*chunk] for i in range(N_ISLANDS)]
        print(f"🏝️  Island model: {N_ISLANDS} islands x {chunk} sequences each")
    else:
        islands = [population]


    # Mark a new run in mutation history
    with open(MUTATION_HISTORY_FILE, 'a') as f:
        f.write(f"\n--- New Run Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f.write(f"Population size: {population_size}, Generations: {generations}, Peptide length: {peptide_length}\n")
        f.write("Generation,Mutation Rate\n")


    stagnant_generations = 0
    best_fitness_so_far = 0
    run_simulation._last_peak_fitness = 0.0
    run_simulation._abandonment_fired = False
    best_avg_so_far = 0
    global_best_score = -np.inf
    global_best_peptide = None

    # One MAP-Elites grid per island to preserve island diversity
    map_elites_grid = {}
    for gen in range(1, generations + 1):
        gen_start = datetime.now()
        print(f"\\n🌱 Generation {gen}...")

        # ── Per-gen agent rotation ──────────────────────────────────────
        if USE_ISLANDS and agents is not None:
            # Each gen, the "primary" agent rotates so logs reflect diversity
            agent = agents[gen % N_ISLANDS]

        # ── Goal mode overrides ─────────────────────────────────────────
        if gen < 200:
            tournament_k       = 2
            explore_parent_prob = 0.30
        else:
            tournament_k       = 3
            explore_parent_prob = 0.15


        for isl_idx, isl_agent in enumerate(agents if (USE_ISLANDS and agents) else ([agent] if agent is not None else [])):
            if USE_ISLANDS and agents:
                mode = isl_agent.goal_mode
            if mode == "explore":
                explore_parent_prob = 0.40; tournament_k = 2
                isl_agent.delta_tie = 0.06
                isl_agent.r = min(4.0, isl_agent.r + 0.02)
            elif mode == "exploit":
                explore_parent_prob = 0.05; tournament_k = 4
                isl_agent.delta_tie = 0.01
                isl_agent.r = max(3.6, isl_agent.r - 0.02)
            elif mode == "stabilize":
                explore_parent_prob = 0.10; tournament_k = 3
                isl_agent.r = max(3.6, isl_agent.r - 0.03)

        # ── Score each island independently ────────────────────────────
        island_dfs   = []   # per-island DataFrames
        island_new   = []   # per-island new populations

        for isl_idx, island in enumerate(islands):
            isl_agent = agents[isl_idx] if (USE_ISLANDS and agents) else agent

            encoded = np.array([encode_int(p, max_len=max_len) for p in island])
            amp_sc  = amp_model.predict(encoded, verbose=0).flatten()
            tox_sc  = toxicity_model.predict(encoded, verbose=0).flatten()
            stab_sc = stability_model.predict(encoded, verbose=0).flatten()
            nat_sc  = naturalness_model.predict(encoded, verbose=0).flatten()

            scores = []
            for idx, pep in enumerate(island):
                amp_i, tox_i, stab_i = float(amp_sc[idx]), float(tox_sc[idx]), float(stab_sc[idx])
                result = score_peptide(
                    pep, amp_i, tox_i, stab_i,
                    previous_peptides=island,
                    agent=isl_agent if USE_AGENT else None,
                    archive_kmers=archive_kmers,
                )

                amp_v, tox_v, stab_v, fitness, sol, agg, pI, boman, net_charge, hydrophobicity, sol_tag, agg_tag, realism_score, hydro_moment = result
                mic_score = predict_mic_score(pep)
                update_salient_motifs(pep, fitness)
                map_elites_update(map_elites_grid, pep, fitness,
                                amp=amp_v, tox=tox_v, stab=stab_v, real=realism_score)

                crowding_update(pep, fitness)
                tag = assess_peptide_quality(pep)
                scores.append((
                    pep, amp_v, tox_v, stab_v, sol, agg, pI, boman,
                    net_charge, hydrophobicity, sol_tag, agg_tag,
                    fitness, realism_score, tag,
                    hydro_moment, mic_score,
                ))

            isl_df = pd.DataFrame(scores, columns=[
                'Peptide',
                'AMP_Score', 'Toxicity_Score', 'Stability_Score',
                'Solubility_Score', 'Aggregation_Risk', 'Isoelectric_Point',
                'Boman_Index', 'Net_Charge', 'Hydrophobicity',
                'Solubility_Tag', 'Aggregation_Tag',
                'Fitness_Score', 'Realism_Score', 'Quality_Tag',
                'Hydrophobic_Moment', 'MIC_Score',
            ])
            isl_df['Generation'] = gen
            isl_df['Island']     = isl_idx
            island_dfs.append(isl_df)

        # ── Combine for global logging/stats ────────────────────────────
        generation_df = pd.concat(island_dfs, ignore_index=True)
        population    = generation_df['Peptide'].tolist()

        # Novelty archive update
        archive_add_sequences(generation_df["Peptide"].tolist(), k=ARCHIVE_K)

        # World model update (uses combined data)
        if USE_AGENT:
            pwm_add_samples(generation_df)
            pwm_train_one_epoch()

        # Junk cut
        junk_cut = 0.05 if gen <= 20 else 0.10
        avg_realism = generation_df['Realism_Score'].mean()
        generation_df = generation_df[generation_df["Realism_Score"] >= junk_cut].copy()

        avg_fitness = generation_df['Fitness_Score'].mean()
        max_fitness = generation_df['Fitness_Score'].max()
        std_fitness = generation_df['Fitness_Score'].std()
        avg_charge  = generation_df['Net_Charge'].mean()
        avg_hydro   = generation_df['Hydrophobicity'].mean()
        avg_agg     = generation_df['Aggregation_Risk'].mean()

        if avg_realism < 0.10 and gen > 20:
            print("❌ Avg realism extremely low — stopping.")
            break

        if gen == 1:
            with open(trait_log_path, 'w') as f:
                f.write("Generation,NetCharge,Hydrophobicity,AggregationRisk,Realism\n")
        with open(trait_log_path, 'a') as f:
            f.write(f"{gen},{avg_charge:.4f},{avg_hydro:.4f},{avg_agg:.4f},{avg_realism:.4f}\n")

        print(f"📊 Std Dev Fitness: {std_fitness:.4f}")

        _sim_sample = random.sample(population, min(100, len(population)))
        avg_sim, min_sim, max_sim = similarity_stats(_sim_sample)
        gen_time = datetime.now() - gen_start

        # Goal mode for primary agent
        if USE_AGENT and agent is not None:
            mode = agent.maybe_switch_goal_mode(stagnant_generations, avg_sim, avg_realism)

        if USE_AGENT:
            total_err = log_world_model_error(generation_df, gen)
            if total_err is None: total_err = 999.0
        else:
            total_err = 0.0

        confident_chaos = (avg_sim > 0.70 and total_err < 0.18)
        emergency_chaos = (avg_sim > 0.78 and stagnant_generations > 40)
        low_diversity_flag = confident_chaos or emergency_chaos


        with open(SIMILARITY_LOG_PATH, 'a') as sim_log:
            if gen == 1: sim_log.write("Generation,Similarity\n")
            sim_log.write(f"{gen},{avg_sim:.4f}\n")

        log_novelty_stats(avg_sim, min_sim, max_sim, gen)
        log_entropy_stats(population, gen)
        update_position_entropy(population)

        # Niche clearing: if population has converged hard, archive the dominant cluster
        # and activate penalty to discourage re-convergence to the same scaffold
        global niche_archive, niche_penalty_active
        # Trigger on convergence OR prolonged stagnation
        stagnation_trigger = stagnant_generations > 200 and niche_penalty_active == 0
        similarity_trigger = avg_sim > 0.72 and niche_penalty_active == 0
        if similarity_trigger or stagnation_trigger:
            top_seqs = generation_df.sort_values('Fitness_Score', ascending=False).head(20)['Peptide'].tolist()
            dominant_kmers = set()
            for s in top_seqs:
                dominant_kmers.update(kmer_set(s, ARCHIVE_K))
            niche_archive.append((dominant_kmers, float(generation_df['Fitness_Score'].max())))
            niche_penalty_active = NICHE_PENALTY_GENS
            trigger_reason = "similarity" if similarity_trigger else "stagnation"
            print(f"🔴 Niche cleared at gen {gen} ({trigger_reason}, avg_sim={avg_sim:.3f}, stagnant={stagnant_generations}). Penalty active for {NICHE_PENALTY_GENS} gens.")

        mean_H, max_H = compute_population_entropy(population)

        if gen < 30 and avg_sim > 0.55:
            print(f"⚠️ Early convergence (avg_sim={avg_sim:.3f}) — injecting exploratory peptides.")
            extra = [''.join(random.choices(amino_acids, k=peptide_length))
                     for _ in range(int(0.2 * population_size))]
            extra = [p for p in extra if is_realistic(p, gen)]
            population.extend(extra)

        if gen % 50 == 0:
            prune_salient_motifs()
            top_peps_path = f"data/top_peptides_gen_{gen:04d}_{RUN_TAG}.csv"
            generation_df.head(5).to_csv(top_peps_path, index=False)
            print(f"🏅 Saved top peptides snapshot: {top_peps_path}")

        # Mutation rate schedule
        
        initial_mutation = 0.4; final_mutation = 0.08
        # Use gen relative to last restart for cosine decay
        _restart_gen = getattr(run_simulation, '_restart_gen', 0)
        gens_since_restart = gen - _restart_gen
        cosine_decay = 0.5 * (1 + math.cos(math.pi * min(gens_since_restart / generations, 1.0)))
        base_mutation = final_mutation + (initial_mutation - final_mutation) * cosine_decay
        
        mutation_rate = base_mutation * (1.2 if avg_sim > 0.65 else 0.8 if avg_sim < 0.5 else 1.0)
        mutation_rate *= min(max(avg_realism, 0.4), 1.0)
        if stagnant_generations > 100:
            boost = min(0.1 + stagnant_generations / 1000, 0.5)
            mutation_rate = min(0.6, mutation_rate + boost)

        if gens_since_restart > 0.7 * generations:
            mutation_rate = min(mutation_rate, 0.18)
        
        mutation_rate = float(np.clip(mutation_rate, 0.10, 0.65))

        print(f"\\n{'-'*60}")
        print(f"🌱 Generation {gen}")
        print(f"{'-'*60}")
        print(f"📈 Avg Fitness: {avg_fitness:.4f} | Best: {max_fitness:.4f}")

        print("\\n🏅 Top Peptides:")
        top_peptides = generation_df.sort_values('Fitness_Score', ascending=False).head(3)
        for rank, (_, row) in enumerate(top_peptides.iterrows(), 1):
            isl = int(row.get('Island', -1))
            print(f"  {rank}. {row['Peptide']} | Fit: {row['Fitness_Score']:.4f} | Island {isl}")

        top_row = generation_df.sort_values('Fitness_Score', ascending=False).iloc[0]
        if top_row['Fitness_Score'] > global_best_score:
            global_best_score   = top_row['Fitness_Score']
            global_best_peptide = top_row['Peptide']
            print(f"💎 New global best! Fitness: {global_best_score:.4f} — {global_best_peptide}")


        n_peaks = len(getattr(run_simulation, '_historical_peaks', []))
        peak_str = f" | Peak #{n_peaks} active" if n_peaks > 0 else ""
        print(f"🔬 Avg similarity: {avg_sim:.4f}{peak_str}")

        # Aggregate MAP-Elites stats across all island grids

        all_grid_entries = [e for front in map_elites_grid.values() for e in front]
        me_filled = len(map_elites_grid)
        me_total = len(CHARGE_BINS) * len(HYDRO_BINS)
        me_mean = float(np.mean([e["fitness"] for e in all_grid_entries])) if all_grid_entries else 0.0
        me_max  = float(np.max([e["fitness"]  for e in all_grid_entries])) if all_grid_entries else 0.0


        print(f"🗺️  MAP-Elites: {me_filled}/{me_total} cells | mean: {me_mean:.4f} | best: {me_max:.4f}")

        # ── Adaptive archive pruning ────────────────────────────────────
        # Every 100 gens, evict lowest-fitness cells when archive is >80% full
        # This creates negative space for fresh exploration
        if gen % 100 == 0 and me_filled > 0:
            occupancy = me_filled / me_total
            if occupancy > 0.80:
                # Sort cells by mean fitness, evict bottom 15%
                cell_means = {
                    k: float(np.mean([e["fitness"] for e in front]))
                    for k, front in map_elites_grid.items()
                }
                sorted_cells = sorted(cell_means.items(), key=lambda x: x[1])
                n_evict = max(5, int(me_filled * 0.15))
                evicted = 0
                for k, _ in sorted_cells[:n_evict]:
                    # Only evict if cell fitness is below archive mean
                    if cell_means[k] < me_mean * 0.90:
                        del map_elites_grid[k]
                        evicted += 1
                if evicted > 0:
                    print(f"🗑️  Archive pruned: evicted {evicted} low-fitness cells (occupancy {occupancy:.0%} → {len(map_elites_grid)/me_total:.0%})")


        top5_mean = generation_df.sort_values('Fitness_Score', ascending=False) \
                         .head(5)['Fitness_Score'].mean()

        if stagnant_generations % 100 == 0 and stagnant_generations > 0:
            print(f"⚠️  {stagnant_generations} stagnant generations")

        if top5_mean > best_fitness_so_far + 0.005 or avg_fitness > best_avg_so_far + 0.005:
            best_fitness_so_far = max(best_fitness_so_far, top5_mean)
            best_avg_so_far = max(best_avg_so_far, avg_fitness)
            stagnant_generations = 0
            _stagnant_gens_global = 0
        else:
            stagnant_generations += 1
            _stagnant_gens_global = stagnant_generations
            if stagnant_generations == 1:
                print(f"📉 Stagnation started at gen {gen} (top5={top5_mean:.4f}, avg={avg_fitness:.4f}, best_avg={best_avg_so_far:.4f})")


        if USE_AGENT:
            for isl_agent in (agents if (USE_ISLANDS and agents) else [agent]):
                isl_agent.update_progress(top5_mean, best_fitness_so_far, avg_sim, 1.0 - avg_sim)


        # ── Deliberate peak abandonment — adaptive trigger ──────────────
        # Trigger when improvement rate drops below threshold, not at fixed gen count
        # Minimum 100 stagnant gens before considering abandonment
        _last = getattr(run_simulation, '_last_peak_fitness', None)
        if _last is None:
            run_simulation._last_peak_fitness = best_fitness_so_far
            _last = best_fitness_so_far
        _improvement_rate = (best_fitness_so_far - _last) / max(stagnant_generations, 1)
        _abandonment_ready = (
            stagnant_generations >= 100 and
            _improvement_rate < 0.0001 and  # essentially flat
            stagnant_generations >= 150     # grace period to fully exploit
        )

        if _abandonment_ready and not getattr(run_simulation, '_abandonment_fired', False):
            run_simulation._abandonment_fired = True
            run_simulation._last_peak_fitness = best_fitness_so_far
            print(f"\n🏔️  PEAK ABANDONMENT at gen {gen} — rate={_improvement_rate:.6f}, stagnant={stagnant_generations}.")

            top_seqs = generation_df.sort_values('Fitness_Score', ascending=False).head(30)['Peptide'].tolist()
            peak_kmers = set()
            for s in top_seqs:
                peak_kmers.update(kmer_set(s, ARCHIVE_K))
            if not hasattr(run_simulation, '_historical_peaks'):
                run_simulation._historical_peaks = []


            run_simulation._historical_peaks.append((peak_kmers, float(generation_df['Fitness_Score'].max())))
            if not hasattr(run_simulation, '_historical_peak_ages'):
                run_simulation._historical_peak_ages = []
            run_simulation._historical_peak_ages.append(stagnant_generations)
            print(f"   Peak #{len(run_simulation._historical_peaks)} archived (fitness {generation_df['Fitness_Score'].max():.4f}).")

            dominant_kmers = set()
            
            for s in top_seqs[:20]:
                dominant_kmers.update(kmer_set(s, ARCHIVE_K))
            niche_archive.append((dominant_kmers, float(generation_df['Fitness_Score'].max())))
            niche_penalty_active = 300

            with open(PEAK_EVENTS_FILE, 'a') as f:
                f.write(f"{gen},abandonment,{generation_df['Fitness_Score'].max():.4f},{len(run_simulation._historical_peaks)},adaptive abandonment rate={_improvement_rate:.6f}\n")
            print(f"   Niche penalty activated for 300 gens.")
            print(f"   Fitness will DROP — this is intentional. Finding new basin...\n")




        # Reset _last_peak_fitness when stagnation counter resets
        if stagnant_generations == 0:
            run_simulation._abandonment_fired = False
            run_simulation._last_peak_fitness = None  # will be re-initialized next stagnation

        # Early stop check
        # ── True restart at 400 stagnant gens ──────────────────────────
        RESTART_TRIGGER = 400
        if stagnant_generations >= RESTART_TRIGGER:

            print(f"\n🔄 TRUE RESTART at gen {gen} — {stagnant_generations} stagnant gens.")

            # Collect top 10% of archive as seeds
            all_archive = [e for front in map_elites_grid.values() for e in front]

            # Pick elites that span diverse MAP-Elites cells, not just top fitness
            # This prevents restart seeding the same biochemical neighborhood
            n_keep = max(10, int(population_size * 0.10))
            all_keys = list(map_elites_grid.keys())
            random.shuffle(all_keys)
            elite_seqs = []
            seen_cells = set()
            # First pass: one best sequence per cell, prioritize underrepresented cells
            for k in all_keys:
                if k not in seen_cells:
                    front = sorted(map_elites_grid[k], key=lambda e: e["fitness"], reverse=True)
                    elite_seqs.append(front[0]["peptide"])
                    seen_cells.add(k)
                if len(elite_seqs) >= n_keep:
                    break
            # If not enough, fill with top fitness from remaining
            if len(elite_seqs) < n_keep:
                all_archive.sort(key=lambda e: e["fitness"], reverse=True)
                for e in all_archive:
                    if e["peptide"] not in elite_seqs:
                        elite_seqs.append(e["peptide"])
                    if len(elite_seqs) >= n_keep:
                        break

            print(f"   Keeping {len(elite_seqs)} elite sequences from archive.")

            # Save peak to historical peaks list for long-term novelty
            peak_kmers = set()
            for s in elite_seqs:
                peak_kmers.update(kmer_set(s, ARCHIVE_K))
            if not hasattr(run_simulation, '_historical_peaks'):
                run_simulation._historical_peaks = []
            run_simulation._historical_peaks.append((peak_kmers, best_fitness_so_far))
            if not hasattr(run_simulation, '_historical_peak_ages'):
                run_simulation._historical_peak_ages = []
            run_simulation._historical_peak_ages.append(stagnant_generations)
            print(f"   Archived peak #{len(run_simulation._historical_peaks)} (fitness {best_fitness_so_far:.4f}) to historical peaks.")

            # Reinitialize 90% of population

            n_random = population_size - len(elite_seqs)
            new_randoms = []
            while len(new_randoms) < n_random:
                # 50% pure random, 50% elite-seeded (mutate an elite heavily)
                if random.random() < 0.5 or not elite_seqs:
                    seq = ''.join(random.choices(amino_acids, k=peptide_length))
                else:
                    # Wild mutate to escape biochemical neighborhood, not just sequence space
                    base = list(random.choice(elite_seqs))
                    for i in range(len(base)):
                        if random.random() < 0.6:
                            base[i] = random.choice(amino_acids)  # fully random, not group-conservative
                    seq = ''.join(base)
                if is_realistic(seq, gen):
                    new_randoms.append(seq)

            new_population = elite_seqs + new_randoms
            random.shuffle(new_population)

            # Redistribute across islands
            chunk = len(new_population) // N_ISLANDS
            islands = [new_population[i*chunk:(i+1)*chunk] for i in range(N_ISLANDS)]
            new_islands = islands


            # Reset stagnation counters and decay clock
            stagnant_generations = 0
            _stagnant_gens_global = 0
            run_simulation._restart_gen = gen

            # Reset stagnation counters and decay clock
            stagnant_generations = 0
            _stagnant_gens_global = 0
            run_simulation._restart_gen = gen
            run_simulation._last_peak_fitness = None
            
            # Aggressively decay salient motifs so old peak bias is cleared
            global salient_motifs
            salient_motifs = {k: v * 0.1 for k, v in salient_motifs.items() if v * 0.1 > 0.001}
            print(f"   Salient motifs decayed: {len(salient_motifs)} motifs retained at 10% strength.")
            # Decay cell visit counts so crowding penalties don't carry over to new peak
            global _cell_visit_counts
            _cell_visit_counts = {k: max(1, v // 4) for k, v in _cell_visit_counts.items()}
            print(f"   Cell visit counts decayed to 25% — crowding penalties reset for new basin.")

            # Reset world model weights to prevent old peak bias in dreamer rollouts
            if world_model is not None:
                vocab_size = len(amino_acids) + 1
                world_model = build_world_model(max_len, vocab_size)
                global pwm_buffer
                pwm_buffer = []
                print(f"   World model reset — dreamer rollouts unbiased for new basin exploration.")

            # Reset fitness tracking so stagnation detection recalibrates
            best_fitness_so_far = 0.0
            best_avg_so_far = 0.0
            # Clear MAP-Elites grid to allow fresh exploration
            map_elites_grid = {}
            # Prepopulate archive from elites to prevent novelty inflation post-restart
            archive_add_sequences(elite_seqs, k=ARCHIVE_K)
            # Population size modulation — temporarily expand to accelerate new peak discovery
            run_simulation._expanded_population_size = int(population_size * 1.5)
            run_simulation._expansion_active_until = gen + 100
            with open(PEAK_EVENTS_FILE, 'a') as f:
                f.write(f"{gen},restart,{best_fitness_so_far:.4f},{len(run_simulation._historical_peaks)},full population restart\n")
            print(f"   MAP-Elites grid cleared. Population reinitialized with {len(elite_seqs)} elites + {len(new_randoms)} new sequences.")
            print(f"   Archive prepopulated with {len(elite_seqs)} elite sequences.")
            print(f"   Population temporarily expanded to {run_simulation._expanded_population_size} for 100 gens.")
            print(f"   Fitness tracking reset. Stagnation counter reset. Continuing from gen {gen+1}...\n")


        # Early stop check — only trigger after restart has been attempted
        min_gen_for_stop = int(0.40 * generations)

        if (gen >= min_gen_for_stop
                and best_fitness_so_far >= 0.60
                and stagnant_generations >= STAGNANT_LIMIT):
            print(f"🛑 Early stopping at gen {gen}.")
            break

        if USE_AGENT:
            for isl_agent in (agents if (USE_ISLANDS and agents) else [agent]):
                isl_agent.adapt_chaos(stagnant_generations, avg_sim, total_err, avg_realism)
                # Kick z out of attractor every 100 gens with a small random perturbation
                if gen % 100 == 0:
                    isl_agent.z = _clamp(isl_agent.z + random.uniform(-0.15, 0.15), 0.01, 0.99)

        print(f"🧬 Mutation rate: {mutation_rate:.2f}")
        if gen % 100 == 0:
            print(f"📌 Checkpoint Gen {gen}, Top: {top_row['Fitness_Score']:.4f}")

        append_mutation_rate(gen, mutation_rate)
        append_generation_to_master(gen, generation_df.to_dict('records'))
        generation_df.to_csv(THIS_RUN_FILE, mode='a',
                             header=not os.path.exists(THIS_RUN_FILE), index=False)

        repro_start = datetime.now()
        # ── Per-island reproduction ─────────────────────────────────────
        new_islands = []
        for isl_idx, (island, isl_df) in enumerate(zip(islands, island_dfs)):
            isl_agent = agents[isl_idx] if (USE_ISLANDS and agents) else agent

            # Island-level survivor selection from MAP-Elites + top island seqs
            grid_seqs = map_elites_sample(map_elites_grid, n=len(island) // 2)
            top_isl   = isl_df.sort_values('Fitness_Score', ascending=False) \
                              .head(len(island) // 4)['Peptide'].tolist()
            survivors = list(dict.fromkeys(grid_seqs + top_isl))
            if not survivors:
                survivors = isl_df.sort_values('Fitness_Score', ascending=False) \
                                  .head(len(island) // 2)['Peptide'].tolist()

            # Realism gate
            filtered_df = isl_df[isl_df['Peptide'].isin(survivors)].copy()
            if len(filtered_df) < len(island) // 4:
                filtered_df = isl_df.sort_values('Fitness_Score', ascending=False) \
                                     .head(len(island) // 4)

            weights = get_survivor_weights(survivors, filtered_df,
                                           min_realism_gate=0.10, realism_power=2.0)
            if len(weights) != len(survivors) or sum(weights) <= 0:
                weights = [1.0] * len(survivors)

            # Use expanded population size if in post-restart exploration window
            _exp_size = getattr(run_simulation, '_expanded_population_size', population_size)
            _exp_until = getattr(run_simulation, '_expansion_active_until', 0)
            _active_pop_size = _exp_size if gen <= _exp_until else population_size
            island_target = _active_pop_size // N_ISLANDS
            new_pop = []
            attempts = 0

            while len(new_pop) < island_target:
                attempts += 1
                if attempts > 3000:
                    new_pop.append(''.join(random.choices(amino_acids, k=peptide_length)))
                    attempts = 0
                    continue

                # Chaos injection
                if low_diversity_flag and random.random() < 0.35:
                    base = random.choice(survivors)
                    cm = min(mutation_rate * (1.8 if emergency_chaos and total_err > 0.25 else 2.5), 0.60)
                    wc = smart_mutate(base, cm)
                    if is_realistic(wc, gen) and novelty_floor_check(wc, archive_kmers):
                        new_pop.append(wc)
                        with open(ACTION_LOG, 'a') as f: f.write(f"{gen},chaos_mutate\n")
                        continue

                # Crossover
                if random.random() < 0.5 and len(survivors) >= 2:
                    try:
                        if random.random() < explore_parent_prob:
                            p1, p2 = random.choice(survivors), random.choice(survivors)
                        elif random.random() < 0.5:
                            p1 = tournament_pick(survivors, isl_df, k=tournament_k)
                            p2 = tournament_pick(survivors, isl_df, k=tournament_k)
                        else:
                            p1, p2 = random.choices(survivors, weights=weights, k=2)

                        tries = 0
                        while jaccard_kmer_similarity(p1, p2) > 0.85 and tries < 5:
                            p2 = random.choice(survivors); tries += 1

                    except ValueError:
                        weights = [1.0] * len(survivors)
                        p1, p2 = random.choices(survivors, weights=weights, k=2)

                    child = crossover(p1, p2)
                    if not is_realistic(child, gen) or not novelty_floor_check(child, archive_kmers):
                        continue
                    f1 = isl_df[isl_df['Peptide'] == p1]['Fitness_Score'].values
                    f2 = isl_df[isl_df['Peptide'] == p2]['Fitness_Score'].values
                    if f1.size > 0 and f2.size > 0:
                        # Use world model for fast fitness estimate if available and trained
                        if world_model is not None and len(pwm_buffer) >= PWM_BATCH_SIZE * 4:
                            enc = np.array([encode_int(child, max_len=max_len)], dtype='int32')
                            pwm_pred = world_model.predict(enc, verbose=0)[0]
                            amp_c, tox_c = float(pwm_pred[0]), float(pwm_pred[1])
                            cf_fast = amp_c * 0.5 + (1.0 - tox_c) * 0.5  # rough proxy
                            if cf_fast < 0.45:  # fast reject obvious failures
                                continue
                        enc = np.array([encode_int(child, max_len=max_len)])
                        amp_c = float(amp_model.predict(enc, verbose=0)[0][0])
                        tox_c = float(toxicity_model.predict(enc, verbose=0)[0][0])
                        stab_c = 0.0
                        cf = score_peptide(child, amp_c, tox_c, stab_c,
                                          agent=isl_agent if USE_AGENT else None)[3]
                        if cf >= 0.80 * max(f1[0], f2[0]):
                            new_pop.append(child)
                            with open(ACTION_LOG, 'a') as f: f.write(f"{gen},crossover\n")
                            continue

                # Mutation
                parent = random.choice(survivors)
                if USE_AGENT and isl_agent is not None:

                    mutant, reasons, source, all_scored = cognitive_mutate(
                        parent, mutation_rate, isl_agent,
                        amp_model, toxicity_model, stability_model,
                        archive_kmers, peptide_length)
                    if is_realistic(mutant, gen) and novelty_floor_check(mutant, archive_kmers):
                        new_pop.append(mutant)

                        # Only log 1 in 5 mutations to keep reason log manageable
                        if random.random() < 0.20:
                            with open(REASON_LOG, 'a') as f:
                                # Log winner
                                f.write(f"{gen},{parent},{mutant},{source},"
                                        f"{reasons['amp']:.4f},{reasons['safety']:.4f},"
                                        f"{reasons['stability']:.4f},{reasons['realism']:.4f},"
                                        f"{reasons['novelty']:.4f},"
                                        f"{isl_agent.reason_score(reasons):.4f}\n")
                                # Log rejected candidates for contrastive prophet training
                                for _score, _source, _cand, _reasons in all_scored[1:]:
                                    f.write(f"{gen},{parent},{_cand},{_source}_rejected,"
                                            f"{_reasons['amp']:.4f},{_reasons['safety']:.4f},"
                                            f"{_reasons['stability']:.4f},{_reasons['realism']:.4f},"
                                            f"{_reasons['novelty']:.4f},{_score:.4f}\n")

                        with open(ACTION_LOG, 'a') as f: f.write(f"{gen},mutate_{source}\n")                 
                        continue

                else:
                    mutant = smart_mutate(parent, mutation_rate)
                    if is_realistic(mutant, gen) and novelty_floor_check(mutant, archive_kmers):
                        new_pop.append(mutant)
                        with open(ACTION_LOG, 'a') as f: f.write(f"{gen},mutate_smart\n")
                        continue

            new_islands.append(new_pop)
            print(f"✅ Island {isl_idx}: built {len(new_pop)} sequences")

        repro_time = (datetime.now() - repro_start).total_seconds()
        total_gen_time = (datetime.now() - gen_start).total_seconds()
        proj_remaining = total_gen_time * (generations - gen)
        proj_h = int(proj_remaining // 3600)
        proj_m = int((proj_remaining % 3600) // 60)
        print(f"⏱️ Gen time: {total_gen_time:.1f}s (score: {gen_time.total_seconds():.1f}s repro: {repro_time:.1f}s) | Projected: {proj_h}h {proj_m}m")

        # ── Agent updates (per-island) ──────────────────────────────────
        if USE_AGENT:
            for isl_idx, isl_df in enumerate(island_dfs):
                isl_agent = agents[isl_idx] if (USE_ISLANDS and agents) else agent
                isl_agent.update_weights(isl_df)
                if gen % META_INTROSPECTION_INTERVAL == 0:
                    avg_novelty = 1.0 - avg_sim
                    max_H_th    = math.log2(len(amino_acids))
                    div_norm    = max(0.0, min(1.0, mean_H / max_H_th)) if max_H_th > 0 else 0.0
                    isl_agent.introspect_and_remap({
                        "plateau":          min(stagnant_generations / STAGNANT_LIMIT, 1.0),
                        "novelty":          max(0.0, min(1.0, avg_novelty)),
                        "diversity":        div_norm,
                        "salience_entropy": salience_entropy(),
                        "chaos":            max(0.0, min(1.0, 4.0 * isl_agent.z * (1.0 - isl_agent.z))),
                    })
                isl_agent.w = np.maximum(isl_agent.w, 1e-6)
                isl_agent.w /= isl_agent.w.sum()
                cur_idx = isl_agent.motives.index("curiosity")
                amp_idx = isl_agent.motives.index("amp")
                saf_idx = isl_agent.motives.index("safety")

                isl_agent.w[cur_idx] = min(isl_agent.w[cur_idx], isl_agent.MAX_CUR)
                stab_idx = isl_agent.motives.index("stability")
                isl_agent.w[stab_idx] = max(isl_agent.w[stab_idx], 0.12)
                s = isl_agent.w[amp_idx] + isl_agent.w[saf_idx]

                if s < 0.30:
                    boost = (0.30 - s) / 2.0
                    isl_agent.w[amp_idx] += boost
                    isl_agent.w[saf_idx] += boost
                isl_agent.w = np.maximum(isl_agent.w, isl_agent.MIN_W)
                isl_agent.w /= isl_agent.w.sum()

            # Log primary agent state
            with open(AGENT_LOG, 'a') as f:
                
                f.write(
                    f"{gen}," +
                    ",".join(f"{w:.5f}" for w in agent.w) +
                    f",{agent.z:.6f},{agent.r:.4f},"
                    f"{stagnant_generations},{avg_sim:.4f},"
                    f"{mean_H:.4f},{total_err:.4f},{avg_realism:.4f}\n"
                )

        # ── Snapshots, grid saves, wildcards ───────────────────────────
        if gen % 500 == 0:
            snap = f"data/gen_{gen:04d}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            generation_df.to_csv(snap, index=False)
            print(f"💾 Snapshot: {snap}")

        if gen % 100 == 0:
            grid_path = f"data/map_elites_grid_gen_{gen:04d}_{RUN_TAG}.csv"
            grid_rows = [{'charge_bin': CHARGE_BINS[k[0]], 'hydro_bin': HYDRO_BINS[k[1]],
                    'fitness': e["fitness"], 'peptide': e["peptide"],
                    'amp': e["amp"], 'safety': e["safety"],
                    'stability': e["stability"], 'realism': e["realism"]}
                    for k, front in map_elites_grid.items() for e in front]
            pd.DataFrame(grid_rows).to_csv(grid_path, index=False)
            print(f"🗺️  Shared grid saved at gen {gen}")


        if gen % 100 == 0 and avg_realism >= 0.6:
            wildcards = [''.join(random.choices(amino_acids, k=peptide_length)) for _ in range(20)]
            wildcards = [w for w in wildcards if is_realistic(w, gen)]
            n_inject  = min(15, len(wildcards))
            if n_inject > 0:
                # Spread wildcards across islands
                per_island = max(1, n_inject // N_ISLANDS)
                for i in range(len(new_islands)):
                    new_islands[i].extend(wildcards[i*per_island:(i+1)*per_island])
                print(f"🧬 Injected {n_inject} wildcards across islands")


        if stagnant_generations > 100:
            print(f"🔥 Anti-stagnation batch after {stagnant_generations} stagnant gens.")

            # Scale injection size with stagnation depth
            if stagnant_generations < 300:
                inject_frac = 0.10   # 10% of population
            elif stagnant_generations < 500:
                inject_frac = 0.20   # 20%
            else:
                inject_frac = 0.30   # 30% — heavy intervention for deep stagnation

            n_target = max(15, int(population_size * inject_frac))

            # Scaffold perturbations from top sequences
            top_seqs = generation_df.sort_values('Fitness_Score', ascending=False).head(20)['Peptide'].tolist()
            perturbed = []
            for seq in top_seqs:
                # More aggressive mutation the longer we've been stuck
                mut_strength = min(0.7, mutation_rate * (2.0 + stagnant_generations / 400.0))
                for _ in range(5):
                    m = smart_mutate(seq, mut_strength)
                    if is_realistic(m, gen) and novelty_floor_check(m, archive_kmers):
                        perturbed.append(m)

            # Pure randoms — scale with stagnation
            n_random = max(10, n_target // 2)
            wild_batch = [''.join(random.choices(amino_acids, k=peptide_length))
                          for _ in range(n_random * 3)]
            filtered_w = [w for w in wild_batch if is_realistic(w, gen)][:n_random]

            injections = (perturbed + filtered_w)[:n_target]
            print(f"🧪 {len(injections)} anti-stagnation sequences ({len(perturbed)} perturbed, {len(filtered_w)} random) [{inject_frac*100:.0f}% injection].")

            # Distribute evenly across islands
            for i, w in enumerate(injections):
                new_islands[i % len(new_islands)].append(w)

        # Niche penalty countdown
        if niche_penalty_active > 0:
            niche_penalty_active -= 1

        with open(FITNESS_STATS_FILE, 'a') as f:
            if gen == 1: f.write("Generation,AvgFitness,MaxFitness,StdFitness\n")
            f.write(f"{gen},{avg_fitness:.5f},{max_fitness:.5f},{std_fitness:.5f}\n")

        islands = new_islands
        save_population([p for isl in islands for p in isl])
        # Save run state for resume
        import json

        with open('data/run_state.json', 'w') as f:
            _peaks_serializable = [
                ([list(s) for s in pk], fit)
                for pk, fit in getattr(run_simulation, '_historical_peaks', [])
            ]
            json.dump({
                'gen': gen,
                'best_fitness_so_far': best_fitness_so_far,
                'best_avg_so_far': best_avg_so_far,
                'stagnant_generations': stagnant_generations,
                'global_best_score': global_best_score,
                'global_best_peptide': global_best_peptide,
                'historical_peaks': _peaks_serializable,
                'historical_peak_ages': getattr(run_simulation, '_historical_peak_ages', []),
                'restart_gen': getattr(run_simulation, '_restart_gen', 0),

            }, f)

        # ── Migration ──────────────────────────────────────────────────

        if USE_ISLANDS:
            # Scale migration frequency and rate with stagnation
            if stagnant_generations > 300:
                effective_interval = 10   # every 10 gens when deeply stuck
                effective_rate = 0.30     # 30% migration
            elif stagnant_generations > 150:
                effective_interval = 25   # every 25 gens
                effective_rate = 0.20     # 20% migration
            else:
                effective_interval = MIGRATION_INTERVAL  # normal 50 gens
                effective_rate = MIGRATION_RATE          # normal 10%

            if gen % effective_interval == 0:
                islands = migrate_islands(islands, effective_rate)
                print(f"🏝️  Migration at gen {gen}: {N_ISLANDS} islands exchanged migrants (rate={effective_rate:.0%}, interval={effective_interval})")


        if check_early_stop():
            print("🛑 stop.txt detected — finalizing.")
            os.remove(EARLY_STOP_FILE)
            break

    save_salient_motifs()
    print(f"🧠 Salience memory saved: {len(salient_motifs)} motifs")

    delete_saved_population()
    elapsed = datetime.now() - run_start_time

    total_peptides = gen * population_size  # gen is your actual completed generation count

    with open(f"data/evolution_histories/run_summary_{RUN_TAG}.txt", 'w') as f:
        f.write(f"Run started: {run_start_time}\n")
        f.write(f"Run ended: {datetime.now()}\n")
        f.write(f"Generations: {gen}\n")
        f.write(f"Peptides simulated: {total_peptides:,}\n")
        f.write(f"Best peptide: {global_best_peptide}\n")
        f.write(f"Best fitness: {global_best_score:.4f}\n")

    print("🏁 Evolution complete! History appended to 'data/master_evolution_history.csv'")
    print(f"📊 Peptides simulated: {total_peptides:,}")
    print(f"🏅 Final global best peptide: {global_best_peptide} | Fitness: {global_best_score:.4f}")
    print(f"⏱️ Run finished in {int(elapsed.total_seconds()//3600)}h {(elapsed.total_seconds()%3600)//60:.0f}m")

    # 🧼 Clean up checkpoints from this run
    import glob
    for f in glob.glob(f"data/top_peptides_gen_*_{RUN_TAG}.csv"):
        os.remove(f)
    for f in glob.glob(f"data/gen_*_{RUN_TAG}.csv"):
        os.remove(f)
    if os.path.exists('data/run_state.json'):
        os.remove('data/run_state.json')
    print("🗑️ Checkpoints cleaned after successful run.")

# Only run this if evolve.py is executed directly, not imported
if __name__ == "__main__":
    run_simulation()