import os
import random
import string
import math
import numpy as np
import pandas as pd
from tensorflow import keras
from datetime import datetime

import sys

# Allow importing from project root (../external/pyampa_integration.py)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from external.pyampa_integration import pyampa_scores



world_model = None
naturalness_model = None 
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


WORLD_MODEL_ERROR_FILE = 'data/world_model_error.csv'
NOVELTY_STATS_FILE = 'data/novelty_stats.csv'
ENTROPY_STATS_FILE = 'data/entropy_stats.csv'


os.makedirs('data', exist_ok=True)


EARLY_STOP_FILE = 'stop.txt'

# 👁️ Attentional motif salience memory
salient_motifs = {}  # Stores k-mers and their evolving salience values



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

# Settings
amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
population_size = 150
generations = 2000
peptide_length = 25
max_len = 50

STAGNANT_LIMIT = 600

# How often the agent performs meta-cognitive introspection
META_INTROSPECTION_INTERVAL = 50  # e.g. every 25 generations


# How long to wait with no improvement before entering exploration mode
EXPLORATION_STAGNANT_TRIGGER = 150   # you can tune this
MAX_EXPLORATION_BOOST = 1.4         # cap on how hard we shift to curiosity/novelty



# One-hot encoding function (you might not be using this directly in the scoring)
aa_to_idx = {aa: idx for idx, aa in enumerate(amino_acids)}
# Scoring function
def encode_int(seq, max_len=50):
    int_seq = np.zeros(max_len, dtype=np.int32)
    for i, aa in enumerate(seq[:max_len]):
        idx = aa_to_idx.get(aa)
        if idx is not None:
            int_seq[i] = idx + 1  # 1-indexed
    return int_seq

def predict_single_value(model, int_encoded):
    """Helper to predict a single float output cleanly."""
    return float(model.predict(int_encoded, verbose=0)[0][0])


def score_peptide(
    peptide,
    amp_score,
    tox_score,
    stab_score,
    previous_peptides=None,
    agent=None,
    pyamp=None,
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
    
    # Structure Bonus: Encourage amphipathic helices
    structure_bonus = 1.0
    if hydro_moment > 0.45:
        structure_bonus = 1.15
    elif hydro_moment > 0.35:
        structure_bonus = 1.05
    elif hydro_moment < 0.20:
        structure_bonus = 0.85 

    # 👽 Turing Test (Naturalness Check)
    # Uses global naturalness_model if available
    turing_bonus = 1.0
    global naturalness_model
    if naturalness_model is not None:
        try:
            # Predict takes shape (1, 50)
            nat_pred = float(naturalness_model.predict(encode_int(peptide).reshape(1, -1), verbose=0)[0][0])
            
            if nat_pred < 0.3:
                turing_bonus = 0.5  # Heavy penalty for "Alien" looking junk
            elif nat_pred > 0.8:
                turing_bonus = 1.1  # Reward for "Natural" looking grammar
        except Exception:
            pass # Safety pass if model fails

    # --- Optional PyAMPA calibration ---
    if pyamp is not None:
        ext_amp = float(pyamp.get("amp", amp_score))
        ext_tox = float(pyamp.get("tox", tox_score))
        ext_hem = float(pyamp.get("hemolysis", 0.0))

        # Soft blend
        amp_score = 0.7 * amp_score + 0.3 * ext_amp
        tox_blend = 0.7 * tox_score + 0.3 * ext_tox
        tox_score = min(1.0, 0.5 * tox_blend + 0.5 * ext_hem)

    # --- Novelty term ---
    if previous_peptides:
        max_sim = max(sequence_identity(peptide, p) for p in previous_peptides)
        novelty = 1.0 - max_sim
    else:
        novelty = 0.5

    # --- Weights ---
    if agent is not None:
        m = agent.motives
        w_amp = agent.w[m.index("amp")]
        w_safety = agent.w[m.index("safety")]
        w_stab = agent.w[m.index("stability")]
        w_novel = agent.w[m.index("novelty")]
        w_quality = 0.5 * (w_amp + w_safety)
    else:
        w_amp, w_safety, w_stab, w_quality, w_novel = 0.35, 0.25, 0.20, 0.10, 0.10

    weights = np.array([w_amp, w_safety, w_stab, w_quality, w_novel], dtype=float)
    weights = weights / weights.sum()
    w_amp, w_safety, w_stab, w_quality, w_novel = weights

    # --- Core Value ---
    value = (
        w_amp    * amp_score +
        w_safety * (1.0 - tox_score) +
        w_stab   * stab_score +
        w_quality * quality_score +
        w_novel * novelty
    )

    # --- Gating (Realism * Structure * Turing) ---
    gated = value * realism * structure_bonus * turing_bonus

    # --- Logistic compression ---
    beta = 6.0
    bias = 0.5
    fitness = 1.0 / (1.0 + math.exp(-beta * (gated - bias)))

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
        if salient_motifs.get(motif, 0) > 0.02:
            # Attention bias: mutate within similar group
            for j in range(3):
                aa = peptide[idx + j]
                group = aa_to_group.get(aa, amino_acids)
                peptide[idx + j] = random.choice(group)

    return ''.join(peptide)

def realism_penalty_score(peptide):
    """Returns realism weight between 0.4 and 1.0 based on structural plausibility."""
    penalty = 1.0

    # Max 2 Cysteines (disulfide control)
    if peptide.count('C') > 2:
        penalty *= 0.85

    # Max 4 Prolines (structure disruption)
    if peptide.count('P') > 4:
        penalty *= 0.85

    # Excess hydrophobicity
    hydrophobicity = calculate_hydrophobicity(peptide)
    if hydrophobicity > 0.7:
        penalty *= 0.85

    # Excess polarity
    polar_content = calculate_polar_content(peptide)
    if polar_content > 0.7:
        penalty *= 0.85

    # Forbidden motifs
    for motif in ['PPPP', 'CCCC']:
        if motif in peptide:
            penalty *= 0.7

    # Low diversity
    if len(set(peptide)) < 6:
        penalty *= 0.8
    elif len(set(peptide)) < 8:
        penalty *= 0.9

    return max(penalty, 0.4)

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
    if realism < 0.6:
        return 0.4  # or even 0.3 if you're harsher
    # harsh cutoff fallback

    score = 1.0

    net_charge = calculate_net_charge(peptide)
    hydrophobicity = calculate_hydrophobicity(peptide)
    polar_content = calculate_polar_content(peptide)
    unique_aas = len(set(peptide))

    # Charge range bonus zone
    if 2 <= net_charge <= 8:
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
    elif unique_aas < 8:
        score *= 0.95
    elif unique_aas < 6:
        score *= 0.85

    score *= realism  # realism = realism_penalty_score(peptide)
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
        'PyAMPA_AMP', 'PyAMPA_Toxicity', 'PyAMPA_Hemolysis', 'PyAMPA_CPP',
        'Surprise_AMP', 'Surprise_Toxicity',
        'Hydrophobic_Moment',  # <-- NEW
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
    return sub.iloc[0]['Peptide']






# Diversity helper
def average_similarity(population):
    if len(population) < 2:
        return 0.0  # not enough to compare
    similarities = [
        sequence_identity(population[i], population[j])
        for i in range(len(population))
        for j in range(i + 1, len(population))
    ]
    return np.mean(similarities)



# 🔽 INSERT HERE:
def similarity_penalty(pop):
    sims = [sequence_identity(a, b) for i, a in enumerate(pop) for b in pop[i+1:]]
    return np.mean(sims)




def diverse_subset(peptides, threshold=0.85):
    unique = []
    for p in peptides:
        if all(sequence_identity(p, u) < threshold for u in unique):
            unique.append(p)
        if len(unique) >= int(0.1 * population_size):
            break
    return unique



def similarity_stats(population):
    """
    Returns (avg_sim, min_sim, max_sim) over all pairwise sequence identities.
    """
    if len(population) < 2:
        return 0.0, 0.0, 0.0
    sims = [
        sequence_identity(population[i], population[j])
        for i in range(len(population))
        for j in range(i + 1, len(population))
    ]
    sims = np.array(sims, dtype=float)
    return float(sims.mean()), float(sims.min()), float(sims.max())


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




# Evolution loop
stagnant_generations = 0
best_fitness_so_far = 0

global_best_peptide = None
global_best_score = -np.inf

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
    base = realism_penalty_score(peptide)
    # much gentler early threshold
    min_realism = 0.25 + 0.25 * min(gen / 2000, 1.0)
    return base >= min_realism



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





def _deterministic_point_mutation(seq, pos, aa_idx):
    seq = list(seq)
    seq[pos] = amino_acids[aa_idx]
    return ''.join(seq)


# =========================
# 🧠 Cognitive Agent Module
# =========================
class AgentController:
    def __init__(self):
        # Align these motives with your scoring dimensions
        self.motives = ["amp","safety","stability","realism","novelty","parsimony","curiosity"]
        self.w = np.ones(len(self.motives)) / len(self.motives)  # start uniform

        # Chaotic latent
        self.z = 0.347
        self.r = 3.9          # slightly below full chaos; will be modulated per-gen
        self.delta_tie = 0.02

        # 🔒 global curiosity cap (lowered a bit)
        self.MAX_CUR = 0.25
        # 🔒 minimum weight for any motive (simplex floor)
        self.MIN_W   = 0.04


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
                0.22,  # amp
                0.18,  # safety
                0.15,  # stability
                0.10,  # realism
                0.12,  # novelty
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

            # === GRADUAL UPDATE: move a little toward target ===
            eta = 0.02  # slower drift
            self.w = (1 - eta) * self.w + eta * target

            # === apply floors + curiosity ceiling ===
            cur_idx = self.motives.index("curiosity")
            
            # 🔒 HARD CAP: Curiosity can never exceed 25% of the total drive
            # This prevents the "Curiosity Singularity" where the agent ignores fitness.
            MAX_CUR = 0.25 
            
            self.w = np.maximum(self.w, self.MIN_W)
            self.w[cur_idx] = min(self.w[cur_idx], MAX_CUR)

            # (optional) ensure AMP+Safety never collapse
            i_amp = self.motives.index("amp")
            i_saf = self.motives.index("safety")
            min_amp_saf = 0.30
            s = self.w[i_amp] + self.w[i_saf]
            if s < min_amp_saf:
                boost = (min_amp_saf - s) / 2.0
                self.w[i_amp] += boost
                self.w[i_saf] += boost

            # final renormalization
            self.w /= self.w.sum()



    def introspect_and_remap(self, signals: dict):
        """
        Meta-cognitive step: use slow-timescale signals to reshape motive weights.
        `signals` contains:
          - plateau: 0–1 (how long we’ve been stuck)
          - novelty: 0–1 (1 = very novel, 0 = identical)
          - diversity: 0–1 (entropy-based sequence diversity)
          - salience_entropy: 0–1 (motif memory spread)
          - chaos: 0–1 (how chaotic the latent z regime is)
        """
        m_idx = self.motives.index

        plateau          = float(signals.get("plateau", 0.0))
        novelty          = float(signals.get("novelty", 0.5))
        diversity        = float(signals.get("diversity", 0.5))
        sal_ent          = float(signals.get("salience_entropy", 0.5))
        chaos            = float(signals.get("chaos", 0.5))

        # Start from current weights (this is truly meta, not reset)
        target = self.w.copy()

        # 1) Long plateau + low novelty -> push exploration motives
        explore_pressure = plateau * (1.0 - novelty)  # big when stuck & redundant
        if explore_pressure > 0:
            target[m_idx("curiosity")] += 0.20 * explore_pressure
            target[m_idx("novelty")]   += 0.15 * explore_pressure

        # 2) Diversity collapse + motif lock-in (low diversity, low salience entropy)
        collapse = (1.0 - diversity) * (1.0 - sal_ent)
        if collapse > 0:
            # pull back a bit on pure exploitation
            target[m_idx("amp")]      -= 0.10 * collapse
            target[m_idx("safety")]   -= 0.05 * collapse
            # encourage breaking out
            target[m_idx("curiosity")] += 0.10 * collapse

        # 3) Chaos rhythm:
        #    - If chaos is very high, we cool down curiosity/novelty
        #    - If chaos is low, we warm them up (add “jitter”)
        if chaos > 0.8:
            target[m_idx("curiosity")] -= 0.10
            target[m_idx("novelty")]   -= 0.05
            target[m_idx("realism")]   += 0.08
            target[m_idx("parsimony")] += 0.05
        elif chaos < 0.2:
            target[m_idx("curiosity")] += 0.10
            target[m_idx("novelty")]   += 0.05

        # --- safety: clamp + smooth move toward target ---
        target = np.maximum(target, 1e-6)

        # small meta learning rate so this is MUCH slower than per-gen homeostasis
        eta_meta = 0.02
        self.w = (1.0 - eta_meta) * self.w + eta_meta * target


        cur_idx = m_idx("curiosity")
        self.w = np.maximum(self.w, self.MIN_W)
        self.w[cur_idx] = min(self.w[cur_idx], self.MAX_CUR)

        self.w /= self.w.sum()







def _estimate_deltas(candidate: str,
                     amp_model, tox_model, stab_model,
                     population: list, target_len: int) -> dict:
    """
    Deterministic 'reasons' for/against a candidate mutation.
    All terms are quick-to-compute proxies you already use elsewhere.
    """
    int_enc = np.array([encode_int(candidate)])
    amp = float(amp_model.predict(int_enc, verbose=0)[0][0])
    tox = float(tox_model.predict(int_enc, verbose=0)[0][0])
    stab = float(stab_model.predict(int_enc, verbose=0)[0][0])

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
                     population: list, target_len: int) -> str:
    # Generate k candidates deterministically from agent.z (no RNG)
    k = 2
    candidates = []
    length = len(parent)
    for _ in range(k):
        # choose a position via chaos
        agent.z, pos_idx = _chaotic_index(agent.z, agent.r, length)
        # choose an amino acid via chaos
        agent.z, aa_idx = _chaotic_index(agent.z, agent.r, len(amino_acids))
        cand = _deterministic_point_mutation(parent, pos_idx, aa_idx)
        # occasional (deterministic) swap to mimic your smart_mutate flavor
        agent.z, swap_a = _chaotic_index(agent.z, agent.r, length)
        agent.z, swap_b = _chaotic_index(agent.z, agent.r, length)
        if swap_a != swap_b and mutation_rate > 0.2:  # only sometimes
            c = list(cand); c[swap_a], c[swap_b] = c[swap_b], c[swap_a]; cand = ''.join(c)
        candidates.append(cand)

    # Score candidates
    scored = []
    for cand in candidates:
        deltas = _estimate_deltas(cand, amp_model, tox_model, stab_model, population, target_len)
        score = agent.reason_score(deltas)
        scored.append((score, cand, deltas)) # <--- Add deltas here

    scored.sort(reverse=True, key=lambda x: x[0])
    
    # Chaotic Pick (now returns tuple)
    best_score, best_cand, best_reasons = scored[0] # Simplification: Agent picks top reason-based choice
    
    # Check chaos for tie-breaking if you want, but for logging, let's track the "Winner"
    return best_cand, best_reasons


def run_simulation():

    # Load models
    print("📦 Loading models...")

    global world_model, naturalness_model  # 👈 add this

    amp_model = keras.models.load_model('models/amp_model.keras', compile=False)
    toxicity_model = keras.models.load_model('models/toxicity_cnn_model.keras', compile=False)
    stability_model = keras.models.load_model('models/stability_cnn_model.keras', compile=False)
    naturalness_model = keras.models.load_model('models/naturalness_discriminator.keras')
    print("✅ Models loaded: AMP, Toxicity, Stability, Naturalness Discriminator")

    # 🧠 Build internal Peptide World Model (PWM)
    global world_model

    vocab_size = len(amino_acids) + 1
    world_model = build_world_model(max_len, vocab_size)
    print("✅ Internal world model initialized")

    run_start_time = datetime.now()
    # Similarity log path (unique per run)
    os.makedirs('data/similarity_logs', exist_ok=True)
    RUN_TAG = run_start_time.strftime('%Y%m%d_%H%M')
    SIMILARITY_LOG_PATH = f"data/similarity_logs/similarity_log_{RUN_TAG}.csv"


    # 🔗 Make the other logs run-specific as well
    global WORLD_MODEL_ERROR_FILE, NOVELTY_STATS_FILE, ENTROPY_STATS_FILE, MUTATION_HISTORY_FILE
    WORLD_MODEL_ERROR_FILE = f"data/world_model_error_{RUN_TAG}.csv"
    NOVELTY_STATS_FILE     = f"data/novelty_stats_{RUN_TAG}.csv"
    ENTROPY_STATS_FILE     = f"data/entropy_stats_{RUN_TAG}.csv"
    MUTATION_HISTORY_FILE  = f"data/mutation_rate_history_{RUN_TAG}.txt"
    FITNESS_STATS_FILE     = f"data/fitness_stats_{RUN_TAG}.csv"


    # 🧠 Initialize cognitive controller (deterministic but sensitive)
    agent = AgentController()

    # Per-run logs
    AGENT_LOG  = f"data/agent_state_{RUN_TAG}.csv"
    ACTION_LOG = f"data/action_log_{RUN_TAG}.csv"

    # create headers if new
    if not os.path.exists(AGENT_LOG):
        with open(AGENT_LOG, 'w') as f:
            f.write(
                "Generation," +
                ",".join(agent.motives) +
                ",z,r,stagnant,avg_sim,mean_entropy,total_err,avg_realism\n"
            )



    if not os.path.exists(ACTION_LOG):
        with open(ACTION_LOG, 'w') as f:
            f.write("Generation,ActionType\n")

    REASON_LOG = f"data/agent_reasoning_{RUN_TAG}.csv"
    with open(REASON_LOG, 'w') as f:
        f.write("Generation,Parent,Child,Delta_AMP,Delta_Safety,Delta_Stab,Delta_Realism,Delta_Novelty,Total_Score\n")

    # Create trait stats output folder if needed
    os.makedirs('data/trait_stats', exist_ok=True)

    # Create trait stats output folder if needed
    os.makedirs('data/trait_stats', exist_ok=True)
    os.makedirs('data/evolution_histories', exist_ok=True)
    trait_log_path = f"data/trait_stats/trait_stats_{RUN_TAG}.csv"
    THIS_RUN_FILE = f"data/evolution_histories/evolution_run_{RUN_TAG}.csv"

    # Initialize population
    population = load_population()
    if not population:

        population = [''.join(random.choices(amino_acids, k=peptide_length)) for _ in range(population_size)]
        print(f"🌱 Created pure random population: {population_size} sequences.")

    else:
        print(f"🌟 Resuming from saved population with {len(population)} peptides.")

    # Mark a new run in mutation history
    with open(MUTATION_HISTORY_FILE, 'a') as f:
        f.write(f"\n--- New Run Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f.write(f"Population size: {population_size}, Generations: {generations}, Peptide length: {peptide_length}\n")
        f.write("Generation,Mutation Rate\n")

    stagnant_generations = 0
    best_fitness_so_far = 0
    global_best_score = -np.inf
    global_best_peptide = None

    for gen in range(1, generations + 1):
        gen_start = datetime.now()
        print(f"🌱 Generation {gen}...")
        
        # (Optional) Log r to see the "heartbeat" of the agent
        # You can add 'r' to your AGENT_LOG columns if you want


        # 🎯 Parent selection schedule: softer early, sharper later
        if gen < 200:
            tournament_k = 2
            explore_parent_prob = 0.30   # 30% of the time: purely random parents
        else:
            tournament_k = 3
            explore_parent_prob = 0.15   # still some randomness, but less


        encoded_population = np.array([encode_int(p) for p in population])
        amp_scores = amp_model.predict(encoded_population, verbose=0).flatten()
        tox_scores = toxicity_model.predict(encoded_population, verbose=0).flatten()
        stab_scores = stability_model.predict(encoded_population, verbose=0).flatten()


        scores = []
        for idx, pep in enumerate(population):
            amp_int = float(amp_scores[idx])
            tox_int = float(tox_scores[idx])
            stab_int = float(stab_scores[idx])

            # 🔗 External model: PyAMPA
            try:
                py = pyampa_scores(pep)
            except Exception as e:
                # If anything goes wrong, fall back to "no external info"
                print(f"⚠️ PyAMPA failed for {pep}: {e}")
                py = {"amp": amp_int, "tox": tox_int, "hemolysis": 0.0, "cpp": 0.0}

            # Core scoring (Agent motives + PyAMPA-blended scores)
            amp, tox, stab, fitness, sol, agg, pI, boman, net_charge, hydrophobicity, sol_tag, agg_tag, realism_score, hydro_moment = score_peptide(
                pep,
                amp_int,
                tox_int,
                stab_int,
                previous_peptides=population,
                agent=agent,
                pyamp=py,
            )





            # Surprise = disagreement between internal CNNs and PyAMPA
            surprise_amp = abs(amp_int - float(py.get("amp", amp_int)))
            surprise_tox = abs(tox_int - float(py.get("tox", tox_int)))

            update_salient_motifs(pep, fitness)

            if gen == 1 and idx < 5:
                print(
                    f"⚙️ AMP_int={amp_int:.2f}, AMP_ext={py.get('amp', 0):.2f}, "
                    f"TOX_int={tox_int:.2f}, TOX_ext={py.get('tox', 0):.2f}, "
                    f"FITNESS={fitness:.3f}"
                )

            tag = assess_peptide_quality(pep)

            scores.append((
                pep, amp, tox, stab, sol, agg, pI, boman,
                net_charge, hydrophobicity, sol_tag, agg_tag,
                fitness, realism_score, tag,
                float(py.get("amp", 0.0)),
                float(py.get("tox", 0.0)),
                float(py.get("hemolysis", 0.0)),
                float(py.get("cpp", 0.0)),
                surprise_amp,
                surprise_tox,
                hydro_moment,  # <-- NEW
            ))



        generation_df = pd.DataFrame(scores, columns=[
            'Peptide',
            'AMP_Score', 'Toxicity_Score', 'Stability_Score',
            'Solubility_Score', 'Aggregation_Risk', 'Isoelectric_Point',
            'Boman_Index', 'Net_Charge', 'Hydrophobicity',
            'Solubility_Tag', 'Aggregation_Tag',
            'Fitness_Score', 'Realism_Score', 'Quality_Tag',
            'PyAMPA_AMP', 'PyAMPA_Toxicity', 'PyAMPA_Hemolysis', 'PyAMPA_CPP',
            'Surprise_AMP', 'Surprise_Toxicity',
            'Hydrophobic_Moment',  # <-- NEW
        ])




        # === Update Peptide World Model from this generation ===
        pwm_add_samples(generation_df)
        pwm_train_one_epoch()


        generation_df["Generation"] = gen

        # Track generation fitness stats
        avg_fitness = generation_df['Fitness_Score'].mean()
        max_fitness = generation_df['Fitness_Score'].max()

        std_fitness = generation_df['Fitness_Score'].std()

        # 🧬 Log trait drift over time
        avg_charge = generation_df['Net_Charge'].mean()
        avg_hydro = generation_df['Hydrophobicity'].mean()
        avg_agg = generation_df['Aggregation_Risk'].mean()
        avg_realism = generation_df['Realism_Score'].mean()

        if avg_realism < 0.5:
            print("❌ Average realism score too low — stopping early to preserve plausible peptides.")
            break

        # First generation: write new file with header
        if gen == 1:
            with open(trait_log_path, 'w') as trait_log:
                trait_log.write("Generation,NetCharge,Hydrophobicity,AggregationRisk,Realism\n")
        # Append data each generation
        with open(trait_log_path, 'a') as trait_log:
            trait_log.write(f"{gen},{avg_charge:.4f},{avg_hydro:.4f},{avg_agg:.4f},{avg_realism:.4f}\n")

        print(f"📊 Std Dev Fitness: {std_fitness:.4f}")

        # 🔍 Similarity / novelty stats
        avg_sim, min_sim, max_sim = similarity_stats(population)
        gen_time = datetime.now() - gen_start



        # 🧠 World-model learning diagnostics
        total_err = log_world_model_error(generation_df, gen)
        if total_err is None:
            total_err = 999.0  # world model not ready yet

        # Chaos mode only when population is converged AND PWM is confident
        low_diversity_flag = (avg_sim > 0.70 and total_err < 0.18)

        with open(SIMILARITY_LOG_PATH, 'a') as sim_log:
            if gen == 1:
                sim_log.write("Generation,Similarity\n")
            sim_log.write(f"{gen},{avg_sim:.4f}\n")


        # 🌱 Novelty & entropy logs
        log_novelty_stats(avg_sim, min_sim, max_sim, gen)
        log_entropy_stats(population, gen)

        print(f"🧪 Logged PWM error for gen {gen}")
        print(f"🧬 Logged novelty: avg={avg_sim:.3f}, min={min_sim:.3f}, max={max_sim:.3f}")
        mean_H, max_H = compute_population_entropy(population)
        print(f"📚 Logged entropy: mean={mean_H:.3f}, max={max_H:.3f}")

        # 🛡️ Early convergence guard: if population collapses too soon, inject exploration
        if gen < 30 and avg_sim > 0.35:
            print(f"⚠️ Early convergence detected (avg_sim={avg_sim:.3f}) — injecting exploratory peptides.")
            extra = [
                ''.join(random.choices(amino_acids, k=peptide_length))
                for _ in range(int(0.2 * population_size))
            ]
            extra = [p for p in extra if is_realistic(p, gen)]
            population.extend(extra)
            # Recompute similarity roughly (we don't need to log again)
            print(f"🧬 Added {len(extra)} exploratory sequences to slow convergence.")

        if gen % 50 == 0:
            top_peps_path = f"data/top_peptides_gen_{gen:04d}_{RUN_TAG}.csv"
            generation_df.head(5).to_csv(top_peps_path, index=False)
            print(f"🏅 Saved top peptides snapshot: {top_peps_path}")

        # === Dynamic decay over time ===
        initial_mutation = 0.4
        final_mutation = 0.08
        decay_duration = generations  # Total generations to decay across

        # Cosine decay curve
        cosine_decay = 0.5 * (1 + math.cos(math.pi * min(gen / decay_duration, 1.0)))
        base_mutation = final_mutation + (initial_mutation - final_mutation) * cosine_decay

        # === Adaptive tweak based on similarity and stagnation ===
        if avg_sim > 0.65:
            mutation_rate = base_mutation * 1.2
        elif avg_sim < 0.5:
            mutation_rate = base_mutation * 0.8
        else:
            mutation_rate = base_mutation

        # === Realism Penalty Slope ===
        realism_weight = min(max(avg_realism, 0.4), 1.0)
        mutation_rate *= realism_weight
        print(f"🧠 Realism-Adjusted Mutation Rate: {mutation_rate:.4f} (based on realism {avg_realism:.3f})")

        if stagnant_generations > 100:
            boost = min(0.1 + stagnant_generations / 1000, 0.5)
            mutation_rate = min(0.6, mutation_rate + boost)

        # Late-stage convergence clamp
        if gen > 0.7 * generations:
            mutation_rate = min(mutation_rate, 0.18)

        # Hard clamp: keep a decent floor and prevent absurd spikes
        mutation_rate = float(np.clip(mutation_rate, 0.10, 0.65))



        print(f"\n{'-'*60}")
        print(f"🌱 Generation {gen}")
        print(f"{'-'*60}")
        print(f"📈 Avg Fitness: {avg_fitness:.4f} | Best: {max_fitness:.4f}")
        print(f"⏱️ Runtime: {gen_time.total_seconds():.2f}s")

        # Print top 3 peptides compactly
        print("\n🏅 Top Peptides:")
        top_peptides = generation_df.sort_values(by='Fitness_Score', ascending=False).head(3)
        for i, row in top_peptides.iterrows():
            print(f"  {i+1}. {row['Peptide']} | Fit: {row['Fitness_Score']:.4f} | AMP: {row['AMP_Score']:.2f} | TOX: {row['Toxicity_Score']:.2f} | STAB: {row['Stability_Score']:.2f} | Charge: {row['Net_Charge']} | Tag: {row['Quality_Tag']}")

        # Global best tracking
        top_row = generation_df.sort_values(by='Fitness_Score', ascending=False).iloc[0]
        if top_row['Fitness_Score'] > global_best_score:
            global_best_score = top_row['Fitness_Score']
            global_best_peptide = top_row['Peptide']
            print(f"💎 New global best peptide found! Fitness: {global_best_score:.4f}")
            print(f"🧬 Sequence: {global_best_peptide}")

        print(f"🔬 Average similarity in Generation {gen}: {avg_sim:.4f}")
        top5_mean = generation_df.head(5)['Fitness_Score'].mean()

        print(f"⏱️ Gen {gen} runtime: {gen_time.total_seconds():.2f}s")

        if stagnant_generations % 100 == 0 and stagnant_generations > 0:
            print(f"⚠️  {stagnant_generations} stagnant generations (no improvement in top-5 mean)")



        if top5_mean > best_fitness_so_far + 1e-5:
            best_fitness_so_far = top5_mean
            stagnant_generations = 0
        else:
            stagnant_generations += 1

            # 🧱 Dynamic early-stop conditions:
            # 1) Don't even consider early stopping before some minimum fraction of the run.
            # 2) Only stop if we've reached a reasonable fitness regime (not stagnating in garbage).
            min_gen_for_stop = int(0.25 * generations)   # e.g. after 25% of planned gens
            min_quality_for_stop = 0.60                  # require at least this top-5 mean
                
            # Extra guards so we don't stop while still diverse or still "hot"
            min_gen_for_stop      = int(0.25 * generations)  # e.g. after 25% of run
            min_quality_for_stop  = 0.60
            min_div_for_stop      = 0.65   # require reasonably converged population
            min_mut_for_stop      = 0.12   # and fairly low mutation rate

            if (
                gen >= min_gen_for_stop
                and best_fitness_so_far >= min_quality_for_stop
                and stagnant_generations >= STAGNANT_LIMIT
                and avg_sim >= min_div_for_stop
                and mutation_rate <= min_mut_for_stop
            ):
                print(
                    f"🛑 Early stopping: No fitness improvement for {stagnant_generations} generations "
                    f"(limit={STAGNANT_LIMIT}), avg_sim={avg_sim:.2f}, "
                    f"mutation_rate={mutation_rate:.3f}."
                )
                break

        # 🔄 Self-referential chaos update (now uses real signals)
        agent.adapt_chaos(
            stagnant_generations,
            avg_sim=avg_sim,
            total_err=total_err,
            avg_realism=avg_realism,
        )

        # 🔁 Log mutation rate and checkpoint info
        print(f"🧬 Mutation rate for Gen {gen}: {mutation_rate:.2f}")
        if gen % 100 == 0:
            print(f"📌 Checkpoint: Gen {gen}, Top Fitness: {top_row['Fitness_Score']:.4f}")

        append_mutation_rate(gen, mutation_rate)
        append_generation_to_master(gen, generation_df.to_dict('records'))

        # Also log this generation to the run-specific file

        generation_df.to_csv(
            THIS_RUN_FILE,
            mode='a',
            header=not os.path.exists(THIS_RUN_FILE),
            index=False
        )

       
        # 🧪 Use a gentler realism threshold early on
        realism_threshold = 0.2 + 0.6 * min(gen / 5000, 1.0)
       
        # --- Soft tiered survivor pool (slows convergence, keeps "bad" peptides alive) ---

        # Sort population globally
        sorted_df = generation_df.sort_values(by="Fitness_Score", ascending=False)

        n = len(sorted_df)

        top_cut   = int(0.30 * n)   # top 30% fully eligible
        mid_cut   = int(0.70 * n)   # next 40% eligible but down-weighted

        top_df = sorted_df.iloc[:top_cut]
        mid_df = sorted_df.iloc[top_cut:mid_cut]
        low_df = sorted_df.iloc[mid_cut:]

        # survival probabilities by tier
        top_df = top_df.sample(frac=1.0)            # keep all
        mid_df = mid_df.sample(frac=0.40)           # keep ~40% of mid-tier
        low_df = low_df.sample(frac=0.10)           # keep 10% bottom-tier

        filtered_df = pd.concat([top_df, mid_df, low_df])


        # ✅ SAFETY: ensure minimum survivors
        min_survivors = population_size // 4
        if len(filtered_df) < min_survivors:
            print(f"⚠️ Too few survivors ({len(filtered_df)}). Falling back to top-{min_survivors} by fitness.")
            filtered_df = generation_df.sort_values(
                by="Fitness_Score", ascending=False
            ).head(min_survivors)


        # 🧠 Agent adapts motives (baseline homeostatic rules)
        agent.update_weights(generation_df)

        # 🪞 Periodic meta-cognition: slower, self-reflective remapping
        if gen % META_INTROSPECTION_INTERVAL == 0:
            # novelty proxy: 1 - avg_sim (we already computed avg_sim above)
            avg_novelty = 1.0 - avg_sim

            # diversity: normalize mean_H by max entropy over 20 amino acids
            # (compute_population_entropy already gave us mean_H earlier)
            max_H_theoretical = math.log2(len(amino_acids))
            diversity_norm = 0.0
            if max_H_theoretical > 0:
                diversity_norm = max(0.0, min(1.0, mean_H / max_H_theoretical))

            meta_signals = {
                "plateau": min(stagnant_generations / STAGNANT_LIMIT, 1.0),
                "novelty": max(0.0, min(1.0, avg_novelty)),
                "diversity": diversity_norm,
                "salience_entropy": salience_entropy(),
                # chaos index from logistic map: peaks ~1 at z=0.5
                "chaos": max(0.0, min(1.0, 4.0 * agent.z * (1.0 - agent.z))),
            }

            agent.introspect_and_remap(meta_signals)

        # 🚀 Gentle exploration bump: no multiplicative blow-ups
        if stagnant_generations > EXPLORATION_STAGNANT_TRIGGER:
            excess = stagnant_generations - EXPLORATION_STAGNANT_TRIGGER
            # 0 → 1 as stagnation grows
            strength = min(excess / 50.0, 1.0)

            i_nov = agent.motives.index("novelty")
            i_cur = agent.motives.index("curiosity")
            i_amp = agent.motives.index("amp")
            i_saf = agent.motives.index("safety")
            i_stab = agent.motives.index("stability")

            # Add a little exploration
            agent.w[i_nov] += 0.03 * strength
            agent.w[i_cur] += 0.05 * strength

            # Gently soften pure exploitation
            damp = 0.02 * strength
            agent.w[i_amp] *= (1.0 - damp)
            agent.w[i_saf] *= (1.0 - damp)
            agent.w[i_stab] *= (1.0 - damp)

            print(f"🧭 Exploration bump: stagnant={stagnant_generations}, strength={strength:.2f}")


        # Always clamp + renormalize
        agent.w = np.maximum(agent.w, 1e-6)
        agent.w = agent.w / agent.w.sum()

        # 🔒 Global curiosity cap + AMP/safety floor (applied AFTER all tweaks)
        cur_idx = agent.motives.index("curiosity")
        amp_idx = agent.motives.index("amp")
        saf_idx = agent.motives.index("safety")

        # cap curiosity
        agent.w[cur_idx] = min(agent.w[cur_idx], agent.MAX_CUR)

        # ensure AMP + safety together never collapse
        min_amp_saf = 0.30
        s = agent.w[amp_idx] + agent.w[saf_idx]
        if s < min_amp_saf:
            boost = (min_amp_saf - s) / 2.0
            agent.w[amp_idx] += boost
            agent.w[saf_idx] += boost

        # enforce per-motive floor and renormalize
        agent.w = np.maximum(agent.w, agent.MIN_W)
        agent.w = agent.w / agent.w.sum()

        # Log agent state each generation (after meta + exploration tweaks)
        with open(AGENT_LOG, 'a') as f:
            f.write(
                f"{gen}," +
                ",".join(f"{w:.5f}" for w in agent.w) +
                f",{agent.z:.6f}," +
                f"{agent.r:.4f}," +
                f"{stagnant_generations}," +
                f"{avg_sim:.4f}," +
                f"{mean_H:.4f}," +
                f"{total_err:.4f}," +
                f"{avg_realism:.4f}\n"
            )



        print(f"🧪 Realism threshold: {realism_threshold:.2f} | Survivors found: {len(filtered_df)}")


        # Safety fallback if not enough survivors
        if len(filtered_df) >= population_size // 2:
            survivors = filtered_df.sample(n=population_size // 2, weights='Fitness_Score')['Peptide'].tolist()
        else:
            print(f"⚠️ Using all {len(filtered_df)} survivors without sampling.")
            survivors = filtered_df['Peptide'].tolist()
        # ✅ Patch: Inject fallback survivors if empty
        if not survivors:
            print("🚨 No valid survivors found — injecting new random population.")
            survivors = [''.join(random.choices(amino_acids, k=peptide_length)) for _ in range(population_size)]

            # 🔍 Score fallback survivors for proper crossover weighting
            encoded = np.array([encode_int(p) for p in survivors])
            amp_scores = amp_model.predict(encoded, verbose=0).flatten()
            tox_scores = toxicity_model.predict(encoded, verbose=0).flatten()
            stab_scores = stability_model.predict(encoded, verbose=0).flatten()

            weights = np.array([
                score_peptide(p, amp_scores[i], tox_scores[i], stab_scores[i], agent=agent)[3]
                for i, p in enumerate(survivors)
            ])


            print(f"🧪 Injected {len(survivors)} new survivors with fallback fitness scores.")
        
        print(f"👥 Survivors going into reproduction: {len(survivors)}")

        

        high_sim = similarity_penalty(survivors)
        if high_sim > 0.8:
            print(f"⚠️ High survivor similarity detected (pairwise ~{high_sim:.2f}). Chaos mode may inject exploratory children this gen.")

        # === Build next generation ===
        new_population = []
        attempts = 0  # 👈 already there

        # 👇 progress cadence
        progress_step = 25  # print every 25 children
        last_print = 0

        while len(new_population) < population_size:
            attempts += 1

            # 👇 DEBUG: show how many children we've built so far
            if len(new_population) >= last_print + progress_step:
                print(f"  🧬 Built {len(new_population)} / {population_size} children...")
                last_print = len(new_population)

            # 🚨 Emergency fallback: if we've tried too many times without filling the population,
            # inject a random peptide (so we never get stuck in an infinite loop).
            if attempts > 3000:
                rand = ''.join(random.choices(amino_acids, k=peptide_length))
                new_population.append(rand)
                attempts = 0
                print("  ⚠️ Too many failed attempts, injecting random peptide to keep things moving.")
                continue

            # 🌪️ Chaos children: only when low_diversity_flag is set
            if low_diversity_flag and random.random() < 0.35:
                # pick a base parent but push it hard with smart_mutate
                base_parent = random.choice(survivors)
                # higher effective mutation just for this child
                chaos_mutation_rate = min(mutation_rate * 2.5, 0.6)
                wild_child = smart_mutate(base_parent, chaos_mutation_rate)
                if is_realistic(wild_child, gen):
                    new_population.append(wild_child)
                    with open(ACTION_LOG, 'a') as f:
                        f.write(f"{gen},chaos_mutate\n")
                    continue
                # if wild_child fails realism, we drop through into normal path

            # === SAFE SURVIVOR + WEIGHT PREP FUNCTION ===
            def get_weighted_survivors(survivors, generation_df):
                weights = []
                for s in survivors:
                    row = generation_df[generation_df['Peptide'] == s]
                    if not row.empty:
                        weights.append(row['Fitness_Score'].values[0])
                    else:
                        weights.append(1.0)  # fallback for injected/random peptides
                return survivors, weights

            # Get valid survivors and ensure weights match
            survivors, weights = get_weighted_survivors(survivors, generation_df)


            # --- reproduction phase ---
            if random.random() < 0.5 and len(survivors) >= 2:
                try:
                    mode_roll = random.random()

                    # Extra exploratory branch: sometimes pick parents totally at random
                    if random.random() < explore_parent_prob:
                        p1 = random.choice(survivors)
                        p2 = random.choice(survivors)
                    elif mode_roll < 0.5:
                        # Tournament selection (soft but biased)
                        p1 = tournament_pick(survivors, generation_df, k=tournament_k)
                        p2 = tournament_pick(survivors, generation_df, k=tournament_k)
                    else:
                        # Mix of uniform + weighted
                        #   20% uniform (fitness-blind), 80% weighted by fitness
                        if random.random() < 0.20:
                            p1 = random.choice(survivors)
                            p2 = random.choice(survivors)
                        else:
                            p1, p2 = random.choices(survivors, weights=weights, k=2)

                    # Ensure parents aren't near-clones; try a few times
                    tries = 0

                    while sequence_identity(p1, p2) > 0.8 and tries < 5:
                        # resample p2 (keep p1 fixed)
                        if random.random() < explore_parent_prob:
                            p2 = random.choice(survivors)
                        elif mode_roll < 0.5:
                            p2 = tournament_pick(survivors, generation_df, k=tournament_k)
                        else:
                            if random.random() < 0.20:
                                p2 = random.choice(survivors)
                            else:
                                p2 = random.choices(survivors, weights=weights, k=1)[0]
                        tries += 1

                except ValueError as e:
                    print(f"⚠️ Weight mismatch during crossover: {e}")
                    # Fallback: just use equal weights so we don't crash
                    weights = np.ones(len(survivors))
                    p1, p2 = random.choices(survivors, weights=weights, k=2)


                child = crossover(p1, p2)
                int_encoded = np.array([encode_int(child)])
                amp = float(amp_model.predict(int_encoded, verbose=0)[0][0])
                tox = float(toxicity_model.predict(int_encoded, verbose=0)[0][0])
                stab = float(stability_model.predict(int_encoded, verbose=0)[0][0])

                child_fitness = score_peptide(
                    child,
                    amp,
                    tox,
                    stab,
                    previous_peptides=population,
                    agent=agent,
                )[3]


                f1 = generation_df[generation_df['Peptide'] == p1]['Fitness_Score'].values
                f2 = generation_df[generation_df['Peptide'] == p2]['Fitness_Score'].values

                if f1.size > 0 and f2.size > 0:
                    if child_fitness >= 0.8 * max(f1[0], f2[0]) and is_realistic(child, gen):
                        new_population.append(child)
                        with open(ACTION_LOG, 'a') as f:
                            f.write(f"{gen},crossover\n")
                        continue  # go back to while loop to fill next slot





            # --- cognitive mutation fallback ---
            parent = random.choice(survivors)
            
            # Unpack the new return tuple (mutant AND reasons)
            mutant, reasons = cognitive_mutate(
                parent, mutation_rate, agent,
                amp_model, toxicity_model, stability_model,
                population, peptide_length
            )
            
            if is_realistic(mutant, gen):
                new_population.append(mutant)
                
                # 🧠 LOG THE THOUGHT PROCESS
                with open(REASON_LOG, 'a') as f:
                    f.write(f"{gen},{parent},{mutant},"
                            f"{reasons['amp']:.4f},{reasons['safety']:.4f},"
                            f"{reasons['stability']:.4f},{reasons['realism']:.4f},"
                            f"{reasons['novelty']:.4f},{agent.reason_score(reasons):.4f}\n")
                
                with open(ACTION_LOG, 'a') as f:
                    f.write(f"{gen},mutate\n")
                continue

        print(f"✅ Finished building next population for Generation {gen} (size={len(new_population)})")

        if gen % 500 == 0:
            snap_path = f"data/gen_{gen:04d}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            generation_df.to_csv(snap_path, index=False)
            print(f"💾 Snapshot saved: {snap_path}")

        # 🌪️ Exploration injection: add 5 wildcards every 250 generations
        if gen % 100 == 0 and avg_realism >= 0.6:
            wildcards = [''.join(random.choices(amino_acids, k=peptide_length)) for _ in range(20)]
            wildcards = [w for w in wildcards if is_realistic(w, gen)]

            n_inject = min(15, len(wildcards))
            print(f"🧬 Injecting {n_inject} exploratory wildcards into population.")
            new_population.extend(wildcards[:n_inject])


            # 🛡️ Safety filter after wildcard injection
            next_pop_encoded = np.array([encode_int(p) for p in new_population])
            amp_scores_next = amp_model.predict(next_pop_encoded, verbose=0).flatten()
            tox_scores_next = toxicity_model.predict(next_pop_encoded, verbose=0).flatten()
            stab_scores_next = stability_model.predict(next_pop_encoded, verbose=0).flatten()

            scored = []
            for i, p in enumerate(new_population):
                f = score_peptide(
                    p,
                    amp_scores_next[i],
                    tox_scores_next[i],
                    stab_scores_next[i],
                    agent=agent,
                )[3]
                scored.append((p, f))

            # Drop bottom 20%
            scored.sort(key=lambda x: x[1], reverse=True)
            filtered = [p for p, _ in scored[:int(0.8 * len(scored))]]
            print(f"🧹 Removed {len(new_population) - len(filtered)} peptides with lowest fitness post-wildcards.")
            new_population = filtered

        else:
            if gen % 100 != 0:
                print(f"⏩ Skipped wildcard injection — not scheduled this generation (gen {gen})")
            elif avg_realism < 0.6:
                print(f"⛔ Skipped wildcard injection — realism too low (requires ≥ 0.6, got {avg_realism:.3f})")

            # 🔥 Extra wildcards if we’ve been stuck too long
        if stagnant_generations > 100:
            print(f"🔥 Injecting anti-stagnation wildcard batch after {stagnant_generations} stagnant generations.")
            wild_batch = [''.join(random.choices(amino_acids, k=peptide_length)) for _ in range(30)]
            filtered = [w for w in wild_batch if is_realistic(w, gen)]
            print(f"🧪 {len(filtered[:10])} wildcards passed realism checks.")
            new_population.extend(filtered[:10])

        with open(FITNESS_STATS_FILE, 'a') as stat_file:
            if gen == 1:
                stat_file.write("Generation,AvgFitness,MaxFitness,StdFitness\n")
            stat_file.write(f"{gen},{avg_fitness:.5f},{max_fitness:.5f},{std_fitness:.5f}\n")

        population = new_population
        save_population(population)

        # 🛑 Check for early stop request
        if check_early_stop():
            print("🛑 Early stop signal detected (stop.txt). Finalizing run gracefully...")
            os.remove(EARLY_STOP_FILE)
            break

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
    print("🗑️ Checkpoints cleaned after successful run.")

# Only run this if evolve.py is executed directly, not imported
if __name__ == "__main__":
    run_simulation()
