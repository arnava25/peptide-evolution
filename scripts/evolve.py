import os
import random
import string
import numpy as np
import pandas as pd
from tensorflow import keras
from datetime import datetime

def sequence_identity(seq1, seq2):
    """Calculates % identity between two sequences (simple)."""
    matches = sum(a == b for a, b in zip(seq1, seq2))
    return matches / min(len(seq1), len(seq2))

CURRENT_POP_FILE = 'data/current_population.txt'
MUTATION_HISTORY_FILE = 'data/mutation_rate_history.txt'

EARLY_STOP_FILE = 'stop.txt'


# 👁️ Attentional motif salience memory
salient_motifs = {}  # Stores k-mers and their evolving salience values


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
population_size = 600
generations = 2000
peptide_length = 25
max_len = 50

STAGNANT_LIMIT = 800

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

def score_peptide(peptide, amp_score, tox_score, stab_score, previous_peptides=None):
    solubility_score = estimate_solubility(peptide)
    aggregation_risk = estimate_aggregation_risk(peptide)
    net_charge = calculate_net_charge(peptide)
    hydrophobicity = calculate_hydrophobicity(peptide)
    polar_content = calculate_polar_content(peptide)
    pI = calculate_isoelectric_point(peptide)
    boman = calculate_boman_index(peptide)
    unique_aas = len(set(peptide))

    max_sim = 0.0
    min_sim = 1.0

    # Base fitness
    fitness = (amp_score + 1e-4) * (1 - tox_score + 1e-3) * (stab_score + 1e-3)

    penalty_score = 1.0  # Start from 1, reduce gently

    # === Trait Soft Penalties ===
    if not (2 <= net_charge <= 8):
        penalty_score *= 0.95
    if hydrophobicity < 0.15 or hydrophobicity > 0.7:
        penalty_score *= 0.95
    if polar_content < 0.1 or polar_content > 0.7:
        penalty_score *= 0.95
    if solubility_score < 0.4:
        penalty_score *= 0.95
    if aggregation_risk > 0.5:
        penalty_score *= 0.95
    if boman > 0.6:
        penalty_score *= 0.95
    if pI < 4.5 or pI > 9.5:
        penalty_score *= 0.95
    if unique_aas < 8:
        penalty_score *= 0.9  # mild diversity clamp
    if unique_aas >= 12:
        fitness *= 1.05  # diversity reward

    for aa in set(peptide):
        if aa * 4 in peptide:
            penalty_score *= 0.7  # hard motif penalty

    # === Trait Bonuses ===
    if 0.2 <= polar_content <= 0.5:
        fitness *= 1.05
    if solubility_score >= 0.6 and aggregation_risk <= 0.4:
        fitness *= 1.05

    # === Diversity Bonus ===
    if previous_peptides:
        similarities = [sequence_identity(peptide, p) for p in previous_peptides]
        max_sim = max(similarities)
        min_sim = min(similarities)
        if max_sim < 0.4:
            fitness *= 1.1
        elif max_sim > 0.8:
            fitness *= 0.95
        elif min_sim < 0.5:
            fitness *= 1.05
    
    ## Compute expected fitness from model scores alone
    expected_fitness = (amp_score + 1e-4) * (1 - tox_score + 1e-3) * (stab_score + 1e-3)
    prediction_error = abs(fitness - expected_fitness)

    # Reward surprising peptides
    if 0.05 < prediction_error < 0.2:
        fitness *= 1.05  # gentle reward for moderate surprise
    elif prediction_error >= 0.2:
        fitness *= 1.1  # stronger reward for big surprises


    # Curiosity index = novelty * surprise
    novelty_score = 1 - max_sim
    surprise_score = abs(hydrophobicity - 0.4)  # assume 0.4 is expected average

    curiosity_index = novelty_score * surprise_score

    if 0.05 < curiosity_index < 0.2:
        fitness *= 1.1
    elif curiosity_index > 0.3:
        fitness *= 0.95


    # === Final Quality Modifier ===
    quality_score = compute_quality_score(peptide)

    # === Soft Realism Modifier ===
    realism_weight = realism_penalty_score(peptide)
    penalty_score *= realism_weight
    quality_score *= realism_weight

    fitness *= penalty_score
    fitness *= quality_score
    if 0.4 <= quality_score <= 0.8:
        fitness *= 1.05  # Reward plausible but not peak peptides
    if quality_score > 0.85:
        fitness *= 1.05

    fitness = min(fitness, 1.0)  # Final cap
    sol_tag = "✅ Soluble" if solubility_score >= 0.5 else "🔴 Low Solubility"
    agg_tag = "✅ Safe" if aggregation_risk <= 0.5 else "🔴 Risky"

    return (
        amp_score, tox_score, stab_score, fitness,
        solubility_score, aggregation_risk, pI, boman,
        net_charge, hydrophobicity, sol_tag, agg_tag, realism_weight
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
        'Peptide', 'AMP_Score', 'Toxicity_Score', 'Stability_Score',
        'Solubility_Score', 'Aggregation_Risk', 'Isoelectric_Point',
        'Boman_Index', 'Net_Charge', 'Hydrophobicity',
        'Solubility_Tag', 'Aggregation_Tag',
        'Fitness_Score', 'Realism_Score', 'Quality_Tag'
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

def append_mutation_rate(generation_number, mutation_rate, stagnant_gen=None):
    with open(MUTATION_HISTORY_FILE, 'a') as f:
        if stagnant_gen is not None:
            f.write(f"{generation_number},{mutation_rate},{stagnant_gen}\n")
        else:
            f.write(f"{generation_number},{mutation_rate}\n")

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

def diverse_subset(peptides, threshold=0.85):
    unique = []
    for p in peptides:
        if all(sequence_identity(p, u) < threshold for u in unique):
            unique.append(p)
        if len(unique) >= int(0.1 * population_size):
            break
    return unique

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
    min_realism = 0.4 + 0.3 * min(gen / 5000, 1.0)
    return base >= min_realism

def run_simulation():

    # Load models

    print("📦 Loading models...")

    amp_model = keras.models.load_model('models/amp_model.keras')
    toxicity_model = keras.models.load_model('models/toxicity_cnn_model.keras')
    stability_model = keras.models.load_model('models/stability_cnn_model.keras')

    print("✅ Models loaded: AMP, Toxicity, Stability")

    run_start_time = datetime.now()
    # Similarity log path (unique per run)
    os.makedirs('data/similarity_logs', exist_ok=True)
    RUN_TAG = run_start_time.strftime('%Y%m%d_%H%M')
    SIMILARITY_LOG_PATH = f"data/similarity_logs/similarity_log_{RUN_TAG}.csv"
    # Create trait stats output folder if needed
    os.makedirs('data/trait_stats', exist_ok=True)
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

        encoded_population = np.array([encode_int(p) for p in population])
        amp_scores = amp_model.predict(encoded_population, verbose=0).flatten()
        tox_scores = toxicity_model.predict(encoded_population, verbose=0).flatten()
        stab_scores = stability_model.predict(encoded_population, verbose=0).flatten()

        scores = []
        for idx, pep in enumerate(population):
            amp = amp_scores[idx]
            tox = tox_scores[idx]
            stab = stab_scores[idx]
            amp, tox, stab, fitness, sol, agg, pI, boman, net_charge, hydrophobicity, sol_tag, agg_tag, realism_score = score_peptide(pep, amp, tox, stab, previous_peptides=population)
            
            update_salient_motifs(pep, fitness)

            if gen == 1 and idx < 5:
                print(f"⚙️ AMP={amp:.2f}, TOX={tox:.2f}, STAB={stab:.2f}, FITNESS={fitness:.3f}")

            tag = assess_peptide_quality(pep)
            scores.append((
                pep, amp, tox, stab, sol, agg, pI, boman,
                net_charge, hydrophobicity, sol_tag, agg_tag,
                fitness, realism_score, tag
))
        generation_df = pd.DataFrame(scores, columns=[
            'Peptide', 'AMP_Score', 'Toxicity_Score', 'Stability_Score',
            'Solubility_Score', 'Aggregation_Risk', 'Isoelectric_Point',
            'Boman_Index', 'Net_Charge', 'Hydrophobicity',
            'Solubility_Tag', 'Aggregation_Tag',
            'Fitness_Score', 'Realism_Score', 'Quality_Tag'
        ])

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

        avg_sim = average_similarity(population)
        gen_time = datetime.now() - gen_start

        with open(SIMILARITY_LOG_PATH, 'a') as sim_log:
            if gen == 1:
                sim_log.write("Generation,Similarity\n")
            sim_log.write(f"{gen},{avg_sim:.4f}\n")

        if gen % 50 == 0:
            top_peps_path = f"data/top_peptides_gen_{gen:04d}_{RUN_TAG}.csv"
            generation_df.head(5).to_csv(top_peps_path, index=False)
            print(f"🏅 Saved top peptides snapshot: {top_peps_path}")

        # === Dynamic decay over time ===
        import math
        initial_mutation = 0.4
        final_mutation = 0.1
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
            mutation_rate = min(mutation_rate, 0.2)

        mutation_rate = min(max(mutation_rate, 0.05), 0.6)  # Clamp within bounds

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
            if stagnant_generations >= STAGNANT_LIMIT:
                print("🛑 Early stopping: No fitness improvement for 800 generations.")
                break

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
        realism_threshold = 0.2 + 0.6 * min(gen / 5000, 1.0)  # ramps up to 0.8

        filtered_df = generation_df[
            (generation_df["Fitness_Score"] > 0.4) &
            (generation_df["Realism_Score"] >= realism_threshold)
        ]

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
                score_peptide(p, amp_scores[i], tox_scores[i], stab_scores[i])[3]
                for i, p in enumerate(survivors)
            ])
            print(f"🧪 Injected {len(survivors)} new survivors with fallback fitness scores.")

        # 🔽 INSERT HERE:
        def similarity_penalty(pop):
            sims = [sequence_identity(a, b) for i, a in enumerate(pop) for b in pop[i+1:]]
            return np.mean(sims)

        if similarity_penalty(survivors) > 0.8:
            print("⚠️ Low diversity detected — applying harsher mutations or random reinjection.")
            # Optional: add 20% new randoms or increase mutation rate
            extra_randoms = [''.join(random.choices(amino_acids, k=peptide_length)) for _ in range(int(0.2 * population_size))]
            survivors.extend(extra_randoms)

        new_population = []
        while len(new_population) < population_size:
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

            new_population = []
            while len(new_population) < population_size:
                if random.random() < 0.5 and len(survivors) >= 2:
                    try:
                        p1, p2 = random.choices(survivors, weights=weights, k=2)
                    except ValueError as e:
                        print(f"⚠️ Weight mismatch during crossover: {e}")
                        weights = np.ones(len(survivors))  # fallback
                        p1, p2 = random.choices(survivors, weights=weights, k=2)

                    child = crossover(p1, p2)
                    int_encoded = np.array([encode_int(child)])
                    amp = amp_model.predict(int_encoded, verbose=0)[0][0]
                    tox = toxicity_model.predict(int_encoded, verbose=0)[0][0]
                    stab = stability_model.predict(int_encoded, verbose=0)[0][0]
                    child_fitness = score_peptide(child, amp, tox, stab, previous_peptides=population)[3]

                    f1 = generation_df[generation_df['Peptide'] == p1]['Fitness_Score'].values
                    f2 = generation_df[generation_df['Peptide'] == p2]['Fitness_Score'].values

                    if f1.size > 0 and f2.size > 0:
                        if child_fitness >= 0.8 * max(f1[0], f2[0]) and is_realistic(child, gen):
                            new_population.append(child)
                            continue

                # fallback: mutation from a survivor
                while True:
                    parent = random.choice(survivors)
                    mutant = smart_mutate(parent, mutation_rate)
                    if is_realistic(mutant, gen):
                        new_population.append(mutant)
                        break

        if gen % 500 == 0:
            snap_path = f"data/gen_{gen:04d}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            generation_df.to_csv(snap_path, index=False)
            print(f"💾 Snapshot saved: {snap_path}")

        # 🌪️ Exploration injection: add 5 wildcards every 250 generations
        if gen % 100 == 0 and avg_realism >= 0.6:
            wildcards = [''.join(random.choices(amino_acids, k=peptide_length)) for _ in range(30)]
            wildcards = [w for w in wildcards if is_realistic(w, gen)]
            print(f"🧬 Injecting {len(wildcards[:20])} exploratory wildcards into population.")
            new_population.extend(wildcards[:20])

            # 🛡️ Safety filter after wildcard injection
            next_pop_encoded = np.array([encode_int(p) for p in new_population])
            amp_scores_next = amp_model.predict(next_pop_encoded, verbose=0).flatten()
            tox_scores_next = toxicity_model.predict(next_pop_encoded, verbose=0).flatten()
            stab_scores_next = stability_model.predict(next_pop_encoded, verbose=0).flatten()

            scored = []
            for i, p in enumerate(new_population):
                f = score_peptide(p, amp_scores_next[i], tox_scores_next[i], stab_scores_next[i])[3]
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

        with open('data/fitness_stats.csv', 'a') as stat_file:
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
