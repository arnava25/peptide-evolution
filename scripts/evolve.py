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
generations = 7000
peptide_length = 25
max_len = 50

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

    # Base fitness
    fitness = (amp_score + 1e-4) * (1 - tox_score + 1e-3) * (stab_score + 1e-3)
    penalty = 1.0

    # Repeats of 4+ same AA
    for aa in set(peptide):
        if aa * 4 in peptide:
            penalty *= 0.7  # harsh penalty for unrealistic motif

    # Unique AA check
    if unique_aas < 8:
        penalty *= 0.8
    elif unique_aas >= 12:
        fitness *= 1.05

    # Hydrophobicity
    if hydrophobicity < 0.15 or hydrophobicity > 0.7:
        penalty *= 0.9

    # Polar content
    if polar_content > 0.7 or polar_content < 0.1:
        penalty *= 0.9
    elif 0.2 <= polar_content <= 0.5:
        fitness *= 1.05

    # Charge range
    if net_charge < 2 or net_charge > 8:
        penalty *= 0.9

    # Solubility and aggregation
    if solubility_score < 0.4:
        penalty *= 0.9
    if aggregation_risk > 0.5:
        penalty *= 0.9

    # Boman index
    if boman > 0.6:
        penalty *= 0.95

    # pI extremes
    if pI < 4.5 or pI > 9.5:
        penalty *= 0.95

    # Realism filter
    if not is_realistic(peptide):
        penalty *= 0.5

    # Diversity bonus
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

    # Apply penalties and realism
    fitness *= penalty
    quality_score = compute_quality_score(peptide)
    fitness *= quality_score

    if quality_score > 0.8:
        fitness *= 1.05

    sol_tag = "✅ Soluble" if solubility_score >= 0.5 else "🔴 Low Solubility"
    agg_tag = "✅ Safe" if aggregation_risk <= 0.5 else "🔴 Risky"

    return (
        amp_score, tox_score, stab_score, fitness,
        solubility_score, aggregation_risk, pI, boman,
        net_charge, hydrophobicity, sol_tag, agg_tag
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

    return ''.join(peptide)

def is_realistic(peptide):
    """Simple realism filter to reject weird/unrealistic peptides."""
    # Rule 1: No more than 2 cysteines (C) to avoid disulfide mess
    if peptide.count('C') > 2:
        return False

    # Rule 2: No more than 4 prolines (P) — too many disrupts structure
    if peptide.count('P') > 4:
        return False

    # Rule 3: Avoid super high hydrophobicity (>70%)
    hydrophobic = set('AILMVFWY')
    hydrophobic_count = sum(aa in hydrophobic for aa in peptide)
    if hydrophobic_count / len(peptide) > 0.7:
        return False

    # Rule 4: Avoid overly polar sequences (>70%)
    polar = set('STNQ')
    polar_count = sum(aa in polar for aa in peptide)
    if polar_count / len(peptide) > 0.7:
        return False

    # Rule 5: Reject forbidden motifs like "PPPP" or "CCCC"
    if 'PPPP' in peptide or 'CCCC' in peptide:
        return False

    # Rule 6: Minimum variety — should have at least 10 unique amino acids
    if len(set(peptide)) < 8:
        return False

    return True



def assess_peptide_quality(peptide):
    """Returns a tag if peptide has good properties."""
    net_charge = calculate_net_charge(peptide)
    hydrophobicity = calculate_hydrophobicity(peptide)

    if (2 <= net_charge <= 8) and (0.25 <= hydrophobicity <= 0.55) and is_realistic(peptide):
        return "✅ Good"
    else:
        return "❌ Needs Review"

def compute_quality_score(peptide):
    """Returns a float between 0 (bad) and 1 (excellent) based on realism traits."""
    score = 1.0

    net_charge = calculate_net_charge(peptide)
    hydrophobicity = calculate_hydrophobicity(peptide)
    polar_content = calculate_polar_content(peptide)
    unique_aas = len(set(peptide))

    if not is_realistic(peptide):
        return 0.4  # fallback penalty

    # Ideal charge
    if not (2 <= net_charge <= 8):
        score *= 0.9

    # Ideal hydrophobicity
    if not (0.20 <= hydrophobicity <= 0.60):
        score *= 0.9

    # Polar content bonus
    if polar_content > 0.2:
        score *= 1.05
    elif polar_content < 0.1:
        score *= 0.95

    # Bonus for diversity
    if unique_aas >= 12:
        score *= 1.05
    elif unique_aas < 8:
        score *= 1.02
    elif unique_aas < 6:
        score *= 0.85

    return min(max(score, 0.0), 1.0)



def append_generation_to_master(generation_number, generation_data):
    """Appends the current generation's data to the master evolution file."""
    df = pd.DataFrame(generation_data)

    expected_cols = [
        'Peptide', 'AMP_Score', 'Toxicity_Score', 'Stability_Score',
        'Solubility_Score', 'Aggregation_Risk', 'Isoelectric_Point',
        'Boman_Index', 'Net_Charge', 'Hydrophobicity',
        'Solubility_Tag', 'Aggregation_Tag',
        'Fitness_Score', 'Quality_Tag'
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

def append_mutation_rate(generation_number, mutation_rate, stagnant_gen=None):
    with open(MUTATION_HISTORY_FILE, 'a') as f:
        if stagnant_gen is not None:
            f.write(f"{generation_number},{mutation_rate},{stagnant_gen}\n")
        else:
            f.write(f"{generation_number},{mutation_rate}\n")


# Diversity helper
def average_similarity(population):
    similarities = []
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            sim = sequence_identity(population[i], population[j])
            similarities.append(sim)
    return np.mean(similarities) if similarities else 0

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


def run_simulation():

    # Load models

    print("📦 Loading models...")

    amp_model = keras.models.load_model('models/amp_model.keras')
    toxicity_model = keras.models.load_model('models/toxicity_cnn_model.keras')
    stability_model = keras.models.load_model('models/stability_cnn_model.keras')

    print("✅ Models loaded: AMP, Toxicity, Stability")
    
    run_start_time = datetime.now()
    RUN_TAG = run_start_time.strftime('%Y%m%d_%H%M')
    THIS_RUN_FILE = f"data/evolution_histories/evolution_run_{RUN_TAG}.csv"


    # Initialize population
    population = load_population()        
    if not population:
        top_peps = []
        try:
            df_master = pd.read_csv(MASTER_EVOLUTION_FILE, low_memory=False)
            df_master = df_master.sort_values(by='Fitness_Score', ascending=False)
            # Optional: filter only high-quality tags if needed
            df_master = df_master[df_master['Quality_Tag'].str.contains('Good|Review')]
            
            top_peps = df_master['Peptide'].drop_duplicates().head(int(0.1 * population_size)).tolist()
        except Exception as e:
            print(f"⚠️ Could not load top peptides from master history: {e}")

        num_randoms = population_size - len(top_peps)
        new_randoms = [''.join(random.choices(amino_acids, k=peptide_length)) for _ in range(num_randoms)]

        population = new_randoms + top_peps
        print(f"🌱 Created hybrid population: {len(new_randoms)} random + {len(top_peps)} elite.")

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
            amp, tox, stab, fitness, sol, agg, pI, boman, net_charge, hydrophobicity, sol_tag, agg_tag = score_peptide(pep, amp, tox, stab, previous_peptides=population)
            if gen == 1 and idx < 5:
                print(f"⚙️ AMP={amp:.2f}, TOX={tox:.2f}, STAB={stab:.2f}, FITNESS={fitness:.3f}")

            tag = assess_peptide_quality(pep)
            scores.append((
                pep, amp, tox, stab, sol, agg, pI, boman,
                net_charge, hydrophobicity, sol_tag, agg_tag,
                fitness, tag
            ))



        generation_df = pd.DataFrame(scores, columns=[
            'Peptide', 'AMP_Score', 'Toxicity_Score', 'Stability_Score',
            'Solubility_Score', 'Aggregation_Risk', 'Isoelectric_Point',
            'Boman_Index', 'Net_Charge', 'Hydrophobicity',
            'Solubility_Tag', 'Aggregation_Tag',
            'Fitness_Score', 'Quality_Tag'
        ])

        # Track generation fitness stats
        avg_fitness = generation_df['Fitness_Score'].mean()
        max_fitness = generation_df['Fitness_Score'].max()
        
        avg_sim = average_similarity(population)  
        gen_time = datetime.now() - gen_start


        if gen % 50 == 0:
            top_peps_path = f"data/top_peptides_gen_{gen:04d}_{RUN_TAG}.csv"
            generation_df.head(5).to_csv(top_peps_path, index=False)
            print(f"🏅 Saved top peptides snapshot: {top_peps_path}")

        # === Dynamic decay over time ===
        import math
        initial_mutation = 0.35
        final_mutation = 0.1
        decay_duration = 5000  # Total generations to decay across
        
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

        if stagnant_generations > 150:
            mutation_rate = max(mutation_rate, 0.5)  # override spike to re-diversify

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
            if stagnant_generations >= 500:
                print("🛑 Early stopping: No fitness improvement for 500 generations.")
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

        # Selection + Reproduction
        elite_cutoff = int(0.1 * population_size)
        diverse_cutoff = int(0.1 * population_size)

        sorted_df = generation_df.sort_values(by='Fitness_Score', ascending=False)
        elite = sorted_df.head(elite_cutoff)['Peptide'].tolist()
        rest = sorted_df.iloc[elite_cutoff:]['Peptide'].tolist()
        diverse = random.sample(rest, diverse_cutoff) if len(rest) >= diverse_cutoff else rest

        survivors = elite + diverse

        if global_best_peptide and global_best_peptide not in survivors:
            survivors.append(global_best_peptide)

        new_population = []
        while len(new_population) < population_size:
            if random.random() < 0.5 and len(survivors) >= 2:
                p1, p2 = random.choice(survivors), random.choice(survivors)
                child = crossover(p1, p2)

                int_encoded = np.array([encode_int(child)])
                amp = amp_model.predict(int_encoded, verbose=0)[0][0]
                tox = toxicity_model.predict(int_encoded, verbose=0)[0][0]
                stab = stability_model.predict(int_encoded, verbose=0)[0][0]

                child_fitness = score_peptide(child, amp, tox, stab, previous_peptides=population)[3]

                f1 = generation_df[generation_df['Peptide'] == p1]['Fitness_Score'].values
                f2 = generation_df[generation_df['Peptide'] == p2]['Fitness_Score'].values

                if f1.size > 0 and f2.size > 0:
                    if child_fitness >= max(f1[0], f2[0]) and is_realistic(child):
                        new_population.append(child)
                        continue

            while True:
                parent = random.choice(survivors)
                mutant = smart_mutate(parent, mutation_rate)
                if is_realistic(mutant):
                    new_population.append(mutant)
                    break




        if gen % 500 == 0:
            snap_path = f"data/gen_{gen:04d}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            generation_df.to_csv(snap_path, index=False)
            print(f"💾 Snapshot saved: {snap_path}")
        

        # 🌪️ Exploration injection: add 5 wildcards every 250 generations
        if gen % 250 == 0:
            wildcards = [''.join(random.choices(amino_acids, k=peptide_length)) for _ in range(10)]
            wildcards = [w for w in wildcards if is_realistic(w)]
            print(f"🧬 Injecting {len(wildcards[:5])} exploratory wildcards into population.")
            new_population.extend(wildcards[:5])

        population = new_population
        save_population(population)

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
