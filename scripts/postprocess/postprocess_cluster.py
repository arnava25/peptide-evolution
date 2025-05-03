# scripts/postprocess_cluster.py
import pandas as pd
import os
from sklearn.cluster import KMeans
import numpy as np

MASTER_EVOLUTION_FILE = 'data/master_evolution_history.csv'
CLUSTERED_OUTPUT_FILE = 'data/master_evolution_clustered.csv'
NUM_CLUSTERS = 8  # Adjust this if you want more or fewer clusters

amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
aa_to_idx = {aa: idx for idx, aa in enumerate(amino_acids)}

def encode_peptide(seq, max_len=50):
    int_seq = np.zeros(max_len, dtype=np.float32)
    for i, aa in enumerate(seq[:max_len]):
        idx = aa_to_idx.get(aa)
        if idx is not None:
            int_seq[i] = idx + 1
    return int_seq

def cluster_peptides():
    if not os.path.exists(MASTER_EVOLUTION_FILE):
        print(f"❌ File not found: {MASTER_EVOLUTION_FILE}")
        return

    df = pd.read_csv(MASTER_EVOLUTION_FILE, low_memory=False)

    if 'Peptide' not in df.columns:
        print("❌ Missing 'Peptide' column in input.")
        return

    print("📊 Encoding peptides...")
    encoded_peptides = np.array([encode_peptide(p) for p in df['Peptide']])

    print(f"🔢 Clustering into {NUM_CLUSTERS} groups...")
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(encoded_peptides)

    df['Cluster'] = clusters

    print("📈 Sorting by cluster and fitness...")
    if 'Fitness_Score' in df.columns:
        df = df.sort_values(by=['Cluster', 'Fitness_Score'], ascending=[True, False])

    df.to_csv(CLUSTERED_OUTPUT_FILE, index=False)
    print(f"✅ Saved: {CLUSTERED_OUTPUT_FILE}")

if __name__ == "__main__":
    cluster_peptides()
