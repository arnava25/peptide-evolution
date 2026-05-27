import pandas as pd
import random

VALID_SOURCES = {'LAMP', 'DBAASP', 'APD3', 'ToxinPred', 'Hemolytik', 'DRAMP_Stability', 'Synthetic', 'DRAMP', 'APD6'}
VALID_AAS = set('ACDEFGHIKLMNPQRSTVWY')

def is_valid_sequence(seq):
    if not isinstance(seq, str): return False
    seq = seq.strip().upper()
    if len(seq) < 5 or len(seq) > 100: return False
    return all(c in VALID_AAS for c in seq)

def clean_existing(path):
    df = pd.read_csv(path)
    print(f"Original rows: {len(df)}")
    df = df[df['Source'].isin(VALID_SOURCES)].copy()
    print(f"After source filter: {len(df)}")
    df = df[df['Sequence'].apply(is_valid_sequence)].copy()
    df['Sequence'] = df['Sequence'].str.strip().str.upper()
    print(f"After sequence filter: {len(df)}")
    df = df.drop_duplicates(subset='Sequence', keep='first')
    print(f"After dedup: {len(df)}")
    return df

def load_apd_fasta(fasta_path):
    sequences = []
    with open(fasta_path) as f:
        seq = ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq and is_valid_sequence(seq):
                    sequences.append(seq.upper())
                seq = ""
            else:
                seq += line.strip()
        if seq and is_valid_sequence(seq):
            sequences.append(seq.upper())
    print(f"APD FASTA loaded: {len(sequences)} valid sequences")
    return sequences

def load_dramp_general(path):
    sequences = []
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                seq = parts[1].strip().upper()
                if is_valid_sequence(seq):
                    sequences.append(seq)
    print(f"DRAMP general loaded: {len(sequences)} valid sequences")
    return sequences

base = 'data/model_trainers'
df = clean_existing(f'{base}/amp_dataset_v2.csv')
existing_seqs = set(df['Sequence'].tolist())

apd_seqs = load_apd_fasta('data/apd_sequences.fasta')
apd_new = [s for s in apd_seqs if s not in existing_seqs]
print(f"New APD sequences to add: {len(apd_new)}")

apd_df = pd.DataFrame({'Sequence': apd_new, 'Name': '', 'Source': 'APD6', 'Label': 1})

dramp_seqs = load_dramp_general(f'{base}/dramp_general.txt')
dramp_new = [s for s in dramp_seqs if s not in existing_seqs and s not in set(apd_new)]
print(f"New DRAMP sequences to add: {len(dramp_new)}")

dramp_df = pd.DataFrame({'Sequence': dramp_new, 'Name': '', 'Source': 'DRAMP', 'Label': 1})

df_combined = pd.concat([df, apd_df, dramp_df], ignore_index=True)
df_combined = df_combined.drop_duplicates(subset='Sequence', keep='first')
print(f"Combined total: {len(df_combined)}")

pos = (df_combined['Label'] == 1).sum()
neg = (df_combined['Label'] == 0).sum()
print(f"Positives: {pos}, Negatives: {neg}")

if pos > neg:
    pos_df = df_combined[df_combined['Label'] == 1].sample(n=neg, random_state=42)
    neg_df = df_combined[df_combined['Label'] == 0]
else:
    neg_df = df_combined[df_combined['Label'] == 0].sample(n=pos, random_state=42)
    pos_df = df_combined[df_combined['Label'] == 1]

df_balanced = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Balanced dataset: {len(df_balanced)} ({len(pos_df)} pos, {len(neg_df)} neg)")

df_balanced.to_csv(f'{base}/amp_dataset_v3.csv', index=False)
print(f"Saved to {base}/amp_dataset_v3.csv")
