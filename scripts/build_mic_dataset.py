import pandas as pd
import os
import re

dbaasp_dir = os.path.expanduser('~/Desktop/ampevolv3/data/dbaasp')
output_path = 'data/model_trainers/mic_ecoli_dataset.csv'

VALID_AAS = set('ACDEFGHIKLMNPQRSTVWY')

def is_valid_sequence(seq):
    if not isinstance(seq, str): return False
    seq = seq.strip().upper()
    if len(seq) < 5 or len(seq) > 60: return False
    return all(c in VALID_AAS for c in seq)

def parse_mic(val):
    """Parse MIC value to float. Handle ranges by taking geometric mean."""
    val = str(val).strip()
    if '-' in val:
        parts = val.split('-')
        try:
            a, b = float(parts[0]), float(parts[1])
            return (a * b) ** 0.5  # geometric mean
        except:
            return None
    try:
        return float(val)
    except:
        return None

# Load all peptide segments
print("Loading peptide sequences...")
pep_dfs = []
for i in range(1, 14):
    path = os.path.join(dbaasp_dir, f'peptidesSEG{i}.csv')
    if os.path.exists(path):
        df = pd.read_csv(path, on_bad_lines='skip')
        pep_dfs.append(df)
pep_df = pd.concat(pep_dfs, ignore_index=True)
print(f"Total peptides: {len(pep_df)}")

# Load all activity segments
print("Loading activity data...")
act_dfs = []
for i in range(1, 14):
    path = os.path.join(dbaasp_dir, f'activity-against-target-speciesSEG{i}.csv')
    if os.path.exists(path):
        df = pd.read_csv(path, on_bad_lines='skip')
        act_dfs.append(df)
act_df = pd.concat(act_dfs, ignore_index=True)
print(f"Total activity rows: {len(act_df)}")

# Filter E. coli MIC only
ecoli = act_df[
    act_df['Target Species'].str.contains('Escherichia coli', case=False, na=False) &
    (act_df['Activity Measure'] == 'MIC')
].copy()
print(f"E. coli MIC rows: {len(ecoli)}")

# Parse MIC values
ecoli['MIC_float'] = ecoli['Activity'].apply(parse_mic)
ecoli = ecoli.dropna(subset=['MIC_float'])

# Keep only µM or µg/ml, exclude weird units
ecoli = ecoli[ecoli['Unit'].isin(['µM', 'µg/ml', 'μM', 'μg/ml'])]

# Take lowest MIC per peptide (most potent measurement)
ecoli_min = ecoli.groupby('Peptide ID')['MIC_float'].min().reset_index()
ecoli_min.columns = ['ID', 'MIC']

# Merge with sequences
pep_df['ID'] = pep_df['ID'].astype(str)
ecoli_min['ID'] = ecoli_min['ID'].astype(str)
merged = ecoli_min.merge(pep_df[['ID', 'SEQUENCE']], on='ID', how='inner')

# Clean sequences - remove spaces, multimers etc
merged['SEQUENCE'] = merged['SEQUENCE'].str.replace(r'\s+', '', regex=True).str.upper()
merged = merged[merged['SEQUENCE'].apply(is_valid_sequence)]
merged = merged.drop_duplicates(subset='SEQUENCE', keep='first')

print(f"Final dataset: {len(merged)} peptides with E. coli MIC values")
print(f"MIC range: {merged['MIC'].min():.2f} - {merged['MIC'].max():.2f}")
print(merged[['SEQUENCE', 'MIC']].head(10))

merged[['SEQUENCE', 'MIC']].to_csv(output_path, index=False)
print(f"Saved to {output_path}")
