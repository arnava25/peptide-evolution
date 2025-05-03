import pandas as pd
import os

MASTER_EVOLUTION_FILE = 'data/master_evolution_history.csv'

def summarize_run():
    if not os.path.exists(MASTER_EVOLUTION_FILE):
        print("❌ Master evolution file not found.")
        return

    print("📊 Summarizing evolution run...")

    # Load without dtype constraints to avoid casting issues
    df = pd.read_csv(MASTER_EVOLUTION_FILE, low_memory=False)

    # Safely coerce numeric columns
    numeric_cols = ['Fitness_Score', 'Aggregation_Risk', 'Solubility_Score', 'Boman_Index']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows where fitness is missing (core metric)
    df = df.dropna(subset=['Fitness_Score'])

    # Summary metrics
    total_peptides = len(df)
    avg_fitness = df['Fitness_Score'].mean()
    avg_aggregation = df['Aggregation_Risk'].mean()
    avg_solubility = df['Solubility_Score'].mean()
    avg_boman = df['Boman_Index'].mean()

    print("📈 Summary of Evolution:")
    print(f"• Total peptides evaluated: {total_peptides:,}")
    print(f"• Average fitness: {avg_fitness:.4f}")
    print(f"• Average aggregation risk: {avg_aggregation:.4f}")
    print(f"• Average solubility score: {avg_solubility:.4f}")
    print(f"• Average Boman index: {avg_boman:.4f}")

if __name__ == "__main__":
    summarize_run()
