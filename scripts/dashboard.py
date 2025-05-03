# dashboard.py (updated with solubility and aggregation plots)

import pandas as pd
import matplotlib.pyplot as plt

MASTER_EVOLUTION_FILE = 'data/master_evolution_history.csv'

def load_data():
    try:
        df = pd.read_csv(MASTER_EVOLUTION_FILE)
        return df
    except Exception as e:
        print(f"⚠️ Error loading master evolution file: {e}")
        return None

def explore_database(df):
    while True:
        print("\n--- 📊 Database Explorer ---")
        print("[1] View Top Peptides by Fitness")
        print("[2] Plot Fitness Distribution")
        print("[3] Plot AMP Score vs Stability Score")
        print("[4] Plot Fitness Score vs Toxicity Score")
        print("[5] Plot Solubility vs Aggregation Risk")
        print("[6] Exit")

        choice = input("Select an option: ")

        if choice == '1':
            print(df.sort_values(by='Fitness_Score', ascending=False).head(10))

        elif choice == '2':
            plt.figure(figsize=(8,5))
            plt.hist(df['Fitness_Score'], bins=50, edgecolor='black')
            plt.title('Fitness Score Distribution')
            plt.xlabel('Fitness Score')
            plt.ylabel('Count')
            plt.show()

        elif choice == '3':
            plt.figure(figsize=(8,5))
            plt.scatter(df['AMP_Score'], df['Stability_Score'], alpha=0.5)
            plt.title('AMP Score vs Stability Score')
            plt.xlabel('AMP Score')
            plt.ylabel('Stability Score')
            plt.show()

        elif choice == '4':
            plt.figure(figsize=(8,5))
            plt.scatter(df['Fitness_Score'], df['Toxicity_Score'], alpha=0.5, color='red')
            plt.title('Fitness Score vs Toxicity Score')
            plt.xlabel('Fitness Score')
            plt.ylabel('Toxicity Score')
            plt.show()

        elif choice == '5':
            plt.figure(figsize=(8,5))
            plt.scatter(df['Solubility_Score'], df['Aggregation_Risk'], alpha=0.5, color='purple')
            plt.title('Solubility vs Aggregation Risk')
            plt.xlabel('Solubility Score')
            plt.ylabel('Aggregation Risk')
            plt.show()

        elif choice == '6':
            break

        else:
            print("Invalid option. Please try again.")

def main():
    print("==== 🧬 Evolution Database Explorer ====")

    df = load_data()

    if df is not None:
        explore_database(df)
    else:
        print("❌ Could not load data. Exiting.")

if __name__ == '__main__':
    main()
