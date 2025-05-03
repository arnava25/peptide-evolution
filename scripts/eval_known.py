import pandas as pd
import numpy as np
from tensorflow import keras
from evolve import (
    encode_int,
    score_peptide,
    assess_peptide_quality,
)

# === Load updated AMP model ===
print("📦 Loading updated model...")
amp_model = keras.models.load_model('models/amp_model.keras')
toxicity_model = keras.models.load_model('models/toxicity_cnn_model.keras')
stability_model = keras.models.load_model('models/stability_cnn_model.keras')
print("✅ Models loaded.")

# === Load input peptide CSV ===
input_file = "data/known_peptide_input.csv"
df = pd.read_csv(input_file)

assert 'Peptide' in df.columns and 'True_Label' in df.columns, "CSV must contain 'Peptide' and 'True_Label'."

results = []

for _, row in df.iterrows():
    peptide = str(row["Peptide"])
    true_label = row["True_Label"]

    int_encoded = np.array([encode_int(peptide)])
    amp_score = amp_model.predict(int_encoded, verbose=0)[0][0]
    tox_score = toxicity_model.predict(int_encoded, verbose=0)[0][0]
    stab_score = stability_model.predict(int_encoded, verbose=0)[0][0]

    _, _, _, fitness, *_ = score_peptide(peptide, amp_score, tox_score, stab_score)
    quality = assess_peptide_quality(peptide)

    results.append({
        "Peptide": peptide,
        "True_Label": true_label,
        "AMP_Model": amp_score,
        "Toxicity_Model": tox_score,
        "Stability_Model": stab_score,
        "Fitness_Score": fitness,
        "Quality_Tag": quality
    })

# === Save & Display ===
output_df = pd.DataFrame(results)
print("\n🧪 Evaluation Results:")
print(output_df[["Peptide", "True_Label", "AMP_Model", "Fitness_Score", "Quality_Tag"]])

output_path = "data/eval_output_with_predictions.csv"
output_df.to_csv(output_path, index=False)
print(f"\n📁 Saved results to {output_path}")
