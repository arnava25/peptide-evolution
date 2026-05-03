import os
from joblib import load
import numpy as np
from scipy import sparse


import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names*",
    category=UserWarning,
    module="sklearn"
)

BASE = os.path.join(os.path.dirname(__file__), "PyAMPA")

# Load vectorizers + models
# We wrap loads in try/except to prevent crashes if files are missing/corrupt
try:
    amp_model = load(os.path.join(BASE, "activities_model.pkl"))
    amp_vec   = load(os.path.join(BASE, "activities_vectorizer.pkl"))

    tox_model = load(os.path.join(BASE, "tox_model.pkl"))
    tox_vec   = load(os.path.join(BASE, "tox_vectorizer.pkl"))

    hemo_model = load(os.path.join(BASE, "hemolysis_model.pkl"))
    hemo_vec   = load(os.path.join(BASE, "hemolysis_vectorizer.pkl"))

    cpp_model = load(os.path.join(BASE, "cpp_model.pkl"))
    cpp_vec   = load(os.path.join(BASE, "cpp_vectorizer.pkl"))
except Exception as e:
    print(f"⚠️ Error loading PyAMPA models: {e}")


def _get_2mers(seq: str) -> str:
    """
    Converts 'LLGDFF' -> 'LL LG GD DF FF'
    This is the specific grammar PyAMPA vectorizers were trained on.
    """
    if len(seq) < 2:
        return seq
    return " ".join([seq[i:i+2] for i in range(len(seq) - 1)])


def _score_with_model(seq: str, model, vec) -> float:
    """
    Tokenizes sequence, adapts dimensionality, and predicts.
    Handles both Regressors (predict) and Classifiers (predict_proba).
    """
    # 1. Apply the 2-mer Tokenization (CRITICAL FIX)
    tokenized_seq = _get_2mers(seq)
    
    # 2. Vectorize
    X = vec.transform([tokenized_seq])

    # 3. Dimensionality Fix (Padding/Trimming)
    # Some PyAMPA models expect exact feature counts
    n_model = getattr(model, "n_features_in_", None)
    if n_model is not None and X.shape[1] != n_model:
        if X.shape[1] < n_model:
            diff = n_model - X.shape[1]
            if sparse.issparse(X):
                pad = sparse.csr_matrix((X.shape[0], diff))
                X = sparse.hstack([X, pad])
            else:
                pad = np.zeros((X.shape[0], diff))
                X = np.hstack([X, pad])
        elif X.shape[1] > n_model:
            # Slice
            X = X[:, :n_model]

    # 4. Predict
    try:
        # Try probabilistic classification first (for Tox/Hemo if they are classifiers)
        if hasattr(model, "predict_proba"):
            # Usually Class 1 = Positive (Toxic)
            return float(model.predict_proba(X)[0][1])
    except:
        pass

    # Fallback to Regression/Predict (For AMP model)
    val = float(model.predict(X)[0])
    return val

def pyampa_scores(seq: str):
    """
    Return PyAMPA scores.
    """
    # Raw Outputs
    raw_amp  = _score_with_model(seq, amp_model,  amp_vec)
    raw_tox  = _score_with_model(seq, tox_model,  tox_vec)
    raw_hemo = _score_with_model(seq, hemo_model, hemo_vec)
    raw_cpp  = _score_with_model(seq, cpp_model,  cpp_vec)

    # --- NORMALIZATION & POLARITY CORRECTION ---
    
    # 1. AMP: Lower is Better (MIC). Convert to "Higher is Better" [0-1]
    #    Example: If raw_amp is -0.5 (Super Potent), exp is 1.64 -> Clip to 1.0
    amp_score = np.exp(-raw_amp)
    amp_score = float(np.clip(amp_score, 0.0, 1.0))  # <--- THE FIX

    # 2. Tox/Hemo/CPP: 
    #    Clamp to [0,1]
    tox_score  = max(0.0, min(1.0, raw_tox))
    hemo_score = max(0.0, min(1.0, raw_hemo))
    cpp_score  = max(0.0, min(1.0, raw_cpp))

    return {
        "amp":       amp_score,
        "tox":       tox_score,
        "hemolysis": hemo_score,
        "cpp":       cpp_score,
    }
