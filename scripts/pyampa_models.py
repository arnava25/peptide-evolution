#!/usr/bin/env python3
import os
from typing import List, Dict, Any

import numpy as np
from joblib import load  # scikit-learn models are usually saved via joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYAMPA_DIR = os.path.join(BASE_DIR, "external", "PyAMPA")

# Paths to models & vectorizers
PATHS = {
    "amp_validate_model": os.path.join(PYAMPA_DIR, "AMPValidate.pkl"),
    "amp_validate_vec": os.path.join(PYAMPA_DIR, "amp_validate_vectorizer.pkl"),

    "activity_model": os.path.join(PYAMPA_DIR, "activities_model.pkl"),
    "activity_vec": os.path.join(PYAMPA_DIR, "activities_vectorizer.pkl"),
    "label_encoder": os.path.join(PYAMPA_DIR, "label_encoder.pkl"),

    "tox_model": os.path.join(PYAMPA_DIR, "tox_model.pkl"),
    "tox_vec": os.path.join(PYAMPA_DIR, "tox_vectorizer.pkl"),

    "hem_model": os.path.join(PYAMPA_DIR, "hemolysis_model.pkl"),
    "hem_vec": os.path.join(PYAMPA_DIR, "hemolysis_vectorizer.pkl"),
}


class PyAMPAWrapper:
    def __init__(self):
        # Load everything once
        self.amp_validate_model = load(PATHS["amp_validate_model"])
        self.amp_validate_vec = load(PATHS["amp_validate_vec"])

        self.activity_model = load(PATHS["activity_model"])
        self.activity_vec = load(PATHS["activity_vec"])
        self.label_encoder = load(PATHS["label_encoder"])

        self.tox_model = load(PATHS["tox_model"])
        self.tox_vec = load(PATHS["tox_vec"])

        self.hem_model = load(PATHS["hem_model"])
        self.hem_vec = load(PATHS["hem_vec"])

    def _binary_prob(self, model, X, positive_label_index: int = 1):
        """Generic helper to get P(positive) from a binary classifier."""
        proba = model.predict_proba(X)
        # proba shape = (n_samples, n_classes)
        # We don't yet know which column is 'positive' until we inspect classes_.
        return proba[:, positive_label_index]

    def predict(self, peptides: List[str]) -> Dict[str, Any]:
        """
        Return PyAMPA predictions for a batch of peptides.

        Output keys (proposed):
          - amp_validate_prob: P(“is AMP”) from AMPValidate model
          - activity_probs: dict(activity_label -> prob)
          - tox_prob: P(“toxic”)
          - hemolysis_prob: P(“hemolytic”)
        """
        # AMP / non-AMP
        X_amp = self.amp_validate_vec.transform(peptides)
        amp_proba = self.amp_validate_model.predict_proba(X_amp)
        # We still need to inspect amp_validate_model.classes_ once
        # to know which column is AMP vs non-AMP.
        # For now we'll expose the full matrix:
        amp_classes = self.amp_validate_model.classes_

        # Multi-class activity (e.g. antibacterial / antifungal / etc.)
        X_act = self.activity_vec.transform(peptides)
        act_proba = self.activity_model.predict_proba(X_act)
        act_labels = self.label_encoder.inverse_transform(
            np.arange(act_proba.shape[1])
        )

        # Toxicity
        X_tox = self.tox_vec.transform(peptides)
        tox_proba = self.tox_model.predict_proba(X_tox)
        tox_classes = self.tox_model.classes_

        # Hemolysis
        X_hem = self.hem_vec.transform(peptides)
        hem_proba = self.hem_model.predict_proba(X_hem)
        hem_classes = self.hem_model.classes_

        return {
            "amp_proba": amp_proba,
            "amp_classes": amp_classes,
            "activity_proba": act_proba,
            "activity_labels": act_labels,
            "tox_proba": tox_proba,
            "tox_classes": tox_classes,
            "hem_proba": hem_proba,
            "hem_classes": hem_classes,
        }


if __name__ == "__main__":
    # Quick smoke test – you can run:
    #   python3 scripts/pyampa_models.py
    test_peps = [
        "AGHAILRVGRDLLRALWKRRKLLKC",
        "AAAAAAAAGGGGGGGG",
    ]
    wrapper = PyAMPAWrapper()
    preds = wrapper.predict(test_peps)
    print("AMP classes:", preds["amp_classes"])
    print("Tox classes:", preds["tox_classes"])
    print("Hemolysis classes:", preds["hem_classes"])
    print("Activity labels:", preds["activity_labels"])

