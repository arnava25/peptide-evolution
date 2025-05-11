# Peptide Evolution Framework

This repository contains the source code and minimal training data for the AMP evolution system described in:

**A Realism-Constrained Framework for De Novo Antimicrobial Peptide Evolution via Multi-Objective Fitness**  
(Arnav Amit, 2025)

The framework uses evolutionary simulation, multi-objective scoring, and neural predictors to optimize de novo antimicrobial peptides under biochemical constraints.

## Running the Simulation

Install dependencies (Python 3.8+):

```bash
pip install -r requirements.txt
```

Run:

```bash
python scripts/evolve.py
```

Example outputs are saved to:

- `data/top_20_simulated_peptides.csv`

## Model Files

Pretrained models (`.keras`) are **not included** due to file size and licensing constraints. To retrain them, use:

- `scripts/train_amp_model_v2.py`
- `scripts/train_toxicity_model_v2.py`
- `scripts/train_stability_model_v2.py`

Minimal training datasets are included in:

- `data/model_trainers/`

## License

MIT License

## Citation

If you use this framework, please cite:

```
Amit, A. (2025). A Realism-Constrained Framework for De Novo Antimicrobial Peptide Evolution via Multi-Objective Fitness.
```
