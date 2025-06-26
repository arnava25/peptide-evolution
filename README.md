# Peptide Evolution Framework

The main branch contains the source code and minimal training data for the AMP evolution system described in:

**Multi-Objective Evolution of Antimicrobial Peptides Guided by Cognitive Principles and Realism Constraints**  
(Arnav Amit, 2025)
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5313864

The dev branch includes cognition inspired heuristics in order to avoid convergence on local maxima.

The framework uses evolutionary simulation, multi-objective scoring, and neural predictors to optimize de novo antimicrobial peptides under biochemical constraints.

To run the simulation, first train the CNN models, then execute evolve.py for peptide generation and evaluation.

## Model Files

Pretrained models (`.keras`) are **not included** due to file size and licensing constraints. To retrain them, use:

- `scripts/train_amp_model_v2.py`
- `scripts/train_toxicity_model_v2.py`
- `scripts/train_stability_model_v2.py`

Minimal training datasets are included in:

- `data/model_trainers/`

Running these python files will automatically read the model_trainers datasets and generate the required models that will be called in evolve.py

## Running the Simulation

Install dependencies (Python 3.8+):

```bash
pip3 install -r requirements.txt
```

Run:

```bash
python3 scripts/evolve.py
```

Settings are set to the following by default:
  initial population = 600
  generations = 2000
  peptide length = 25

These can be changed in evolve.py. To safely terminate a long-running simulation without losing progress, you can trigger an early stop by creating a stop.txt file in the project root. In a separate window in the same directory run:

  touch stop.txt

This will signal the simulation to stop after the current generation finishes, save all logs and data, and exit cleanly.

Example outputs from are saved to:

- `data/top_20_simulated_peptides.csv`

## License

MIT License

## Citation

If you use this framework, please cite:

```
Amit, A. (2025). A Realism-Constrained Framework for De Novo Antimicrobial Peptide Evolution via Multi-Objective Fitness.
```
