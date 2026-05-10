# Peptide Evolution Framework

> **Cognitively inspired de novo antimicrobial peptide design using multi-model ensemble scoring, internal world modeling, and imagination based mutation planning.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![SSRN Preprint](https://img.shields.io/badge/Preprint-SSRN-red)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5313864)

## The Problem

Antibiotic resistance is one of the most urgent threats in modern medicine. Antimicrobial peptides (AMPs), short proteins capable of disrupting bacterial membranes, represent a promising alternative to conventional antibiotics. But discovering effective AMPs through laboratory screening is slow, expensive, and largely trial and error.

Computational evolution offers a way to search the peptide design space intelligently, but most existing approaches use standard genetic algorithms that lack mechanisms to avoid convergence, balance competing biological objectives, or reason ahead about which mutations are worth making.

This framework addresses that gap.

## What This Is

A fully agentic, cognitively inspired evolutionary engine for designing novel antimicrobial peptide candidates *in silico*, before any lab synthesis.

Rather than a standard genetic algorithm, the system runs an **AgentController** that dynamically balances seven competing biological motives (antimicrobial activity, safety, stability, realism, novelty, parsimony, and curiosity), adapts its own behavior over time, and uses **imagination based planning** to reason about mutations before committing to them.

Candidate peptides are scored by a multi-model ensemble, filtered for biochemical realism, and evolved across thousands of generations, producing sequences with predicted high antimicrobial activity, low toxicity, and structural plausibility.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    AgentController                       │
│  7 dynamic motive weights (amp, safety, stability,      │
│    realism, novelty, parsimony, curiosity)              │
│  Logistic chaos parameter r (self-tuned)                │
│  Periodic meta-introspection and weight remapping       │
│  Goal mode switching (explore / exploit / stabilize)    │
└───────────────────┬─────────────────────────────────────┘
                    │
        ┌───────────▼────────────┐
        │   Mutation Engine      │
        │  1. Prophet model      │  learned position/AA distributions
        │  2. Dreamer planning   │  imagination rollouts via world model
        │  3. Chaos mutation     │  deterministic logistic map
        └───────────┬────────────┘
                    │
        ┌───────────▼────────────────────────────┐
        │        Scoring Ensemble                 │
        │  Internal CNNs: AMP, Toxicity,          │
        │    Stability, Naturalness               │
        │  External: PyAMPA (AMP, tox,            │
        │    hemolysis, CPP validation)           │
        │  Biophysical heuristics: charge,        │
        │    hydrophobicity, hydrophobic moment,  │
        │    Boman index, pI, aggregation risk    │
        │  Ensemble disagreement penalty          │
        └───────────┬────────────────────────────┘
                    │
        ┌───────────▼────────────┐
        │   World Model (PWM)    │  trains incrementally each generation
        │   Predicts: AMP, Tox,  │  used for Dreamer planning
        │   Stability, Realism   │  curiosity = prediction error
        └───────────┬────────────┘
                    │
        ┌───────────▼────────────┐
        │  Novelty Archive       │  k-mer based, reservoir-sampled
        │  Motif Salience Memory │  attentional decay over triplets
        └────────────────────────┘
```

## Key Features

### Cognitive Agent
The `AgentController` maintains seven weighted motives that shift dynamically based on population state. If toxicity rises, safety weight increases. If the population converges, curiosity and novelty weights increase. A logistic chaos parameter `r` is self-tuned to push the system toward edge of chaos dynamics when stagnant, and pulled back when the world model becomes unreliable.

Periodic **meta-introspection** (every N generations) examines slow-timescale signals including plateau duration, novelty, diversity, motif entropy, and chaos rhythm, to reshape motive weights at a higher level of abstraction. The agent distinguishes between being stuck and redundant (explore more) versus stuck but already wandering (exploit more).

### Imagination Based Mutation Planning (Dreamer)
Before committing to a mutation, the agent rolls out candidate sequences in imagination using the internal world model, simulating multiple mutation steps ahead, scoring predicted outcomes by agent-weighted motives, and selecting the best imagined endpoint. This is analogous to model-based reinforcement learning applied to sequence design.

### Prophet Model
A learned neural model that predicts, given a parent peptide, which positions and which amino acids are most likely to yield improvement. It is trained on past high-fitness transitions and guides the mutation process with experience accumulated across runs.

### Multi-Model Ensemble with Disagreement Penalty
Internal CNN predictions are blended with external PyAMPA scores (AMP activity, toxicity, hemolysis, cell-penetrating peptide probability). When the two pipelines strongly disagree on a candidate, a **disagreement penalty** is applied to its fitness, conservatively flagging sequences the ensemble cannot confidently endorse.

### Attentional Motif Salience
A salience memory tracks tripeptide motifs weighted by their association with high-fitness outcomes, with exponential decay. This creates an attention-like mechanism where motifs that historically produce good peptides receive preferential treatment during mutation, while the system remains capable of exploring new motifs when those become stagnant.

### Biochemical Realism Constraints
Every candidate is evaluated against a smooth, multi-factor realism score covering amino acid diversity, hydrophobicity range, polar content, charge window, cysteine/proline counts, hydrophobic run length, and amphipathic plausibility via hydrophobic moment. Sequences below a generation-scaled minimum realism threshold are culled before they enter the breeding pool.

## Results

The framework was benchmarked against a standard genetic algorithm baseline (same scoring, no cognitive agent). Detailed results, figures, and statistical analysis are available in the preprint:

> Amit, A. (2025). *Multi-Objective Evolution of Antimicrobial Peptides Guided by Cognitive Principles and Realism Constraints.* SSRN.
> https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5313864

Example top candidates from a completed run are saved to `data/top_20_simulated_peptides.csv`.

## Installation

**Requirements:** Python 3.8+, TensorFlow/Keras, NumPy, Pandas

```bash
git clone https://github.com/arnava25/peptide-evolution.git
cd peptide-evolution
pip install -r requirements.txt
```

Pretrained CNN models (`.keras`) are **not included** due to file size. To train them from the included datasets:

```bash
python scripts/train_amp_model_v2.py
python scripts/train_toxicity_model_v2.py
python scripts/train_stability_model_v2.py
python scripts/train_naturalness_model.py
```

Training data is in `data/model_trainers/`. Running each script will generate the required `.keras` files in `models/`.

The Prophet mutation model can be trained after an initial run:

```bash
python scripts/train_prophet.py
```

## Running the Simulation

```bash
python scripts/evolve.py
```

Default settings (editable in `evolve.py`):

| Parameter | Default |
|---|---|
| Population size | 150 |
| Generations | 2000 |
| Peptide length | 25 |
| Cognitive agent | ON |

To run the standard GA baseline for comparison, set `USE_AGENT = False` in `evolve.py`, then run the same command.

To stop a long run without losing progress:

```bash
touch stop.txt
```

The simulation will finish the current generation, save all logs and data, and exit cleanly.

## Output Files

Each run generates a tagged set of output files in `data/`:

| File | Contents |
|---|---|
| `master_evolution_history.csv` | All peptides across all generations |
| `agent_state_*.csv` | Motive weights, chaos parameter, stagnation per generation |
| `agent_reasoning_*.csv` | Per-mutation reasoning: parent, child, source, motive scores |
| `action_log_*.csv` | Action type per generation (crossover / mutate / chaos) |
| `fitness_stats_*.csv` | Avg, max, std fitness per generation |
| `trait_stats/` | Charge, hydrophobicity, aggregation risk, realism drift |
| `novelty_stats_*.csv` | Population pairwise similarity over time |
| `entropy_stats_*.csv` | Sequence entropy per position over time |
| `world_model_error_*.csv` | Internal world model prediction error per generation |
| `similarity_logs/` | Per-generation population convergence |

Run analysis scripts:

```bash
python scripts/analyze_agent.py
python scripts/analyze.py
python scripts/extract.py
```

## Repository Structure

```
peptide-evolution/
├── scripts/
│   ├── evolve.py
│   ├── evolveAGENT.py
│   ├── train_amp_model_v2.py
│   ├── train_toxicity_model_v2.py
│   ├── train_stability_model_v2.py
│   ├── train_naturalness_model.py
│   ├── train_prophet.py
│   ├── analyze_agent.py
│   ├── analyze.py
│   └── extract.py
├── models/
├── external/
│   ├── PyAMPA/
│   └── pyampa_integration.py
├── data/
│   ├── model_trainers/
│   └── top_20_simulated_peptides.csv
└── README.md
```

## Branches

**main** Stable release. Source code and datasets as described in the preprint.

**dev** Active development. Extended cognitive modules including attentional salience, surprise scoring, and Dreamer planning.

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{amit2025ampevol,
  author  = {Amit, Arnav},
  title   = {Multi-Objective Evolution of Antimicrobial Peptides Guided by Cognitive Principles and Realism Constraints},
  journal = {SSRN},
  year    = {2025},
  url     = {https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5313864}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

Questions, collaborations, or feedback welcome. Open an issue or reach out via GitHub.