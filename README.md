# AMPEvol 🧬

A lightweight evolutionary simulation engine that designs and optimizes antimicrobial peptides (AMPs) based on pretrained models for antimicrobial activity, toxicity, and stability.

## 🚀 Features
- AMP/Toxicity/Stability predictors (Keras models)
- Smart mutation with biochemical constraints
- Fitness-based evolution with diversity encouragement
- Logging, visualization, and batch evaluation tools

## 🗂️ Folder Structure

- `scripts/`: Main code (evolution, training, eval, postprocessing)
- `data/`: Peptide logs, training datasets
- `models/`: Pretrained `.keras` models

## 📦 To Run

```bash
python3 scripts/evolve.py
```


🧪 Datasets Used

(You must download these separately)
	-	DRAMP
	-	APD3
	-	DBAASP
	-	ToxinPred

## License
MIT
