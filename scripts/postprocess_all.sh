#!/bin/bash

echo "🧠 Running: Summarize Evolution Run..."
python3 scripts/postprocess_summary.py

echo "🔬 Running: Cluster Peptides..."
python3 scripts/postprocess_cluster.py

echo "✅ Postprocessing complete!"
