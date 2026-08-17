# ChessBehaviors

ChessBehaviors is a research codebase for modeling and evaluating human chess decision-making with Maia-2 and several adaptation/search variants.

## What this repository does

The project supports an end-to-end workflow:

1. Build datasets from PGN files.
2. Generate model predictions for player-specific move choices.
3. Evaluate models with:
   - Move accuracy
   - Jensen-Shannon divergence (JSD) on move distributions
   - Style divergence in a learned style space
4. Export publication-ready tables and heatmaps.

## Repository layout

- `ChessBehaviors/build_dataset.py`  
  Builds `data/metadata.csv` and `data/positions.csv` from raw PGN files.
- `ChessBehaviors/ground_truth.py`  
  Computes empirical baseline predictions from observed moves.
- `ChessBehaviors/maia2_*.py`  
  Runs Maia-2 model variants (baseline, fine-tuned, nucleus-pruned, search-augmented, MoE-LoRA).
- `ChessBehaviors/evaluate_*.py`  
  Computes accuracy/JSD/style metrics from prediction files.
- `ChessBehaviors/generate_*_heatmaps.py`  
  Produces heatmap figures.
- `ChessBehaviors/tab_*.py`  
  Generates LaTeX tables from evaluation outputs.
- `ChessBehaviors/models`, `ChessBehaviors/search`, `ChessBehaviors/utils`  
  Core model wrappers, search algorithms, and shared utilities.

## Requirements

- Python `>=3.13,<3.14`
- Dependencies are defined in `ChessBehaviors/pyproject.toml`.
- GPU-oriented packages are configured through uv package indexes (NVIDIA + PyTorch CUDA wheels).

## Setup

Using `uv`:

```bash
uv sync
```

## Expected data structure

The dataset builder expects PGN files under:

```text
data/
  raw/
    <user_id>/
      <game_id>.pgn
```

## Typical workflow

From `ChessBehaviors`:

```bash
# 1) Build metadata + position dataset
python build_dataset.py

# 2) Run one or more prediction pipelines
python ground_truth.py
python maia2_baseline.py
python maia2_ft.py
python maia2_nucleus.py
python maia2_descent.py
python maia2_mcts.py
python maia2_moe_lora.py

# 3) Evaluate model outputs
python evaluate_cp_error.py
python evaluate_jsd.py
python evaluate_style.py

# 4) Generate tables and figures
python tab_move_accuracy.py
python tab_cp_error.py
python tab_jsd.py
python tab_style.py
python generate_jsd_heatmaps.py
python generate_style_heatmaps.py
```

## Outputs

Most scripts write results to:

- `data/` for prediction and metric CSV files
- `tables/` for LaTeX tables
- `figures/` for exported heatmaps

## Notes

- Scripts use default file paths in `data/` unless modified in code.
- Some pipelines are computationally heavy and intended for GPU execution.
