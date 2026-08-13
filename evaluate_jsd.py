import ast
import gc
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import tqdm

from utils.data import createPlayerDict, getPlayers

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PREDICTION_FILES: dict[str, str] = {
    "Maia-2 Baseline": "data/maia2_predictions.csv",
    "Maia-2 FT": "data/maia2_ft_predictions.csv",
    "Maia-2 Nucleus": "data/maia2_nucleus_predictions.csv",
    "Maia-2 Descent": "data/maia2_descent_50_predictions.csv",
    "Maia-2 N. + Descent": "data/maia2_nucleus_descent_50_predictions.csv",
    "Maia-2 MCTS": "data/maia2_mcts_50_predictions.csv",
    "Maia-2 N. + MCTS": "data/maia2_nucleus_mcts_50_predictions.csv",
    "Maia-2 FT + N. + Descent": "data/maia2_ft_nucleus_descent_50_predictions.csv",
    "Maia-2 FT + N. + MCTS": "data/maia2_ft_nucleus_mcts_50_predictions.csv",
    "Maia-2 MoE-LoRA": "data/maia2_moe_lora_predictions.csv",
    "Maia-2 MoE-LoRA N. + Descent": (
        "data/maia2_moe_lora_nucleus_descent_50_predictions.csv"
    ),
    "Maia-2 MoE-LoRA N. + MCTS": (
        "data/maia2_moe_lora_nucleus_mcts_50_predictions.csv"
    ),
}


def compute_jensen_shannon_divergence_fast(
    p_dict: dict[str, float], q_dict: dict[str, float], eps: float = 1e-12
) -> float:
    """Compute vectorized Jensen-Shannon Divergence (JSD) using NumPy optimizations.

    Calculates the symmetric divergence between empirical action distribution $P$ and
    predicted model distribution $Q$ over the union of supported actions.

    Args:
        p_dict (Dict[str, float]): Empirical target move probability distribution.
        q_dict (Dict[str, float]): Model predicted move probability distribution.
        eps (float, optional): Epsilon threshold to prevent numerical instability during
            logarithmic transformations. Defaults to 1e-12.

    Returns:
        float: Computed Jensen-Shannon Divergence metric constrained to $[0, 1]$.
    """
    all_moves = list(set(p_dict.keys()).union(set(q_dict.keys())))
    if not all_moves:
        return 0.0

    p_vals = np.array([p_dict.get(m, 0.0) for m in all_moves], dtype=np.float64)
    q_vals = np.array([q_dict.get(m, 0.0) for m in all_moves], dtype=np.float64)

    sum_p, sum_q = np.sum(p_vals), np.sum(q_vals)
    if sum_p > 0:
        p_vals /= sum_p
    if sum_q > 0:
        q_vals /= sum_q

    m_vals = 0.5 * (p_vals + q_vals)

    p_clipped = np.clip(p_vals, eps, 1.0)
    q_clipped = np.clip(q_vals, eps, 1.0)
    m_clipped = np.clip(m_vals, eps, 1.0)

    kl_p_m = float(np.sum(p_vals * np.log2(p_clipped / m_clipped)))
    kl_q_m = float(np.sum(q_vals * np.log2(q_clipped / m_clipped)))

    return max(0.0, 0.5 * kl_p_m + 0.5 * kl_q_m)


def parse_probs(raw_probs: str | dict[str, float] | Any) -> dict[str, float]:
    """Parse raw probability serialized objects into structured Python dictionaries.

    Args:
        raw_probs (Union[str, Dict[str, float], Any]): Raw representation of move probabilities,
            either string-serialized or native dictionary format.

    Returns:
        Dict[str, float]: Parsed dictionary containing candidate moves and associated probabilities.
    """
    if isinstance(raw_probs, str):
        try:
            return ast.literal_eval(raw_probs)
        except Exception:
            return {}
    elif isinstance(raw_probs, dict):
        return raw_probs
    return {}


def evaluate_model_jsd(
    model_name: str, csv_path: str, players_dict: dict[str, int]
) -> None:
    """Evaluate Jensen-Shannon Divergence for candidate model predictions per subject cohort.

    Aggregates empirical board position frequencies, extracts corresponding model probability
    distributions, and computes mean divergence across unique state representations.

    Args:
        model_name (str): Identifier for candidate evaluation architecture.
        csv_path (str): File system path leading to target prediction records.
        players_dict (Dict[str, int]): Mapping of subject names to index identifiers.
    """
    path = Path(csv_path)
    if not path.exists():
        logger.warning(
            f"Prediction record file not found: {csv_path}. Skipping model evaluation for: {model_name}."
        )
        return

    logger.info(
        f"Initiating optimized JSD evaluation pipeline for candidate model: {model_name}"
    )

    lazy_df = pl.scan_csv(csv_path)
    results: list[dict[str, Any]] = []

    for player_name, player_index in tqdm.tqdm(players_dict.items(), desc=model_name):
        # 1. Native Polars aggregation to compute empirical move frequencies per board state (FEN)
        player_counts_df = (
            lazy_df.filter(pl.col("player_index") == player_index)
            .group_by(["fen", "move"])
            .agg(pl.len().alias("count"))
            .collect()
        )

        if len(player_counts_df) == 0:
            continue

        # Reconstruct empirical move distributions P(data) efficiently
        fen_real_probs: dict[str, dict[str, float]] = {}
        fen_totals: dict[str, int] = {}

        for row in player_counts_df.iter_rows(named=True):
            fen, move, cnt = row["fen"], row["move"], row["count"]
            fen_totals[fen] = fen_totals.get(fen, 0) + cnt
            if fen not in fen_real_probs:
                fen_real_probs[fen] = {}
            fen_real_probs[fen][move] = cnt

        # Relative probability normalization
        for fen, moves in fen_real_probs.items():
            tot = fen_totals[fen]
            for m in moves:
                moves[m] /= tot

        # 2. Query distinct predicted move probability distributions Q(model) per board state
        model_probs_df = (
            lazy_df.filter(pl.col("player_index") == player_index)
            .select(["fen", "moves_probs"])
            .unique(subset=["fen"])
            .collect()
        )

        fen_model_probs = {
            row["fen"]: parse_probs(row["moves_probs"])
            for row in model_probs_df.iter_rows(named=True)
        }

        # 3. Vectorized JSD computation across unique board positions
        jsd_values = [
            compute_jensen_shannon_divergence_fast(p_real, fen_model_probs.get(fen, {}))
            for fen, p_real in fen_real_probs.items()
        ]

        mean_jsd = sum(jsd_values) / len(jsd_values) if jsd_values else 0.0

        results.append(
            {
                "player_index": player_index,
                "player_name": player_name,
                "num_unique_positions": len(jsd_values),
                "mean_jsd": round(mean_jsd, 6),
                "jsd_distance": round(math.sqrt(mean_jsd), 6),
            }
        )

        del player_counts_df, model_probs_df, fen_real_probs, fen_model_probs
        gc.collect()

    if results:
        output_df = pl.DataFrame(results)
        clean_name = (
            model_name.lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace(".", "")
            .replace("+", "")
        )
        out_file = f"data/{clean_name}_jsd.csv"
        output_df.write_csv(out_file)
        logger.info(f"Evaluation metrics successfully exported to: {out_file}")


if __name__ == "__main__":
    players = getPlayers("data/metadata.csv")
    players_dict = createPlayerDict(players)

    for model_name, path_str in PREDICTION_FILES.items():
        evaluate_model_jsd(model_name, path_str, players_dict)
