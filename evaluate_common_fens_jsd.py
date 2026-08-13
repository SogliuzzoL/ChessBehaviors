import ast
import gc
import logging
import math
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import tqdm

from utils.data import createPlayerDict, getPlayers

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PREDICTION_FILES = {
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


def compute_kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    r"""Compute the Kullback-Leibler (KL) divergence between two discrete probability distributions.

    Calculates $D_{KL}(P \parallel Q)$ using a base-2 logarithm to express divergence in bits.
    Numerical stability is ensured by clipping input distributions with an epsilon parameter.
    """
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log2(p / q)))


def compute_jensen_shannon_divergence(
    p_dict: dict[str, float], q_dict: dict[str, float]
) -> float:
    """Compute the symmetric Jensen-Shannon Divergence (JSD) between two categorical move probability distributions.

    Evaluates $JSD(P \\parallel Q) = \\frac{1}{2} D_{KL}(P \\parallel M) + \\frac{1}{2} D_{KL}(Q \\parallel M)$,
    where $M = \\frac{1}{2}(P + Q)$ represents the mixture distribution across the union of support moves.
    """
    all_moves = list(set(p_dict.keys()).union(set(q_dict.keys())))
    if not all_moves:
        return 0.0

    p_vals = np.array([p_dict.get(m, 0.0) for m in all_moves], dtype=np.float64)
    q_vals = np.array([q_dict.get(m, 0.0) for m in all_moves], dtype=np.float64)

    sum_p = np.sum(p_vals)
    sum_q = np.sum(q_vals)

    if sum_p > 0:
        p_vals /= sum_p
    if sum_q > 0:
        q_vals /= sum_q

    m_vals = 0.5 * (p_vals + q_vals)

    kl_p_m = compute_kl_divergence(p_vals, m_vals)
    kl_q_m = compute_kl_divergence(q_vals, m_vals)

    return max(0.0, 0.5 * kl_p_m + 0.5 * kl_q_m)


def evaluate_model_jsd(model_name: str, csv_path: str, players_dict: dict[str, int]):
    """Evaluate model predictive performance via Jensen-Shannon Divergence across common Forsyth-Edwards Notation (FEN) states.

    Performs a comparative evaluation of model move probability distributions against empirical move frequency
    distributions. The analysis is strictly constrained to board states (FENs) that are mutually observed across all
    subjects in the evaluation cohort.
    """
    path = Path(csv_path)
    if not path.exists():
        logger.warning(
            f"Prediction record missing at target path: {csv_path}. Skipping evaluation for model: {model_name}."
        )
        return

    logger.info(
        f"Executing JSD evaluation pipeline (lazy computation paradigm) for candidate model: {model_name}"
    )

    valid_player_indices = set(players_dict.values())
    lazy_df = pl.scan_csv(csv_path).filter(
        pl.col("player_index").is_in(valid_player_indices)
    )

    # 1. Identification of board positions (FENs) shared across all cohort subjects
    logger.info(
        "Identifying common Forsyth-Edwards Notation (FEN) positions across all player cohorts..."
    )
    common_fens_df = (
        lazy_df.select(["player_index", "fen"])
        .unique()
        .group_by("fen")
        .agg(pl.count("player_index").alias("player_count"))
        .filter(pl.col("player_count") == len(valid_player_indices))
        .collect()
    )

    common_fens = set(common_fens_df["fen"].to_list())
    logger.info(
        f"Identified {len(common_fens)} mutually shared board positions (FENs)."
    )

    if not common_fens:
        logger.warning(
            "Intersection query yielded zero common FEN positions across subjects. Terminating model evaluation."
        )
        return

    # 2. Global filtering on common board positions (FENs)
    filtered_lazy_df = lazy_df.filter(pl.col("fen").is_in(common_fens)).select(
        ["player_index", "fen", "move", "moves_probs"]
    )

    results = []

    for player_name, player_index in tqdm.tqdm(players_dict.items(), desc=model_name):
        player_data = filtered_lazy_df.filter(
            pl.col("player_index") == player_index
        ).collect()

        if len(player_data) == 0:
            del player_data
            gc.collect()
            continue

        fen_groups: dict[str, list[str]] = {}
        fen_model_probs: dict[str, dict[str, float]] = {}

        for row in player_data.iter_rows(named=True):
            fen = row["fen"]
            target_move = row["move"]

            if fen not in fen_groups:
                fen_groups[fen] = []
                raw_probs = row["moves_probs"]
                if isinstance(raw_probs, str):
                    try:
                        fen_model_probs[fen] = ast.literal_eval(raw_probs)
                    except Exception:
                        fen_model_probs[fen] = {}
                elif isinstance(raw_probs, dict):
                    fen_model_probs[fen] = raw_probs
                else:
                    fen_model_probs[fen] = {}

            fen_groups[fen].append(target_move)

        jsd_values = []
        for fen, moves in fen_groups.items():
            total_fen_moves = len(moves)
            p_data = {m: cnt / total_fen_moves for m, cnt in Counter(moves).items()}
            p_model = fen_model_probs.get(fen, {})

            jsd = compute_jensen_shannon_divergence(p_data, p_model)
            jsd_values.append(jsd)

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

        del player_data
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
        out_file = f"data/{clean_name}_common_fens_jsd.csv"
        output_df.write_csv(out_file)
        logger.info(f"Evaluation metrics successfully exported to: {out_file}")


if __name__ == "__main__":
    players = getPlayers("data/metadata.csv")
    players_dict = createPlayerDict(players)

    for model_name, path_str in PREDICTION_FILES.items():
        evaluate_model_jsd(model_name, path_str, players_dict)
