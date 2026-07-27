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
    """Calcule la divergence de Kullback-Leibler KL(P || Q)."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def compute_jensen_shannon_divergence(
    p_dict: dict[str, float], q_dict: dict[str, float]
) -> float:
    """Calcule la divergence de Jensen-Shannon entre deux distributions de probabilités."""
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

    jsd = 0.5 * kl_p_m + 0.5 * kl_q_m
    return max(0.0, jsd)


def evaluate_model_jsd(model_name: str, csv_path: str, players_dict: dict[str, int]):
    """Calcule la JSD moyenne par joueur en mode Lazy Polars."""
    path = Path(csv_path)
    if not path.exists():
        logger.warning(
            f"Fichier de prédictions introuvable : {csv_path}, passage au suivant."
        )
        return

    logger.info(f"Évaluation JSD (Lazy Mode) pour le modèle : {model_name}")
    lazy_df = pl.scan_csv(csv_path)
    results = []

    for player_name, player_index in tqdm.tqdm(players_dict.items(), desc=model_name):
        player_data = (
            lazy_df.filter(pl.col("player_index") == player_index)
            .select(["fen", "move", "moves_probs"])
            .collect()
        )

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
            counts = Counter(moves)
            total_fen_moves = len(moves)

            p_data = {move: cnt / total_fen_moves for move, cnt in counts.items()}
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
        out_file = f"data/{clean_name}_jsd.csv"
        output_df.write_csv(out_file)
        logger.info(f"Résultats JSD enregistrés dans : {out_file}")


if __name__ == "__main__":
    players = getPlayers("data/metadata.csv")
    players_dict = createPlayerDict(players)

    for model_name, path_str in PREDICTION_FILES.items():
        evaluate_model_jsd(model_name, path_str, players_dict)
