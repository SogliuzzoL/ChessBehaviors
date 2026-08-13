import gc
import logging
from pathlib import Path

import numpy as np
import polars as pl
import torch
import tqdm

from models.style import evaluate_style_pipeline
from utils.data import createPlayerDict, getPlayers
from utils.transitions import get_transition_vector

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


def evaluate_model_style(model_name: str, csv_path: str, players_dict: dict[str, int]):
    path = Path(csv_path)
    if not path.exists():
        logger.warning(
            f"Fichier de prédictions introuvable : {csv_path}, passage au suivant."
        )
        return

    logger.info(f"Évaluation du style (AE + cuML UMAP + JSD) : {model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lazy_df = pl.scan_csv(csv_path)

    results = []
    for player_name, player_index in tqdm.tqdm(players_dict.items(), desc=model_name):
        player_data = (
            lazy_df.filter(pl.col("player_index") == player_index)
            .select(["fen", "move", "predicted_move"])
            .collect()
        )

        if len(player_data) == 0:
            del player_data
            gc.collect()
            continue

        fens = player_data["fen"].to_list()
        player_moves = player_data["move"].to_list()
        model_moves = player_data["predicted_move"].to_list()

        p_vecs = []
        m_vecs = []

        for fen, p_move, m_move in zip(fens, player_moves, model_moves):
            p_vec = get_transition_vector(fen, p_move)
            m_vec = get_transition_vector(fen, m_move)

            if p_vec is not None and m_vec is not None:
                p_vecs.append(p_vec)
                m_vecs.append(m_vec)

        if not p_vecs or not m_vecs:
            del player_data, fens, player_moves, model_moves
            gc.collect()
            continue

        p_arr = np.array(p_vecs, dtype=np.float32)
        m_arr = np.array(m_vecs, dtype=np.float32)

        # Calcul des métriques de style via le pipeline
        metrics = evaluate_style_pipeline(p_arr, m_arr, device=device)

        results.append(
            {
                "player_index": player_index,
                "player_name": player_name,
                "num_positions": len(p_vecs),
                "style_jsd": metrics["style_jsd"],
                "style_jsd_distance": metrics["style_jsd_distance"],
            }
        )

        del player_data, fens, player_moves, model_moves, p_vecs, m_vecs, p_arr, m_arr
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
        out_file = f"data/{clean_name}_style.csv"
        output_df.write_csv(out_file)
        logger.info(f"Résultats de style enregistrés dans : {out_file}")


if __name__ == "__main__":
    players = getPlayers("data/metadata.csv")
    players_dict = createPlayerDict(players)

    for model_name, path_str in PREDICTION_FILES.items():
        evaluate_model_style(model_name, path_str, players_dict)
