"""
Main evaluation pipeline for quantifying stylistic divergence (Autoencoder + cuML UMAP + JSD)
across model variants and historical world chess champions utilizing memory-mapped disk storage.
"""

import gc
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import tqdm

from models.style import GlobalStyleSpace, evaluate_style_with_space
from utils.data import createPlayerDict, getPlayers
from utils.transitions import get_transition_vector

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


def set_seed(seed: int = 42) -> None:
    """Fix random seeds across all dependencies to ensure deterministic experiment reproducibility.

    Args:
        seed (int, optional): Random seed value. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_global_reference_space(
    positions_file: str,
    players_dict: dict[str, int],
    device: torch.device,
    memmap_path: str = "data/reference_transitions.dat",
    seed: int = 42,
) -> GlobalStyleSpace:
    """Construct a shared low-dimensional global style manifold from raw position datasets.

    Extracts high-dimensional transition vectors across all player cohorts, streams them
    to a disk-backed memory-mapped array, and fits an Autoencoder followed by a UMAP projection space.

    Args:
        positions_file (str): Path to CSV containing candidate position representations.
        players_dict (Dict[str, int]): Map of subject identifiers to numerical cohort indices.
        device (torch.device): PyTorch compute hardware target (CPU or CUDA GPU).
        memmap_path (str, optional): Target binary file path for memory-mapped storage.
            Defaults to "data/reference_transitions.dat".
        seed (int, optional): Random seed for transformation space fitting. Defaults to 42.

    Returns:
        GlobalStyleSpace: Fitted latent manifold pipeline for downstream style projections.
    """
    logger.info("Scanning position dataset to determine global cohort record volume...")
    lazy_pos = pl.scan_csv(positions_file)
    valid_indices = set(players_dict.values())

    total_positions = (
        lazy_pos.filter(pl.col("player_index").is_in(valid_indices))
        .select(pl.len())
        .collect()
        .item()
    )
    logger.info(
        "Identified %d reference positions for manifold construction.", total_positions
    )

    os.makedirs(os.path.dirname(memmap_path), exist_ok=True)
    fp = np.memmap(
        memmap_path, dtype="float32", mode="w+", shape=(total_positions, 2304)
    )

    written_count = 0
    player_pbar = tqdm.tqdm(
        players_dict.items(),
        desc="Streaming Reference Transitions to Disk",
        unit="player",
    )

    for player_name, player_index in player_pbar:
        player_pbar.set_postfix({"player": player_name})
        player_pos = (
            lazy_pos.filter(pl.col("player_index") == player_index)
            .select(["fen", "move"])
            .collect()
        )

        for row in player_pos.iter_rows(named=True):
            vec = get_transition_vector(row["fen"], row["move"])
            if vec is not None:
                fp[written_count] = vec
                written_count += 1

        del player_pos
        gc.collect()

    fp.flush()
    del fp
    gc.collect()

    logger.info(
        "Successfully persisted %d valid transition vectors to memory-mapped storage: %s",
        written_count,
        memmap_path,
    )

    ref_memmap = np.memmap(
        memmap_path, dtype="float32", mode="r", shape=(written_count, 2304)
    )

    global_space = GlobalStyleSpace.fit_from_memmap(
        ref_memmap, ae_epochs=10, device=device, seed=seed
    )

    del ref_memmap
    gc.collect()
    torch.cuda.empty_cache()  # Explicit GPU VRAM cleanup post-fitting procedure

    return global_space


def evaluate_model_style(
    model_name: str,
    csv_path: str,
    players_dict: dict[str, int],
    global_space: GlobalStyleSpace,
) -> None:
    """Evaluate stylistic divergence between human empirical moves and model predictions.

    Maps ground-truth human transitions and model-predicted transitions into the global
    style space to compute Jensen-Shannon Divergence across low-dimensional projections.

    Args:
        model_name (str): Identifier for candidate evaluation architecture.
        csv_path (str): File system path leading to model prediction records.
        players_dict (Dict[str, int]): Map of subject names to numerical cohort indices.
        global_space (GlobalStyleSpace): Fitted reference style manifold.
    """
    path = Path(csv_path)
    if not path.exists():
        logger.warning(
            "Prediction record missing at target path: %s. Skipping model evaluation.",
            csv_path,
        )
        return

    logger.info(
        "Initiating stylistic divergence evaluation pipeline for candidate model: %s",
        model_name,
    )
    lazy_df = pl.scan_csv(csv_path)

    results: list[dict[str, Any]] = []
    player_pbar = tqdm.tqdm(
        players_dict.items(),
        desc=f"Evaluating {model_name}",
        unit="player",
    )

    for player_name, player_index in player_pbar:
        player_pbar.set_postfix({"current_player": player_name})

        player_data = (
            lazy_df.filter(pl.col("player_index") == player_index)
            .select(["fen", "move", "predicted_move"])
            .collect()
        )

        if len(player_data) == 0:
            del player_data
            gc.collect()
            continue

        fens: list[str] = player_data["fen"].to_list()
        player_moves: list[str] = player_data["move"].to_list()
        model_moves: list[str] = player_data["predicted_move"].to_list()

        p_vecs: list[np.ndarray] = []
        m_vecs: list[np.ndarray] = []

        encoding_pbar = tqdm.tqdm(
            zip(fens, player_moves, model_moves),
            total=len(fens),
            desc=f"  -> Extracting transition representations [{player_name}]",
            leave=False,
            unit="pos",
        )

        for fen, p_move, m_move in encoding_pbar:
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

        metrics = evaluate_style_with_space(global_space, p_arr, m_arr)

        results.append(
            {
                "player_index": player_index,
                "player_name": player_name,
                "num_positions": len(p_vecs),
                "style_jsd": metrics["style_jsd"],
                "style_jsd_distance": metrics["style_jsd_distance"],
            }
        )

        player_pbar.set_postfix(
            {
                "latest_JSD": f"{metrics['style_jsd']:.4f}",
                "positions": len(p_vecs),
            }
        )

        del player_data, fens, player_moves, model_moves, p_vecs, m_vecs, p_arr, m_arr
        gc.collect()
        torch.cuda.empty_cache()  # Explicit GPU VRAM cleanup per subject iteration

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
        logger.info(
            "Stylistic divergence evaluation metrics successfully exported to: %s",
            out_file,
        )


if __name__ == "__main__":
    GLOBAL_SEED: int = 42
    set_seed(GLOBAL_SEED)

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    players = getPlayers("data/metadata.csv")
    players_dict = createPlayerDict(players)

    global_space = build_global_reference_space(
        "data/positions.csv", players_dict, device, seed=GLOBAL_SEED
    )

    models_pbar = tqdm.tqdm(
        PREDICTION_FILES.items(),
        desc="Overall Model Evaluation Pipeline Progress",
        unit="model",
    )

    for model_name, path_str in models_pbar:
        evaluate_model_style(model_name, path_str, players_dict, global_space)
