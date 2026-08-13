"""
Script for generating stylistic alignment heatmaps between model-conditioned
predicted move transitions and ground-truth human player board transitions.
Leverages a unified GlobalStyleSpace (AutoEncoder + cuML UMAP + Discrete Grid JSD)[cite: 1, 2].
"""

import gc
import logging
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import torch
import tqdm

from models.style import GlobalStyleSpace, evaluate_style_with_space
from utils.data import createPlayerDict, getPlayers
from utils.transitions import get_transition_vector

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("figures")
DATA_DIR = Path("data")

PREDICTION_FILES = {
    "Maia-2 Baseline": DATA_DIR / "maia2_predictions.csv",
    "Maia-2 FT": DATA_DIR / "maia2_ft_predictions.csv",
    "Maia-2 Nucleus": DATA_DIR / "maia2_nucleus_predictions.csv",
    "Maia-2 Descent": DATA_DIR / "maia2_descent_50_predictions.csv",
    "Maia-2 N. + Descent": DATA_DIR / "maia2_nucleus_descent_50_predictions.csv",
    "Maia-2 MCTS": DATA_DIR / "maia2_mcts_50_predictions.csv",
    "Maia-2 N. + MCTS": DATA_DIR / "maia2_nucleus_mcts_50_predictions.csv",
    "Maia-2 FT + N. + Descent": DATA_DIR
    / "maia2_ft_nucleus_descent_50_predictions.csv",
    "Maia-2 FT + N. + MCTS": DATA_DIR / "maia2_ft_nucleus_mcts_50_predictions.csv",
    "Maia-2 MoE-LoRA": DATA_DIR / "maia2_moe_lora_predictions.csv",
    "Maia-2 MoE-LoRA N. + Descent": (
        DATA_DIR / "maia2_moe_lora_nucleus_descent_50_predictions.csv"
    ),
    "Maia-2 MoE-LoRA N. + MCTS": (
        DATA_DIR / "maia2_moe_lora_nucleus_mcts_50_predictions.csv"
    ),
}


def set_seed(seed: int = 42) -> None:
    """
    Enforces global determinism across Python, NumPy, PyTorch, and CUDA runtime.
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
    positions_file: Path,
    players_dict: dict[str, int],
    device: torch.device,
    memmap_path: str = "data/reference_transitions.dat",
    seed: int = 42,
) -> GlobalStyleSpace:
    """
    Constructs the global invariant reference space (AutoEncoder + cuML UMAP)[cite: 1, 2]
    from ground-truth human transitions streamed directly to a memory-mapped file.
    """
    if Path(memmap_path).exists():
        logger.info(
            f"Existing reference binary found at {memmap_path}. Loading memory-mapped space..."
        )
        # Count rows based on existing file size: file_size_bytes / (2304 float32 values * 4 bytes)
        file_bytes = os.path.getsize(memmap_path)
        written_count = file_bytes // (2304 * 4)
    else:
        logger.info("Scanning position dataset to compute memory allocation size...")
        lazy_pos = pl.scan_csv(positions_file)
        valid_indices = set(players_dict.values())

        total_positions = (
            lazy_pos.filter(pl.col("player_index").is_in(valid_indices))
            .select(pl.len())
            .collect()
            .item()
        )
        logger.info(f"Total candidate reference positions: {total_positions}")

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
            f"Flushed {written_count} valid reference transition vectors to {memmap_path}."
        )

    ref_memmap = np.memmap(
        memmap_path, dtype="float32", mode="r", shape=(written_count, 2304)
    )

    global_space = GlobalStyleSpace.fit_from_memmap(
        ref_memmap, ae_epochs=10, device=device, seed=seed
    )

    del ref_memmap
    gc.collect()
    torch.cuda.empty_cache()

    return global_space


def extract_player_transition_matrices(
    prediction_file: Path,
    target_player_index: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Extracts transition vector arrays for a model conditioned on target_player_index
    evaluated against every real target player dataset[cite: 1, 2].
    """
    lazy_df = pl.scan_csv(prediction_file)
    player_data = (
        lazy_df.filter(pl.col("player_index") == target_player_index)
        .select(["fen", "move", "predicted_move"])
        .collect()
    )

    if len(player_data) == 0:
        return {}

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

    del player_data
    gc.collect()

    if not p_vecs or not m_vecs:
        return {}

    return {
        target_player_index: (
            np.array(p_vecs, dtype=np.float32),
            np.array(m_vecs, dtype=np.float32),
        )
    }


def build_model_vs_player_style_matrix(
    model_predictions_file: Path,
    global_space: GlobalStyleSpace,
    players_dict: dict[str, int],
) -> tuple[np.ndarray, list[str]]:
    """
    Computes the NxN Stylistic JSD divergence matrix comparing model outputs[cite: 1, 2]
    conditioned on Player i against real ground-truth transitions of Player j[cite: 1, 2].
    """
    player_names = list(players_dict.keys())
    n_players = len(player_names)
    matrix = np.zeros((n_players, n_players), dtype=np.float64)

    lazy_df = pl.scan_csv(model_predictions_file)

    pbar = tqdm.tqdm(
        enumerate(players_dict.items()),
        total=n_players,
        desc="Processing Model vs. Player Matrix",
        unit="model_cond",
        leave=False,
    )

    for i, (model_player_name, model_p_idx) in pbar:
        pbar.set_postfix({"conditioned_on": model_player_name})

        # Load predicted moves by the model conditioned on player_i
        model_data = (
            lazy_df.filter(pl.col("player_index") == model_p_idx)
            .select(["fen", "predicted_move"])
            .collect()
        )
        model_pred_map = {
            row["fen"]: row["predicted_move"]
            for row in model_data.iter_rows(named=True)
            if row["predicted_move"]
        }
        del model_data
        gc.collect()

        for j, (real_player_name, real_p_idx) in enumerate(players_dict.items()):
            real_data = (
                lazy_df.filter(pl.col("player_index") == real_p_idx)
                .select(["fen", "move"])
                .collect()
            )

            p_vecs = []
            m_vecs = []

            for row in real_data.iter_rows(named=True):
                fen = row["fen"]
                real_move = row["move"]

                if fen in model_pred_map:
                    model_move = model_pred_map[fen]
                    p_vec = get_transition_vector(fen, real_move)
                    m_vec = get_transition_vector(fen, model_move)

                    if p_vec is not None and m_vec is not None:
                        p_vecs.append(p_vec)
                        m_vecs.append(m_vec)

            del real_data
            gc.collect()

            if not p_vecs or not m_vecs:
                matrix[i, j] = np.nan
                continue

            p_arr = np.array(p_vecs, dtype=np.float32)
            m_arr = np.array(m_vecs, dtype=np.float32)

            metrics = evaluate_style_with_space(global_space, p_arr, m_arr)
            matrix[i, j] = metrics["style_jsd"]

            del p_vecs, m_vecs, p_arr, m_arr
            gc.collect()
            torch.cuda.empty_cache()

    return matrix, player_names


def plot_jsd_heatmap(
    jsd_matrix: np.ndarray,
    player_names: list[str],
    output_path: Path,
    title: str,
    y_label: str = "Model Conditioned Player",
    is_inter_player: bool = False,
) -> None:
    """
    Generates and saves publication-quality stylistic alignment heatmaps.
    """
    plt.figure(figsize=(11, 8.5))

    y_labels = (
        player_names if is_inter_player else [f"Model ({p})" for p in player_names]
    )
    cmap = "YlGnBu" if is_inter_player else "viridis_r"

    sns.heatmap(
        jsd_matrix,
        xticklabels=player_names,
        yticklabels=y_labels,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        cbar_kws={"label": "Jensen-Shannon Divergence"},
        linewidths=0.5,
    )

    plt.title(title, fontsize=12, pad=12, fontweight="bold")
    plt.xlabel("Real Target Player", fontsize=10, fontweight="bold")
    plt.ylabel(y_label, fontsize=10, fontweight="bold")

    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Heatmap successfully saved to: {output_path}")


if __name__ == "__main__":
    GLOBAL_SEED = 42
    set_seed(GLOBAL_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata_file = DATA_DIR / "metadata.csv"
    positions_file = DATA_DIR / "positions.csv"

    if not metadata_file.exists() or not positions_file.exists():
        logger.error(
            "Required dataset files (metadata.csv or positions.csv) were not found."
        )
        exit(1)

    players_df = getPlayers(str(metadata_file))
    players_dict = createPlayerDict(players_df)

    # 1. Construct the invariant global style space (AutoEncoder + cuML UMAP)[cite: 1, 2]
    global_space = build_global_reference_space(
        positions_file, players_dict, device, seed=GLOBAL_SEED
    )

    # 2. Iterate through all model prediction CSV files and generate stylistic heatmaps
    models_pbar = tqdm.tqdm(
        PREDICTION_FILES.items(),
        desc="Generating Stylistic Alignment Heatmaps",
        unit="model",
    )

    for model_name, path in models_pbar:
        if not path.exists():
            logger.warning(
                f"Prediction file missing for model '{model_name}': {path}. Skipping."
            )
            continue

        models_pbar.set_postfix({"current_model": model_name})
        logger.info(f"Generating stylistic heatmap matrix for model: {model_name}")

        model_matrix, player_names = build_model_vs_player_style_matrix(
            path, global_space, players_dict
        )

        clean_name = (
            model_name.lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace(".", "")
            .replace("+", "")
        )
        out_file = OUTPUT_DIR / f"{clean_name}_style_vs_player_heatmap.pdf"

        plot_jsd_heatmap(
            model_matrix,
            player_names,
            out_file,
            title=f"Stylistic Alignment Matrix: {model_name} vs Real Players",
            is_inter_player=False,
        )

        del model_matrix
        gc.collect()
        torch.cuda.empty_cache()
