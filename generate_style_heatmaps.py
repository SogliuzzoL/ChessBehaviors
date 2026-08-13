"""
Main evaluation pipeline for constructing stylistic alignment heatmaps between
model-conditioned move transitions and ground-truth human player board state transitions.
Leverages a unified GlobalStyleSpace manifold (Autoencoder + cuML UMAP + Discrete Grid JSD).
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

OUTPUT_DIR: Path = Path("figures")
DATA_DIR: Path = Path("data")

PREDICTION_FILES: dict[str, Path] = {
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
    """Enforce global determinism across Python, NumPy, PyTorch, and CUDA execution contexts.

    Args:
        seed (int, optional): Integer seed value for pseudo-random number generators. Defaults to 42.
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
    memmap_path: Path = DATA_DIR / "reference_transitions.dat",
    seed: int = 42,
) -> GlobalStyleSpace:
    """Construct the global invariant reference space manifold (Autoencoder + cuML UMAP) from
    ground-truth human transition representations streamed directly to disk-backed memory-mapped storage.

    Args:
        positions_file (Path): Path leading to raw position dataset records.
        players_dict (Dict[str, int]): Subject mapping linking cohort labels to integer indices.
        device (torch.device): Compute target device (CPU or CUDA GPU) for model execution.
        memmap_path (Path, optional): Target binary storage path for transition vector matrices.
            Defaults to DATA_DIR / "reference_transitions.dat".
        seed (int, optional): Random seed parameter for manifold dimensionality reduction. Defaults to 42.

    Returns:
        GlobalStyleSpace: Instance of the fitted reference manifold pipeline.
    """
    memmap_path = Path(memmap_path)
    if memmap_path.exists():
        logger.info(
            "Existing reference binary detected at target path: %s. Loading memory-mapped space.",
            memmap_path,
        )
        file_bytes = os.path.getsize(memmap_path)
        written_count = file_bytes // (2304 * 4)
    else:
        logger.info(
            "Scanning position dataset to calculate binary disk storage allocation bounds..."
        )
        lazy_pos = pl.scan_csv(positions_file)
        valid_indices = set(players_dict.values())

        total_positions = (
            lazy_pos.filter(pl.col("player_index").is_in(valid_indices))
            .select(pl.len())
            .collect()
            .item()
        )
        logger.info(
            "Total candidate reference positions identified: %d", total_positions
        )

        os.makedirs(memmap_path.parent, exist_ok=True)
        fp = np.memmap(
            memmap_path, dtype="float32", mode="w+", shape=(total_positions, 2304)
        )

        written_count = 0
        player_pbar = tqdm.tqdm(
            players_dict.items(),
            desc="Streaming Reference Transitions to Disk Storage",
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
            "Flushed %d valid reference transition vectors to storage location: %s",
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
    torch.cuda.empty_cache()

    return global_space


def build_model_vs_player_style_matrix(
    model_predictions_file: Path,
    global_space: GlobalStyleSpace,
    players_dict: dict[str, int],
) -> tuple[np.ndarray, list[str]]:
    """Compute the square stylistic JSD divergence matrix comparing model predictions conditioned
    on subject profile i against ground-truth empirical transitions of subject profile j.

    Args:
        model_predictions_file (Path): File path pointing to evaluation prediction data.
        global_space (GlobalStyleSpace): Pre-fitted global style reference manifold.
        players_dict (Dict[str, int]): Subject mapping dictionary linking cohort names to indices.

    Returns:
        Tuple[np.ndarray, List[str]]: Asymmetric matrix of pairwise stylistic JSD divergence values
            and list of ordered subject cohort labels.
    """
    player_names = list(players_dict.keys())
    n_players = len(player_names)
    matrix = np.zeros((n_players, n_players), dtype=np.float64)

    lazy_df = pl.scan_csv(model_predictions_file)

    pbar = tqdm.tqdm(
        enumerate(players_dict.items()),
        total=n_players,
        desc="Computing Model vs. Player Matrix",
        unit="model_cond",
        leave=False,
    )

    for i, (model_player_name, model_p_idx) in pbar:
        pbar.set_postfix({"conditioned_on": model_player_name})

        # Filter candidate predictions generated by model conditioned on profile i
        model_data = (
            lazy_df.filter(pl.col("player_index") == model_p_idx)
            .select(["fen", "predicted_move"])
            .collect()
        )

        # Build map with strict non-null move validation
        model_pred_map = {
            row["fen"]: row["predicted_move"]
            for row in model_data.iter_rows(named=True)
            if row["predicted_move"] and isinstance(row["predicted_move"], str)
        }
        del model_data
        gc.collect()

        for j, (real_player_name, real_p_idx) in enumerate(players_dict.items()):
            real_data = (
                lazy_df.filter(pl.col("player_index") == real_p_idx)
                .select(["fen", "move"])
                .collect()
            )

            p_vecs: list[np.ndarray] = []
            m_vecs: list[np.ndarray] = []

            for row in real_data.iter_rows(named=True):
                fen = row["fen"]
                real_move = row["move"]

                if fen in model_pred_map:
                    model_move = model_pred_map[fen]
                    if not real_move or not model_move:
                        continue

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
    """Generate and export publication-ready stylistic alignment heatmap visualizations.

    Args:
        jsd_matrix (np.ndarray): 2D square array containing calculated pairwise JSD divergence metrics.
        player_names (List[str]): Ordered subject cohort names for axis labelling.
        output_path (Path): Destination file system path for exported figure format.
        title (str): Descriptive plot title.
        y_label (str, optional): Vertical axis label string. Defaults to "Model Conditioned Player".
        is_inter_player (bool, optional): Configuration flag switching color map schemes for
            inter-human baseline comparisons. Defaults to False.
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
    logger.info(
        "Heatmap visualization successfully saved to destination path: %s", output_path
    )


if __name__ == "__main__":
    GLOBAL_SEED: int = 42
    set_seed(GLOBAL_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata_file = DATA_DIR / "metadata.csv"
    positions_file = DATA_DIR / "positions.csv"

    if not metadata_file.exists() or not positions_file.exists():
        logger.error(
            "Required dataset dependency files (metadata.csv or positions.csv) were not found."
        )
        exit(1)

    players_df = getPlayers(str(metadata_file))
    players_dict = createPlayerDict(players_df)

    # 1. Construct the invariant global reference manifold space (Autoencoder + cuML UMAP)
    global_space = build_global_reference_space(
        positions_file, players_dict, device, seed=GLOBAL_SEED
    )

    # 2. Iterate through all model prediction evaluation files and compute stylistic heatmaps
    models_pbar = tqdm.tqdm(
        PREDICTION_FILES.items(),
        desc="Generating Stylistic Alignment Heatmaps",
        unit="model",
    )

    for model_name, path in models_pbar:
        if not path.exists():
            logger.warning(
                "Prediction record missing for candidate architecture '%s': %s. Skipping evaluation.",
                model_name,
                path,
            )
            continue

        models_pbar.set_postfix({"current_model": model_name})
        logger.info(
            "Generating stylistic heatmap matrix for candidate model: %s", model_name
        )

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
