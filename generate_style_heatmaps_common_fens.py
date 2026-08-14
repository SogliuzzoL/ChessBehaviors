"""
Pipeline for constructing pairwise stylistic alignment heatmaps between
model-conditioned predictions and empirical subject transitions exclusively across Common FENs.
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
    """Fix random seeds for deterministic execution.

    Args:
        seed (int, optional): Integer seed value. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_common_fens(positions_file: Path, players_dict: dict[str, int]) -> set[str]:
    """Retrieve the common FEN intersection observed across all subjects.

    Args:
        positions_file (Path): Master positions CSV path.
        players_dict (Dict[str, int]): Subject mapping dictionary.

    Returns:
        Set[str]: Set of common FEN board strings.
    """
    lazy_pos = pl.scan_csv(positions_file)
    valid_indices = set(players_dict.values())

    df_counts = (
        lazy_pos.filter(pl.col("player_index").is_in(valid_indices))
        .select(["player_index", "fen"])
        .unique()
        .group_by("fen")
        .agg(pl.len().alias("p_count"))
        .filter(pl.col("p_count") == len(valid_indices))
        .collect()
    )
    return set(df_counts["fen"].to_list())


def build_global_reference_space_common_fens(
    positions_file: Path,
    common_fens: set[str],
    players_dict: dict[str, int],
    device: torch.device,
    memmap_path: Path = DATA_DIR / "reference_transitions_common_fens.dat",
    seed: int = 42,
) -> GlobalStyleSpace:
    """Load or fit global reference space on common FEN transitions.

    Args:
        positions_file (Path): Dataset path.
        common_fens (Set[str]): Intersected FEN set.
        players_dict (Dict[str, int]): Cohort mapping.
        device (torch.device): Compute device.
        memmap_path (Path, optional): Memory-mapped file path.
            Defaults to DATA_DIR / "reference_transitions_common_fens.dat".
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        GlobalStyleSpace: Fitted latent manifold space.
    """
    memmap_path = Path(memmap_path)
    if memmap_path.exists():
        file_bytes = os.path.getsize(memmap_path)
        written_count = file_bytes // (2304 * 4)
    else:
        lazy_pos = pl.scan_csv(positions_file)
        valid_indices = set(players_dict.values())

        total_positions = (
            lazy_pos.filter(
                (pl.col("player_index").is_in(valid_indices))
                & (pl.col("fen").is_in(common_fens))
            )
            .select(pl.len())
            .collect()
            .item()
        )

        os.makedirs(memmap_path.parent, exist_ok=True)
        fp = np.memmap(
            memmap_path, dtype="float32", mode="w+", shape=(total_positions, 2304)
        )

        written_count = 0
        for player_name, player_index in players_dict.items():
            player_pos = (
                lazy_pos.filter(
                    (pl.col("player_index") == player_index)
                    & (pl.col("fen").is_in(common_fens))
                )
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


def build_model_vs_player_common_fens_matrix(
    model_predictions_file: Path,
    common_fens: set[str],
    global_space: GlobalStyleSpace,
    players_dict: dict[str, int],
) -> tuple[np.ndarray, list[str]]:
    """Compute square stylistic divergence matrix on common FEN positions.

    Args:
        model_predictions_file (Path): Path to model prediction dataset.
        common_fens (Set[str]): Intersected shared position set.
        global_space (GlobalStyleSpace): Style reference manifold.
        players_dict (Dict[str, int]): Subject mapping dictionary.

    Returns:
        Tuple[np.ndarray, List[str]]: Divergence matrix and list of player names.
    """
    player_names = list(players_dict.keys())
    n_players = len(player_names)
    matrix = np.zeros((n_players, n_players), dtype=np.float64)

    lazy_df = pl.scan_csv(model_predictions_file).filter(
        pl.col("fen").is_in(common_fens)
    )

    pbar = tqdm.tqdm(
        enumerate(players_dict.items()),
        total=n_players,
        desc="Computing Common FEN Style Matrix",
        unit="model_cond",
        leave=False,
    )

    for i, (model_player_name, model_p_idx) in pbar:
        pbar.set_postfix({"conditioned_on": model_player_name})

        model_data = (
            lazy_df.filter(pl.col("player_index") == model_p_idx)
            .select(["fen", "predicted_move"])
            .collect()
        )

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
    """Render publication-ready JSD heatmap visualization.

    Args:
        jsd_matrix (np.ndarray): 2D square matrix of calculated divergences.
        player_names (List[str]): Subject cohort labels.
        output_path (Path): Destination file path.
        title (str): Plot title string.
        y_label (str, optional): Vertical axis label. Defaults to "Model Conditioned Player".
        is_inter_player (bool, optional): Flag switching palette for inter-human baseline.
            Defaults to False.
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
    logger.info("Heatmap successfully saved to: %s", output_path)


if __name__ == "__main__":
    GLOBAL_SEED: int = 42
    set_seed(GLOBAL_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata_file = DATA_DIR / "metadata.csv"
    positions_file = DATA_DIR / "positions.csv"

    if not metadata_file.exists() or not positions_file.exists():
        logger.error(
            "Required datasets (metadata.csv or positions.csv) were not found."
        )
        exit(1)

    players_df = getPlayers(str(metadata_file))
    players_dict = createPlayerDict(players_df)

    common_fens = extract_common_fens(positions_file, players_dict)
    if not common_fens:
        logger.error(
            "Zero common FEN positions identified across subjects. Terminating."
        )
        exit(1)

    # 1. Construct or load the common FEN reference manifold space
    global_space = build_global_reference_space_common_fens(
        positions_file, common_fens, players_dict, device, seed=GLOBAL_SEED
    )

    # 2. Iterate through prediction files and plot stylistic heatmaps
    models_pbar = tqdm.tqdm(
        PREDICTION_FILES.items(),
        desc="Generating Common FEN Stylistic Heatmaps",
        unit="model",
    )

    for model_name, path in models_pbar:
        if not path.exists():
            logger.warning(
                "Prediction record missing for model '%s': %s. Skipping.",
                model_name,
                path,
            )
            continue

        models_pbar.set_postfix({"current_model": model_name})
        logger.info("Generating common FEN style heatmap for model: %s", model_name)

        model_matrix, player_names = build_model_vs_player_common_fens_matrix(
            path, common_fens, global_space, players_dict
        )

        clean_name = (
            model_name.lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace(".", "")
            .replace("+", "")
        )
        out_file = OUTPUT_DIR / f"{clean_name}_common_fens_style_vs_player_heatmap.pdf"

        plot_jsd_heatmap(
            model_matrix,
            player_names,
            out_file,
            title=f"Stylistic Alignment Matrix (Common FENs): {model_name} vs Real Players",
            is_inter_player=False,
        )

        del model_matrix
        gc.collect()
        torch.cuda.empty_cache()
