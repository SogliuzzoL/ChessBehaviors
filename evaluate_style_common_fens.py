"""
Main evaluation pipeline for quantifying stylistic divergence (Autoencoder + cuML UMAP + Discrete Grid JSD)
across model variants conditioned on mutually observed board positions (Common FENs) across all human subjects.
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

DATA_DIR: Path = Path("data")

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
    """Enforce strict global determinism across random number generators and hardware backends.

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


def extract_common_fens(positions_file: Path, players_dict: dict[str, int]) -> set[str]:
    """Identify the exact intersection set of board positions (FENs) observed across all player cohorts.

    Args:
        positions_file (Path): Path leading to master positional records CSV.
        players_dict (Dict[str, int]): Mapping of subject names to integer cohort indices.

    Returns:
        Set[str]: Set of FEN strings mutually shared by all participating subjects.
    """
    logger.info("Computing common FEN intersection across subject cohorts...")
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

    common_fens: set[str] = set(df_counts["fen"].to_list())
    logger.info(
        "Identified %d mutually observed common FEN positions across all %d subjects.",
        len(common_fens),
        len(valid_indices),
    )
    return common_fens


def build_global_reference_space_common_fens(
    positions_file: Path,
    common_fens: set[str],
    players_dict: dict[str, int],
    device: torch.device,
    memmap_path: Path = DATA_DIR / "reference_transitions_common_fens.dat",
    seed: int = 42,
) -> GlobalStyleSpace:
    """Construct global invariant reference manifold restricted strictly to common board positions.

    Args:
        positions_file (Path): Path to ground-truth positional records.
        common_fens (Set[str]): Intersection set of mutually shared FEN strings.
        players_dict (Dict[str, int]): Subject mapping dictionary.
        device (torch.device): Computation device target.
        memmap_path (Path, optional): Destination binary path for memory-mapped storage.
            Defaults to DATA_DIR / "reference_transitions_common_fens.dat".
        seed (int, optional): Random seed parameter. Defaults to 42.

    Returns:
        GlobalStyleSpace: Fitted latent manifold space.
    """
    memmap_path = Path(memmap_path)
    if memmap_path.exists():
        logger.info(
            "Pre-existing common FEN memory-mapped reference detected at: %s. Loading binary.",
            memmap_path,
        )
        file_bytes = os.path.getsize(memmap_path)
        written_count = file_bytes // (2304 * 4)
    else:
        logger.info(
            "Filtering reference dataset positions down to mutual common FEN subset..."
        )
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
        logger.info(
            "Identified %d candidate reference moves on shared FEN states.",
            total_positions,
        )

        os.makedirs(memmap_path.parent, exist_ok=True)
        fp = np.memmap(
            memmap_path, dtype="float32", mode="w+", shape=(total_positions, 2304)
        )

        written_count = 0
        player_pbar = tqdm.tqdm(
            players_dict.items(),
            desc="Streaming Common FEN Transitions to Disk",
            unit="player",
        )

        for player_name, player_index in player_pbar:
            player_pbar.set_postfix({"player": player_name})
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
        logger.info(
            "Persisted %d valid common FEN reference vectors to: %s",
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


def evaluate_model_style_common_fens(
    model_name: str,
    csv_path: str,
    common_fens: set[str],
    players_dict: dict[str, int],
    global_space: GlobalStyleSpace,
) -> None:
    """Compute stylistic alignment metrics for candidate model against ground truth on shared FENs.

    Args:
        model_name (str): Model architecture identifier.
        csv_path (str): File system path to model prediction records.
        common_fens (Set[str]): Intersected shared board state set.
        players_dict (Dict[str, int]): Subject mapping dictionary.
        global_space (GlobalStyleSpace): Reference style manifold.
    """
    path = Path(csv_path)
    if not path.exists():
        logger.warning(
            "Prediction record missing at target path: %s. Skipping model evaluation.",
            csv_path,
        )
        return

    logger.info(
        "Evaluating common FEN stylistic divergence for candidate model: %s", model_name
    )
    lazy_df = pl.scan_csv(csv_path).filter(pl.col("fen").is_in(common_fens))

    results: list[dict[str, Any]] = []
    player_pbar = tqdm.tqdm(
        players_dict.items(),
        desc=f"Evaluating {model_name} (Common FENs)",
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
        model_moves: list[str | None] = player_data["predicted_move"].to_list()

        p_vecs: list[np.ndarray] = []
        m_vecs: list[np.ndarray] = []

        encoding_pbar = tqdm.tqdm(
            zip(fens, player_moves, model_moves),
            total=len(fens),
            desc=f"  -> Extracting common FEN representations [{player_name}]",
            leave=False,
            unit="pos",
        )

        for fen, p_move, m_move in encoding_pbar:
            if not p_move or not m_move:
                continue

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
        torch.cuda.empty_cache()

    if results:
        output_df = pl.DataFrame(results)
        clean_name = (
            model_name.lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace(".", "")
            .replace("+", "")
        )
        out_file = f"data/{clean_name}_common_fens_style.csv"
        output_df.write_csv(out_file)
        logger.info(
            "Stylistic evaluation metrics on common FENs successfully exported to: %s",
            out_file,
        )


if __name__ == "__main__":
    GLOBAL_SEED: int = 42
    set_seed(GLOBAL_SEED)

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata_path = DATA_DIR / "metadata.csv"
    positions_path = DATA_DIR / "positions.csv"

    if not metadata_path.exists() or not positions_path.exists():
        logger.error(
            "Required dataset files (metadata.csv or positions.csv) were not found."
        )
        exit(1)

    players_df = getPlayers(str(metadata_path))
    players_dict = createPlayerDict(players_df)

    # 1. Identify common FEN positions shared across all subjects
    common_fens = extract_common_fens(positions_path, players_dict)
    if not common_fens:
        logger.error(
            "No mutually observed positions found across all cohorts. Aborting."
        )
        exit(1)

    # 2. Build or load the global reference manifold fitted on common FENs
    global_space = build_global_reference_space_common_fens(
        positions_path, common_fens, players_dict, device, seed=GLOBAL_SEED
    )

    # 3. Evaluate each model prediction file on common FENs
    models_pbar = tqdm.tqdm(
        PREDICTION_FILES.items(),
        desc="Evaluating Models on Common FEN Style Space",
        unit="model",
    )

    for model_name, path_str in models_pbar:
        evaluate_model_style_common_fens(
            model_name, path_str, common_fens, players_dict, global_space
        )
