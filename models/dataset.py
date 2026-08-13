"""
PyTorch Dataset implementation for preparing board state representations and targeted move labels
during subject-level fine-tuning and adaptation procedures.
"""

from typing import Any

import pandas as pd
from maia2.inference import preprocessing
from maia2.utils import mirror_move
from torch.utils.data import Dataset


class PlayerTrainDataset(Dataset):
    """PyTorch Dataset wrapper processing positional observation records into tensor representations.

    Constructs preprocessed board state inputs and target action indices for fine-tuning
    the Maia-2 architecture on subject-specific decision histories. Mirroring transformations
    are applied to standard board orientations when Black is the active player.

    Attributes:
        samples (List[Tuple[Any, int]]): List of tuples containing preprocessed board
            tensor representations and corresponding target move indices.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        all_moves_dict: dict[str, int],
        elo_dict: dict[Any, Any],
    ) -> None:
        """Initialize the subject dataset and preprocess target position observations.

        Args:
            train_df (pd.DataFrame): Dataframe containing positional records with 'board'
                (FEN) and 'move' (UCI) columns.
            all_moves_dict (Dict[str, int]): Vocabulary mapping UCI move strings to integer indices.
            elo_dict (Dict[Any, Any]): Subject rating metadata dictionary required by Maia-2 preprocessing.
        """
        self.samples: list[tuple[Any, int]] = []

        for row in train_df.itertuples():
            fen: str = row.board
            target_move: str = row.move

            # Extract spatial board tensor representation via Maia-2 preprocessing pipeline
            board_input, _, _, _ = preprocessing(
                fen, 2500, 2500, elo_dict, all_moves_dict
            )

            # Apply perspective normalization (board mirroring) if active side is Black
            if fen.split(" ")[1] == "b":
                target_move = mirror_move(target_move)

            # Map target candidate move to corresponding categorical index
            if target_move in all_moves_dict:
                move_idx: int = all_moves_dict[target_move]
                self.samples.append((board_input, move_idx))

    def __len__(self) -> int:
        """Return total number of valid positional samples in dataset.

        Returns:
            int: Dataset sample count.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        """Fetch preprocessed board tensor and candidate move label by index.

        Args:
            idx (int): Sample position index.

        Returns:
            Tuple[Any, int]: Tuple containing preprocessed board representation tensor
                and target move index label.
        """
        return self.samples[idx]
