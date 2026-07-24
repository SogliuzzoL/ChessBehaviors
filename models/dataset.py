import pandas as pd
from maia2.inference import preprocessing
from maia2.utils import mirror_move
from torch.utils.data import Dataset


class PlayerTrainDataset(Dataset):
    def __init__(self, train_df: pd.DataFrame, all_moves_dict: dict, elo_dict: dict):
        self.samples = []
        for row in train_df.itertuples():
            fen = row.board
            target_move = row.move

            board_input, _, _, _ = preprocessing(
                fen, 2500, 2500, elo_dict, all_moves_dict
            )

            if fen.split(" ")[1] == "b":
                target_move = mirror_move(target_move)

            if target_move in all_moves_dict:
                move_idx = all_moves_dict[target_move]
                self.samples.append((board_input, move_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
