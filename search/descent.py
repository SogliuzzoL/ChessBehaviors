"""
Descent Search algorithm module for trajectory optimization and value-guided tree search over chess state spaces.
"""

import time
from dataclasses import dataclass, field

import chess
import numpy as np

from models import ChessModel


@dataclass
class TranspositionNode:
    """Node data structure stored in the search transposition table.

    Attributes:
        v (float): Scalar position evaluation or value score.
        c (float): Convergence state score or solved game value.
        r (int): Resolution status flag indicating whether state trajectory is fully determined.
        n (Dict[str, int]): Map of candidate UCI action strings to cumulative visit counts.
    """

    v: float = 0.0
    c: float = 0.0
    r: int = 0
    n: dict[str, int] = field(default_factory=dict)


class DescentSearch:
    """Value-guided Descent Search optimization agent operating on chess state spaces.

    Iteratively expands search trajectories using value-guided exploration,
    populating transposition nodes and resolving game states via dynamic programming.

    Attributes:
        model (ChessModel): Base policy and position evaluation model.
        T (Dict[str, TranspositionNode]): Global transposition table mapping FEN strings to nodes.
    """

    def __init__(self, model: ChessModel) -> None:
        """Initialize the Descent Search engine.

        Args:
            model (ChessModel): Base neural evaluation engine supplying action priors and value estimates.
        """
        self.model = model
        self.T: dict[str, TranspositionNode] = {}

    def _get_terminal_gain(self, board: chess.Board) -> float:
        """Compute objective terminal payoff score for ended game positions.

        Args:
            board (chess.Board): Terminal board state instance.

        Returns:
            float: Terminal reward value (+1.0 for White win, -1.0 for Black win, 0.0 for draw).
        """
        outcome = board.outcome()
        if outcome is None or outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner == chess.WHITE else -1.0

    def _get_child_node(
        self, board: chess.Board, move_uci: str
    ) -> tuple[str, TranspositionNode]:
        """Fetch child FEN representation and corresponding transposition node for a candidate move.

        Args:
            board (chess.Board): Current board state.
            move_uci (str): Candidate action in Universal Chess Interface (UCI) string format.

        Returns:
            Tuple[str, TranspositionNode]: Pair containing child FEN identifier and child node reference.
        """
        board.push_uci(move_uci)
        child_s = board.fen()
        board.pop()
        return child_s, self.T[child_s]

    def _select_best_action(
        self, board: chess.Board, moves: list[str], dual: bool = False
    ) -> str:
        """Select optimal candidate action maximizing/minimizing target search criteria.

        Args:
            board (chess.Board): Current board state instance.
            moves (List[str]): Candidate action set in UCI string format.
            dual (bool, optional): Inverts visit count criteria for tie-breaking/exploration. Defaults to False.

        Returns:
            str: Selected candidate move UCI string.
        """
        s = board.fen()
        is_white = board.turn == chess.WHITE

        def key_fn(move: str) -> tuple[float, float, float]:
            _, child_node = self._get_child_node(board, move)
            n_val = float(self.T[s].n.get(move, 0))
            if dual:
                n_val = -n_val
            return (child_node.c, child_node.v, n_val if is_white else -n_val)

        return max(moves, key=key_fn) if is_white else min(moves, key=key_fn)

    def _backup_resolution(self, board: chess.Board, moves: list[str]) -> int:
        """Calculate resolution status flag across child branches.

        Args:
            board (chess.Board): Current board state.
            moves (List[str]): Candidate move set.

        Returns:
            int: Minimum child resolution status flag value.
        """
        s = board.fen()
        if abs(self.T[s].c) == 1.0:
            return 1

        children_r = [self._get_child_node(board, m)[1].r for m in moves]
        return min(children_r) if children_r else 0

    def _update_node_values(self, board: chess.Board, moves: list[str]) -> None:
        """Update node evaluation metrics based on best available candidate action.

        Args:
            board (chess.Board): Current board state instance.
            moves (List[str]): Candidate move set.
        """
        s = board.fen()
        best_move = self._select_best_action(board, moves, dual=False)
        _, best_child = self._get_child_node(board, best_move)

        self.T[s].c = best_child.c
        self.T[s].v = best_child.v
        self.T[s].r = self._backup_resolution(board, moves)

    def iteration(self, board: chess.Board) -> None:
        """Execute a single recursive descent search expansion iteration.

        Args:
            board (chess.Board): Current target board state instance.
        """
        s = board.fen()

        # Handle terminal leaf states
        if board.is_game_over():
            gain = self._get_terminal_gain(board)
            self.T[s] = TranspositionNode(v=gain, c=gain, r=1)
            return

        # Perform node expansion and prior initialization
        if s not in self.T:
            moves_probs, value = self.model.predict(board)
            self.T[s] = TranspositionNode(
                v=value, c=0.0, r=0, n={m: 0 for m in moves_probs}
            )

            for move_uci, prob in moves_probs.items():
                board.push_uci(move_uci)
                child_s = board.fen()

                if board.is_game_over():
                    gain = self._get_terminal_gain(board)
                    self.T[child_s] = TranspositionNode(v=gain, c=gain, r=1)
                elif child_s not in self.T:
                    self.T[child_s] = TranspositionNode(v=prob, c=0.0, r=0)

                board.pop()

            legal_moves = list(moves_probs.keys())
            if legal_moves:
                self._update_node_values(board, legal_moves)
            return

        # Recurse down unresolved subtrees
        if self.T[s].r == 0:
            legal_moves = list(self.T[s].n.keys())
            unresolved = [
                m for m in legal_moves if self._get_child_node(board, m)[1].r == 0
            ]

            if unresolved:
                selected_move = self._select_best_action(board, unresolved, dual=True)
                self.T[s].n[selected_move] += 1

                board.push_uci(selected_move)
                self.iteration(board)
                board.pop()

                self._update_node_values(board, legal_moves)

    def get_policy(
        self, board: chess.Board, temperature: float = 1.0
    ) -> dict[str, float]:
        """Compute Softmax-scaled policy distribution over child node values.

        Args:
            board (chess.Board): Target root board state.
            temperature (float, optional): Softmax policy temperature scaling parameter. Defaults to 1.0.

        Returns:
            Dict[str, float]: Policy probability distribution mapping move UCI strings to probabilities.
        """
        s = board.fen()
        if s not in self.T or not self.T[s].n:
            return {}

        moves = list(self.T[s].n.keys())
        values = [self._get_child_node(board, m)[1].v for m in moves]

        values_arr = np.array(values, dtype=np.float64) / max(temperature, 1e-6)
        exp_values = np.exp(values_arr - np.max(values_arr))
        probs = exp_values / np.sum(exp_values)

        return {move: float(p) for move, p in zip(moves, probs)}

    def search(
        self, board: chess.Board, max_iterations: int = 100, timeout: float = 2.0
    ) -> tuple[dict[str, float], float]:
        """Execute iterative descent search optimization budget over board position.

        Args:
            board (chess.Board): Target board state instance.
            max_iterations (int, optional): Maximum iteration step count. Defaults to 100.
            timeout (float, optional): Maximum allowed search execution time in seconds. Defaults to 2.0.

        Returns:
            Tuple[Dict[str, float], float]: Tuple containing search-refined policy distribution
                and root state evaluation value.
        """
        start_time = time.time()
        for _ in range(max_iterations):
            if (
                time.time() - start_time >= timeout
                or self.T.get(board.fen(), TranspositionNode()).r == 1
            ):
                break
            self.iteration(board)

        policy = self.get_policy(board)
        root_value = self.T.get(board.fen(), TranspositionNode()).v
        return policy, root_value
