"""
Monte Carlo Tree Search (MCTS) module for upper confidence bound action selection
and policy refinement over chess state spaces.
"""

import time
from dataclasses import dataclass, field

import chess
import numpy as np

from models import ChessModel


@dataclass
class MCTSNode:
    """Data structure representing a node in the Monte Carlo search tree.

    Attributes:
        v (float): Prior value estimate for the board state.
        n_sum (int): Total cumulative visit count across all outgoing child actions.
        n (Dict[str, int]): Action visit counts mapping candidate UCI strings to integers.
        w (Dict[str, float]): Action action-value accumulation mapping UCI strings to total rewards.
        p (Dict[str, float]): Prior policy distribution mapping UCI strings to probabilities.
        is_terminal (bool): Flag indicating whether the state represents a terminal game outcome.
        terminal_value (float): Objective terminal reward value if the position is terminal.
    """

    v: float = 0.0
    n_sum: int = 0
    n: dict[str, int] = field(default_factory=dict)
    w: dict[str, float] = field(default_factory=dict)
    p: dict[str, float] = field(default_factory=dict)
    is_terminal: bool = False
    terminal_value: float = 0.0


class MCTSSearch:
    """Monte Carlo Tree Search engine leveraging Predictor-Upper Confidence Bounds for Trees (PUCT).

    Executes tree traversal, node expansion, neural evaluation backpropagation, and
    visit-count based policy refinement over chess position representations.

    Attributes:
        model (ChessModel): Neural network model evaluating priors and position values.
        cpuct (float): Exploration constant balancing exploitation and prior exploration.
        T (Dict[str, MCTSNode]): Transposition table mapping FEN strings to search tree nodes.
    """

    def __init__(self, model: ChessModel, cpuct: float = 1.5) -> None:
        """Initialize the MCTSSearch engine.

        Args:
            model (ChessModel): Base neural network providing prior move policies and state values.
            cpuct (float, optional): PUCT exploration constant parameter. Defaults to 1.5.
        """
        self.model = model
        self.cpuct = cpuct
        self.T: dict[str, MCTSNode] = {}

    def _get_terminal_gain(self, board: chess.Board) -> float:
        """Compute terminal payoff score for game-ending board positions.

        Args:
            board (chess.Board): Terminal board state instance.

        Returns:
            float: Terminal reward value (+1.0 for White victory, -1.0 for Black victory, 0.0 for draw).
        """
        outcome = board.outcome()
        if outcome is None or outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner == chess.WHITE else -1.0

    def _select_puct_action(self, board: chess.Board, node: MCTSNode) -> str:
        """Select candidate action maximizing or minimizing the PUCT objective score.

        Args:
            board (chess.Board): Current chess board state instance.
            node (MCTSNode): Corresponding search tree node.

        Returns:
            str: Selected optimal action string in UCI format.
        """
        is_white = board.turn == chess.WHITE
        best_action: str | None = None
        best_puct = -float("inf") if is_white else float("inf")
        sqrt_total = float(np.sqrt(max(1, node.n_sum)))

        for a, prob in node.p.items():
            visit_count = node.n[a]
            q_val = node.w[a] / visit_count if visit_count > 0 else node.v
            u_val = self.cpuct * prob * sqrt_total / (1 + visit_count)
            score = q_val + u_val if is_white else q_val - u_val

            if is_white:
                if score > best_puct:
                    best_puct = score
                    best_action = a
            else:
                if score < best_puct:
                    best_puct = score
                    best_action = a

        return best_action or list(node.p.keys())[0]

    def _simulation(self, board: chess.Board) -> None:
        """Execute a single MCTS simulation pass (Selection, Expansion, Evaluation, Backpropagation).

        Args:
            board (chess.Board): Target root chess board state instance.
        """
        path: list[tuple[str, str]] = []
        curr_board = board.copy()
        s = curr_board.fen()

        # Selection and Expansion phase
        while True:
            if curr_board.is_game_over():
                if s not in self.T:
                    gain = self._get_terminal_gain(curr_board)
                    self.T[s] = MCTSNode(is_terminal=True, terminal_value=gain)
                break

            if s not in self.T:
                moves_probs, value = self.model.predict(curr_board)
                self.T[s] = MCTSNode(
                    v=value,
                    n={m: 0 for m in moves_probs},
                    w={m: 0.0 for m in moves_probs},
                    p=moves_probs,
                )
                break

            node = self.T[s]
            if not node.p:
                break

            best_action = self._select_puct_action(curr_board, node)
            path.append((s, best_action))
            curr_board.push_uci(best_action)
            s = curr_board.fen()

        # Evaluation phase
        if s in self.T and self.T[s].is_terminal:
            v = self.T[s].terminal_value
        elif s in self.T:
            v = self.T[s].v
        else:
            v = 0.0

        # Backpropagation phase
        for state_fen, action in reversed(path):
            node = self.T[state_fen]
            node.n_sum += 1
            node.n[action] += 1
            node.w[action] += v

    def get_policy(self, board: chess.Board) -> dict[str, float]:
        """Derive posterior policy distribution proportional to search tree visit frequencies.

        Args:
            board (chess.Board): Target root chess board state instance.

        Returns:
            Dict[str, float]: Normalized action policy mapping move UCI strings to visit probabilities.
        """
        s = board.fen()
        if s not in self.T or not self.T[s].n:
            return {}
        node = self.T[s]
        total_visits = sum(node.n.values())
        if total_visits == 0:
            return {m: prob for m, prob in node.p.items()}
        return {m: visits / total_visits for m, visits in node.n.items()}

    def search(
        self, board: chess.Board, max_iterations: int = 50, timeout: float = 1.0
    ) -> tuple[dict[str, float], float]:
        """Execute iterative Monte Carlo simulation budget over a given position.

        Args:
            board (chess.Board): Target root board state instance.
            max_iterations (int, optional): Maximum simulation count budget. Defaults to 50.
            timeout (float, optional): Maximum search time budget in seconds. Defaults to 1.0.

        Returns:
            Tuple[Dict[str, float], float]: Tuple containing visit-count posterior policy distribution
                and root state evaluation value.
        """
        start_time = time.time()
        for _ in range(max_iterations):
            if time.time() - start_time >= timeout:
                break
            self._simulation(board)

        policy = self.get_policy(board)
        root_value = self.T.get(board.fen(), MCTSNode()).v
        return policy, root_value
