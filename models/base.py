"""
Core architectural abstractions and search inference wrappers for evaluating
chess model policy prediction distributions and position value estimates.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable

import chess
from maia2 import inference


class ChessModel(ABC):
    """Abstract base class defining the standard prediction interface for chess models.

    Defines the contract for forward-pass policy inference and position evaluation,
    returning candidate move probability distributions and root state values.
    """

    @abstractmethod
    def predict(self, board: chess.Board) -> tuple[dict[str, float], float]:
        """Predict candidate move probabilities and position value for a given board state.

        Args:
            board (chess.Board): Current chess board state representation.

        Returns:
            Tuple[Dict[str, float], float]: A tuple containing:
                - Dict[str, float]: Normalized move policy distribution mapping UCI strings to probabilities.
                - float: Scalar position evaluation score from the active player's perspective.
        """


class Maia2(ChessModel):
    """Interface wrapper for the baseline pre-trained Maia-2 neural architecture.

    Handles forward-pass inference over raw board states, legal move filtering,
    optional action-space pruning, and relative probability normalization.

    Attributes:
        model: Pre-trained neural model instance.
        prepared: Pre-allocated inference context structures for Maia-2.
        pruning_fn (Optional[Callable[[Dict[str, float]], List[str]]]): Optional action-space
            pruning transformation applied to candidate moves.
    """

    def __init__(
        self,
        model: object,
        pruning_fn: Callable[[dict[str, float]], list[str]] | None = None,
    ) -> None:
        """Initialize the Maia-2 baseline model wrapper.

        Args:
            model (object): Pre-trained Maia-2 neural execution model.
            pruning_fn (Optional[Callable[[Dict[str, float]], list[str]]], optional): Function
                executing policy space pruning (e.g., nucleus pruning). Defaults to None.
        """
        self.model = model
        self.prepared = inference.prepare()
        self.pruning_fn = pruning_fn

    def predict(self, board: chess.Board) -> tuple[dict[str, float], float]:
        """Perform policy inference over legal moves in the given state.

        Args:
            board (chess.Board): Chess state instance.

        Returns:
            Tuple[Dict[str, float], float]: Pair of normalized legal move action distribution
                and scalar root position value score.
        """
        raw_moves, value = inference.inference_each(
            self.model, self.prepared, board.fen(), 2500, 2500
        )

        # Enforce strict legal action filtering in Universal Chess Interface (UCI) format
        legal_uci_moves: set[str] = {m.uci() for m in board.legal_moves}
        legal_moves_dict: dict[str, float] = {
            move: score for move, score in raw_moves.items() if move in legal_uci_moves
        }

        # Apply action-space pruning strategy if specified
        if self.pruning_fn and legal_moves_dict:
            moves_pruned = self.pruning_fn(legal_moves_dict)
            legal_moves_dict = {
                move: legal_moves_dict[move]
                for move in moves_pruned
                if move in legal_moves_dict
            }

        # Renormalize posterior legal move probability distribution
        total = sum(legal_moves_dict.values())
        if total > 0:
            legal_moves_dict = {
                move: score / total for move, score in legal_moves_dict.items()
            }

        return legal_moves_dict, value


class DescentModelWrapper(ChessModel):
    """Inference wrapper enhancing base model policy via descent search optimization.

    Attributes:
        base_model (ChessModel): Target baseline model providing prior guidance.
        max_iterations (int): Maximum optimization step budget per position.
        timeout (float): Time threshold budget in seconds per search call.
    """

    def __init__(
        self, base_model: ChessModel, max_iterations: int = 100, timeout: float = 2.0
    ) -> None:
        """Initialize the descent search wrapper agent.

        Args:
            base_model (ChessModel): Base policy evaluation model.
            max_iterations (int, optional): Iteration budget for descent search. Defaults to 100.
            timeout (float, optional): Maximum execution time in seconds. Defaults to 2.0.
        """
        self.base_model = base_model
        self.max_iterations = max_iterations
        self.timeout = timeout

    def predict(self, board: chess.Board) -> tuple[dict[str, float], float]:
        """Perform descent-search guided policy prediction.

        Args:
            board (chess.Board): Target chess board state.

        Returns:
            Tuple[Dict[str, float], float]: Tuple containing search-refined policy distribution
                and root state evaluation.
        """
        from search.descent import DescentSearch

        search_engine = DescentSearch(model=self.base_model)
        policy_probs, root_value = search_engine.search(
            board, max_iterations=self.max_iterations, timeout=self.timeout
        )
        return policy_probs, root_value


class MCTSModelWrapper(ChessModel):
    """Inference wrapper augmenting base model evaluation via Monte Carlo Tree Search (MCTS).

    Attributes:
        base_model (ChessModel): Target baseline policy and value evaluation model.
        max_iterations (int): Maximum node expansion simulation budget.
        timeout (float): Maximum search time limit in seconds per decision point.
        cpuct (float): Upper Confidence Bound (UCB) exploration constant.
    """

    def __init__(
        self,
        base_model: ChessModel,
        max_iterations: int = 50,
        timeout: float = 1.0,
        cpuct: float = 1.5,
    ) -> None:
        """Initialize the MCTS search wrapper agent.

        Args:
            base_model (ChessModel): Base neural evaluation engine.
            max_iterations (int, optional): Simulation budget count. Defaults to 50.
            timeout (float, optional): Maximum time budget in seconds. Defaults to 1.0.
            cpuct (float, optional): UCB exploration trade-off parameter. Defaults to 1.5.
        """
        self.base_model = base_model
        self.max_iterations = max_iterations
        self.timeout = timeout
        self.cpuct = cpuct

    def predict(self, board: chess.Board) -> tuple[dict[str, float], float]:
        """Perform MCTS simulation search to extract tree visit policy distributions.

        Args:
            board (chess.Board): Target chess board state.

        Returns:
            Tuple[Dict[str, float], float]: Tuple containing MCTS visit-count policy distribution
                and tree root valuation.
        """
        from search.mcts import MCTSSearch

        search_engine = MCTSSearch(model=self.base_model, cpuct=self.cpuct)
        policy_probs, root_value = search_engine.search(
            board, max_iterations=self.max_iterations, timeout=self.timeout
        )
        return policy_probs, root_value
