"""Baseline and LLM-guided MCTS implementations for Tic-Tac-Toe."""

from __future__ import annotations

import json
import math
import os
import random
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from mcts_tictactoe import (
    TicTacToeState,
    apply_move,
    board_to_prompt_text,
    is_terminal,
    legal_moves,
    other_player,
    winner,
)


@dataclass
class MCTSNode:
    """Single tree node used by MCTS search."""

    state: TicTacToeState
    parent: Optional["MCTSNode"] = None
    move: Optional[int] = None
    children: Dict[int, "MCTSNode"] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0

    # Returns the player who made the move that created this node.
    def player_just_moved(self) -> str:
        # For non-root nodes, the parent's current player made the move.
        if self.parent is not None:
            return self.parent.state.current_player
        # For the root, define "just moved" as the opponent of side-to-move.
        return other_player(self.state.current_player)

    # Returns the mean value stored at this node.
    def average_value(self) -> float:
        # Unvisited nodes have no mean value yet.
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    # Returns all legal moves that have not been expanded yet.
    def untried_moves(self) -> List[int]:
        # Any legal move absent from children is still untried.
        expanded = set(self.children.keys())
        return [move for move in legal_moves(self.state) if move not in expanded]

    # Returns True when every legal move has already been expanded.
    def is_fully_expanded(self) -> bool:
        # Node is fully expanded when no untried moves remain.
        return len(self.untried_moves()) == 0

    # Selects the best child using UCB1.
    def best_child_ucb(self, exploration_constant: float) -> "MCTSNode":
        # Guard against selecting from an empty children dictionary.
        if not self.children:
            raise ValueError("Cannot select best child from a node without children")

        # Use at least log(1) to avoid math errors before visits accumulate.
        parent_log = math.log(max(1, self.visits))
        best_score = float("-inf")
        best_node: Optional[MCTSNode] = None

        # Compute UCB score for each child and keep the maximum.
        for child in self.children.values():
            if child.visits == 0:
                score = float("inf")
            else:
                exploit = child.average_value()
                explore = exploration_constant * math.sqrt(parent_log / child.visits)
                score = exploit + explore

            if score > best_score:
                best_score = score
                best_node = child

        if best_node is None:
            raise RuntimeError("UCB selection failed to choose a child")
        return best_node


class BaselineMCTS:
    """Standard MCTS with random rollout simulation."""

    # Initializes baseline MCTS search settings.
    def __init__(self, iterations: int = 1000, exploration_constant: float = math.sqrt(2), seed: Optional[int] = None):
        # Store algorithm hyperparameters.
        self.iterations = iterations
        self.exploration_constant = exploration_constant
        # Use a dedicated RNG so behavior is reproducible when seed is set.
        self.rng = random.Random(seed)

    # Chooses the move with the highest visit count after MCTS search.
    def choose_move(self, state: TicTacToeState) -> int:
        # Terminal states do not have legal next moves.
        if is_terminal(state):
            raise ValueError("Cannot choose a move from a terminal state")

        # Build and search the tree from the given root state.
        root = MCTSNode(state=state)
        self._run_search(root)

        # Select move by highest visits (standard MCTS action rule).
        if not root.children:
            raise RuntimeError("MCTS produced no children from a non-terminal state")
        best_move, _ = max(root.children.items(), key=lambda item: (item[1].visits, item[1].average_value()))
        return best_move

    # Runs repeated MCTS iterations from the provided root node.
    def _run_search(self, root: MCTSNode) -> None:
        # Execute the four MCTS phases for each iteration.
        for _ in range(self.iterations):
            leaf, path = self._select_and_expand(root)
            leaf_player = leaf.state.current_player
            leaf_value = self._leaf_value(leaf.state)
            self._backpropagate_leaf_value(path, leaf_player, leaf_value)

    # Performs Selection and Expansion and returns the resulting leaf plus path.
    def _select_and_expand(self, root: MCTSNode) -> Tuple[MCTSNode, List[MCTSNode]]:
        # Start at root and keep track of full path for backpropagation.
        node = root
        path = [root]

        # Selection: follow UCB while node is non-terminal and fully expanded.
        while not is_terminal(node.state) and node.is_fully_expanded():
            node = node.best_child_ucb(self.exploration_constant)
            path.append(node)

        # Expansion: add exactly one child when node is non-terminal.
        if not is_terminal(node.state):
            node = self._expand_one_child(node)
            path.append(node)

        return node, path

    # Expands one random untried move under the given node.
    def _expand_one_child(self, node: MCTSNode) -> MCTSNode:
        # Pick one untried move uniformly at random.
        untried = node.untried_moves()
        if not untried:
            raise RuntimeError("Expand called on fully expanded node")
        move = self.rng.choice(untried)

        # Create and store the child node for this move.
        child_state = apply_move(node.state, move)
        child_node = MCTSNode(state=child_state, parent=node, move=move)
        node.children[move] = child_node
        return child_node

    # Returns the leaf value in [0,1] for the leaf's current player.
    def _leaf_value(self, state: TicTacToeState) -> float:
        # If terminal, compute exact result directly.
        if is_terminal(state):
            return self._score_from_winner(state.current_player, winner(state))

        # Otherwise run random rollout to a terminal outcome.
        rollout_winner = self._simulate_random_rollout(state)
        return self._score_from_winner(state.current_player, rollout_winner)

    # Runs a random playout until terminal and returns winner marker or None for draw.
    def _simulate_random_rollout(self, state: TicTacToeState) -> Optional[str]:
        # Copy state reference and keep applying random legal moves.
        current = state
        while not is_terminal(current):
            move = self.rng.choice(legal_moves(current))
            current = apply_move(current, move)
        return winner(current)

    # Converts winner outcome into score for a specific player.
    def _score_from_winner(self, player: str, game_winner: Optional[str]) -> float:
        # Draw yields neutral value.
        if game_winner is None:
            return 0.5
        # Win gives 1.0, loss gives 0.0.
        return 1.0 if game_winner == player else 0.0

    # Backpropagates leaf value to every node in the selected path.
    def _backpropagate_leaf_value(self, path: List[MCTSNode], leaf_player: str, leaf_value: float) -> None:
        # Update from leaf back to root.
        for node in reversed(path):
            node.visits += 1
            node_player = node.player_just_moved()

            # Keep values from each node's player perspective.
            if node_player == leaf_player:
                node.value_sum += leaf_value
            else:
                node.value_sum += 1.0 - leaf_value


class LLMPositionEvaluator:
    """LLM wrapper that estimates win probability for current player."""

    # Configures provider/model credentials and request behavior.
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        timeout_seconds: int = 30,
    ):
        # Normalize provider names to reduce CLI friction.
        self.provider = provider.strip().lower()
        self.model = model
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

        # Read API key from environment when not passed directly.
        if api_key is None:
            if self.provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
            elif self.provider == "anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            raise ValueError("Missing API key. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.")
        self.api_key = api_key

        # Cache repeated evaluations of identical states.
        self._cache: Dict[Tuple[Tuple[str, ...], str], float] = {}

    # Returns win probability in [0,1] for the state's current player.
    def evaluate(self, state: TicTacToeState) -> float:
        # Use cache first to avoid repeated paid API calls.
        key = (state.board, state.current_player)
        if key in self._cache:
            return self._cache[key]

        # Build a strict JSON-response prompt for robust parsing.
        prompt = self._build_prompt(state)

        # Send request to the selected provider.
        if self.provider == "openai":
            raw_text = self._call_openai(prompt)
        elif self.provider == "anthropic":
            raw_text = self._call_anthropic(prompt)
        else:
            raise ValueError("provider must be 'openai' or 'anthropic'")

        # Parse and clamp the probability output.
        probability = self._extract_probability(raw_text)
        self._cache[key] = probability
        return probability

    # Builds a deterministic prompt asking for win probability JSON.
    def _build_prompt(self, state: TicTacToeState) -> str:
        # Keep output schema very small for easier parsing.
        return (
            "You are evaluating a Tic-Tac-Toe position.\n"
            "The board is shown below as a 3x3 grid.\n"
            "Rate this position for the CURRENT PLAYER on a 0 to 1 scale:\n"
            "0.0 = certain loss, 0.5 = draw/unclear, 1.0 = certain win.\n"
            "Return ONLY valid JSON with one key: win_probability\n"
            "Example: {\"win_probability\": 0.73}\n\n"
            f"{board_to_prompt_text(state)}"
        )

    # Calls OpenAI chat completions endpoint and returns text content.
    def _call_openai(self, prompt: str) -> str:
        # Prepare request payload for a deterministic short response.
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": "Respond with JSON only."},
                {"role": "user", "content": prompt},
            ],
        }

        # Create HTTP request with auth and content headers.
        request = urllib.request.Request(
            url="https://api.openai.com/v1/chat/completions",
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload).encode("utf-8"),
        )

        # Execute request and decode JSON response.
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))

        # Extract assistant content safely.
        message = data["choices"][0]["message"]["content"]
        if isinstance(message, str):
            return message
        return json.dumps(message)

    # Calls Anthropic messages endpoint and returns text content.
    def _call_anthropic(self, prompt: str) -> str:
        # Prepare request payload in Anthropic format.
        payload = {
            "model": self.model,
            "max_tokens": 128,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        # Create HTTP request with required Anthropic headers.
        request = urllib.request.Request(
            url="https://api.anthropic.com/v1/messages",
            method="POST",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            data=json.dumps(payload).encode("utf-8"),
        )

        # Execute request and decode JSON response.
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))

        # Extract concatenated text blocks from content array.
        parts = data.get("content", [])
        texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
        return "\n".join(texts).strip()

    # Extracts win_probability from model output and clamps to [0,1].
    def _extract_probability(self, raw_text: str) -> float:
        # First try direct JSON parse.
        try:
            parsed = json.loads(raw_text)
            value = float(parsed["win_probability"])
            return min(1.0, max(0.0, value))
        except Exception:
            pass

        # Next try to find an embedded JSON object.
        json_match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                value = float(parsed["win_probability"])
                return min(1.0, max(0.0, value))
            except Exception:
                pass

        # Last fallback: first number in the output.
        number_match = re.search(r"[-+]?\d*\.?\d+", raw_text)
        if number_match:
            value = float(number_match.group(0))
            # If model returned percentage, map it to [0,1].
            if value > 1.0:
                value = value / 100.0
            return min(1.0, max(0.0, value))

        # If parsing fails completely, return a neutral estimate.
        return 0.5


class LLMGuidedMCTS(BaselineMCTS):
    """MCTS variant where Simulation is replaced by LLM position evaluation."""

    # Initializes LLM-guided MCTS with an evaluator instance.
    def __init__(
        self,
        evaluator: LLMPositionEvaluator,
        iterations: int = 300,
        exploration_constant: float = math.sqrt(2),
        seed: Optional[int] = None,
    ):
        # Reuse baseline setup for selection/expansion/backprop logic.
        super().__init__(iterations=iterations, exploration_constant=exploration_constant, seed=seed)
        self.evaluator = evaluator

    # Returns the leaf value in [0,1] for the leaf's current player.
    def _leaf_value(self, state: TicTacToeState) -> float:
        # Preserve exact outcomes for terminal states.
        if is_terminal(state):
            return self._score_from_winner(state.current_player, winner(state))

        # Replace random rollout with direct LLM position evaluation.
        return self.evaluator.evaluate(state)
