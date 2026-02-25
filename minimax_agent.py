"""Simple minimax reference agent for Tic-Tac-Toe."""

from __future__ import annotations

from typing import Optional, Tuple

from mcts_tictactoe import TicTacToeState, apply_move, is_terminal, legal_moves, winner


# Chooses the best move for the current player using full minimax search.
def choose_minimax_move(state: TicTacToeState) -> int:
    # Terminal states have no legal move to return.
    if is_terminal(state):
        raise ValueError("Cannot choose a move from a terminal state")

    # Keep the best move and score found so far.
    best_move: Optional[int] = None
    best_score = float("-inf")

    # Evaluate each legal move by recursive minimax.
    for move in legal_moves(state):
        child = apply_move(state, move)
        score = _minimax_value(child, maximizing_player=state.current_player)
        if score > best_score:
            best_score = score
            best_move = move

    if best_move is None:
        raise RuntimeError("No move selected by minimax")
    return best_move


# Recursively computes minimax value from a state for a fixed maximizing player.
def _minimax_value(state: TicTacToeState, maximizing_player: str) -> float:
    # Base case: exact terminal utility.
    if is_terminal(state):
        return _terminal_score(state, maximizing_player)

    # If it's maximizing player's turn, maximize; otherwise minimize.
    if state.current_player == maximizing_player:
        value = float("-inf")
        for move in legal_moves(state):
            child = apply_move(state, move)
            value = max(value, _minimax_value(child, maximizing_player))
        return value

    value = float("inf")
    for move in legal_moves(state):
        child = apply_move(state, move)
        value = min(value, _minimax_value(child, maximizing_player))
    return value


# Converts a terminal state into minimax utility for maximizing player.
def _terminal_score(state: TicTacToeState, maximizing_player: str) -> float:
    # Determine game winner for exact utility.
    game_winner = winner(state)

    # Draw utility is neutral.
    if game_winner is None:
        return 0.0

    # Win for maximizing player is +1; loss is -1.
    return 1.0 if game_winner == maximizing_player else -1.0
