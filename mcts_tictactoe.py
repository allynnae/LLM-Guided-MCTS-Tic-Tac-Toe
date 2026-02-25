"""Simple Tic-Tac-Toe state helpers used by MCTS agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


# Constant board marker for an empty square.
EMPTY = " "


# All 8 winning line combinations for Tic-Tac-Toe.
WINNING_LINES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


@dataclass(frozen=True)
class TicTacToeState:
    """Immutable game state with a 9-cell board and side to move."""

    board: Tuple[str, ...]
    current_player: str


# Returns the initial empty game state with X to move.
def initial_state() -> TicTacToeState:
    # Start with all empty squares and X as the first player.
    return TicTacToeState(board=(EMPTY,) * 9, current_player="X")


# Returns the opponent marker for a given player marker.
def other_player(player: str) -> str:
    # Tic-Tac-Toe only has two players, X and O.
    return "O" if player == "X" else "X"


# Returns a list of legal move indices for the given state.
def legal_moves(state: TicTacToeState) -> List[int]:
    # Any empty index is a legal move.
    return [i for i, cell in enumerate(state.board) if cell == EMPTY]


# Applies one move and returns the next immutable state.
def apply_move(state: TicTacToeState, move: int) -> TicTacToeState:
    # Validate index range first.
    if move < 0 or move > 8:
        raise ValueError(f"Move index out of range: {move}")

    # Reject moves played on occupied squares.
    if state.board[move] != EMPTY:
        raise ValueError(f"Illegal move {move}: square is not empty")

    # Copy board to a list, place the marker, then freeze back to tuple.
    new_board = list(state.board)
    new_board[move] = state.current_player
    return TicTacToeState(board=tuple(new_board), current_player=other_player(state.current_player))


# Returns the winner marker ("X" or "O"), or None if no winner exists yet.
def winner(state: TicTacToeState) -> Optional[str]:
    # Check every winning line for a non-empty triple.
    for a, b, c in WINNING_LINES:
        if state.board[a] != EMPTY and state.board[a] == state.board[b] == state.board[c]:
            return state.board[a]
    return None


# Returns True when the game is a draw (full board with no winner).
def is_draw(state: TicTacToeState) -> bool:
    # A draw requires no winner and no empty squares.
    return winner(state) is None and EMPTY not in state.board


# Returns True if the state is terminal (win or draw).
def is_terminal(state: TicTacToeState) -> bool:
    # Terminal states are either winning states or draws.
    return winner(state) is not None or is_draw(state)


# Returns a compact 3-line board string for readable logs.
def board_to_pretty_string(state: TicTacToeState) -> str:
    # Replace empty squares with dot for display clarity.
    view = [cell if cell != EMPTY else "." for cell in state.board]
    return "\n".join(
        [
            f"{view[0]} {view[1]} {view[2]}",
            f"{view[3]} {view[4]} {view[5]}",
            f"{view[6]} {view[7]} {view[8]}",
        ]
    )


# Builds a short board description used in LLM prompts.
def board_to_prompt_text(state: TicTacToeState) -> str:
    # Keep the prompt deterministic and easy for the model to parse.
    return (
        "Board (rows):\n"
        f"{board_to_pretty_string(state)}\n"
        f"Current player: {state.current_player}\n"
        "Coordinate mapping (0-8):\n"
        "0 1 2\n"
        "3 4 5\n"
        "6 7 8\n"
    )


# Parses a 9-char board string (X, O, .) into a Tic-Tac-Toe state.
def state_from_compact_string(board_text: str, current_player: str) -> TicTacToeState:
    # Normalize and validate the input length.
    cleaned = board_text.strip()
    if len(cleaned) != 9:
        raise ValueError("Board string must have exactly 9 characters")

    # Convert "." into EMPTY and keep X/O as-is.
    board_cells: List[str] = []
    for char in cleaned:
        if char == ".":
            board_cells.append(EMPTY)
        elif char in ("X", "O"):
            board_cells.append(char)
        else:
            raise ValueError("Board characters must be X, O, or .")

    # Validate player marker.
    if current_player not in ("X", "O"):
        raise ValueError("current_player must be 'X' or 'O'")

    return TicTacToeState(board=tuple(board_cells), current_player=current_player)
