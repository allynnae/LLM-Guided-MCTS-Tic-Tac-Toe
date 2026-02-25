"""Minimal CLI entrypoint for baseline and LLM-guided MCTS move selection."""

from __future__ import annotations

import argparse

from mcts import BaselineMCTS, LLMGuidedMCTS, LLMPositionEvaluator
from mcts_tictactoe import (
    board_to_pretty_string,
    initial_state,
    legal_moves,
    state_from_compact_string,
)


# Parses command-line arguments for one MCTS move-selection run.
def parse_args() -> argparse.Namespace:
    # Build a small CLI focused on this assignment's two agents.
    parser = argparse.ArgumentParser(description="Run baseline or LLM-guided MCTS for Tic-Tac-Toe.")
    parser.add_argument("--agent", choices=["baseline", "llm"], default="baseline")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--exploration", type=float, default=1.41421356237)
    parser.add_argument("--seed", type=int, default=None)

    # Optional board setup for non-initial states.
    parser.add_argument("--board", type=str, default=None, help="9-char board: X, O, .")
    parser.add_argument("--current-player", choices=["X", "O"], default="X")

    # LLM-specific options.
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    return parser.parse_args()


# Builds either the baseline MCTS agent or the LLM-guided variant.
def build_agent(args: argparse.Namespace):
    # Construct baseline agent with random rollouts.
    if args.agent == "baseline":
        return BaselineMCTS(iterations=args.iterations, exploration_constant=args.exploration, seed=args.seed)

    # Construct LLM evaluator and LLM-guided MCTS agent.
    evaluator = LLMPositionEvaluator(provider=args.provider, model=args.model)
    return LLMGuidedMCTS(
        evaluator=evaluator,
        iterations=args.iterations,
        exploration_constant=args.exploration,
        seed=args.seed,
    )


# Builds a state from CLI inputs, defaulting to the standard empty board.
def build_state(args: argparse.Namespace):
    # Use initial board when no custom board is supplied.
    if args.board is None:
        return initial_state()
    return state_from_compact_string(args.board, args.current_player)


# Runs one move-selection pass and prints the result.
def main() -> None:
    # Parse inputs and build requested state/agent.
    args = parse_args()
    state = build_state(args)
    agent = build_agent(args)

    # Show current board and legal moves for transparency.
    print("Current board:")
    print(board_to_pretty_string(state))
    print(f"Current player: {state.current_player}")
    print(f"Legal moves: {legal_moves(state)}")

    # Ask the agent for one move decision.
    move = agent.choose_move(state)
    print(f"Chosen move: {move}")


# Standard script entrypoint.
if __name__ == "__main__":
    main()
