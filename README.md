# LLM-Guided Monte Carlo Tree Search for Tic-Tac-Toe

This project contains two Tic-Tac-Toe MCTS agents in Python:

- Baseline MCTS with random rollouts in the Simulation phase.
- LLM-guided MCTS that replaces random rollout with direct LLM position evaluation (RAP-inspired).

## Files

- `mcts_tictactoe.py`: game state and board helper functions.
- `mcts.py`: baseline MCTS, LLM-guided MCTS, and LLM evaluator.
- `minimax_agent.py`: simple minimax reference agent.
- `run_experiments.py`: minimal CLI to run one MCTS move selection.
- `tictactoe_gui.py`: Tkinter GUI to play against baseline or LLM-guided MCTS.

## Setup

Use Python 3.10+.

For OpenAI:

```bash
export OPENAI_API_KEY="your_key_here"
```

For Anthropic:

```bash
export ANTHROPIC_API_KEY="your_key_here"
```

## Usage

Baseline MCTS (random rollout):

```bash
python run_experiments.py --agent baseline --iterations 1000
```

LLM-guided MCTS (direct position evaluation):

```bash
python run_experiments.py --agent llm --provider openai --model gpt-4o-mini --iterations 200
```

LLM-guided MCTS with Anthropic:

```bash
python run_experiments.py --agent llm --provider anthropic --model claude-3-5-sonnet-latest --iterations 200
```

Optional board input uses 9 characters (`X`, `O`, or `.`), row-major order.
Example: `XO...O..X`

```bash
python run_experiments.py --agent baseline --board XO...O..X --current-player X
```

## GUI

Run the GUI:

```bash
python tictactoe_gui.py
```

In the GUI:
- You play as `X`.
- AI plays as `O`.
- Choose `baseline` for random-rollout MCTS.
- Choose `llm` for LLM evaluation (requires API key in environment).

## AI Usage 

Codex was used in the development of this code. Prompts include:
- Can you help develop a simple TicTacToe GUI?
- How would you create a class for a single tree node used by MCTS search?
