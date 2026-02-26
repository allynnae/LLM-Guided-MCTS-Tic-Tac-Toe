"""Simple Tkinter GUI for playing Tic-Tac-Toe against MCTS."""

from __future__ import annotations

import math
import threading
import tkinter as tk
from tkinter import messagebox

from mcts import BaselineMCTS, LLMGuidedMCTS, LLMPositionEvaluator
from mcts_tictactoe import EMPTY, apply_move, initial_state, is_terminal, legal_moves, winner


class TicTacToeGUI:
    """Small GUI app where a human plays against an MCTS agent."""

    # Creates the window, control widgets, and board buttons.
    def __init__(self) -> None:
        # Create the top-level Tk window.
        self.root = tk.Tk()
        self.root.title("Tic-Tac-Toe (MCTS)")
        # Register an explicit close handler for the window X button.
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Keep game state in memory; human is X and goes first.
        self.state = initial_state()
        self.human_player = "X"
        self.ai_player = "O"
        # Track whether an AI computation is in progress.
        self.ai_thinking = False
        # Track whether the window is closing to ignore late callbacks.
        self.is_closing = False
        # Invalidate stale background results when game resets.
        self.ai_request_id = 0

        # Variables connected to the agent controls.
        self.agent_var = tk.StringVar(value="baseline")
        self.iterations_var = tk.StringVar(value="100")
        self.provider_var = tk.StringVar(value="openai")
        self.model_var = tk.StringVar(value="gpt-4o-mini")
        self.status_var = tk.StringVar(value="Your turn (X).")

        # Build the top controls and the main board.
        self._build_controls()
        self._build_board()

        # Draw the starting board state.
        self._refresh_board()

    # Builds dropdowns/entries for selecting agent settings.
    def _build_controls(self) -> None:
        # Put controls in a single top frame.
        controls = tk.Frame(self.root, padx=10, pady=10)
        controls.pack(fill="x")

        # Agent type control.
        tk.Label(controls, text="Agent:").grid(row=0, column=0, sticky="w")
        tk.OptionMenu(controls, self.agent_var, "baseline", "llm").grid(row=0, column=1, sticky="w")

        # Iteration count control.
        tk.Label(controls, text="Iterations:").grid(row=0, column=2, padx=(12, 0), sticky="w")
        tk.Entry(controls, textvariable=self.iterations_var, width=8).grid(row=0, column=3, sticky="w")

        # LLM provider control.
        tk.Label(controls, text="Provider:").grid(row=1, column=0, pady=(8, 0), sticky="w")
        tk.OptionMenu(controls, self.provider_var, "openai", "anthropic").grid(row=1, column=1, pady=(8, 0), sticky="w")

        # LLM model control.
        tk.Label(controls, text="Model:").grid(row=1, column=2, padx=(12, 0), pady=(8, 0), sticky="w")
        tk.Entry(controls, textvariable=self.model_var, width=24).grid(row=1, column=3, pady=(8, 0), sticky="w")

        # Reset button to start a new game quickly.
        tk.Button(controls, text="New Game", command=self._reset_game).grid(row=0, column=4, rowspan=2, padx=(16, 0))

        # Status line for turn prompts and errors.
        tk.Label(self.root, textvariable=self.status_var, anchor="w", padx=10).pack(fill="x")

    # Builds a 3x3 grid of clickable buttons for board cells.
    def _build_board(self) -> None:
        # Board frame keeps the 9 cells grouped and centered.
        board_frame = tk.Frame(self.root, padx=10, pady=10)
        board_frame.pack()

        # Create one button per square and store for refresh updates.
        self.cell_buttons = []
        for idx in range(9):
            # Bind each button to its board index.
            button = tk.Button(
                board_frame,
                text="",
                width=4,
                height=2,
                font=("Arial", 24),
                command=lambda i=idx: self._on_human_click(i),
            )
            button.grid(row=idx // 3, column=idx % 3, padx=3, pady=3)
            self.cell_buttons.append(button)

    # Handles human clicks and triggers AI move if the game is still active.
    def _on_human_click(self, move: int) -> None:
        # Ignore clicks once game is over.
        if is_terminal(self.state):
            return

        # Ignore clicks while AI is still thinking.
        if self.ai_thinking:
            return

        # Ignore clicks when it is not the human's turn.
        if self.state.current_player != self.human_player:
            return

        # Ignore illegal clicks on occupied cells.
        if move not in legal_moves(self.state):
            return

        # Apply the human move and redraw.
        self.state = apply_move(self.state, move)
        self._refresh_board()

        # End immediately if human just finished the game.
        if is_terminal(self.state):
            self._show_game_result()
            return

        # Start AI move computation after status text updates.
        self.status_var.set("AI thinking...")
        self.ai_thinking = True
        self._refresh_board()
        self.root.after(10, self._make_ai_move)

    # Builds and returns the currently selected agent instance.
    def _build_agent(self):
        # Parse iterations with a safe fallback.
        try:
            iterations = int(self.iterations_var.get())
        except ValueError:
            iterations = 100
            self.iterations_var.set("100")

        # Enforce at least one iteration.
        iterations = max(1, iterations)

        # Baseline uses random rollouts and does not call any API.
        if self.agent_var.get() == "baseline":
            return BaselineMCTS(iterations=iterations, exploration_constant=math.sqrt(2))

        # LLM mode uses direct position evaluation for simulation replacement.
        provider = self.provider_var.get().strip().lower()
        model = self.model_var.get().strip() or "gpt-4o-mini"
        evaluator = LLMPositionEvaluator(provider=provider, model=model)
        return LLMGuidedMCTS(evaluator=evaluator, iterations=iterations, exploration_constant=math.sqrt(2))

    # Computes and applies the AI move based on selected agent settings.
    def _make_ai_move(self) -> None:
        # Capture current state snapshot and request id for this worker.
        state_snapshot = self.state
        request_id = self.ai_request_id + 1
        self.ai_request_id = request_id

        # Build agent on main thread (Tk variables are not thread-safe).
        try:
            agent = self._build_agent()
        except Exception as exc:
            self.ai_thinking = False
            messagebox.showerror("Agent Error", str(exc))
            self.status_var.set("Could not make AI move. Check settings.")
            self._refresh_board()
            return

        # Compute AI move in a thread so UI remains responsive.
        worker = threading.Thread(
            target=self._compute_ai_move_worker,
            args=(agent, state_snapshot, request_id),
            daemon=True,
        )
        worker.start()

    # Computes AI move in background and sends result back to Tk thread.
    def _compute_ai_move_worker(self, agent, state_snapshot, request_id: int) -> None:
        # Choose move off the UI thread.
        try:
            ai_move = agent.choose_move(state_snapshot)
            error_text = None
        except Exception as exc:
            ai_move = None
            error_text = str(exc)

        # Marshal result safely to main thread if app is still open.
        if not self.is_closing:
            try:
                self.root.after(0, lambda: self._finish_ai_move(request_id, state_snapshot, ai_move, error_text))
            except tk.TclError:
                # Window may have closed between check and callback scheduling.
                return

    # Applies background AI result in the Tk thread if result is still valid.
    def _finish_ai_move(self, request_id: int, state_snapshot, ai_move, error_text: str | None) -> None:
        # Ignore callbacks after close or from old/stale requests.
        if self.is_closing or request_id != self.ai_request_id:
            return

        # Clear thinking flag first so UI can be interacted with again.
        self.ai_thinking = False

        # Show an error if AI computation failed.
        if error_text is not None:
            messagebox.showerror("Agent Error", error_text)
            self.status_var.set("Could not make AI move. Check settings.")
            self._refresh_board()
            return

        # Ignore result if game state changed while AI was thinking.
        if self.state != state_snapshot:
            self._refresh_board()
            return

        # Apply AI move and redraw.
        self.state = apply_move(self.state, ai_move)
        self._refresh_board()

        # Announce result if game ended, otherwise hand turn back to human.
        if is_terminal(self.state):
            self._show_game_result()
        else:
            self.status_var.set("Your turn (X).")

    # Updates button text/states so UI matches the current game state.
    def _refresh_board(self) -> None:
        # A terminal board locks all buttons.
        game_over = is_terminal(self.state)
        # While AI is thinking, lock board interaction.
        board_locked = game_over or self.ai_thinking

        # Refresh each cell from the state tuple.
        for idx, button in enumerate(self.cell_buttons):
            marker = self.state.board[idx]

            # Show empty cells as blank text.
            button_text = "" if marker == EMPTY else marker
            button.configure(text=button_text)

            # Enable only legal human moves during active human turn.
            if (not board_locked) and (self.state.current_player == self.human_player) and (idx in legal_moves(self.state)):
                button.configure(state="normal")
            else:
                button.configure(state="disabled")

    # Displays the final result message when a game reaches terminal state.
    def _show_game_result(self) -> None:
        # Determine winner marker, or draw when no winner.
        game_winner = winner(self.state)
        if game_winner is None:
            result_text = "Draw."
        elif game_winner == self.human_player:
            result_text = "You win."
        else:
            result_text = "AI wins."

        # Update status line and show a popup result.
        self.status_var.set(result_text)
        messagebox.showinfo("Game Over", result_text)

    # Resets state and UI so a new game can begin.
    def _reset_game(self) -> None:
        # Reinitialize to empty board with human turn first.
        self.ai_request_id += 1
        self.ai_thinking = False
        self.state = initial_state()
        self.status_var.set("Your turn (X).")
        self._refresh_board()

    # Handles a window close request from the title-bar X button.
    def _on_close(self) -> None:
        # Mark closing and invalidate pending AI callbacks.
        self.is_closing = True
        self.ai_request_id += 1
        # Terminate Tk loop and destroy window immediately.
        self.root.quit()
        self.root.destroy()

    # Starts the Tk event loop.
    def run(self) -> None:
        # Enter GUI main loop until the window is closed.
        self.root.mainloop()


# Creates and runs the GUI application.
def main() -> None:
    # Keep the script entrypoint very small.
    app = TicTacToeGUI()
    app.run()


# Standard script entrypoint.
if __name__ == "__main__":
    main()
