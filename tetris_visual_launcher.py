
import argparse
import time

from nes_tetris import NESTetris, ORIENTATIONS, evaluate_genome


class VisualTetris(NESTetris):
    COLORS = {
        "T": "#ba75ff",
        "J": "#648aff",
        "Z": "#ff667b",
        "O": "#ffd65c",
        "S": "#68df9a",
        "L": "#ffa75b",
        "I": "#60d9ef",
    }

    def __init__(self, seed=1, level=0, fps=30, render_every=2):
        import tkinter as tk

        self.tk = tk
        self.window = tk.Tk()
        self.window.title("NES Tetris - Live Board")
        self.window.resizable(False, False)

        self.canvas = tk.Canvas(
            self.window,
            width=360,
            height=548,
            bg="#101827",
            highlightthickness=0,
        )
        self.canvas.pack()
        self.window.protocol("WM_DELETE_WINDOW", self.hide_window)

        self._render_ready = False
        self._window_closed = False
        self._last_draw = None
        self.fps = max(1, float(fps))
        self.render_every = max(1, int(render_every))

        super().__init__(seed=seed, level=level)

        self._render_ready = True
        self.draw()

    def hide_window(self):
        self._window_closed = True
        self.window.destroy()

    def frame(self, *buttons):
        super().frame(*buttons)

        if (
            self._render_ready
            and not self._window_closed
            and self.frames % self.render_every == 0
        ):
            self.draw(pace=True)

    def draw(self, pace=False):
        if self._window_closed:
            return

        if pace and self._last_draw is not None:
            delay = 1.0 / self.fps - (
                time.monotonic() - self._last_draw
            )
            time.sleep(max(0.0, delay))

        c = self.canvas
        c.delete("all")

        c.create_text(
            16, 20,
            text="NES TETRIS - live board",
            anchor="w",
            fill="white",
            font=("Arial", 12, "bold"),
        )

        board = self.board()

        # Draw the 10x20 playfield.
        for row in range(20):
            for col in range(10):
                x = 16 + col * 24
                y = 48 + row * 24

                color = (
                    "#8da4bc" if board[row, col]
                    else "#1c2a3c"
                )

                c.create_rectangle(
                    x, y, x + 22, y + 22,
                    width=0,
                    fill=color,
                )

        # Draw the active falling piece.
        if self._active() and self.current_piece in self.COLORS:
            px, py = self.piece_xy

            for dy, dx in ORIENTATIONS[self.orientation_id][1]:
                row = py + dy
                col = px + dx

                if 0 <= row < 20 and 0 <= col < 10:
                    x = 16 + col * 24
                    y = 48 + row * 24

                    c.create_rectangle(
                        x, y, x + 22, y + 22,
                        width=0,
                        fill=self.COLORS[self.current_piece],
                    )

        # Sidebar statistics.
        labels = (
            f"Score: {self.score}",
            f"Lines: {self.lines}",
            f"Level: {self.level}",
            f"Next: {self.next_piece}",
            f"Pieces: {self.pieces_placed}",
            f"Frames: {self.frames}",
        )

        for i, label in enumerate(labels):
            c.create_text(
                266, 56 + 30 * i,
                text=label,
                anchor="w",
                fill="white",
                font=("Arial", 9),
            )

        if self.game_over:
            c.create_text(
                266, 256,
                text="GAME OVER",
                anchor="w",
                fill="#ff667b",
            )

        try:
            self.window.update_idletasks()
            self.window.update()
        except self.tk.TclError:
            self._window_closed = True

        self._last_draw = time.monotonic()

    def close(self):
        if not self._window_closed:
            self.hide_window()
        super().close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--render", action="store_true")
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--render-every", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--max-pieces", type=int, default=200)

    args = parser.parse_args()

    weights = {
        "aggregate_height": -0.51,
        "lines": 0.76,
        "holes": -0.36,
        "bumpiness": -0.18,
    }

    if args.render:
        game = VisualTetris(
            seed=args.seed,
            fps=args.fps,
            render_every=args.render_every,
        )
    else:
        game = NESTetris(seed=args.seed)

    try:
        result = evaluate_genome(
            weights,
            seed=args.seed,
            max_pieces=args.max_pieces,
            game=game,
        )
        print(result)
    finally:
        game.close()


if __name__ == "__main__":
    main()
