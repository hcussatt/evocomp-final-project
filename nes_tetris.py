"""
nes_tetris.py — run NES Tetris headlessly or watch the original NES graphics.

Runs the original 1989 NES Tetris in the nes-py emulator (via gym-tetris, which ships
the ROM) and exposes what an EC loop needs:

    game = NESTetris(seed=7)
    game.column_heights()            # [0, 0, 3, 2, ...] read live from NES RAM
    game.current_piece, game.next_piece
    game.play_moves(["cw", "left", "left"])   # tap inputs, then soft-drop to lock
    game.place(rotation=1, x=3)               # or name the target placement
    game.legal_placements()                   # every reachable (rotation, x)
    game.simulate_placement(rotation, x)      # fast pure-Python lookahead
    snap = game.snapshot(); ...; game.restore(snap)   # exact emulator lookahead

Conventions
- rotation = number of clockwise (A button) presses from the spawn orientation, 0-3
- x       = the piece's pivot column (0-9) as stored by the game at RAM $0040
- Board row 0 is the top, row 19 the bottom; heights count up from the floor.

Usage:
  python nes_tetris.py                   # headless, fastest
  python nes_tetris.py --render          # watch the original NES graphics
  python nes_tetris.py --render --fps 120 --pieces 50

In your own Python code, use NESTetris(render=True), and call
set_render(False) or set_render(True) to switch at runtime.

Install:  pip install gym-tetris   (brings nes-py and the ROM)
  Python 3.13+ gets gym-tetris 4.x (recommended). On older Python pip installs the
  legacy 3.x stack, which also needs:  pip install "numpy<2"  (snapshot() unavailable)

RAM addresses used (verified against the running ROM):
  $0040/$0041  piece pivot X / Y        $0042  piece orientation id (0-18)
  $0060-$0062  player-1 copies of those (the game copies them into $40-$42 every frame)
  $0044        level                    $0048  play state (1 = piece under player control)
  $0050-$0051  lines (BCD)              $0053-$0055  score (BCD)
  $0058        game-over flag           $00BF  next piece orientation id
  $0400-$04C7  10x20 playfield, 0xEF = empty (the falling piece is NOT in this array)
"""

from __future__ import annotations

import contextlib
import copy
import io
import time

import numpy as np

# On Python < 3.13, pip installs the legacy stack (gym-tetris 3.x + nes-py 8.x + gym),
# whose import prints a "Gym has been unmaintained" notice. Hide that notice; the
# version check below handles the real incompatibility.
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from gym_tetris.tetris_env import TetrisEnv as _GymTetris
    from nes_py import NESEnv as _NESEnv

LEGACY_STACK = not hasattr(_NESEnv, "dump_state")   # nes-py < 9 (gym-tetris 3.x)
if LEGACY_STACK and int(np.__version__.split(".")[0]) >= 2:
    raise ImportError(
        "You have the legacy gym-tetris 3.x / nes-py 8.x stack, which crashes under "
        f"NumPy {np.__version__}. Fix with ONE of:\n"
        "  (recommended) use Python 3.13+, then:  pip install -U gym-tetris\n"
        '  (keep your Python)                     pip install "numpy<2"'
    )

# ---------------------------------------------------------------------------
# Tables extracted from the ROM (see derive script in the project notes)
# ---------------------------------------------------------------------------
EMPTY = 0xEF
BTN = {"right": 0x80, "left": 0x40, "down": 0x20, "up": 0x10,
       "start": 0x08, "select": 0x04, "B": 0x02, "A": 0x01}

# orientation id -> (piece kind, cell offsets (dy, dx) relative to the pivot)
ORIENTATIONS = {
    0: ("T", [(-1, 0), (0, -1), (0, 0), (0, 1)]),
    1: ("T", [(-1, 0), (0, 0), (0, 1), (1, 0)]),
    2: ("T", [(0, -1), (0, 0), (0, 1), (1, 0)]),
    3: ("T", [(-1, 0), (0, -1), (0, 0), (1, 0)]),
    4: ("J", [(-1, 0), (0, 0), (1, -1), (1, 0)]),
    5: ("J", [(-1, -1), (0, -1), (0, 0), (0, 1)]),
    6: ("J", [(-1, 0), (-1, 1), (0, 0), (1, 0)]),
    7: ("J", [(0, -1), (0, 0), (0, 1), (1, 1)]),
    8: ("Z", [(0, -1), (0, 0), (1, 0), (1, 1)]),
    9: ("Z", [(-1, 1), (0, 0), (0, 1), (1, 0)]),
    10: ("O", [(0, -1), (0, 0), (1, -1), (1, 0)]),
    11: ("S", [(0, 0), (0, 1), (1, -1), (1, 0)]),
    12: ("S", [(-1, 0), (0, 0), (0, 1), (1, 1)]),
    13: ("L", [(-1, 0), (0, 0), (1, 0), (1, 1)]),
    14: ("L", [(0, -1), (0, 0), (0, 1), (1, -1)]),
    15: ("L", [(-1, -1), (-1, 0), (0, 0), (1, 0)]),
    16: ("L", [(-1, 1), (0, -1), (0, 0), (0, 1)]),
    17: ("I", [(-2, 0), (-1, 0), (0, 0), (1, 0)]),
    18: ("I", [(0, -2), (0, -1), (0, 0), (0, 1)]),
}
ROTATE_CW = {0: 1, 1: 2, 2: 3, 3: 0, 4: 5, 5: 6, 6: 7, 7: 4, 8: 9, 9: 8, 10: 10,
             11: 12, 12: 11, 13: 14, 14: 15, 15: 16, 16: 13, 17: 18, 18: 17}
ROTATE_CCW = {0: 3, 1: 0, 2: 1, 3: 2, 4: 7, 5: 4, 6: 5, 7: 6, 8: 9, 9: 8, 10: 10,
              11: 12, 12: 11, 13: 16, 14: 13, 15: 14, 16: 15, 17: 18, 18: 17}
SPAWN_ID = {"T": 2, "J": 7, "Z": 8, "O": 10, "S": 11, "L": 14, "I": 18}
UNIQUE_ROTATIONS = {"T": 4, "J": 4, "L": 4, "Z": 2, "S": 2, "I": 2, "O": 1}
SPAWN_X, SPAWN_Y = 5, 0


# A loaded nes-py snapshot keeps a raw pointer to the ROM data of the emulator it was
# saved from. If that emulator were garbage-collected, any game that loaded the
# snapshot would run on freed memory and crash ("failed to execute opcode",
# segfault). Each emulator therefore holds a reference to the emulator whose ROM it
# currently points at, which keeps that one alive.
def _rom_owner(env):
    return env.__dict__.get("_rom_owner", env)


def _rom_owner_after_reset(env):
    return env.__dict__.get("_backup_owner", env)


class NESTetris:
    """NES Tetris with optional live graphics and turn-level control."""

    MAX_WAIT_FRAMES = 3000  # safety bound for any wait loop

    def __init__(self, seed: int | None = None, level: int = 0,
                 render: bool = False, render_fps: float = 60,
                 render_scale: int = 3):
        """Create the game. Set render=True to watch the original NES screen.

        render_fps=60 runs at the NES's normal speed; 0 removes the FPS cap.
        render_scale changes the initial window size (e.g., 3 = 768x720).
        None of these options change the game logic in headless mode.
        """
        if render_fps < 0:
            raise ValueError("render_fps must be nonnegative (0 means uncapped)")
        if render_scale < 1:
            raise ValueError("render_scale must be at least 1")
        self.render_enabled = bool(render)
        self.render_fps = render_fps
        self.render_scale = int(render_scale)
        self._next_render_time = None
        self._render_scaled = False
        # nes-py 9+ uses render_mode at creation; nes-py 8 uses render(mode=...).
        options = {} if LEGACY_STACK else {
            "render_mode": "human" if render else None
        }
        self.env = _GymTetris(deterministic=False, **options)
        if not LEGACY_STACK and NESTetris._BOOT is None:
            # A private emulator holding the post-title-screen state that reset()
            # returns to, so copies made with copy.deepcopy() can reset() too.
            owner = _GymTetris(deterministic=False)
            owner._restore()
            NESTetris._BOOT, NESTetris._BOOT_OWNER = owner.dump_state(), owner
        self.level_start = level
        self.reset(seed)

    # ---------------------------------------------------------- live graphics
    def set_render(self, enabled: bool, fps: float | None = None):
        """Turn the viewer on/off while the game is running.

        Only call on the main thread; the NES viewer uses pyglet/OpenGL.
        """
        if fps is not None:
            if fps < 0:
                raise ValueError("fps must be nonnegative")
            self.render_fps = fps
        self.render_enabled = bool(enabled)
        self._next_render_time = None
        if not LEGACY_STACK:
            self.env.render_mode = "human" if enabled else None
        if enabled:
            self._show_frame()
        else:
            viewer = getattr(self.env, "viewer", None)
            if viewer is not None:
                viewer.close()
                self.env.viewer = None
            self._render_scaled = False

    def _show_frame(self):
        """Show one actual emulator frame; completely skipped when headless."""
        if not self.render_enabled:
            return
        if self.render_fps > 0:
            now = time.perf_counter()
            if self._next_render_time is not None and now < self._next_render_time:
                time.sleep(self._next_render_time - now)
            self._next_render_time = time.perf_counter() + 1 / self.render_fps
        if LEGACY_STACK:
            self.env.render(mode="human")
        else:
            self.env.render()
        # The nes-py viewer defaults to 256x240. Enlarge the window once;
        # its image blitter automatically scales subsequent frames.
        if not self._render_scaled:
            viewer = getattr(self.env, "viewer", None)
            window = getattr(viewer, "_window", None)
            if window is not None:
                window.set_size(256 * self.render_scale, 240 * self.render_scale)
            self._render_scaled = True

    # ------------------------------------------------------------------ setup
    def reset(self, seed: int | None = None):
        """Start a new A-type game. The same seed gives the same piece sequence."""
        self.env.reset(seed=seed)
        self.env._rom_owner = _rom_owner_after_reset(self.env)
        if self.level_start:
            self._poke(0x44, self.level_start)
            self._poke(0x64, self.level_start)
        self.frames = 0
        self.pieces_placed = 0
        self._next_render_time = None
        self._wait_for_control()
        self._show_frame()
        return self.get_state()

    # ----------------------------------------------------------- RAM helpers
    @property
    def ram(self) -> np.ndarray:
        return self.env.ram  # always re-read: snapshots can swap the buffer

    def _poke(self, addr: int, value: int):
        self.env.ram[addr] = value

    def _bcd(self, addr: int, n: int) -> int:
        v = 0
        for i in range(n):
            b = int(self.ram[addr + i])
            v += (10 ** (2 * i + 1)) * (b >> 4) + (10 ** (2 * i)) * (b & 0x0F)
        return v

    # --------------------------------------------------------------- reading
    def board(self) -> np.ndarray:
        """20x10 bool array of locked cells (row 0 = top). Excludes the falling piece."""
        return self.ram[0x400:0x4C8].reshape(20, 10) != EMPTY

    def column_heights(self, board: np.ndarray | None = None) -> np.ndarray:
        b = self.board() if board is None else board
        has = b.any(axis=0)
        return np.where(has, 20 - np.argmax(b, axis=0), 0)

    def holes(self, board: np.ndarray | None = None) -> int:
        b = self.board() if board is None else board
        return int((np.maximum.accumulate(b, axis=0) & ~b).sum())

    def features(self, board: np.ndarray | None = None) -> dict:
        h = self.column_heights(board)
        return {
            "heights": h.tolist(),
            "aggregate_height": int(h.sum()),
            "max_height": int(h.max()),
            "holes": self.holes(board),
            "bumpiness": int(np.abs(np.diff(h)).sum()),
        }

    @property
    def orientation_id(self) -> int:
        return int(self.ram[0x42])

    @property
    def current_piece(self) -> str | None:
        o = ORIENTATIONS.get(self.orientation_id)
        return o[0] if o else None

    @property
    def next_piece(self) -> str | None:
        o = ORIENTATIONS.get(int(self.ram[0xBF]))
        return o[0] if o else None

    @property
    def piece_xy(self) -> tuple[int, int]:
        return int(self.ram[0x40]), int(self.ram[0x41])

    @property
    def lines(self) -> int:
        return self._bcd(0x50, 2)

    @property
    def score(self) -> int:
        return self._bcd(0x53, 3)

    @property
    def level(self) -> int:
        return int(self.ram[0x44])

    @property
    def game_over(self) -> bool:
        return bool(self.ram[0x58])

    def get_state(self) -> dict:
        return {
            "board": self.board(),
            "column_heights": self.column_heights().tolist(),
            "current": self.current_piece,
            "next": self.next_piece,
            "piece_xy": self.piece_xy,
            "lines": self.lines,
            "score": self.score,
            "level": self.level,
            "pieces_placed": self.pieces_placed,
            "frames": self.frames,
            "game_over": self.game_over,
        }

    # -------------------------------------------------------- frame control
    def frame(self, *buttons: str):
        """Advance exactly one frame holding the given buttons (e.g. "left", "A")."""
        mask = 0
        for b in buttons:
            mask |= BTN[b]
        self.env._frame_advance(mask)
        self.frames += 1
        self._show_frame()

    def tap(self, action: str):
        """One tap = 1 frame pressed + 1 frame released (NES reads new presses).
        action: "left", "right", "cw" (A), "ccw" (B), or "noop"."""
        btn = {"left": "left", "right": "right", "cw": "A", "ccw": "B", "noop": None}[action]
        if btn:
            self.frame(btn)
        else:
            self.frame()
        self.frame()

    def _active(self) -> bool:
        return int(self.ram[0x48]) == 1

    def _wait_for_control(self):
        for _ in range(self.MAX_WAIT_FRAMES):
            if self.game_over or self._active():
                return
            self.frame()
        raise RuntimeError("timed out waiting for the next piece")

    # --------------------------------------------------------- turn control
    def play_moves(self, moves: list[str]) -> dict:
        """Tap each input for the current piece, then soft-drop until it locks and the
        next piece is under control. Returns what happened this turn."""
        if self.game_over:
            return {"lines_cleared": 0, "game_over": True}
        lines_before = self.lines
        for m in moves:
            if m == "drop":
                break
            self.tap(m)
            if not self._active():
                break  # piece locked early (high gravity)
        # soft drop: hold down until the piece leaves player control
        for _ in range(self.MAX_WAIT_FRAMES):
            if not self._active() or self.game_over:
                break
            self.frame("down")
        self._wait_for_control()  # buttons released, so 'down' re-triggers next piece
        self.pieces_placed += 1
        return {"lines_cleared": self.lines - lines_before, "game_over": self.game_over}

    def moves_for(self, rotation: int, x: int) -> list[str]:
        rotation %= 4
        moves = ["ccw"] if rotation == 3 else ["cw"] * rotation
        dx = x - self.piece_xy[0]
        moves += ["right"] * dx if dx > 0 else ["left"] * (-dx)
        return moves

    def place(self, rotation: int, x: int, teleport: bool = False) -> dict:
        """Put the current piece at (rotation, pivot x) and drop it.
        teleport=False: press real buttons (faithful to the game, incl. blocked moves).
        teleport=True : write the resolved landing orientation/X/Y into RAM, so the game
                        locks it within a frame or two (a hard drop). Several times
                        fewer frames per piece and immune to high-level gravity; line
                        clears, scoring, RNG and spawning are still the ROM's own code.
                        Unreachable targets fall back to button presses."""
        if teleport and not self.game_over:
            res = self._resolve(rotation, x)
            if res is not None:
                oid, px, py = res
                for a, v in ((0x42, oid), (0x62, oid), (0x40, px), (0x60, px),
                             (0x41, py), (0x61, py)):
                    self._poke(a, v)
                return self.play_moves([])
        return self.play_moves(self.moves_for(rotation, x))

    # -------------------------------------------------------- lookahead
    def _orientation_after(self, rotation: int) -> int:
        oid = SPAWN_ID[self.current_piece]
        for _ in range(rotation % 4):
            oid = ROTATE_CW[oid]
        return oid

    @staticmethod
    def _fits(board, oid, x, y) -> bool:
        for dy, dx in ORIENTATIONS[oid][1]:
            r, c = y + dy, x + dx
            if c < 0 or c > 9 or r > 19:
                return False
            if r >= 0 and board[r, c]:
                return False
        return True

    def _resolve(self, rotation: int, x: int, board=None):
        """Python model of rotate -> slide -> drop from spawn. Returns (oid, x, y) of the
        lock position, or None if the target is unreachable. NES Tetris has no wall
        kicks: a blocked rotation or shift simply fails."""
        b = self.board() if board is None else board
        oid = SPAWN_ID[self.current_piece]
        px, py = SPAWN_X, SPAWN_Y
        steps = [ROTATE_CCW] if rotation % 4 == 3 else [ROTATE_CW] * (rotation % 4)
        for table in steps:
            if not self._fits(b, table[oid], px, py):
                return None
            oid = table[oid]
        while px != x:
            nx = px + (1 if x > px else -1)
            if not self._fits(b, oid, nx, py):
                return None
            px = nx
        while self._fits(b, oid, px, py + 1):
            py += 1
        return oid, px, py

    def legal_placements(self) -> list[tuple[int, int]]:
        """Every distinct (rotation, x) the current piece can reach."""
        if self.game_over or self.current_piece is None:
            return []
        b = self.board()
        return [(rot, x)
                for rot in range(UNIQUE_ROTATIONS[self.current_piece])
                for x in range(10)
                if self._resolve(rot, x, b) is not None]

    def simulate_placement(self, rotation: int, x: int):
        """(board_after, lines_cleared, topped_out) without touching the emulator,
        or None if unreachable. Matches place() at the speeds where taps aren't
        interrupted by gravity (verified against the ROM at level 0)."""
        b = self.board()
        res = self._resolve(rotation, x, b)
        if res is None:
            return None
        oid, px, py = res
        nb = b.copy()
        topped = False
        for dy, dx in ORIENTATIONS[oid][1]:
            r = py + dy
            if r < 0:
                topped = True
            else:
                nb[r, px + dx] = True
        full = nb.all(axis=1)
        n = int(full.sum())
        if n:
            nb = np.vstack([np.zeros((n, 10), bool), nb[~full]])
        return nb, n, topped

    # -------------------------------------------------- emulator snapshots
    def snapshot(self):
        """Exact emulator state (incl. RNG), restorable within this process.
        Needs nes-py 9+ (gym-tetris 4.x, Python 3.13+)."""
        if LEGACY_STACK:
            raise NotImplementedError(
                "snapshot()/restore() need gym-tetris 4.x (Python 3.13+). The legacy "
                "emulator has only one save slot, which reset() already uses. "
                "simulate_placement() works on both stacks for lookahead.")
        return (self.env.dump_state(), self.frames, self.pieces_placed,
                _rom_owner(self.env))

    def restore(self, snap):
        """Load a snapshot from this game or from any other NESTetris game."""
        state, self.frames, self.pieces_placed, owner = snap
        self.env.load_state(state)
        self.env._rom_owner = owner   # see _rom_owner() below
        self._show_frame()

    _BOOT = None        # shared post-title-screen emulator state (see __init__)
    _BOOT_OWNER = None  # the private emulator that state came from
    _COPY_ATTRS = ("_b_type", "_reward_score", "_current_score", "_reward_lines",
                   "_current_lines", "_penalize_height", "_current_height",
                   "deterministic")

    def __deepcopy__(self, memo):
        """copy.deepcopy(game) gives an independent game in the exact same state
        (board, piece, RNG). The emulator itself can't be deep-copied, so this builds
        a fresh one and loads this game's state into it (about 1 ms)."""
        if LEGACY_STACK:
            raise NotImplementedError("copying a game needs gym-tetris 4.x (Python 3.13+)")
        env = _GymTetris.__new__(_GymTetris)
        _NESEnv.__init__(env, self.env._rom_path)       # bare emulator, no boot-up
        for k in self._COPY_ATTRS:
            setattr(env, k, getattr(self.env, k))
        env.np_random = copy.deepcopy(self.env.np_random, memo)
        env.load_state(NESTetris._BOOT)                  # give the copy its reset point
        env._backup()
        env._backup_owner = NESTetris._BOOT_OWNER
        env.load_state(self.env.dump_state())            # then this game's current state
        env._rom_owner = _rom_owner(self.env)
        new = object.__new__(NESTetris)
        new.env = env
        new.level_start = self.level_start
        new.frames = self.frames
        new.pieces_placed = self.pieces_placed
        # Clones are simulation workers, not additional GUI windows.
        new.render_enabled = False
        new.render_fps = self.render_fps
        new.render_scale = self.render_scale
        new._next_render_time = None
        new._render_scaled = False
        memo[id(self)] = new
        return new

    def clone(self) -> "NESTetris":
        """Same as copy.deepcopy(game)."""
        return copy.deepcopy(self)

    def close(self):
        self.env.close()

    # ---------------------------------------------------------- debugging
    def render_text(self) -> str:
        b = self.board()
        grid = [["#" if v else "." for v in row] for row in b]
        if self._active() and self.current_piece:
            x, y = self.piece_xy
            for dy, dx in ORIENTATIONS[self.orientation_id][1]:
                if 0 <= y + dy < 20 and 0 <= x + dx < 10:
                    grid[y + dy][x + dx] = "@"
        rows = ["|" + "".join(r) + "|" for r in grid]
        rows.append("+----------+")
        rows.append(" " + "".join(str(h % 10) for h in self.column_heights()))
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# Example fitness function: a weighted-feature genome playing the real ROM
# ---------------------------------------------------------------------------
def evaluate_genome(weights: dict, seed: int = 0, max_pieces: int = 300,
                    teleport: bool = False, game: NESTetris | None = None) -> dict:
    g = game or NESTetris(seed=seed)
    if game:
        g.reset(seed)
    while not g.game_over and g.pieces_placed < max_pieces:
        best, best_s = None, -float("inf")
        for rot, x in g.legal_placements():
            board, lines, topped = g.simulate_placement(rot, x)
            if topped:
                continue
            f = g.features(board)
            s = weights.get("lines", 0) * lines + sum(
                weights.get(k, 0) * f[k]
                for k in ("aggregate_height", "holes", "bumpiness", "max_height"))
            if s > best_s:
                best, best_s = (rot, x), s
        if best is None:
            break
        g.place(*best, teleport=teleport)
    return {"lines": g.lines, "score": g.score, "pieces": g.pieces_placed,
            "frames": g.frames}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NES Tetris (headless or graphical)")
    parser.add_argument("--render", action="store_true",
                        help="show the original NES game while the agent plays")
    parser.add_argument("--fps", type=float, default=60,
                        help="viewer FPS (60 = normal speed; 0 = uncapped)")
    parser.add_argument("--scale", type=int, default=3,
                        help="initial viewer window size multiplier (default: 3)")
    parser.add_argument("--pieces", type=int, default=200,
                        help="maximum pieces for the greedy example")
    parser.add_argument("--seed", type=int, default=3,
                        help="random seed for the greedy example")
    args = parser.parse_args()

    if args.fps < 0 or args.scale < 1 or args.pieces < 1:
        parser.error("fps must be >= 0, scale >= 1, and pieces >= 1")

    w = {"aggregate_height": -0.51, "lines": 0.76, "holes": -0.36,
         "bumpiness": -0.18}
    g = NESTetris(seed=args.seed, render=args.render,
                  render_fps=args.fps, render_scale=args.scale)
    try:
        print("current:", g.current_piece, "next:", g.next_piece)
        start = time.time()
        result = evaluate_genome(w, seed=args.seed, max_pieces=args.pieces, game=g)
        print("greedy agent:", result, f"{time.time() - start:.1f}s")
        if args.render:
            try:
                input("Finished. Press Enter to close the NES window... ")
            except EOFError:
                pass
        else:
            print(g.render_text())
            print("column heights:", g.column_heights().tolist())
    finally:
        g.close()