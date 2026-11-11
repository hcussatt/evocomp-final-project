"""Display every real controller attempt. No retries or hidden candidate trials."""
import time
from nes_tetris import ORIENTATIONS

class BoardViewer:
    def __init__(self, fps=60):
        import tkinter as tk
        self.tk = tk
        self.root = tk.Tk()
        self.root.title('Tetris controller — one continuing game')
        self.root.resizable(False, False)
        self.canvas = tk.Canvas(self.root, width=560, height=550, bg='#101827', highlightthickness=0)
        self.canvas.pack()
        self.tiles = [self.canvas.create_rectangle(16+c*24, 40+r*24, 38+c*24, 62+r*24,
                       fill='#1c2a3c', width=0) for r in range(20) for c in range(10)]
        self.label = self.canvas.create_text(270, 45, anchor='nw', fill='white', width=275, font=('Arial',11))
        self.status_id = self.canvas.create_text(16, 532, anchor='w', fill='white', font=('Arial',10))
        self.last_error = None
        self.target = None
        self.status = 'Starting'
        self.closed = False
        self.fps = fps
        self.last_frame = time.monotonic()
        self.last_pump = 0
        self.root.protocol('WM_DELETE_WINDOW', self.close)
    def close(self):
        if not self.closed:
            self.closed = True
            self.root.destroy()
    def pump(self, force=False):
        if self.closed:
            raise KeyboardInterrupt
        now = time.monotonic()
        if force or now-self.last_pump >= .03:
            self.canvas.itemconfigure(self.status_id, text=self.status)
            self.root.update()
            self.last_pump = now
        if self.closed:
            raise KeyboardInterrupt
    def draw(self, game):
        colors = ['#8da4bc' if filled else '#1c2a3c' for filled in game.board().flat]
        if game._active() and game.orientation_id in ORIENTATIONS:
            x,y = game.piece_xy
            for dy,dx in ORIENTATIONS[game.orientation_id][1]:
                if 0 <= y+dy < 20 and 0 <= x+dx < 10:
                    colors[(y+dy)*10+x+dx] = '#60d9ef'
        for tile,color in zip(self.tiles,colors):
            self.canvas.itemconfigure(tile, fill=color, outline='', width=0)
        if self.target:
            oid,x,y = self.target
            for dy,dx in ORIENTATIONS[oid][1]:
                if 0 <= y+dy < 20 and 0 <= x+dx < 10:
                    self.canvas.itemconfigure(self.tiles[(y+dy)*10+x+dx], outline='#ffd65c', width=3)
        self.canvas.itemconfigure(self.label, text=f'Pieces: {game.pieces_placed}\nLines: {game.lines}\nScore: {game.score}\nNext: {game.next_piece}\n\nCyan: falling piece\nYellow outline: target\n\nLast placement error: {self.last_error}\n\nEvery attempt counts.\nMistakes stay on the board.')
        self.pump(force=True)
    def frame(self, game):
        time.sleep(max(0, 1/self.fps-(time.monotonic()-self.last_frame)))
        self.draw(game)
        self.last_frame = time.monotonic()

if __name__ == '__main__':
    import sys
    from tetris_framework import main
    main(sys.argv[1:] if '--render' in sys.argv else ['--render', *sys.argv[1:]])
