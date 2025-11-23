import random
import numpy as np
from copy import deepcopy

# --------- 2048 game (numpy-based) ----------
UP, DOWN, LEFT, RIGHT = 0,1,2,3

class Game2048:
    def __init__(self, size=4, rng=None):
        self.size = size
        self.rng = rng or random.Random()
        self.reset()

    def reset(self):    # creates board of just zeroes
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.spawn()
        self.spawn()
        return self.board.copy()

    def spawn(self):
        empties = list(zip(*np.where(self.board==0)))
        if not empties:
            return
        r = self.rng.choice(empties)
        # spawn 2 (90%) or 4 (10%)
        self.board[r] = 4 if self.rng.random() < 0.1 else 2

    def can_move(self):
        if np.any(self.board == 0):
            return True
        # check merges in rows
        for i in range(self.size):
            for j in range(self.size-1):
                if self.board[i,j] == self.board[i,j+1]: return True
                if self.board[j,i] == self.board[j+1,i]: return True
        return False

    def move(self, direction):
        rotated = False
        moved = False
        if direction == UP:
            self.board = np.rot90(self.board, -1)
            rotated = True
        elif direction == DOWN:
            self.board = np.rot90(self.board, 1)
            rotated = True
        elif direction == RIGHT:
            self.board = np.fliplr(self.board)

        # Now we only need to handle left moves on rows
        for i in range(self.size):
            row = list(self.board[i])
            new_row, gained, row_moved = self._merge_row(row)
            if row_moved:
                moved = True
            self.board[i] = new_row
            self.score += gained

        # undo transforms
        if direction == RIGHT:
            self.board = np.fliplr(self.board)
        if rotated:
            if direction == UP:
                self.board = np.rot90(self.board, 1)
            else:
                self.board = np.rot90(self.board, -1)
        if moved:
            self.spawn()
        return moved

    def _merge_row(self, row):
        # Takes a list row and returns compressed+merged row, points gained, whether moved
        nonzero = [v for v in row if v != 0]
        merged = []
        gained = 0
        i = 0
        while i < len(nonzero):
            if i+1 < len(nonzero) and nonzero[i] == nonzero[i+1]:
                newv = nonzero[i]*2
                merged.append(newv)
                gained += newv
                i += 2
            else:
                merged.append(nonzero[i])
                i += 1
        # pad zeros
        merged += [0] * (self.size - len(merged))
        moved = merged != row
        return merged, gained, moved

    def step(self, direction):
        moved = self.move(direction)
        done = not self.can_move()
        return self.board.copy(), self.score, done, moved

    def get_max_tile(self):
        return int(self.board.max())
    
# --- Board feature extraction ---
def extract_features(board):
    # returns fixed set of features the evolutionary computing program will receive
    empty = np.sum(board == 0)
    max_tile = board.max()
    sum_tiles = np.sum(board)
    smooth = 0

    # smoothness: penalty for abrupt differences between tiles (monotonicity heuristic)
    for i in range(4):
        for j in range(4):
            if board[i, j] != 0:
                for di, dj in [(1,0), (0,1)]:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < 4 and 0 <= nj < 4 and board[ni, nj] != 0:
                        smooth -= abs(np.log2(board[i,j]) - np.log2(board[ni,nj]))

    return np.array([empty, max_tile, sum_tiles, smooth], dtype=float)