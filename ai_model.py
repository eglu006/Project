from __future__ import annotations
import random
from collections import deque
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import torch
from torch import nn

W, H = 10, 20
N_PIECES = 7
N_ACTIONS = 4 * W
BOARD_FEATURES = W * H
PIECE_FEATURES = N_PIECES * 2
COLUMN_FEATURES = W * 2
GLOBAL_FEATURES = 3
STATE_SIZE = BOARD_FEATURES + PIECE_FEATURES + COLUMN_FEATURES + GLOBAL_FEATURES

RAW_FIGURES = [
    [(-1, 0), (-2, 0), (0, 0), (1, 0)],
    [(0, -1), (-1, -1), (-1, 0), (0, 0)],
    [(-1, 0), (-1, 1), (0, 0), (0, -1)],
    [(0, 0), (-1, 0), (0, 1), (-1, -1)],
    [(0, 0), (0, -1), (0, 1), (-1, -1)],
    [(0, 0), (0, -1), (0, 1), (1, -1)],
    [(0, 0), (0, -1), (0, 1), (-1, 0)],
]

#нормализация координат
def _normalise(cells):
    min_x = min(x for x, _ in cells)
    min_y = min(y for _, y in cells)
    return tuple(sorted((x - min_x, y - min_y) for x, y in cells))

#генерация всех уникальных поворотов для фигуры
def make_rotations():
    all_rotations = []
    for raw in RAW_FIGURES:
        pivot_x, pivot_y = raw[0]
        current = [(x - pivot_x, y - pivot_y) for x, y in raw]
        rotations = []
        for _ in range(4):
            norm = _normalise(current)
            if norm not in rotations:
                rotations.append(norm)
            current = [(-y, x) for x, y in current]
        all_rotations.append(rotations)
    return all_rotations

ROTATIONS = make_rotations()

#id фигур, понятный для ИИ
def one_hot(index: int, size: int = N_PIECES):
    out = np.zeros(size, dtype=np.float32)
    out[int(index)] = 1.0
    return out

#анализ поля с помощью ИИ
def board_features(board):
    flat = np.asarray([1.0 if cell else 0.0 for row in board for cell in row], dtype=np.float32)
    heights, holes_by_col = [], []
    for x in range(W):
        first_block, holes, seen_block = None, 0, False
        for y in range(H):
            occupied = bool(board[y][x])
            if occupied and first_block is None: first_block = y
            if occupied: seen_block = True
            elif seen_block: holes += 1
        heights.append(0 if first_block is None else H - first_block)
        holes_by_col.append(holes)
    aggregate_height = sum(heights)
    total_holes = sum(holes_by_col)
    bumpiness = sum(abs(heights[i] - heights[i + 1]) for i in range(W - 1))
    return np.concatenate((
        flat,
        np.asarray(heights, dtype=np.float32) / H,
        np.asarray(holes_by_col, dtype=np.float32) / H,
        np.asarray([aggregate_height / (W * H), total_holes / (W * H), bumpiness / ((W - 1) * H)], dtype=np.float32),
    ))

#объединение всех признаков поля
def encode_state(board, current_piece: int, next_piece: int):
    return np.concatenate((board_features(board), one_hot(current_piece), one_hot(next_piece))).astype(np.float32, copy=False)

#подсчёт дыр, высот столбцов и неровностей
def board_metrics(board):
    heights, holes = [], 0
    for x in range(W):
        seen_block, height = False, 0
        for y in range(H):
            if board[y][x]:
                if not seen_block: height = H - y
                seen_block = True
            elif seen_block: holes += 1
        heights.append(height)
    bumpiness = sum(abs(heights[i] - heights[i + 1]) for i in range(W - 1))
    return sum(heights), holes, bumpiness

#штраф
def board_cost(board):
        aggregate_height, holes, bumpiness = board_metrics(board)
    
        row_transitions = 0
        for y in range(H):
            prev = True
            for x in range(W):
                occ = bool(board[y][x])
                row_transitions += occ != prev
                prev = occ
            row_transitions += not prev
    
        col_transitions = 0
        for x in range(W):
            prev = True
            for y in range(H):
                occ = bool(board[y][x])
                col_transitions += occ != prev
                prev = occ
            col_transitions += not prev
    
        return (
            0.045 * aggregate_height +
            0.80 * holes +
            0.10 * bumpiness +
            0.025 * row_transitions +
            0.035 * col_transitions
        )

def choose_heuristic_action(env): #эврестический алгоритм(выступает в роли учителя)
    valid = env.valid_actions()

    if not valid:
        return 0

    best_action = valid[0]
    best_score = float("inf")

    for action in valid:
        rotation, x_left = divmod(int(action), W)

        cells = ROTATIONS[env.current_piece][rotation]

        y_top = env.landing_y(cells, x_left)
        if y_top is None:
            continue

        # Копируем поле, не изменяя игровое
        board = [row.copy() for row in env.board]

        # Размещаем фигуру
        for dx, dy in cells:
            x = x_left + dx
            y = y_top + dy

            if 0 <= y < H:
                board[y][x] = env.current_piece + 1

        # Очищаем заполненные линии
        full_rows = [
            i for i, row in enumerate(board)
            if all(row)
        ]
        lines = len(full_rows)

        if lines:
            board = [
                row for row in board
                if not all(row)
            ]
            board = [[0] * W for _ in range(lines)] + board

        # Чем меньше оценка, тем лучше положение
        score = board_cost(board) - 3.0 * lines

        if score < best_score:
            best_score = score
            best_action = action

    return best_action

class TetrisEnv: #среда где ИИ и бот могу тиграть в тетрис
    def __init__(self, seed=None, max_pieces=500):
        self.rng = random.Random(seed)
        self.max_pieces = max_pieces
        self.reset()

    def reset(self): #очистка поля и начало новой игры
        self.board = [[0] * W for _ in range(H)]
        self.current_piece = self.rng.randrange(N_PIECES)
        self.next_piece = self.rng.randrange(N_PIECES)
        self.pieces_played = 0
        self.done = False
        return self.state()

    def state(self): return encode_state(self.board, self.current_piece, self.next_piece) #возврат текущего состояния поля

    def collides(self, cells, x_left, y_top): #проверка сталкивания фигуры со стенками
        for dx, dy in cells:
            x, y = x_left + dx, y_top + dy
            if x < 0 or x >= W or y >= H: return True
            if y >= 0 and self.board[y][x]: return True
        return False

    def landing_y(self, cells, x_left): #определение, на какой высоте окажется фигура при падении
        if self.collides(cells, x_left, 0): return None
        y = 0
        while not self.collides(cells, x_left, y + 1): y += 1
        return y

    def valid_actions(self): #список допустимых положений и поворотов фигуры
        valid = []
        for rotation, cells in enumerate(ROTATIONS[self.current_piece]):
            width = max(x for x, _ in cells) + 1
            for x_left in range(W - width + 1):
                if self.landing_y(cells, x_left) is not None:
                    valid.append(rotation * W + x_left)
        return valid

    def step(self, action): #выполнение выбранного действия, обновление игрового поля и проверка окончания игры

        if self.done:
            return self.state(), 0.0, True, {"lines":0,"pieces":self.pieces_played}

        valid=self.valid_actions()

        if not valid:
            self.done=True
            return self.state(),-8.0,True,{"lines":0,"pieces":self.pieces_played}

        if action not in valid:
            action=valid[0]

        old_cost=board_cost(self.board)

        rotation,x_left=divmod(int(action),W)
        cells=ROTATIONS[self.current_piece][rotation%len(ROTATIONS[self.current_piece])]
        y_top=self.landing_y(cells,x_left)

        if y_top is None:
            self.done=True
            return self.state(),-8.0,True,{"lines":0,"pieces":self.pieces_played}

        for dx,dy in cells:
            x=x_left+dx
            y=y_top+dy
            if 0<=y<H:
                self.board[y][x]=self.current_piece+1

        full=[i for i,row in enumerate(self.board) if all(row)]
        lines=len(full)

        if lines:
            self.board=[row for row in self.board if not all(row)]
            self.board=[[0]*W for _ in range(lines)]+self.board

        new_cost=board_cost(self.board)

        line_reward = {
            0: 0.0,
            1: 10.0,
            2: 25.0,
            3: 45.0,
            4: 70.0,
        }.get(lines, 70.0)

        shaping = 0.25 * (old_cost - new_cost)
        shaping = float(np.clip(shaping, -2.0, 2.0))

        reward = line_reward + shaping - 0.05

        self.pieces_played += 1
        self.current_piece = self.next_piece
        self.next_piece = self.rng.randrange(N_PIECES)

        
        # Проверяем, можно ли разместить следующую фигуру
        topped = not self.valid_actions()

        limit = self.pieces_played >= self.max_pieces
        self.done = topped or limit

        if topped:
            reward -= 15.0


        return self.state(),float(reward),self.done,{
            "lines":lines,
            "pieces":self.pieces_played
        }

class TetrisMLP(nn.Module): #Сама нейронная сеть MLP
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_SIZE, 512), nn.ReLU(), #1 скрытый слой
            nn.Linear(512, 256), nn.ReLU(), #2 скрытый слой
            nn.Linear(256, 128), nn.ReLU(), #3 скрытый слой
            nn.Linear(128, N_ACTIONS),
        )

    def forward(self, x):
        return self.net(x)

@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_mask: np.ndarray

class ReplayBuffer:
    def __init__(self, capacity=200_000):
        self.data = deque(maxlen=capacity)
    def add(self, *args):
        self.data.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.data, batch_size)
    def __len__(self):
        return len(self.data)

def action_mask(env):
    mask = np.zeros(N_ACTIONS, dtype=np.bool_)
    valid = env.valid_actions()
    if valid: mask[valid] = True
    return mask

def choose_action(model, state, valid_actions: Sequence[int], device, epsilon=0.0): #выбор действия нейросетью
    if not valid_actions: return 0
    if random.random() < epsilon: return random.choice(list(valid_actions))
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        values = model(x)[0].clone()
        invalid = torch.ones(N_ACTIONS, dtype=torch.bool, device=device)
        invalid[list(valid_actions)] = False
        values[invalid] = -1e9
        return int(values.argmax().item())