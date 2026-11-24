from pathlib import Path
import random
import json
import math
import pygame
import torch
from datetime import datetime
from ai_model import W, H, ROTATIONS, TetrisMLP, choose_action, encode_state

TILE = 34 #размер одной клетки
FIELD_W, FIELD_H = W * TILE, H * TILE
WINDOW = (1400, 850)
LEFT_X, RIGHT_X, FIELD_Y = 30, 1020, 55
CENTER_X, CENTER_W = 495, 410
FPS = 60
GRAVITY_MS = 450
MATCH_DURATION = 600  # 10 минут в секундах

MODEL_PATH = Path(__file__).with_name("dagger_v4.pth")
LEADERBOARD_FILE = Path(__file__).with_name("leaderboard.json")
SETTINGS_FILE = Path(__file__).with_name("settings.json")

DIFFICULTIES = {
    "Лёгкая": 0.2,
    "Средняя": 0.15,
    "Сложная": 0.03,
}

COLORS = [
    (50, 220, 240), (240, 220, 50), (80, 220, 100),
    (220, 70, 70), (180, 80, 230), (70, 120, 240), (240, 150, 40),
]

# Цвета для UI
UI_BG = (10, 10, 18)
UI_GRID = (25, 25, 45)
UI_CYAN = (0, 240, 255)
UI_MAGENTA = (255, 0, 170)
UI_TEXT = (255, 255, 255)
UI_DIM = (100, 100, 120)

pygame.init()
screen = pygame.display.set_mode(WINDOW)
pygame.display.set_caption("TETRIS — Игрок VS AI")
clock = pygame.time.Clock()
font = pygame.font.SysFont("arial", 32, bold=True)
small = pygame.font.SysFont("arial", 22)
tiny = pygame.font.SysFont("arial", 18)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TetrisMLP().to(device)

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Не найден файл модели: {MODEL_PATH.name}.")
    try:
        weights = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    except TypeError:
        weights = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(weights)
    model.eval()

load_model()

#Настройки
def load_settings():
    if not SETTINGS_FILE.exists():
        return {"controls": "arrows"}
    try:
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {"controls": "arrows"}

def save_settings(controls):
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump({"controls": controls}, f, ensure_ascii=False, indent=2)

def get_key(action, controls):
    if controls == "wasd":
        mapping = {
            "LEFT": pygame.K_a, "RIGHT": pygame.K_d,
            "UP": pygame.K_w, "DOWN": pygame.K_s,
            "CONFIRM": pygame.K_RETURN, "BACK": pygame.K_ESCAPE
        }
    else:
        mapping = {
            "LEFT": pygame.K_LEFT, "RIGHT": pygame.K_RIGHT,
            "UP": pygame.K_UP, "DOWN": pygame.K_DOWN,
            "CONFIRM": pygame.K_RETURN, "BACK": pygame.K_ESCAPE
        }
    return mapping.get(action)

#Экран настроек управления
def settings_menu():
    settings = load_settings()
    controls = settings.get("controls", "arrows")
    current_controls = controls
    options = ["Управление: Стрелочки", "Управление: WASD", "Назад в меню"]
    selected = 0 if current_controls == "arrows" else 1
    
    while True:
        time_ms = pygame.time.get_ticks()
        screen.fill(UI_BG)
        
        grid_offset = (time_ms // 40) % TILE
        for y in range(-TILE, WINDOW[1] + TILE, TILE):
            pygame.draw.line(screen, UI_GRID, (0, y + grid_offset), (WINDOW[0], y + grid_offset), 1)
        for x in range(0, WINDOW[0], TILE):
            pygame.draw.line(screen, UI_GRID, (x, 0), (x, WINDOW[1]), 1)

        title = font.render("НАСТРОЙКИ", True, UI_CYAN)
        screen.blit(title, (WINDOW[0] // 2 - title.get_width() // 2, 150))

        option_rects = []
        mouse_pos = pygame.mouse.get_pos()
        
        for i, name in enumerate(options):
            rect = pygame.Rect(550, 300 + i * 90, 300, 65)
            option_rects.append(rect)
            is_hovered = rect.collidepoint(mouse_pos)
            
            is_current = (i == 0 and current_controls == "arrows") or (i == 1 and current_controls == "wasd")
            
            if i == selected or is_hovered:
                pulse = max(0, min(255, int(150 + 105 * math.sin(time_ms / 250))))
                bg_color = (30, 40, 60)
                border_color = (0, pulse, pulse)
                
                glow_rect = rect.inflate(12, 12)
                glow_surf = pygame.Surface((glow_rect.width, glow_rect.height), pygame.SRCALPHA)
                glow_surf.fill((0, 240, 255, 40))
                screen.blit(glow_surf, (glow_rect.x, glow_rect.y))
                pygame.draw.rect(screen, border_color, rect, 3, border_radius=12)
            else:
                bg_color = (25, 25, 35)
                border_color = (60, 60, 80)
                pygame.draw.rect(screen, border_color, rect, 2, border_radius=12)
                
            pygame.draw.rect(screen, bg_color, rect, border_radius=12)
            
            display_name = name + "" if is_current else name
            label = small.render(display_name, True, UI_TEXT if i == selected or is_hovered else UI_DIM)
            screen.blit(label, (rect.centerx - label.get_width() // 2, rect.centery - label.get_height() // 2))

        hint = tiny.render(f"{get_key('UP', controls) == pygame.K_w and 'W/S' or '↑/↓'} — выбор     Enter — подтвердить", True, UI_DIM)
        screen.blit(hint, (WINDOW[0] // 2 - hint.get_width() // 2, WINDOW[1] - 50))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == get_key("UP", controls):
                    selected = (selected - 1) % len(options)
                elif event.key == get_key("DOWN", controls):
                    selected = (selected + 1) % len(options)
                elif event.key == get_key("CONFIRM", controls):
                    if selected == 0:
                        current_controls = "arrows"
                        controls = "arrows"
                        save_settings("arrows")
                    elif selected == 1:
                        current_controls = "wasd"
                        controls = "wasd"
                        save_settings("wasd")
                    elif selected == 2:
                        return
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for i, rect in enumerate(option_rects):
                    if rect.collidepoint(event.pos):
                        if i == 0:
                            current_controls = "arrows"
                            controls = "arrows"
                            save_settings("arrows")
                        elif i == 1:
                            current_controls = "wasd"
                            controls = "wasd"
                            save_settings("wasd")
                        elif i == 2:
                            return
        clock.tick(FPS)

#Таблица лидеров
def load_leaderboard():
    if not LEADERBOARD_FILE.exists():
        return []
    try:
        with open(LEADERBOARD_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

#Проверка ника
def is_nickname_taken(nickname):
    board = load_leaderboard()
    nickname_lower = nickname.strip().lower()
    return any(entry["nickname"].lower() == nickname_lower for entry in board)

#Если ник есть, то обновляем, иначе добавляем
def save_score(nickname, score):
    board = load_leaderboard()
    nickname_clean = nickname.strip()
    nickname_lower = nickname_clean.lower()
    current_time = datetime.now().strftime("%d.%m.%Y %H:%M")
    
    name_exists = False
    for entry in board:
        if entry["nickname"].lower() == nickname_lower:
            name_exists = True
            if score > entry["score"]:
                entry["score"] = score
                entry["date"] = current_time
            break
            
    if not name_exists:
        board.append({"nickname": nickname_clean, "score": score, "date": current_time})
        
    board.sort(key=lambda x: x["score"], reverse=True)
    board = board[:10]
    
    with open(LEADERBOARD_FILE, 'w', encoding='utf-8') as f:
        json.dump(board, f, ensure_ascii=False, indent=2)

#Отрисовка таблицы лидеров
def show_leaderboard():
    board = load_leaderboard()
    while True:
        time_ms = pygame.time.get_ticks()
        screen.fill(UI_BG)
        
        grid_offset = (time_ms // 40) % TILE
        for y in range(-TILE, WINDOW[1] + TILE, TILE):
            pygame.draw.line(screen, UI_GRID, (0, y + grid_offset), (WINDOW[0], y + grid_offset), 1)
        for x in range(0, WINDOW[0], TILE):
            pygame.draw.line(screen, UI_GRID, (x, 0), (x, WINDOW[1]), 1)

        title = font.render("Таблица лидеров", True, UI_CYAN)
        screen.blit(title, (WINDOW[0] // 2 - title.get_width() // 2, 50))

        if not board:
            no_scores = small.render("Пока нет результатов. Сыграйте первую игру!", True, UI_DIM)
            screen.blit(no_scores, (WINDOW[0] // 2 - no_scores.get_width() // 2, 150))
        else:
            list_rect = pygame.Rect(400, 130, 600, 520)
            s = pygame.Surface((list_rect.width, list_rect.height), pygame.SRCALPHA)
            s.fill((20, 20, 30, 180))
            screen.blit(s, (list_rect.x, list_rect.y))
            pygame.draw.rect(screen, UI_CYAN, list_rect, 2, border_radius=10)

            for i, entry in enumerate(board):
                date_str = entry.get("date", "Дата неизвестна")
                main_text = f"{i+1}. {entry['nickname']} — {entry['score']}"
                main_color = (255, 215, 0) if i == 0 else UI_TEXT
                main_surf = small.render(main_text, True, main_color)
                screen.blit(main_surf, (430, 145 + i * 50))
                
                date_surf = tiny.render(date_str, True, UI_DIM)
                screen.blit(date_surf, (450, 170 + i * 50))

        hint = small.render("Esc — вернуться в меню", True, UI_DIM)
        screen.blit(hint, (WINDOW[0] // 2 - hint.get_width() // 2, WINDOW[1] - 60))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return
        clock.tick(FPS)

#Главное меню
def main_menu():
    nickname = ""
    selected = 0
    options = ["Играть", "Таблица лидеров", "Настройки", "Выход"]
    
    cursor_visible = True
    last_cursor_toggle = pygame.time.get_ticks()
    
    while True:
        time_ms = pygame.time.get_ticks()
        screen.fill(UI_BG)
        
        # Считываем актуальные настройки для навигации
        controls = load_settings().get("controls", "arrows")
        
        grid_offset = (time_ms // 40) % TILE
        for y in range(-TILE, WINDOW[1] + TILE, TILE):
            pygame.draw.line(screen, UI_GRID, (0, y + grid_offset), (WINDOW[0], y + grid_offset), 1)
        for x in range(0, WINDOW[0], TILE):
            pygame.draw.line(screen, UI_GRID, (x, 0), (x, WINDOW[1]), 1)

        title = font.render("Введите свой никнейм", True, UI_CYAN)
        shadow = font.render("Введите свой никнейм", True, (0, 0, 0))
        screen.blit(shadow, (WINDOW[0] // 2 - title.get_width() // 2 + 2, 152))
        screen.blit(title, (WINDOW[0] // 2 - title.get_width() // 2, 150))

        input_rect = pygame.Rect(500, 220, 400, 60)
        pygame.draw.rect(screen, (20, 20, 30), input_rect, border_radius=10)
        
        name_taken = nickname.strip() and is_nickname_taken(nickname)
        
        if selected == 0:
            pulse = max(0, min(255, int(150 + 105 * math.sin(time_ms / 200))))
            border_color = (255, 165, 0) if name_taken else (0, pulse, pulse)
        else:
            border_color = UI_DIM
            
        pygame.draw.rect(screen, border_color, input_rect, 3, border_radius=10)
        
        nick_text = nickname if nickname else "Введите никнейм..."
        nick_color = (255, 200, 0) if name_taken else (UI_TEXT if nickname else UI_DIM)
        nick_surf = font.render(nick_text, True, nick_color)
        screen.blit(nick_surf, (input_rect.x + 20, input_rect.y + 14))

        if name_taken and selected == 0:
            warning = tiny.render("Этот ник уже в таблице. Ваш лучший результат будет обновлен.", True, (255, 200, 0))
            screen.blit(warning, (WINDOW[0] // 2 - warning.get_width() // 2, input_rect.bottom + 10))

        if time_ms - last_cursor_toggle > 500:
            cursor_visible = not cursor_visible
            last_cursor_toggle = time_ms

        if selected == 0 and cursor_visible:
            cursor_x = input_rect.x + 20 + nick_surf.get_width() + 2
            cursor_y = input_rect.y + 16
            cursor_color = (255, 200, 0) if name_taken else UI_CYAN
            pygame.draw.line(screen, cursor_color, (cursor_x, cursor_y), (cursor_x, cursor_y + 28), 3)

        option_rects = []
        mouse_pos = pygame.mouse.get_pos()
        
        for i, name in enumerate(options):
            rect = pygame.Rect(550, 350 + i * 80, 300, 65)
            option_rects.append(rect)
            is_hovered = rect.collidepoint(mouse_pos)
            
            if i == selected:
                pulse = max(0, min(255, int(150 + 105 * math.sin(time_ms / 250))))
                bg_color = (30, 40, 60)
                border_color = (0, pulse, pulse)
                
                glow_rect = rect.inflate(12, 12)
                glow_surf = pygame.Surface((glow_rect.width, glow_rect.height), pygame.SRCALPHA)
                glow_surf.fill((0, 240, 255, 40))
                screen.blit(glow_surf, (glow_rect.x, glow_rect.y))
                pygame.draw.rect(screen, border_color, rect, 3, border_radius=12)
            else:
                bg_color = (25, 25, 35) if not is_hovered else (35, 35, 50)
                border_color = (60, 60, 80) if not is_hovered else (100, 100, 120)
                pygame.draw.rect(screen, border_color, rect, 2, border_radius=12)
                
            pygame.draw.rect(screen, bg_color, rect, border_radius=12)
            label = small.render(name, True, UI_TEXT if i == selected or is_hovered else UI_DIM)
            screen.blit(label, (rect.centerx - label.get_width() // 2, rect.centery - label.get_height() // 2))

        up_key = "W/S" if controls == "wasd" else "↑/↓"
        hint = tiny.render(f"{up_key} — выбор     Enter — подтвердить     Backspace — стереть", True, UI_DIM)
        screen.blit(hint, (WINDOW[0] // 2 - hint.get_width() // 2, WINDOW[1] - 50))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == get_key("UP", controls):
                    selected = (selected - 1) % len(options)
                elif event.key == get_key("DOWN", controls):
                    selected = (selected + 1) % len(options)
                elif event.key == get_key("CONFIRM", controls):
                    if selected == 0:
                        if nickname.strip():
                            return nickname.strip()
                    elif selected == 1:
                        show_leaderboard()
                    elif selected == 2:
                        settings_menu()
                    elif selected == 3:
                        return None
                elif event.key == pygame.K_BACKSPACE:
                    if selected == 0:
                        nickname = nickname[:-1]
                elif selected == 0 and event.key not in (pygame.K_ESCAPE, pygame.K_TAB):
                    if len(nickname) < 15 and event.unicode.isprintable():
                        nickname += event.unicode
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if input_rect.collidepoint(event.pos):
                    selected = 0
                else:
                    for i, rect in enumerate(option_rects):
                        if rect.collidepoint(event.pos):
                            selected = i
                            if i == 0:
                                if nickname.strip(): 
                                    return nickname.strip()
                            elif i == 1:
                                show_leaderboard()
                            elif i == 2:
                                settings_menu()
                            elif i == 3:
                                return None
        clock.tick(FPS)

#Игра
def play_game(nickname, controls):
    difficulty_name = difficulty_menu()
    if difficulty_name is None:
        return "BACK_TO_MENU"
        
    epsilon = DIFFICULTIES[difficulty_name]
    piece_sequence = [random.randrange(7) for _ in range(10_000)]

    #генератор случайных фигур
    def ensure_sequence(index):
        while index + 1 >= len(piece_sequence):
            piece_sequence.extend(random.randrange(7) for _ in range(1000))

    board1, board2 = new_board(), new_board()
    idx1 = idx2 = 0
    ensure_sequence(0)
    p1 = spawn_piece(board1, piece_sequence[0])
    p2 = spawn_piece(board2, piece_sequence[0], True, piece_sequence[1], epsilon)
    next1 = next2 = piece_sequence[1]
    score1 = score2 = 0
    over1 = over2 = False
    time_up = False
    score_saved = False
    last_fall = pygame.time.get_ticks()
    start_time = pygame.time.get_ticks()

    move_dir = 0
    move_start_time = 0
    last_move_time = 0
    MOVE_DELAY = 200
    MOVE_RATE = 60

    animating = False
    anim_start = 0
    anim_duration = 500
    clear_rows_1 = []
    clear_rows_2 = []

    #очистка заполненных линий + начисление очков
    def finalize_turn():
        nonlocal clear_rows_1, clear_rows_2, score1, score2, idx1, idx2
        nonlocal next1, next2, p1, p2, last_fall, animating, over1, over2
        
        lines1 = len(clear_rows_1)
        lines2 = len(clear_rows_2)
        points = {0: 0, 1: 100, 2: 300, 3: 700, 4: 1500}
        
        if lines1 > 0:
            kept1 = [row for i, row in enumerate(board1) if i not in clear_rows_1]
            board1[:] = [[0 for _ in range(W)] for _ in range(lines1)] + kept1
            score1 += points.get(lines1, 1500)
            
        if lines2 > 0:
            kept2 = [row for i, row in enumerate(board2) if i not in clear_rows_2]
            board2[:] = [[0 for _ in range(W)] for _ in range(lines2)] + kept2
            score2 += points.get(lines2, 1500)
            
        idx1 += 1
        idx2 += 1
        ensure_sequence(max(idx1, idx2))
        current_id = piece_sequence[idx1]
        next_id = piece_sequence[idx1 + 1]
        p1 = spawn_piece(board1, current_id)
        p2 = spawn_piece(board2, current_id, True, next_id, epsilon)
        next1 = next2 = next_id
        
        if any(board1[0]): over1 = True
        if any(board2[0]): over2 = True
        
        animating = False
        clear_rows_1 = []
        clear_rows_2 = []
        last_fall = pygame.time.get_ticks()

    #падение или фиксация фигуры игрока = падение фигуры ИИ
    def lock_both():
        nonlocal p1, p2, idx1, idx2, next1, next2
        nonlocal score1, score2, over1, over2, last_fall
        nonlocal animating, anim_start, anim_duration, clear_rows_1, clear_rows_2
        
        p2["y"] = landing_y(board2, p2)
        ok1 = lock_piece(board1, p1)
        ok2 = lock_piece(board2, p2)
        if not ok1: over1 = True
        if not ok2: over2 = True
        if over1 or over2: return

        clear_rows_1 = [i for i, row in enumerate(board1) if all(row)]
        clear_rows_2 = [i for i, row in enumerate(board2) if all(row)]
        
        max_lines = max(len(clear_rows_1), len(clear_rows_2))
        
        if max_lines > 0:
            animating = True
            anim_start = pygame.time.get_ticks()
            anim_duration = 400 + (max_lines * 150) 
        else:
            finalize_turn()

    running = True
    while running:
        time_ms = pygame.time.get_ticks()
        now = time_ms
        
        if not (over1 or over2 or time_up):
            elapsed = (now - start_time) // 1000
            time_left = max(0, MATCH_DURATION - elapsed)
            if time_left == 0 and not time_up:
                time_up = True

        keys = pygame.key.get_pressed()
        
        screen.fill(UI_BG)
        grid_offset = (time_ms // 60) % TILE
        for y in range(-TILE, WINDOW[1] + TILE, TILE):
            pygame.draw.line(screen, UI_GRID, (0, y + grid_offset), (WINDOW[0], y + grid_offset), 1)
        for x in range(0, WINDOW[0], TILE):
            pygame.draw.line(screen, UI_GRID, (x, 0), (x, WINDOW[1]), 1)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and not (over1 or over2 or time_up) and not animating:
                if event.key == get_key("UP", controls):
                    new_rot = (p1["rotation"] + 1) % len(ROTATIONS[p1["id"]])
                    if not collides(board1, p1, rotation=new_rot):
                        p1["rotation"] = new_rot
                elif event.key == get_key("DOWN", controls):
                    while not collides(board1, p1, dy=1):
                        p1["y"] += 1
                    lock_both()

        if not (over1 or over2 or time_up) and not animating:
            current_dir = 0
            if keys[get_key("LEFT", controls)]:
                current_dir = -1
            elif keys[get_key("RIGHT", controls)]:
                current_dir = 1

            if current_dir != 0:
                if move_dir != current_dir:
                    move_dir = current_dir
                    move_start_time = now
                    last_move_time = now
                    if not collides(board1, p1, dx=move_dir):
                        p1["x"] += move_dir
                else:
                    if now - move_start_time >= MOVE_DELAY:
                        if now - last_move_time >= MOVE_RATE:
                            if not collides(board1, p1, dx=move_dir):
                                p1["x"] += move_dir
                                last_move_time = now
            else:
                move_dir = 0

        if not (over1 or over2 or time_up) and not animating and now - last_fall >= GRAVITY_MS:
            last_fall = now
            if not collides(board1, p1, dy=1):
                p1["y"] += 1
            else:
                lock_both()
            if not (over1 or over2 or time_up) and not collides(board2, p2, dy=1):
                p2["y"] += 1

        if animating:
            if now - anim_start > anim_duration:
                finalize_turn()

        anim_progress = 0.0
        if animating:
            anim_progress = min(1.0, (now - anim_start) / anim_duration)

        draw_board(board1, p1, LEFT_X, f"Игрок ({nickname})", score1, next1, 
                   is_player=True, clear_rows=clear_rows_1, anim_progress=anim_progress)
        draw_board(board2, p2, RIGHT_X, "AI", score2, next2, 
                   is_player=False, clear_rows=clear_rows_2, anim_progress=anim_progress)
        
        draw_center(score1, score2, difficulty_name, epsilon, over1, over2, time_left)

        next_box_w = 140
        next_box_h = 120
        next_box_x = 630
        next_box_y = 530
        
        pygame.draw.rect(screen, (30, 30, 45), (next_box_x, next_box_y, next_box_w, next_box_h), border_radius=8)
        pygame.draw.rect(screen, UI_CYAN, (next_box_x, next_box_y, next_box_w, next_box_h), 2, border_radius=8)
        
        next_label = tiny.render("Следующая:", True, UI_TEXT)
        screen.blit(next_label, (next_box_x + (next_box_w - next_label.get_width()) // 2, next_box_y - 22))
        
        cells = ROTATIONS[next1][0]
        min_x = min(c[0] for c in cells)
        max_x = max(c[0] for c in cells)
        min_y = min(c[1] for c in cells)
        max_y = max(c[1] for c in cells)
        
        piece_w = (max_x - min_x + 1) * TILE
        piece_h = (max_y - min_y + 1) * TILE
        
        start_x = next_box_x + (next_box_w - piece_w) // 2 - min_x * TILE
        start_y = next_box_y + (next_box_h - piece_h) // 2 - min_y * TILE
        
        color = COLORS[next1]
        for dx, dy in cells:
            rect = pygame.Rect(start_x + dx * TILE, start_y + dy * TILE, TILE, TILE)
            pygame.draw.rect(screen, color, rect.inflate(-2, -2))
            
            highlight = tuple(min(255, c + 60) for c in color)
            pygame.draw.line(screen, highlight, (rect.left+2, rect.top+2), (rect.right-2, rect.top+2), 2)
            pygame.draw.line(screen, highlight, (rect.left+2, rect.top+2), (rect.left+2, rect.bottom-2), 2)
            
            shadow = tuple(max(0, c - 60) for c in color)
            pygame.draw.line(screen, shadow, (rect.left+2, rect.bottom-2), (rect.right-2, rect.bottom-2), 2)
            pygame.draw.line(screen, shadow, (rect.right-2, rect.top+2), (rect.right-2, rect.bottom-2), 2)

        if animating and (len(clear_rows_1) == 4 or len(clear_rows_2) == 4) and anim_progress < 0.4:
            flash_alpha = int(200 * (1.0 - (anim_progress / 0.4)))
            flash_surf = pygame.Surface(WINDOW, pygame.SRCALPHA)
            flash_surf.fill((255, 255, 255, flash_alpha))
            screen.blit(flash_surf, (0, 0))

        if over1 or over2 or time_up:
            if not score_saved:
                save_score(nickname, score1)
                score_saved = True

            overlay = pygame.Surface(WINDOW, pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            screen.blit(overlay, (0, 0))

            if time_up and not over1 and not over2:
                if score1 > score2:
                    winner = "Игрок победил (по очкам)!"
                    win_color = UI_CYAN
                elif score2 > score1:
                    winner = "AI победил (по очкам)!"
                    win_color = (255, 80, 80)
                else:
                    winner = "НИЧЬЯ (по очкам)!"
                    win_color = (255, 220, 60)
            else:
                if over1 and not over2: 
                    winner = "AI победил!"
                    win_color = (255, 80, 80)
                elif over2 and not over1: 
                    winner = "Игрок победил!"
                    win_color = UI_CYAN
                else: 
                    winner = "НИЧЬЯ!"
                    win_color = (255, 220, 60)
            
            msg = font.render(winner, True, win_color)
            screen.blit(msg, (WINDOW[0] // 2 - msg.get_width() // 2, WINDOW[1] // 2 - 40))
            
            sub_msg = small.render("Нажмите R чтобы вернуться в меню", True, UI_DIM)
            screen.blit(sub_msg, (WINDOW[0] // 2 - sub_msg.get_width() // 2, WINDOW[1] // 2 + 10))
            
            pygame.display.flip()
            
            keys_pressed = pygame.key.get_pressed()
            if keys_pressed[pygame.K_r]:
                return "RESTART"
                
        pygame.display.flip()
        clock.tick(FPS)
        
    return "QUIT"

#создание и возврат поля
def new_board():
    return [[0 for _ in range(W)] for _ in range(H)]

#описание фигуры по координатам
def new_piece(piece_id, rotation=0, x_left=None, y_top=0):
    cells = ROTATIONS[piece_id][rotation % len(ROTATIONS[piece_id])]
    if x_left is None:
        width = max(x for x, y in cells) + 1
        x_left = max(0, (W - width) // 2)
    return {"id": piece_id, "rotation": rotation % len(ROTATIONS[piece_id]),
            "x": x_left, "y": y_top}

#преобразование фигуры
def piece_cells(piece):
    return [(piece["x"] + dx, piece["y"] + dy)
            for dx, dy in ROTATIONS[piece["id"]][piece["rotation"]]]

#проверка столкновения фигуры с границами
def collides(board, piece, dx=0, dy=0, rotation=None):
    rot = piece["rotation"] if rotation is None else rotation % len(ROTATIONS[piece["id"]])
    for ox, oy in ROTATIONS[piece["id"]][rot]:
        x, y = piece["x"] + dx + ox, piece["y"] + dy + oy
        if x < 0 or x >= W or y >= H: return True
        if y >= 0 and board[y][x]: return True
    return False

#падение фигуры
def landing_y(board, piece):
    y = piece["y"]
    while not collides(board, piece, dy=1):
        piece["y"] += 1
    result = piece["y"]
    piece["y"] = y
    return result

#фиксация фигуры на поле
def lock_piece(board, piece):
    cells = piece_cells(piece)
    if any(y < 0 for x, y in cells): return False
    for x, y in cells:
        board[y][x] = COLORS[piece["id"]]
    return True

#перебор варианта размещения фигуры для ИИ
def choose_bot_placement(board, current_id, next_id, epsilon):
    state = encode_state(board, current_id, next_id)
    candidates = []
    for rotation, cells in enumerate(ROTATIONS[current_id]):
        width = max(x for x, y in cells) + 1
        for x_left in range(W - width + 1):
            probe = new_piece(current_id, rotation, x_left, 0)
            if collides(board, probe): continue
            candidates.append((rotation * W + x_left, rotation, x_left))
    if not candidates: return None
    action = choose_action(model, state, [item[0] for item in candidates], device, epsilon=epsilon) #ИИ оценивает всех и выбирает лучший ход
    for action_id, rotation, x_left in candidates:
        if action_id == action: return rotation, x_left
    return None

#создание фигур
def spawn_piece(board, piece_id, is_bot=False, next_id=None, epsilon=0.0):
    piece = new_piece(piece_id)
    if is_bot:
        plan = choose_bot_placement(board, piece_id, next_id, epsilon)
        if plan is not None:
            rotation, x_left = plan
            piece = new_piece(piece_id, rotation, x_left, 0)
    return piece

#рисовка игрового поля
def draw_board(board, piece, x0, label, score, next_id, is_player=True, clear_rows=None, anim_progress=0.0):
    if clear_rows is None:
        clear_rows = []
        
    pygame.draw.rect(screen, (5, 5, 12), (x0, FIELD_Y, FIELD_W, FIELD_H))
    for y in range(H):
        for x in range(W):
            rect = pygame.Rect(x0 + x * TILE, FIELD_Y + y * TILE, TILE, TILE)
            pygame.draw.rect(screen, (20, 20, 35), rect, 1)
            
            if board[y][x]:
                color = board[y][x]
                pygame.draw.rect(screen, color, rect.inflate(-2, -2))
                
                highlight = tuple(min(255, c + 50) for c in color)
                pygame.draw.line(screen, highlight, (rect.left+2, rect.top+2), (rect.right-2, rect.top+2), 2)
                pygame.draw.line(screen, highlight, (rect.left+2, rect.top+2), (rect.left+2, rect.bottom-2), 2)
                
                shadow = tuple(max(0, c - 50) for c in color)
                pygame.draw.line(screen, shadow, (rect.left+2, rect.bottom-2), (rect.right-2, rect.bottom-2), 2)
                pygame.draw.line(screen, shadow, (rect.right-2, rect.top+2), (rect.right-2, rect.bottom-2), 2)

    if piece:
        for x, y in piece_cells(piece):
            if y >= 0:
                rect = pygame.Rect(x0 + x * TILE, FIELD_Y + y * TILE, TILE, TILE)
                color = COLORS[piece["id"]]
                pygame.draw.rect(screen, color, rect.inflate(-2, -2))
                
                highlight = tuple(min(255, c + 50) for c in color)
                pygame.draw.line(screen, highlight, (rect.left+2, rect.top+2), (rect.right-2, rect.top+2), 2)
                pygame.draw.line(screen, highlight, (rect.left+2, rect.top+2), (rect.left+2, rect.bottom-2), 2)
                
                shadow = tuple(max(0, c - 50) for c in color)
                pygame.draw.line(screen, shadow, (rect.left+2, rect.bottom-2), (rect.right-2, rect.bottom-2), 2)
                pygame.draw.line(screen, shadow, (rect.right-2, rect.top+2), (rect.right-2, rect.bottom-2), 2)
    
    if clear_rows and anim_progress < 1.0:
        num_lines = len(clear_rows)
        flash = int(255 * (0.5 + 0.5 * math.sin(anim_progress * math.pi * (3 + num_lines))))
        
        for row_idx in clear_rows:
            rect = pygame.Rect(x0, FIELD_Y + row_idx * TILE, FIELD_W, TILE)
            
            if num_lines == 1:
                color = (255, 255, 255)
            elif num_lines == 2:
                color = (0, 240, 255)
            elif num_lines == 3:
                color = (255, 200, 0)
            else:
                color = (255, 50, 255)
                
            surf = pygame.Surface((FIELD_W, TILE), pygame.SRCALPHA)
            surf.fill((*color, flash))
            screen.blit(surf, (rect.x, rect.y))
            
            if num_lines >= 3:
                expand = int(15 * anim_progress)
                pygame.draw.rect(screen, color, rect.inflate(expand, expand), 2)
                
            if num_lines == 4 and anim_progress < 0.6:
                tetris_surf = font.render("Tetris", True, (255, 255, 255))
                alpha = int(255 * (1.0 - (anim_progress / 0.6)))
                s = pygame.Surface((tetris_surf.get_width(), tetris_surf.get_height()), pygame.SRCALPHA)
                s.fill((*color, alpha))
                screen.blit(s, (x0 + FIELD_W//2 - tetris_surf.get_width()//2, FIELD_Y + row_idx * TILE - 25))
                screen.blit(tetris_surf, (x0 + FIELD_W//2 - tetris_surf.get_width()//2, FIELD_Y + row_idx * TILE - 25))

    border_color = UI_CYAN if is_player else UI_MAGENTA
    pygame.draw.rect(screen, border_color, (x0, FIELD_Y, FIELD_W, FIELD_H), 3, border_radius=4)
    
    label_surf = font.render(label, True, border_color)
    screen.blit(label_surf, (x0 + FIELD_W // 2 - label_surf.get_width() // 2, 12))
    
    score_surf = small.render(f"Счёт: {score}", True, UI_TEXT)
    screen.blit(score_surf, (x0 + 8, FIELD_Y + FIELD_H + 8))

#Экран выбора сложности
def difficulty_menu():
    selected = 1
    names = list(DIFFICULTIES.keys())
    buttons = []
    
    descriptions = {
        "Лёгкая": "Новичок: не знаком с тетрисом",
        "Средняя": "Для любителей",
        "Сложная": "Профи: проверь свои навыки"
    }
    
    while True:
        time_ms = pygame.time.get_ticks()
        screen.fill(UI_BG)
        
        # Считываем актуальные настройки для навигации
        controls = load_settings().get("controls", "arrows")
        
        grid_offset = (time_ms // 40) % TILE
        for y in range(-TILE, WINDOW[1] + TILE, TILE):
            pygame.draw.line(screen, UI_GRID, (0, y + grid_offset), (WINDOW[0], y + grid_offset), 1)
        for x in range(0, WINDOW[0], TILE):
            pygame.draw.line(screen, UI_GRID, (x, 0), (x, WINDOW[1]), 1)

        title = font.render("Выберите сложность ИИ", True, UI_CYAN)
        screen.blit(title, (WINDOW[0] // 2 - title.get_width() // 2, 210))
        buttons.clear()
        
        for i, name in enumerate(names):
            rect = pygame.Rect(230 + i * 320, 340, 300, 90)
            buttons.append(rect)
            
            is_hovered = rect.collidepoint(pygame.mouse.get_pos())
            
            if i == selected:
                pulse = max(0, min(255, int(150 + 105 * math.sin(time_ms / 250))))
                bg_color = (30, 40, 60)
                border_color = (0, pulse, pulse)
                
                glow_rect = rect.inflate(12, 12)
                glow_surf = pygame.Surface((glow_rect.width, glow_rect.height), pygame.SRCALPHA)
                glow_surf.fill((0, 240, 255, 40))
                screen.blit(glow_surf, (glow_rect.x, glow_rect.y))
                pygame.draw.rect(screen, border_color, rect, 3, border_radius=12)
            else:
                bg_color = (25, 25, 35) if not is_hovered else (35, 35, 50)
                border_color = (60, 60, 80) if not is_hovered else (100, 100, 120)
                pygame.draw.rect(screen, border_color, rect, 2, border_radius=12)
                
            pygame.draw.rect(screen, bg_color, rect, border_radius=12)
            
            label = small.render(name, True, UI_TEXT if i == selected or is_hovered else UI_DIM)
            screen.blit(label, (rect.centerx - label.get_width() // 2, rect.centery - 15))
            
            desc = descriptions.get(name, "")
            desc_surf = tiny.render(desc, True, (150, 150, 170) if i == selected else UI_DIM)
            screen.blit(desc_surf, (rect.centerx - desc_surf.get_width() // 2, rect.centery + 15))
            
        left_key = "A/D" if load_settings().get("controls", "arrows") == "wasd" else "←/→"
        hint = small.render(f"{left_key} — выбор     Enter — начать     Esc — назад", True, UI_DIM)
        screen.blit(hint, (WINDOW[0] // 2 - hint.get_width() // 2, 500))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT: return None
            if event.type == pygame.KEYDOWN:
                if event.key == get_key("LEFT", controls):
                    selected = (selected - 1) % len(names)
                elif event.key == get_key("RIGHT", controls):
                    selected = (selected + 1) % len(names)
                elif event.key == get_key("CONFIRM", controls):
                    return names[selected]
                elif event.key == get_key("BACK", controls):
                    return None
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for i, rect in enumerate(buttons):
                    if rect.collidepoint(event.pos): return names[i]
        clock.tick(FPS)

#Центральная панель в игровом поле
def draw_center(score1, score2, difficulty_name, epsilon, over1, over2, time_left):
    panel = pygame.Rect(CENTER_X, FIELD_Y, CENTER_W, FIELD_H)
    
    s = pygame.Surface((CENTER_W, FIELD_H), pygame.SRCALPHA)
    s.fill((10, 10, 18, 200))
    screen.blit(s, (CENTER_X, FIELD_Y))
    
    pygame.draw.rect(screen, UI_CYAN, panel, 2, border_radius=10)
    
    mins = time_left // 60
    secs = time_left % 60
    timer_color = (255, 80, 80) if time_left <= 60 else UI_TEXT
    
    controls = load_settings().get("controls", "arrows")
    if controls == "wasd":
        move_keys = "A / D"
        rotate_key = "W"
        drop_key = "S"
    else:
        move_keys = "← / →"
        rotate_key = "↑"
        drop_key = "↓"
    
    lines = [
        "Игрок VS AI", " ",
        f"Игрок: {score1}", f"AI: {score2}", " ",
        f"Время: {mins:02d}:{secs:02d}",
        f"Сложность: {difficulty_name}",
        f"{move_keys}  движение", f"{rotate_key}  поворот",
        f"{drop_key}  падение", " ",
    ]
    for i, line in enumerate(lines):
        if i == 0:
            color = UI_CYAN
        elif "Время:" in line:
            color = timer_color
        elif "Игрок" in line or "AI" in line or "Сложность" in line:
            color = UI_TEXT
        else:
            color = UI_DIM
            
        surf = small.render(line, True, color)
        screen.blit(surf, (CENTER_X + CENTER_W // 2 - surf.get_width() // 2, FIELD_Y + 55 + i * 35))

def main():
    while True:
        nickname = main_menu()
        if nickname is None:
            break
            
        settings = load_settings()
        controls = settings.get("controls", "arrows")
            
        result = play_game(nickname, controls)
        
        if result == "QUIT":
            break
            
    pygame.quit()

if __name__ == "__main__":
    main()