import numpy as np
import torch

from ai_model import (
    TetrisEnv,
    TetrisMLP,
    choose_action,
    choose_heuristic_action,
)

WEIGHTS_PATH = "dagger_v4.pth" #файл с обученными весами
GAMES = 100 #колиество тестовых игр
START_SEED = 50000 #нач значение ген случ чисел
MAX_PIECES = 500 #миксимум фигур за одну партию

#симуляция одной партии
def play_game(seed, model=None, device=None):
    env = TetrisEnv(seed=seed, max_pieces=MAX_PIECES)
    total_lines = 0
    info = {"pieces": 0}

    while not env.done:
        valid = env.valid_actions() #список допусстимых действий для текущей фигуры
        if not valid:
            break

        if model is None:
            action = choose_heuristic_action(env)
        else:
            action = choose_action(
                model,
                env.state(),
                valid,
                device,
                epsilon=0.0, #отключаем случайные ходы
            )

        _, _, done, info = env.step(action)
        total_lines += int(info.get("lines", 0))

        if done:
            break

    return total_lines, int(info.get("pieces", 0))


def evaluate(name, model=None, device=None): #функция оценки поля + массовое тестирование
    lines_list = []
    pieces_list = []

    for i in range(GAMES):
        seed = START_SEED + i
        lines, pieces = play_game(seed, model, device)
        lines_list.append(lines)
        pieces_list.append(pieces)

    print(f"\n{name}:")
    print(f"Среднее число линий: {np.mean(lines_list):.2f}")
    print(f"Среднее число фигур: {np.mean(pieces_list):.2f}")
    print(f"Максимум линий: {max(lines_list)}")

    return np.mean(lines_list)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Устройство:", device)
    print(f"Тестовых игр: {GAMES}")
    print(f"Seed: от {START_SEED} до {START_SEED + GAMES - 1}")

    model = TetrisMLP().to(device)
    weights = torch.load(
        WEIGHTS_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(weights)
    model.eval()

    heuristic_avg = evaluate("Эвристика")
    neural_avg = evaluate("Нейросеть DAgger", model, device)

    print("\nИтоговое сравнение")
    print(f"Эвристика: {heuristic_avg:.2f} линий")
    print(f"Нейросеть: {neural_avg:.2f} линий")
    print(f"Разница: {neural_avg - heuristic_avg:+.2f} линий")


if __name__ == "__main__":
    main()
