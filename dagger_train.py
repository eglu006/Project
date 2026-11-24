import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ai_model import (
    TetrisEnv,
    TetrisMLP,
    choose_action,
    choose_heuristic_action,
    N_ACTIONS,
)


START_WEIGHTS = "dagger_v3.pth"
OUTPUT_WEIGHTS = "dagger_v4.pth"

ROUNDS = 6 #Количество циклов сбора данных и дообучения
EPISODES_PER_ROUND = 100 #Количество игровых партий в цикле
MAX_PIECES = 500 #Макс количество фигур в патрии
EPOCHS_PER_ROUND = 6 #Количество эпох 
BATCH_SIZE = 256 #Колиество примеров в одном обучающем пакете
LEARNING_RATE = 0.0003 #Скорость изменения параметров нейросети
SEED = 3030 #Нач значение для генераторов случ чисел

DATASET_PATH = "dagger_dataset.npz"

#сбор данных
def collect_data(model, device, episodes, seed, round_index):
    states, labels, masks = [], [], []
    lines_results, pieces_results = [], []

    model.eval()
    epsilon = max(0.03, 0.15 - 0.025 * round_index)

    for episode in range(episodes):
        env = TetrisEnv(seed=seed + episode, max_pieces=MAX_PIECES)
        state = env.state()
        done = False
        total_lines = 0
        info = {"pieces": 0}

        while not done:
            valid = env.valid_actions() #список допустимых действий для фигуры
            if not valid:
                break

            #смотрим как бы эвристический бот поставил фигуру
            teacher_action = choose_heuristic_action(env)

            mask = np.zeros(N_ACTIONS, dtype=np.bool_)
            mask[valid] = True

            states.append(state.copy())
            labels.append(int(teacher_action))
            masks.append(mask)

            #Мы собираем состояния, в которых она ошибается.
            action = choose_action(
                model, state, valid, device, epsilon=epsilon
            )

            state, _, done, info = env.step(action)
            total_lines += int(info.get("lines", 0))

        lines_results.append(total_lines)
        pieces_results.append(int(info.get("pieces", 0)))

    print(
        f"Раунд {round_index + 1}: "
        f"средние линии={np.mean(lines_results):.2f}, "
        f"фигур={np.mean(pieces_results):.2f}, "
        f"примеров={len(states)}"
    )

    return states, labels, masks

#обучение ИИ
def train_dataset(model, device, states, labels, masks, seed):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(states))

    val_count = max(1, int(len(indices) * 0.1))
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]

    x = torch.tensor(np.asarray(states), dtype=torch.float32) #игровые состояния
    y = torch.tensor(np.asarray(labels), dtype=torch.long) #правильное действие учителя
    m = torch.tensor(np.asarray(masks), dtype=torch.bool) #маски допустимых действий

    train_data = TensorDataset( #обучающий набор данных
        x[train_idx], y[train_idx], m[train_idx]
    )
    loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True
    )

    vx = x[val_idx].to(device) #проверенные ис
    vy = y[val_idx].to(device) #правильные ответы учителя
    vm = m[val_idx].to(device) #допустимые действия для этих состояний

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5,
    )

    best_loss = float("inf")
    best_state = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }

    for epoch in range(1, EPOCHS_PER_ROUND + 1):
        model.train()
        loss_sum = 0.0
        count = 0

        for bx, by, bm in loader:
            bx = bx.to(device)
            by = by.to(device)
            bm = bm.to(device)

            logits = model(bx).masked_fill(~bm, -1e9)
            loss = F.cross_entropy(logits, by) #вычисляем функция потерь, для дальнейшего сравнения с правильными действиями учителя

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            loss_sum += loss.item() * len(bx)
            count += len(bx)

        model.eval()
        with torch.no_grad():
            val_logits = model(vx).masked_fill(~vm, -1e9)
            val_loss = F.cross_entropy(val_logits, vy).item()
            val_acc = ( #вычисляем точность предсказаний
                val_logits.argmax(dim=1) == vy
            ).float().mean().item()

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        print(
            f"  Эпоха {epoch}/{EPOCHS_PER_ROUND}: "
            f"train_loss={loss_sum / max(count, 1):.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_accuracy={val_acc:.1%}"
        )

    model.load_state_dict(best_state)
    return model

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    start_path = Path(START_WEIGHTS)

    model = TetrisMLP().to(device)
    model.load_state_dict(
        torch.load(
            start_path,
            map_location=device,
            weights_only=True,
        )
    )

    # Загружаем датасет от предыдущих запусков.
    if Path(DATASET_PATH).exists():
        with np.load(DATASET_PATH) as data:
            all_states = list(data["states"])
            all_labels = data["labels"].tolist()
            all_masks = list(data["masks"])

        print("Загружено прошлых примеров:", len(all_states))
    else:
        all_states, all_labels, all_masks = [], [], []

    for round_index in range(ROUNDS):
        states, labels, masks = collect_data(
            model,
            device,
            EPISODES_PER_ROUND,
            SEED + round_index * 100_000,
            round_index,
        )

        # Добавляем новые примеры к старым.
        all_states.extend(states)
        all_labels.extend(labels)
        all_masks.extend(masks)

        # Сохраняем выборку после каждого раунда.
        np.savez_compressed(
            DATASET_PATH,
            states=np.asarray(all_states, dtype=np.float32),
            labels=np.asarray(all_labels, dtype=np.int64),
            masks=np.asarray(all_masks, dtype=np.bool_),
        )

        print("Всего примеров:", len(all_states))
        print("Обучающая выборка сохранена:", DATASET_PATH)

        model = train_dataset( #передаём накопленные данные в функция обучения
            model,
            device,
            all_states,
            all_labels,
            all_masks,
            SEED + round_index,
        )

        torch.save(model.state_dict(), OUTPUT_WEIGHTS)
        print("Сохранены веса:", Path(OUTPUT_WEIGHTS).resolve())

    print("Итоговый файл:", Path(OUTPUT_WEIGHTS).resolve())

if __name__ == "__main__":
    main()
