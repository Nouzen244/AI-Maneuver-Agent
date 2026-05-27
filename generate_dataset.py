"""
generate_dataset.py — Генератор синтетического датасета для обучения ManeuverMLP

Логика: задаём правила вручную (как должен вести себя хороший водитель)
и генерируем тысячи примеров со случайным шумом.

Запуск:
    python generate_dataset.py --samples 20000 --output dataset.csv
"""

import argparse
import random
import csv
import math
from pathlib import Path

# ─────────────────────────────────────────────
#  Правила "эксперта"
# ─────────────────────────────────────────────
def expert_action(f: list[float]) -> int:
    """
    Принимает вектор из 15 признаков, возвращает action_id.
    Это детерминированные правила — эталон для обучения.

    0=Stay, 1=Left, 2=Right, 3=Brake, 4=Accel
    """
    front_dist    = f[0]
    front_speed   = f[1]
    left_side     = f[2]
    right_side    = f[3]
    current_speed = f[5]
    speed_limit   = f[6]
    nav_s         = f[9]
    nav_l         = f[10]
    nav_r         = f[11]
    lane_l        = f[12]
    lane_r        = f[13]

    # 1. Критическое препятствие → тормозить
    if front_dist < 0.15:
        return 3  # Brake

    # 2. Навигатор требует поворот — выполнить если полоса свободна
    if nav_l > 0.5 and lane_l > 0.5 and left_side < 0.5:
        return 1  # Left
    if nav_r > 0.5 and lane_r > 0.5 and right_side < 0.5:
        return 2  # Right
    # Навигатор хочет повернуть, но манёвр заблокирован → держать полосу
    if nav_l > 0.5 and (left_side >= 0.5 or lane_l < 0.5):
        return 0  # Stay — ждём пока путь освободится
    if nav_r > 0.5 and (right_side >= 0.5 or lane_r < 0.5):
        return 0  # Stay

    # 3. Обгон приоритетнее торможения:
    #    объект заметно медленнее нас, слева свободно и полоса есть
    if (0.15 < front_dist < 0.55
            and front_speed > 10.0
            and left_side < 0.5
            and lane_l > 0.5):
        return 1  # Left

    # 4. Умеренное препятствие, обгон невозможен → тормозить
    if front_dist < 0.35 and front_speed > 5.0:
        return 3  # Brake

    # 5. Скорость значительно ниже лимита → ускориться
    if current_speed < speed_limit * 0.8 and front_dist > 0.6:
        return 4  # Accel

    # 6. По умолчанию — держать полосу
    return 0  # Stay


# ─────────────────────────────────────────────
#  Генератор случайных сценариев
# ─────────────────────────────────────────────
def random_state() -> list[float]:
    """Генерирует реалистичный случайный вектор из 15 признаков."""

    # Навигатор — один из трёх вариантов
    nav_choice = random.choices([0, 1, 2], weights=[0.6, 0.2, 0.2])[0]
    nav_s = 1 if nav_choice == 0 else 0
    nav_l = 1 if nav_choice == 1 else 0
    nav_r = 1 if nav_choice == 2 else 0

    # Полосы — зависят от навигатора
    lane_l = 1 if (nav_l or random.random() > 0.4) else 0
    lane_r = 1 if (nav_r or random.random() > 0.4) else 0
    lane_c = 1

    front_dist  = random.uniform(0.05, 1.0)

    # Если спереди пусто — относительная скорость ~0
    if front_dist > 0.7:
        front_speed = random.uniform(-2.0, 2.0)
    else:
        front_speed = random.uniform(-5.0, 40.0)

    # Боковые препятствия — реже если полосы нет
    left_side  = int(random.random() < (0.3 if lane_l else 0.05))
    right_side = int(random.random() < (0.3 if lane_r else 0.05))
    rear_side  = int(random.random() < 0.25)

    current_speed = random.uniform(0.1, 0.95)
    speed_limit   = random.choice([0.3, 0.5, 0.6, 0.7, 0.9, 1.0])
    lane_offset   = random.gauss(0.0, 0.3)   # обычно близко к центру
    heading_error = math.sin(math.radians(random.gauss(0.0, 5.0)))

    return [
        round(front_dist,    3),
        round(front_speed,   2),
        left_side,
        right_side,
        rear_side,
        round(current_speed, 3),
        round(speed_limit,   2),
        round(lane_offset,   3),
        round(heading_error, 4),
        nav_s, nav_l, nav_r,
        lane_l, lane_r, lane_c,
    ]


# ─────────────────────────────────────────────
#  Генерация и сохранение
# ─────────────────────────────────────────────
def generate(n_samples: int, output: str) -> None:
    header = [f"f{i}" for i in range(15)] + ["action"]

    # Счётчики для баланса классов
    counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for _ in range(n_samples):
            state  = random_state()
            action = expert_action(state)
            writer.writerow(state + [action])
            counts[action] += 1

    print(f"\n[Generator] Создано {n_samples} примеров → {output}")
    print("\n  Распределение классов:")
    names = {0:"Stay", 1:"Left", 2:"Right", 3:"Brake", 4:"Accel"}
    for k, v in counts.items():
        bar = "█" * (v * 40 // n_samples)
        print(f"    {names[k]:5s} ({k}): {v:6d} ({v/n_samples*100:5.1f}%)  {bar}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default=20000, type=int)
    parser.add_argument("--output",  default="dataset.csv")
    args = parser.parse_args()

    random.seed(42)
    generate(args.samples, args.output)
