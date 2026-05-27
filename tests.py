"""
tests.py — Модуль тестирования MLP-агента (без симулятора)

Включает:
  • AISimulatorTest — интерактивное тестирование через консоль
  • 3 встроенных тест-кейса с проверкой ожиданий
  • run_all_tests() — автоматический прогон всех кейсов

Запуск автотестов:
    python tests.py

Интерактивный режим:
    python tests.py --interactive
"""

from __future__ import annotations

import sys
import io
import argparse

# Принудительно UTF-8 для Windows-терминала
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from model import (
    ManeuverMLP, ModelTrainer,
    ACTION_STAY, ACTION_LEFT, ACTION_RIGHT, ACTION_BRAKE, ACTION_ACCEL,
    ACTION_NAMES, FEAT_FRONT_DIST, FEAT_LEFT_SIDE, FEAT_RIGHT_SIDE,
    FEAT_LANE_L, FEAT_LANE_R,
)
from safety_controller import SafetyFilter, StepResult


# ─────────────────────────────────────────────
#  Описание каждого признака для интерактивного ввода
# ─────────────────────────────────────────────
FEATURE_PROMPTS = [
    {
        "name":    "front_dist",
        "index":   0,
        "range":   "[0.0 .. 1.0]",
        "default": 1.0,
        "binary":  False,
        "desc": (
            "Дистанция до ближайшего объекта ВПЕРЕДИ.\n"
            "  0.0 = объект вплотную (≈0 м)\n"
            "  1.0 = путь полностью свободен (≥50 м)\n"
            "  Пример: 0.10 — объект в 5 м (критически близко → Brake)\n"
            "          0.80 — объект в 40 м (можно ехать)"
        ),
    },
    {
        "name":    "front_speed",
        "index":   1,
        "range":   "любое float (обычно -30 .. 40)",
        "default": 0.0,
        "binary":  False,
        "desc": (
            "Относительная скорость к объекту впереди (эго − цель), км/ч.\n"
            "  > 0  — мы едем БЫСТРЕЕ объекта (сближаемся → опасно)\n"
            "  < 0  — объект едет быстрее нас (удаляется → безопасно)\n"
            "    0  — нет объекта или одинаковая скорость\n"
            "  Пример: 20.0 — мы на 20 км/ч быстрее → нужен обгон или торможение"
        ),
    },
    {
        "name":    "left_side",
        "index":   2,
        "range":   "{0, 1}",
        "default": 0,
        "binary":  True,
        "desc": (
            "Есть ли препятствие в ЛЕВОМ боковом секторе (слепая зона).\n"
            "  0 — слева свободно (перестроение влево безопасно)\n"
            "  1 — слева есть объект (перестроение влево ЗАПРЕЩЕНО фильтром)"
        ),
    },
    {
        "name":    "right_side",
        "index":   3,
        "range":   "{0, 1}",
        "default": 0,
        "binary":  True,
        "desc": (
            "Есть ли препятствие в ПРАВОМ боковом секторе (слепая зона).\n"
            "  0 — справа свободно\n"
            "  1 — справа есть объект (перестроение вправо ЗАПРЕЩЕНО фильтром)"
        ),
    },
    {
        "name":    "rear_side",
        "index":   4,
        "range":   "{0, 1}",
        "default": 0,
        "binary":  True,
        "desc": (
            "Есть ли препятствие СЗАДИ.\n"
            "  0 — сзади свободно\n"
            "  1 — сзади есть объект (информационный признак)"
        ),
    },
    {
        "name":    "current_speed",
        "index":   5,
        "range":   "[0.0 .. 1.0]",
        "default": 0.5,
        "binary":  False,
        "desc": (
            "Текущая скорость автомобиля, нормализованная (скорость / 100).\n"
            "  0.0 = стоит\n"
            "  0.5 = 50 км/ч\n"
            "  1.0 = 100 км/ч\n"
            "  Пример: 0.60 → едем 60 км/ч"
        ),
    },
    {
        "name":    "speed_limit",
        "index":   6,
        "range":   "[0.0 .. 1.0]",
        "default": 0.9,
        "binary":  False,
        "desc": (
            "Ограничение скорости на текущем участке, нормализованное (лимит / 100).\n"
            "  0.3 = зона 30 км/ч\n"
            "  0.5 = зона 50 км/ч\n"
            "  0.9 = зона 90 км/ч\n"
            "  Влияет на решение Accel: если current_speed < 0.8×speed_limit → разгон"
        ),
    },
    {
        "name":    "lane_offset",
        "index":   7,
        "range":   "float, метры (обычно -1.5 .. 1.5)",
        "default": 0.0,
        "binary":  False,
        "desc": (
            "Боковое смещение от центра полосы, в метрах.\n"
            "   0.0  — по центру\n"
            "  +0.5  — смещён вправо на 0.5 м\n"
            "  -0.5  — смещён влево на 0.5 м\n"
            "  Используется нейросетью для оценки положения в полосе"
        ),
    },
    {
        "name":    "heading_error",
        "index":   8,
        "range":   "[-1.0 .. 1.0]  (sin угла)",
        "default": 0.0,
        "binary":  False,
        "desc": (
            "Ошибка курса — синус угла между направлением движения и осью дороги.\n"
            "   0.0  — едем строго по дороге\n"
            "  +0.17 ≈ sin(10°) — небольшое отклонение вправо\n"
            "  -0.17 ≈ sin(-10°) — небольшое отклонение влево\n"
            "  Большие значения означают риск выезда из полосы"
        ),
    },
    {
        "name":    "nav_s  (навигатор: ПРЯМО)",
        "index":   9,
        "range":   "{0, 1}",
        "default": 1,
        "binary":  True,
        "desc": (
            "Навигатор командует ЕХАТЬ ПРЯМО.\n"
            "  1 — навигатор говорит «прямо»\n"
            "  0 — другая команда\n"
            "  Ровно один из nav_s / nav_l / nav_r должен быть = 1"
        ),
    },
    {
        "name":    "nav_l  (навигатор: НАЛЕВО)",
        "index":   10,
        "range":   "{0, 1}",
        "default": 0,
        "binary":  True,
        "desc": (
            "Навигатор командует ПОВЕРНУТЬ НАЛЕВО (или перестроиться влево).\n"
            "  1 — навигатор говорит «налево»\n"
            "  0 — другая команда\n"
            "  Если lane_l=0 → фильтр безопасности всё равно запретит поворот"
        ),
    },
    {
        "name":    "nav_r  (навигатор: НАПРАВО)",
        "index":   11,
        "range":   "{0, 1}",
        "default": 0,
        "binary":  True,
        "desc": (
            "Навигатор командует ПОВЕРНУТЬ НАПРАВО (или перестроиться вправо).\n"
            "  1 — навигатор говорит «направо»\n"
            "  0 — другая команда\n"
            "  Если lane_r=0 → фильтр безопасности запретит поворот"
        ),
    },
    {
        "name":    "lane_l  (полоса слева)",
        "index":   12,
        "range":   "{0, 1}",
        "default": 1,
        "binary":  True,
        "desc": (
            "Существует ли полоса движения СЛЕВА от текущей.\n"
            "  1 — полоса слева есть (перестроение влево физически возможно)\n"
            "  0 — полосы нет (край дороги / разделитель)"
        ),
    },
    {
        "name":    "lane_r  (полоса справа)",
        "index":   13,
        "range":   "{0, 1}",
        "default": 1,
        "binary":  True,
        "desc": (
            "Существует ли полоса движения СПРАВА от текущей.\n"
            "  1 — полоса справа есть\n"
            "  0 — полосы нет"
        ),
    },
    {
        "name":    "lane_c  (текущая полоса)",
        "index":   14,
        "range":   "{0, 1}",
        "default": 1,
        "binary":  True,
        "desc": (
            "Находится ли автомобиль в допустимой полосе движения.\n"
            "  1 — да, едем по полосе (норма)\n"
            "  0 — вне разметки (редкий случай)"
        ),
    },
]


# ─────────────────────────────────────────────
#  Утилита: создание заглушки модели
# ─────────────────────────────────────────────
def _build_stub_model(forced_action: int) -> ManeuverMLP:
    """
    Создаёт ManeuverMLP, которая всегда предсказывает `forced_action`.
    Используется ТОЛЬКО для тестов SafetyFilter.
    """
    model = ManeuverMLP()
    with torch.no_grad():
        last_layer: nn.Linear = model.net[-1]
        last_layer.weight.zero_()
        last_layer.bias.zero_()
        last_layer.bias[forced_action] = 10.0
    return model


# ─────────────────────────────────────────────
#  Описание тест-кейса
# ─────────────────────────────────────────────
@dataclass
class TestCase:
    name:             str
    description:      str
    state:            list[float]
    model_action:     int
    expected_safe:    int
    expected_override: bool


# ─────────────────────────────────────────────
#  Три обязательных тест-кейса
# ─────────────────────────────────────────────
def build_test_cases() -> list[TestCase]:
    # ── Кейс 1: "Затор" ──────────────────────────────────────────────
    traffic_jam_state = [
        0.10,        # front_dist   — объект в 5 м
        15.0,        # front_speed  — мы на 15 км/ч быстрее
        0,           # left_side
        0,           # right_side
        0,           # rear_side
        0.60,        # current_speed — 60 км/ч
        0.90,        # speed_limit   — 90 км/ч
        0.0,         # lane_offset
        0.0,         # heading_error
        1, 0, 0,     # nav: прямо
        1, 1, 1,     # все полосы есть
    ]
    case1 = TestCase(
        name              = "Затор (Traffic Jam)",
        description       = "Спереди 0.1, лимит 0.9, nav=Straight → ожидаем Brake",
        state             = traffic_jam_state,
        model_action      = ACTION_STAY,
        expected_safe     = ACTION_BRAKE,
        expected_override = True,
    )

    # ── Кейс 2: "Обгон" ──────────────────────────────────────────────
    overtake_state = [
        0.20,        # front_dist   — объект в 10 м
        20.0,        # front_speed  — мы на 20 км/ч быстрее
        0,           # left_side    — слева свободно ← ключевое
        0,           # right_side
        0,           # rear_side
        0.70,        # current_speed — 70 км/ч
        0.80,        # speed_limit   — 80 км/ч
        0.1,         # lane_offset
        0.0,         # heading_error
        1, 0, 0,     # nav: прямо
        1, 1, 1,     # все полосы есть
    ]
    case2 = TestCase(
        name              = "Обгон (Overtake)",
        description       = "Спереди 0.2, слева свободно → модель Left → фильтр пропускает",
        state             = overtake_state,
        model_action      = ACTION_LEFT,
        expected_safe     = ACTION_LEFT,
        expected_override = False,
    )

    # ── Кейс 3: "Ошибка навигатора" ──────────────────────────────────
    nav_error_state = [
        0.80,        # front_dist   — впереди свободно
        0.0,         # front_speed
        0,           # left_side
        0,           # right_side
        0,           # rear_side
        0.50,        # current_speed
        0.60,        # speed_limit
        0.0,         # lane_offset
        0.0,         # heading_error
        0, 0, 1,     # nav: ПРАВО
        1, 0, 1,     # lane_l=1, lane_r=0 ← полосы справа НЕТ
    ]
    case3 = TestCase(
        name              = "Ошибка навигатора (Nav Error)",
        description       = "Nav=Right, lane_r=0 → модель Right → фильтр → Stay",
        state             = nav_error_state,
        model_action      = ACTION_RIGHT,
        expected_safe     = ACTION_STAY,
        expected_override = True,
    )

    return [case1, case2, case3]


# ─────────────────────────────────────────────
#  Класс тестирования
# ─────────────────────────────────────────────
class AISimulatorTest:
    """
    Тестирует нейросетевого агента двумя способами:
      1. run_all()         — автопрогон встроенных тест-кейсов
      2. run_interactive() — пошаговый ввод каждого признака вручную
    """

    SEPARATOR = "─" * 65

    def __init__(self, model: Optional[ManeuverMLP] = None):
        self.fixed_model = model

    # ── Автоматические тесты ──────────────────────────────────────────
    def run_all(self, verbose: bool = True) -> bool:
        cases  = build_test_cases()
        passed = 0

        print(f"\n{'='*65}")
        print("  АВТОМАТИЧЕСКИЕ ТЕСТ-КЕЙСЫ  (тестируется SafetyFilter)")
        print(f"{'='*65}")

        for i, case in enumerate(cases, 1):
            model  = _build_stub_model(case.model_action)
            result = self._run_one(model, case.state)

            ok_action   = (result.safe_action == case.expected_safe)
            ok_override = (result.override    == case.expected_override)
            ok          = ok_action and ok_override
            status      = "[PASS]" if ok else "[FAIL]"
            passed     += int(ok)

            if verbose:
                print(f"\n[Тест {i}] {case.name} — {status}")
                print(f"  Описание : {case.description}")
                print(self.SEPARATOR)
                print(result)
                print(f"  Ожидалось: safe={ACTION_NAMES[case.expected_safe]}, "
                      f"override={case.expected_override}")
                if not ok:
                    print(f"  [!!] Расхождение! "
                          f"safe_got={ACTION_NAMES[result.safe_action]} "
                          f"override_got={result.override}")
                print(self.SEPARATOR)

        print(f"\n  Итог: {passed}/{len(cases)} тестов прошли\n")
        return passed == len(cases)

    # ── Интерактивный режим ───────────────────────────────────────────
    def run_interactive(self, model: Optional[ManeuverMLP] = None) -> None:
        """
        Пошаговый ввод каждого из 15 признаков с подробным описанием.
        После заполнения всех признаков — выводит предсказание MLP и
        результат фильтра безопасности.
        """
        active_model = model or self.fixed_model
        if active_model is None:
            active_model = ManeuverMLP()
            print("[AISimulatorTest] Обученная модель не указана — используются случайные веса.")

        print(f"\n{'='*65}")
        print("  ИНТЕРАКТИВНЫЙ ТЕСТ MLP-МОДЕЛИ")
        print(f"{'='*65}")
        print("Введите значения признаков по одному.")
        print("Нажмите Enter без ввода, чтобы использовать значение по умолчанию.\n")

        while True:
            values = self._collect_features()
            if values is None:
                break

            result = self._run_one(active_model, values)

            self._describe_scenario(values)

            tensor = torch.tensor(values, dtype=torch.float32)
            proba  = active_model.predict_proba(tensor)

            print(f"\n{self.SEPARATOR}")
            print("  РЕЗУЛЬТАТ:")
            print(result)
            print("\n  Вероятности (softmax):")
            for action_id, p in enumerate(proba.tolist()):
                marker = " <--" if action_id == result.raw_action else ""
                print(f"    {ACTION_NAMES[action_id]:5s} ({action_id}): {p:.3f}{marker}")

            print(self.SEPARATOR)

            again = input("\nПроверить ещё один сценарий? [Y/n]: ").strip().lower()
            if again in ("n", "no", "нет", "н"):
                break

        print("\nВыход из интерактивного режима.")

    def _collect_features(self) -> Optional[list[float]]:
        """
        Запрашивает каждый из 15 признаков по очереди.
        Возвращает список из 15 float, или None если пользователь прервал.
        """
        values = [0.0] * 15

        # Индексы nav-признаков для взаимоисключающей логики
        NAV_INDICES = {9: (10, 11), 10: (9, 11), 11: (9, 10)}

        print(f"\n{'─'*65}")
        print("  Введите признаки (Enter = значение по умолчанию, 'q' = выход):")
        print(f"{'─'*65}")

        for feat in FEATURE_PROMPTS:
            idx      = feat["index"]
            name     = feat["name"]
            rng      = feat["range"]
            default  = feat["default"]
            desc     = feat["desc"]
            is_bin   = feat["binary"]

            # Если это nav-признак и уже выставлен автоматически — пропускаем
            if idx in NAV_INDICES and values[idx] == 0.0:
                # проверяем, не заполнен ли он уже авто-нулём от другого nav
                pass  # будем показывать всегда, но с подсказкой

            print(f"\n[{idx:02d}] {name}  {rng}")
            for line in desc.split("\n"):
                print(f"      {line}")

            # Если другой nav уже выбран — показываем что это поле auto=0
            if idx in NAV_INDICES:
                others = NAV_INDICES[idx]
                if any(values[o] == 1.0 for o in others):
                    print(f"      [авто] Другой навигатор уже выбран → 0")
                    values[idx] = 0.0
                    continue

            print(f"      По умолчанию: {default}")

            while True:
                try:
                    raw = input(f"      Значение [{default}]: ").strip()
                    if raw.lower() in ("q", "quit", "exit"):
                        return None
                    if raw == "":
                        val = float(default)
                    else:
                        val = float(raw)
                    if is_bin and val not in (0, 1):
                        print("      [!] Ожидается 0 или 1. Попробуйте снова.")
                        continue
                    values[idx] = val
                    # Если nav=1 → остальные два nav автоматически 0
                    if idx in NAV_INDICES and val == 1.0:
                        for o in NAV_INDICES[idx]:
                            values[o] = 0.0
                    break
                except ValueError:
                    print("      [!] Введите число. Попробуйте снова.")
                except KeyboardInterrupt:
                    print()
                    return None

        # ── Итоговый вектор ───────────────────────────────────────────
        print(f"\n{'─'*65}")
        print("  Введённый вектор:")
        for feat in FEATURE_PROMPTS:
            i = feat["index"]
            print(f"    [{i:02d}] {feat['name']:30s} = {values[i]}")
        print(f"{'─'*65}")

        return values

    @staticmethod
    def _describe_scenario(values: list[float]) -> None:
        """Выводит человекочитаемое описание введённого сценария."""
        front_dist    = values[0]
        front_speed   = values[1]
        left_side     = values[2]
        right_side    = values[3]
        rear_side     = values[4]
        current_speed = values[5]
        speed_limit   = values[6]
        lane_offset   = values[7]
        nav_s, nav_l, nav_r = values[9], values[10], values[11]
        lane_l, lane_r, lane_c = values[12], values[13], values[14]

        lines = []

        # Скорость относительно объекта впереди
        if front_dist < 1.0:
            dist_m = round(front_dist * 50)
            if front_speed > 0:
                lines.append(f"  Спереди машина в ~{dist_m} м, мы БЫСТРЕЕ на {front_speed:.1f} км/ч")
            elif front_speed < 0:
                lines.append(f"  Спереди машина в ~{dist_m} м, мы МЕДЛЕННЕЕ на {abs(front_speed):.1f} км/ч")
            else:
                lines.append(f"  Спереди машина в ~{dist_m} м, одинаковая скорость")
        else:
            lines.append("  Спереди свободно")

        # Боковые объекты
        if left_side:
            lines.append("  Слева есть объект (слепая зона занята)")
        else:
            lines.append("  Слева свободно")

        if right_side:
            lines.append("  Справа есть объект (слепая зона занята)")
        else:
            lines.append("  Справа свободно")

        if rear_side:
            lines.append("  Сзади есть объект")

        # Наша скорость vs лимит
        our_kmh   = round(current_speed * 100)
        limit_kmh = round(speed_limit   * 100)
        lines.append(f"  Наша скорость: {our_kmh} км/ч  |  Лимит: {limit_kmh} км/ч")

        # Смещение от центра
        if abs(lane_offset) > 0.1:
            side = "вправо" if lane_offset > 0 else "влево"
            lines.append(f"  Смещение от центра полосы: {abs(lane_offset):.2f} м ({side})")

        # Мы в полосе?
        if lane_c:
            lines.append("  Мы в полосе движения")
        else:
            lines.append("  Мы ВНЕ полосы!")

        # Полосы рядом
        sides = []
        if lane_l:
            sides.append("слева")
        if lane_r:
            sides.append("справа")
        if sides:
            lines.append(f"  Соседние полосы: {', '.join(sides)}")
        else:
            lines.append("  Соседних полос нет")

        # Навигатор
        if nav_s:
            lines.append("  Навигатор: ПРЯМО")
        elif nav_l:
            lines.append("  Навигатор: НАЛЕВО")
        elif nav_r:
            lines.append("  Навигатор: НАПРАВО")

        print(f"\n{'─'*65}")
        print("  СЦЕНАРИЙ:")
        for line in lines:
            print(line)
        print(f"{'─'*65}")

    # ── Общий метод запуска одного шага ──────────────────────────────
    @staticmethod
    def _run_one(model: ManeuverMLP, state: list[float]) -> StepResult:
        tensor     = torch.tensor(state, dtype=torch.float32)
        raw_action = model.predict(tensor)
        return SafetyFilter.apply(raw_action, state)


# ─────────────────────────────────────────────
#  Точка входа
# ─────────────────────────────────────────────
def run_all_tests() -> bool:
    tester = AISimulatorTest()
    return tester.run_all(verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Simulator Tests")
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Запустить интерактивный режим после автотестов"
    )
    parser.add_argument(
        "--model", "-m", default=None,
        help="Путь к .pt файлу обученной модели (опционально)"
    )
    args = parser.parse_args()

    model = None
    if args.model:
        try:
            model = ManeuverMLP.load(args.model)
            print(f"[Tests] Модель загружена из {args.model}")
        except Exception as e:
            print(f"[Tests] Не удалось загрузить модель: {e}")

    tester = AISimulatorTest(model)
    all_passed = tester.run_all(verbose=True)

    if args.interactive:
        tester.run_interactive(model)

    sys.exit(0 if all_passed else 1)
