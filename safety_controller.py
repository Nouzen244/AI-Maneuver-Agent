"""
safety_controller.py — Гибридный контроллер с фильтром безопасности

Схема работы:
    1. ManeuverMLP предсказывает raw action_id
    2. SafetyFilter проверяет конфликты и при необходимости меняет действие
    3. HybridController объединяет оба шага в один вызов
"""

from __future__ import annotations

from dataclasses import dataclass

from model import (
    ManeuverMLP, ACTION_STAY, ACTION_LEFT, ACTION_RIGHT,
    ACTION_BRAKE, ACTION_ACCEL, ACTION_NAMES,
    FEAT_FRONT_DIST, FEAT_LEFT_SIDE, FEAT_RIGHT_SIDE,
    FEAT_LANE_L, FEAT_LANE_R,
)

# ─────────────────────────────────────────────
#  Пороговые значения безопасности
# ─────────────────────────────────────────────
DANGER_FRONT_DIST    = 0.20   # < 20% от max_lidar → опасно разгоняться
WARN_FRONT_DIST      = 0.35   # < 35% → рекомендуется торможение
SIDE_OCCUPIED_THRESH = 1      # left_side/right_side = 1 → занято


# ─────────────────────────────────────────────
#  Структура результата шага
# ─────────────────────────────────────────────
@dataclass
class StepResult:
    raw_action:      int    # действие, предложенное нейросетью
    safe_action:     int    # итоговое действие после фильтра
    override:        bool   # была ли замена?
    override_reason: str    # причина замены (пустая строка, если нет)
    raw_name:        str
    safe_name:       str

    def __str__(self) -> str:
        lines = [
            f"  Raw  Prediction : {self.raw_name} (id={self.raw_action})",
            f"  Safe Action     : {self.safe_name} (id={self.safe_action})",
        ]
        if self.override:
            lines.append(f"  [!] Override     : {self.override_reason}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
#  Фильтр безопасности
# ─────────────────────────────────────────────
class SafetyFilter:
    """
    Проверяет предложенное нейросетью действие на конфликты
    и возвращает безопасное действие.

    Правила (в порядке приоритета):
        1. Слепое перестроение влево
        2. Слепое перестроение вправо
        3. Разгон в препятствие
        4. Мягкое предупреждение при сближении
    """

    @staticmethod
    def check(action: int, state: list[float]) -> tuple[int, str]:
        """
        Возвращает (safe_action, reason).
        reason == "" означает, что конфликтов нет.
        """
        front_dist = state[FEAT_FRONT_DIST]
        left_side  = state[FEAT_LEFT_SIDE]
        right_side = state[FEAT_RIGHT_SIDE]
        lane_l     = state[FEAT_LANE_L]
        lane_r     = state[FEAT_LANE_R]

        # ── Правило 1: Слепое перестроение влево ─────────────────────
        if action == ACTION_LEFT:
            if left_side >= SIDE_OCCUPIED_THRESH:
                return ACTION_STAY, "Blind-spot LEFT: препятствие в левом секторе"
            if lane_l < 0.5:
                return ACTION_STAY, "No-Lane LEFT: полосы слева не существует"

        # ── Правило 2: Слепое перестроение вправо ────────────────────
        if action == ACTION_RIGHT:
            if right_side >= SIDE_OCCUPIED_THRESH:
                return ACTION_STAY, "Blind-spot RIGHT: препятствие в правом секторе"
            if lane_r < 0.5:
                return ACTION_STAY, "No-Lane RIGHT: полосы справа не существует"

        # ── Правило 3: Разгон в препятствие ──────────────────────────
        if action == ACTION_ACCEL and front_dist < DANGER_FRONT_DIST:
            return ACTION_BRAKE, (
                f"Obstacle FRONT: front_dist={front_dist:.2f} < {DANGER_FRONT_DIST}"
            )

        # ── Правило 4: Мягкое предупреждение при Stay/Accel вблизи ──
        if action in (ACTION_STAY, ACTION_ACCEL) and front_dist < WARN_FRONT_DIST:
            return ACTION_BRAKE, (
                f"Precaution BRAKE: front_dist={front_dist:.2f} < {WARN_FRONT_DIST}"
            )

        return action, ""

    @classmethod
    def apply(cls, raw_action: int, state: list[float]) -> StepResult:
        """Возвращает StepResult с полным описанием решения."""
        safe_action, reason = cls.check(raw_action, state)
        return StepResult(
            raw_action      = raw_action,
            safe_action     = safe_action,
            override        = (raw_action != safe_action),
            override_reason = reason,
            raw_name        = ACTION_NAMES.get(raw_action,  "?"),
            safe_name       = ACTION_NAMES.get(safe_action, "?"),
        )


# ─────────────────────────────────────────────
#  Главный интерфейс — safe_step
# ─────────────────────────────────────────────
class HybridController:
    """
    Обёртка над нейросетью + фильтром безопасности.

    Использование:
        controller = HybridController(model)
        result = controller.safe_step(state_vector)
    """

    def __init__(self, model: ManeuverMLP):
        self.model  = model
        self.filter = SafetyFilter()

    def safe_step(self, state: list[float]) -> StepResult:
        """
        Полный цикл: признаки → предсказание → фильтр.

        Returns:
            result : StepResult (информация о принятом решении)
        """
        import torch
        tensor     = torch.tensor(state, dtype=torch.float32)
        raw_action = self.model.predict(tensor)
        return SafetyFilter.apply(raw_action, state)
