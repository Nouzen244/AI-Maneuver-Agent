"""
preprocessing.py — Построение входного вектора признаков

Формирует нормализованный массив из 15 чисел для подачи в ManeuverMLP.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────
#  Гиперпараметры нормализации
# ─────────────────────────────────────────────
LIDAR_MAX_RANGE    = 50.0   # метров — максимальная дальность лидара
SPEED_MAX          = 100.0  # км/ч  — для нормализации current_speed и speed_limit
LIDAR_POINT_THRESH = 10     # минимум точек лидара для признания сектора занятым

# Угловые границы секторов лидара (градусы, 0° = вперёд по часовой)
SECTOR_FRONT = (-30,  30)
SECTOR_LEFT  = (-120, -30)
SECTOR_RIGHT = ( 30,  120)
SECTOR_REAR  = (120,  180)   # + (-180, -120) — обрабатывается отдельно


# ─────────────────────────────────────────────
#  Структуры сырых данных
# ─────────────────────────────────────────────
@dataclass
class LidarData:
    """Точки лидара: каждая — (x, y, z) в системе координат автомобиля."""
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))


@dataclass
class VehicleState:
    """Состояние эго-автомобиля."""
    speed_kmh:        float = 0.0    # текущая скорость, км/ч
    speed_limit_kmh:  float = 50.0   # ограничение скорости, км/ч
    lane_offset_m:    float = 0.0    # смещение от центра полосы, метры
    heading_error_deg: float = 0.0   # угол между направлением авто и дорогой, °
    has_lane_left:    bool  = False
    has_lane_right:   bool  = False
    has_lane_center:  bool  = True


@dataclass
class TargetVehicle:
    """Ближайший объект в переднем секторе (если есть)."""
    distance_m: float  = 999.0
    speed_kmh:  float  = 0.0


@dataclass
class NavigatorCommand:
    """Команда навигационного модуля (one-hot)."""
    straight: bool = True
    left:     bool = False
    right:    bool = False

    def to_onehot(self) -> list[int]:
        return [int(self.straight), int(self.left), int(self.right)]


# ─────────────────────────────────────────────
#  Обработка лидарных точек
# ─────────────────────────────────────────────
class LidarProcessor:
    """Разбивает облако точек лидара по секторам и вычисляет признаки."""

    @staticmethod
    def _angle_deg(x: float, y: float) -> float:
        """Угол точки относительно оси X (вперёд), градусы [-180, 180]."""
        return math.degrees(math.atan2(y, x))

    @staticmethod
    def _dist(x: float, y: float) -> float:
        return math.sqrt(x * x + y * y)

    @classmethod
    def _points_in_sector(
        cls, points: np.ndarray, angle_min: float, angle_max: float
    ) -> np.ndarray:
        """Возвращает точки, чьи горизонтальные углы попадают в сектор."""
        if points.shape[0] == 0:
            return points
        angles = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
        mask   = (angles >= angle_min) & (angles < angle_max)
        return points[mask]

    @classmethod
    def compute(
        cls,
        lidar: LidarData,
        ego_speed: float,
        target: Optional[TargetVehicle] = None,
    ) -> dict:
        """
        Возвращает словарь:
            front_dist   : float [0..1]
            front_speed  : float (относительная скорость)
            left_side    : int   {0, 1}
            right_side   : int   {0, 1}
            rear_side    : int   {0, 1}
        """
        pts = lidar.points

        # ── Передний сектор ──────────────────────────────────────────
        front_pts = cls._points_in_sector(pts, *SECTOR_FRONT)
        if front_pts.shape[0] > 0:
            dists      = np.sqrt(front_pts[:, 0]**2 + front_pts[:, 1]**2)
            min_dist_m = float(dists.min())
            front_dist = min(min_dist_m / LIDAR_MAX_RANGE, 1.0)
        else:
            front_dist = 1.0   # путь свободен

        # Относительная скорость
        if target is not None and front_dist < 1.0:
            front_speed = ego_speed - target.speed_kmh
        else:
            front_speed = 0.0

        # ── Боковые / задний секторы ─────────────────────────────────
        left_pts  = cls._points_in_sector(pts, *SECTOR_LEFT)
        right_pts = cls._points_in_sector(pts, *SECTOR_RIGHT)
        # Задний сектор: (120°..180°) и (-180°..-120°)
        rear_pts  = np.vstack([
            cls._points_in_sector(pts,  120,  180),
            cls._points_in_sector(pts, -180, -120),
        ]) if pts.shape[0] > 0 else np.zeros((0, 3))

        return {
            "front_dist":  front_dist,
            "front_speed": front_speed,
            "left_side":   int(left_pts.shape[0]  > LIDAR_POINT_THRESH),
            "right_side":  int(right_pts.shape[0] > LIDAR_POINT_THRESH),
            "rear_side":   int(rear_pts.shape[0]  > LIDAR_POINT_THRESH),
        }


# ─────────────────────────────────────────────
#  Главный строитель вектора признаков
# ─────────────────────────────────────────────
class FeatureBuilder:
    """
    Собирает финальный вектор из 15 признаков из разрозненных источников данных.

    Использование:
        builder = FeatureBuilder()
        vec     = builder.build(lidar, vehicle_state, target, nav_cmd)
        tensor  = torch.tensor(vec, dtype=torch.float32)
    """

    def build(
        self,
        lidar:   LidarData,
        vehicle: VehicleState,
        target:  Optional[TargetVehicle],
        nav:     NavigatorCommand,
    ) -> list[float]:
        """Возвращает список из 15 нормализованных чисел."""

        # 1. Лидарные признаки
        lidar_feats = LidarProcessor.compute(
            lidar, vehicle.speed_kmh, target
        )

        # 2. Скоростные признаки
        current_speed = np.clip(vehicle.speed_kmh / SPEED_MAX, 0.0, 1.0)
        speed_limit   = np.clip(vehicle.speed_limit_kmh / SPEED_MAX, 0.0, 1.0)

        # 3. Геометрические признаки
        lane_offset   = vehicle.lane_offset_m                # уже в метрах
        heading_error = math.sin(math.radians(vehicle.heading_error_deg))

        # 4. Навигатор
        nav_onehot = nav.to_onehot()   # [nav_s, nav_l, nav_r]

        # 5. Разметка полос
        lane_l = int(vehicle.has_lane_left)
        lane_r = int(vehicle.has_lane_right)
        lane_c = int(vehicle.has_lane_center)

        # ── Сборка вектора (порядок важен — должен совпадать с FEAT_* в model.py) ──
        vector: list[float] = [
            lidar_feats["front_dist"],    # [0]
            lidar_feats["front_speed"],   # [1]
            lidar_feats["left_side"],     # [2]
            lidar_feats["right_side"],    # [3]
            lidar_feats["rear_side"],     # [4]
            current_speed,                # [5]
            speed_limit,                  # [6]
            lane_offset,                  # [7]
            heading_error,                # [8]
            nav_onehot[0],               # [9]  nav_s
            nav_onehot[1],               # [10] nav_l
            nav_onehot[2],               # [11] nav_r
            lane_l,                       # [12]
            lane_r,                       # [13]
            lane_c,                       # [14]
        ]

        assert len(vector) == 15, f"Ожидалось 15 признаков, получено {len(vector)}"
        return vector

    @staticmethod
    def from_raw_list(values: list[float]) -> list[float]:
        """
        Принимает уже готовый список из 15 чисел (например, из CSV)
        и проверяет корректность.
        """
        if len(values) != 15:
            raise ValueError(f"Ожидалось 15 значений, получено {len(values)}")
        return [float(v) for v in values]
