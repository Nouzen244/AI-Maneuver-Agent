"""
model.py — MLP-агент для предсказания манёвра
Архитектура: 15 → 64 → 32 → 5
"""

import torch
import torch.nn as nn

# ─────────────────────────────────────────────
#  Константы действий
# ─────────────────────────────────────────────
ACTION_STAY  = 0   # Держать полосу / крейсерская скорость
ACTION_LEFT  = 1   # Перестроение влево
ACTION_RIGHT = 2   # Перестроение вправо
ACTION_BRAKE = 3   # Торможение
ACTION_ACCEL = 4   # Ускорение

ACTION_NAMES = {
    ACTION_STAY:  "Stay",
    ACTION_LEFT:  "Left",
    ACTION_RIGHT: "Right",
    ACTION_BRAKE: "Brake",
    ACTION_ACCEL: "Accel",
}

# Индексы признаков для удобного обращения
FEAT_FRONT_DIST    = 0
FEAT_FRONT_SPEED   = 1
FEAT_LEFT_SIDE     = 2
FEAT_RIGHT_SIDE    = 3
FEAT_REAR_SIDE     = 4
FEAT_CURRENT_SPEED = 5
FEAT_SPEED_LIMIT   = 6
FEAT_LANE_OFFSET   = 7
FEAT_HEADING_ERROR = 8
FEAT_NAV_S         = 9
FEAT_NAV_L         = 10
FEAT_NAV_R         = 11
FEAT_LANE_L        = 12
FEAT_LANE_R        = 13
FEAT_LANE_C        = 14

INPUT_DIM  = 15
HIDDEN1    = 64
HIDDEN2    = 32
OUTPUT_DIM = 5


# ─────────────────────────────────────────────
#  Архитектура MLP
# ─────────────────────────────────────────────
class ManeuverMLP(nn.Module):
    """
    Полносвязная сеть для предсказания манёвра.

    Входной вектор (15 признаков):
        [0]  front_dist      — нормализованная дистанция вперёд   [0..1]
        [1]  front_speed     — относительная скорость к объекту впереди
        [2]  left_side       — препятствие слева (0/1)
        [3]  right_side      — препятствие справа (0/1)
        [4]  rear_side       — препятствие сзади (0/1)
        [5]  current_speed   — скорость авто / 100               [0..1]
        [6]  speed_limit     — лимит / 100                       [0..1]
        [7]  lane_offset     — смещение от центра полосы (м)
        [8]  heading_error   — sin(угол к дороге)
        [9]  nav_s           — навигатор: прямо  (one-hot)
        [10] nav_l           — навигатор: налево (one-hot)
        [11] nav_r           — навигатор: направо(one-hot)
        [12] lane_l          — есть полоса слева  (0/1)
        [13] lane_r          — есть полоса справа (0/1)
        [14] lane_c          — есть текущая полоса(0/1)
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Блок 1
            nn.Linear(INPUT_DIM, HIDDEN1),
            nn.ReLU(),
            nn.BatchNorm1d(HIDDEN1),
            nn.Dropout(p=0.2),

            # Блок 2
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.ReLU(),
            nn.BatchNorm1d(HIDDEN2),

            # Выходной слой
            nn.Linear(HIDDEN2, OUTPUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Возвращает логиты (до softmax)."""
        return self.net(x)

    def predict(self, x: torch.Tensor) -> int:
        """
        Возвращает action_id с наибольшей вероятностью.
        Работает с одним вектором (shape=[15]) или батчем (shape=[N,15]).
        """
        self.eval()
        if x.dim() == 1:
            x = x.unsqueeze(0)          # добавить batch-dim для BatchNorm
        with torch.no_grad():
            logits = self.forward(x)
            return int(torch.argmax(logits, dim=-1).item())

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Возвращает вектор вероятностей (softmax)."""
        self.eval()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1).squeeze(0)

    # ── Утилиты сохранения / загрузки ──────────────────────────────
    def save(self, path: str = "maneuver_mlp.pt") -> None:
        torch.save(self.state_dict(), path)
        print(f"[Model] Сохранено: {path}")

    @classmethod
    def load(cls, path: str = "maneuver_mlp.pt") -> "ManeuverMLP":
        model = cls()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        print(f"[Model] Загружено: {path}")
        return model


# ─────────────────────────────────────────────
#  Тренировочный цикл (Imitation Learning)
# ─────────────────────────────────────────────
class ModelTrainer:
    """Обёртка для обучения ManeuverMLP на собранном датасете."""

    def __init__(self, model: ManeuverMLP, lr: float = 1e-3):
        self.model     = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.history: list[dict] = []

    def train_epoch(self, loader) -> float:
        self.model.train()
        total_loss = 0.0
        for features, labels in loader:
            self.optimizer.zero_grad()
            logits = self.model(features)
            loss   = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(loader), 1)

    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        self.model.eval()
        correct, total = 0, 0
        for features, labels in loader:
            if features.dim() == 1:
                features = features.unsqueeze(0)
            preds   = torch.argmax(self.model(features), dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
        return {"accuracy": correct / total if total > 0 else 0.0}

    def fit(self, train_loader, val_loader=None, epochs: int = 20) -> None:
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            record = {"epoch": epoch, "train_loss": train_loss}
            if val_loader:
                metrics = self.evaluate(val_loader)
                record["val_acc"] = metrics["accuracy"]
                print(f"Epoch {epoch:3d} | loss={train_loss:.4f} | val_acc={metrics['accuracy']:.3f}")
            else:
                print(f"Epoch {epoch:3d} | loss={train_loss:.4f}")
            self.history.append(record)
