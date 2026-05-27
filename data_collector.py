"""
data_collector.py — Загрузчик датасета для обучения ManeuverMLP
"""

from __future__ import annotations


class DrivingDataset:
    """
    Загружает CSV-датасет и возвращает PyTorch DataLoader.

    Использование:
        ds     = DrivingDataset("dataset.csv")
        loader = ds.get_loader(batch_size=64)
        trainer.fit(loader, epochs=20)
    """

    def __init__(self, csv_path: str):
        import pandas as pd
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        df = pd.read_csv(csv_path)
        feat_cols = [f"f{i}" for i in range(15)]

        X = torch.tensor(df[feat_cols].values, dtype=torch.float32)
        y = torch.tensor(df["action"].values,  dtype=torch.long)

        self.dataset = TensorDataset(X, y)
        print(f"[Dataset] Загружено {len(self.dataset)} примеров из {csv_path}")
        self._DataLoader = DataLoader

    def get_loader(self, batch_size: int = 64, shuffle: bool = True):
        return self._DataLoader(
            self.dataset, batch_size=batch_size, shuffle=shuffle
        )

    def split(self, val_ratio: float = 0.15):
        """Возвращает (train_loader, val_loader)."""
        from torch.utils.data import random_split
        n_val   = int(len(self.dataset) * val_ratio)
        n_train = len(self.dataset) - n_val
        train_ds, val_ds = random_split(self.dataset, [n_train, n_val])
        return (
            self._DataLoader(train_ds, batch_size=64, shuffle=True),
            self._DataLoader(val_ds,   batch_size=64, shuffle=False),
        )
