"""
train.py — Обучение ManeuverMLP на собранном датасете

Запуск:
    python train.py --data dataset.csv --epochs 30 --output maneuver_mlp.pt
"""

import argparse
from model import ManeuverMLP, ModelTrainer
from data_collector import DrivingDataset


def main():
    parser = argparse.ArgumentParser(description="Train ManeuverMLP")
    parser.add_argument("--data",    required=True, help="Путь к CSV датасету")
    parser.add_argument("--epochs",  default=30, type=int)
    parser.add_argument("--lr",      default=1e-3, type=float)
    parser.add_argument("--batch",   default=64,  type=int)
    parser.add_argument("--output",  default="maneuver_mlp.pt")
    args = parser.parse_args()

    ds                       = DrivingDataset(args.data)
    train_loader, val_loader = ds.split(val_ratio=0.15)

    model   = ManeuverMLP()
    trainer = ModelTrainer(model, lr=args.lr)

    print(f"\n[Train] Начало обучения: epochs={args.epochs}, lr={args.lr}, batch={args.batch}")
    trainer.fit(train_loader, val_loader, epochs=args.epochs)

    model.save(args.output)
    print("[Train] Готово!")


if __name__ == "__main__":
    main()
