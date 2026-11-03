"""完整训练流程：构建模型、损失、优化器并循环训练，返回最佳模型与 history。"""
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    SELECTED_CLASSES, NUM_EPOCHS, LR, WEIGHT_DECAY, DROPOUT, DEVICE,
)
from model import SmallCNN
from engine import train_one_epoch, evaluate


def train_model(train_loader, val_loader, device=DEVICE):
    train_df = train_loader.dataset.df

    model = SmallCNN(num_classes=len(SELECTED_CLASSES), dropout=DROPOUT).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}", flush=True)

    # 反频率类别权重
    train_class_counts = train_df["label"].value_counts().sort_index().values
    class_weights = 1.0 / train_class_counts
    class_weights = class_weights / class_weights.sum() * len(SELECTED_CLASSES)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print("Train class counts:",
          dict(zip(SELECTED_CLASSES, train_class_counts)), flush=True)
    print("Class weights:", class_weights_tensor.cpu().numpy(), flush=True)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    history = {"train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": [],
               "val_macro_f1": []}

    best_val_f1 = -1.0
    best_state_dict = None

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_macro_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)

        scheduler.step()

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | "
                  f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
                  f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} "
                  f"Macro-F1 {val_macro_f1:.4f}", flush=True)

    print(f"\nBest validation Macro-F1: {best_val_f1:.4f}", flush=True)
    model.load_state_dict(best_state_dict)   # 恢复最佳 epoch 参数

    return model, history
