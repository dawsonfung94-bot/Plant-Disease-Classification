"""测试集评估与可视化。"""
import matplotlib
matplotlib.use("Agg")  # 无界面环境；如需弹窗可删掉这行

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay,
)

from config import SELECTED_CLASSES, IDX_TO_CLASS, DEVICE
from engine import evaluate


def evaluate_on_test(model, test_loader, device=DEVICE):
    """在测试集上跑一遍，打印指标并返回混淆矩阵相关数据。"""
    _, _, _, preds_list, labels_list = evaluate(
        model, test_loader, criterion=None, device=device
    )
    test_preds = np.array(preds_list)
    test_labels = np.array(labels_list)

    test_acc = accuracy_score(test_labels, test_preds)
    test_macro_f1 = f1_score(test_labels, test_preds, average="macro")

    print("=" * 60, flush=True)
    print("TEST SET RESULTS - Model A (Small CNN from scratch)", flush=True)
    print("=" * 60, flush=True)
    print(f"Test Accuracy:  {test_acc:.4f}", flush=True)
    print(f"Test Macro-F1:  {test_macro_f1:.4f}", flush=True)
    print("\nPer-class report:", flush=True)
    print(classification_report(
        test_labels, test_preds,
        target_names=[IDX_TO_CLASS[i] for i in range(len(SELECTED_CLASSES))],
        digits=4,
        zero_division=0,
    ), flush=True)

    cm = confusion_matrix(test_labels, test_preds)
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    class_names_short = [c.split("___", 1)[-1] for c in SELECTED_CLASSES]

    return test_acc, test_macro_f1, cm, cm_normalized, class_names_short


def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_acc"], label="Val Acc")
    axes[1].plot(history["val_macro_f1"], label="Val Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Accuracy / Macro-F1")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curves to {save_path}", flush=True)


def plot_confusion_matrix(cm, cm_normalized, class_names_short, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=class_names_short)
    disp1.plot(ax=axes[0], cmap="Blues", values_format="d", colorbar=False)
    axes[0].set_title("Confusion Matrix (Counts)")

    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_normalized,
                                   display_labels=class_names_short)
    disp2.plot(ax=axes[1], cmap="Blues", values_format=".2f", colorbar=False)
    axes[1].set_title("Confusion Matrix (Row-Normalized)")

    for ax in axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrices to {save_path}", flush=True)

    print("\nRow-normalized matrix (each row sums to 1):", flush=True)
    print(pd.DataFrame(cm_normalized,
                       index=class_names_short,
                       columns=class_names_short).round(3), flush=True)
