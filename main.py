"""入口：串联数据准备 -> 训练 -> 评估 -> 可视化。"""
from config import (
    SEED, DEVICE,
    TRAINING_CURVES_PATH, CONFUSION_MATRIX_PATH,
)
from utils import set_seed
from data_prep import build_dataframe, add_group_stratified_split
from dataset import build_loaders
from train import train_model
from evaluate import (
    evaluate_on_test, plot_training_curves, plot_confusion_matrix,
)


def main():
    set_seed(SEED)
    print(f"Device: {DEVICE}", flush=True)

    # 1) 数据准备
    df = build_dataframe()
    df = add_group_stratified_split(df)

    # 2) DataLoader
    train_loader, val_loader, test_loader = build_loaders(df)

    # 3) 训练
    model, history = train_model(train_loader, val_loader)

    # 4) 可视化训练曲线
    plot_training_curves(history, TRAINING_CURVES_PATH)

    # 5) 测试集评估
    test_acc, test_macro_f1, cm, cm_normalized, class_names_short = evaluate_on_test(
        model, test_loader
    )
    plot_confusion_matrix(cm, cm_normalized, class_names_short,
                          CONFUSION_MATRIX_PATH)


if __name__ == "__main__":
    main()
