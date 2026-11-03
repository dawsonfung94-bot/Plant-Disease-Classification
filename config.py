"""项目全局配置。"""
from pathlib import Path
import torch

# ---------- 路径 ----------
DATASET_ROOT = Path("./PlantVillage-Dataset-master")
COLOR_DIR = DATASET_ROOT / "raw" / "color"
LEAF_MAP_PATH = DATASET_ROOT / "leaf_grouping" / "leaf-map.json"

# ---------- 类别 ----------
SELECTED_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___healthy",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(SELECTED_CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

# ---------- 训练超参 ----------
SEED = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.60, 0.20, 0.20  # 仅作参考，实际由 StratifiedGroupKFold 决定
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 40
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.5

# Windows 下先设 0；确认能跑后再改 2/4
NUM_WORKERS = 0

# ---------- 输出文件 ----------
SPLIT_SAVE_PATH = "plantvillage_tomato_5class_split.csv"
TRAINING_CURVES_PATH = "training_curves.png"
CONFUSION_MATRIX_PATH = "confusion_matrix.png"

# ---------- 设备 ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")