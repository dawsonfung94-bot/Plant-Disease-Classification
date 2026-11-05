"""Prepare grouped PlantVillage data splits."""
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from config import (
    COLOR_DIR, LEAF_MAP_PATH, SELECTED_CLASSES, CLASS_TO_IDX,
    SEED, SPLIT_SAVE_PATH,
)


def _image_key(filename: str) -> str:
    """Make image filenames match the keys in leaf-map.json."""
    name = Path(filename).name.lower().split("___")[-1]
    for extension in (".jpeg", ".jpg"):
        if name.endswith(extension):
            name = name[:-len(extension)]
            break
    return re.sub(r"\s+\([^)]*conflicted copy[^)]*\)$", "", name)


def build_dataframe() -> pd.DataFrame:
    """Load selected RGB classes and assign repository leaf-group IDs."""
    with LEAF_MAP_PATH.open(encoding="utf-8") as file:
        raw_map = json.load(file)

    aliases = {
        _image_key(filename): leaf_ids
        for filename, leaf_ids in raw_map.items()
        if _image_key(filename) != filename
    }

    records = []
    for class_name in SELECTED_CLASSES:
        class_dir = COLOR_DIR / class_name

        for image_path in class_dir.iterdir():
            if image_path.suffix.lower() not in {".jpg", ".jpeg"}:
                continue

            key = _image_key(image_path.name)
            leaf_id = next(
                (
                    value for value in raw_map.get(key, aliases.get(key, []))
                    if value.startswith(f"{class_name}:::")
                ),
                None,
            )
            if leaf_id is None:
                continue

            records.append({
                "image_path": str(image_path),
                "label": CLASS_TO_IDX[class_name],
                "class_name": class_name,
                "leaf_id": leaf_id,
            })

    df = pd.DataFrame(records)
    print(f"Unique leaf IDs: {df['leaf_id'].nunique()}")
    return df


def add_group_stratified_split(
    df: pd.DataFrame,
    seed: int = SEED,
    save_path: str = SPLIT_SAVE_PATH,
) -> pd.DataFrame:
    """Create stratified splits while keeping each leaf group intact."""
    df = df.reset_index(drop=True).copy()

    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=seed
    )
    trainval_idx, test_idx = next(
        splitter.split(df, df["label"], groups=df["leaf_id"])
    )
    trainval = df.iloc[trainval_idx].reset_index(drop=True)

    splitter = StratifiedGroupKFold(
        n_splits=4, shuffle=True, random_state=seed
    )
    train_idx, val_idx = next(
        splitter.split(
            trainval, trainval["label"], groups=trainval["leaf_id"]
        )
    )

    split = pd.Series(index=df.index, dtype="object")
    split.iloc[test_idx] = "test"
    split.iloc[trainval_idx[train_idx]] = "train"
    split.iloc[trainval_idx[val_idx]] = "val"
    df["split"] = split

    print("Class counts by split:\n", pd.crosstab(df["class_name"], df["split"]))
    print(f"Split seed: {seed}")
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Saved split to {save_path}")
    return df
