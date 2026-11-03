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

    leaf_map = {}
    for filename, leaf_ids in raw_map.items():
        leaf_map.setdefault(_image_key(filename), []).extend(leaf_ids)

    records = []
    for class_name in SELECTED_CLASSES:
        class_dir = COLOR_DIR / class_name

        for image_path in class_dir.iterdir():
            if image_path.suffix.lower() not in {".jpg", ".jpeg"}:
                continue

            key = _image_key(image_path.name)
            leaf_id = next(
                (
                    value for value in leaf_map.get(key, [])
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
    """Create 60/20/20 train/validation/test splits by leaf group."""
    groups = df.groupby("leaf_id", as_index=False).agg(label=("label", "first"))

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(
        splitter.split(groups, groups["label"], groups["leaf_id"])
    )
    trainval = groups.iloc[trainval_idx].reset_index(drop=True)
    test = groups.iloc[test_idx]

    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    train_idx, val_idx = next(
        splitter.split(trainval, trainval["label"], trainval["leaf_id"])
    )

    leaf_to_split = {
        leaf_id: split
        for split, subset, indices in (
            ("train", trainval, train_idx),
            ("val", trainval, val_idx),
        )
        for leaf_id in subset.iloc[indices]["leaf_id"]
    }
    leaf_to_split.update({leaf_id: "test" for leaf_id in test["leaf_id"]})

    df = df.copy()
    df["split"] = df["leaf_id"].map(leaf_to_split)

    print("Class counts by split:\n", pd.crosstab(df["class_name"], df["split"]))
    print(f"Split seed: {seed}")
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Saved split to {save_path}")
    return df
