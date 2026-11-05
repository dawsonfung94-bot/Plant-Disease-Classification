"""transforms、Dataset 与 DataLoader 构建。"""
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import IMG_SIZE, BATCH_SIZE, NUM_WORKERS


def _get_train_mean_std(image_paths) -> tuple[list[float], list[float]]:
    """Calculate RGB pixel statistics from training images only."""
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_squared_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for image_path in image_paths:
        with Image.open(image_path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        channel_sum += pixels.sum(axis=(0, 1), dtype=np.float64)
        channel_squared_sum += np.square(pixels).sum(
            axis=(0, 1), dtype=np.float64
        )
        pixel_count += pixels.shape[0] * pixels.shape[1]

    mean = channel_sum / pixel_count
    std = np.sqrt(channel_squared_sum / pixel_count - np.square(mean))
    return mean.tolist(), std.tolist()


def get_train_transform(
    mean: list[float], std: list[float]
) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_eval_transform(
    mean: list[float], std: list[float]
) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


class PlantVillageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = int(row["label"])
        if self.transform:
            image = self.transform(image)
        return image, label


def build_loaders(df):
    """根据 df 里的 split 列构建 train/val/test DataLoader。"""
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    mean, std = _get_train_mean_std(train_df["image_path"])
    print(f"Training RGB mean: {mean}", flush=True)
    print(f"Training RGB std:  {std}", flush=True)

    train_dataset = PlantVillageDataset(
        train_df, transform=get_train_transform(mean, std)
    )
    val_dataset = PlantVillageDataset(
        val_df, transform=get_eval_transform(mean, std)
    )
    test_dataset = PlantVillageDataset(
        test_df, transform=get_eval_transform(mean, std)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    print(f"Train images: {len(train_dataset)}", flush=True)
    print(f"Val images:   {len(val_dataset)}", flush=True)
    print(f"Test images:  {len(test_dataset)}", flush=True)

    return train_loader, val_loader, test_loader
