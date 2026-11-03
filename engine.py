"""单 epoch 训练与评估。"""
import torch
from sklearn.metrics import f1_score


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion=None, device="cpu"):
    """若 criterion 为 None，则跳过 loss 计算（测试时用）。"""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    preds_list, labels_list = [], []
    use_loss = criterion is not None

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)

        if use_loss:
            loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        preds_list.extend(preds.cpu().numpy())
        labels_list.extend(labels.cpu().numpy())

    loss_val = running_loss / total if use_loss else None
    acc = correct / total
    macro_f1 = f1_score(labels_list, preds_list, average="macro")
    return loss_val, acc, macro_f1, preds_list, labels_list
