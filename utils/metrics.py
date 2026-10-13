import torch
import torch.nn as nn


class Dice(nn.Module):
    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, outputs, masks):
        outputs = torch.sigmoid(outputs)
        outputs = outputs.reshape(-1)
        masks = masks.reshape(-1)

        tp = torch.sum(outputs * masks)
        fp = torch.sum(outputs * (1-masks))
        fn = torch.sum((1-outputs) * masks)

        dice = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        return dice.mean()


class Precision(nn.Module):
    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, outputs, masks):
        outputs = torch.sigmoid(outputs)
        outputs = outputs.reshape(-1)
        masks = masks.reshape(-1)

        tp = torch.sum(outputs * masks)
        fp = torch.sum(outputs * (1-masks))
        fn = torch.sum((1-outputs) * masks)

        precision = (tp + self.smooth) / (tp + fp + self.smooth)
        return precision.mean()


class Recall(nn.Module):
    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, outputs, masks):
        outputs = torch.sigmoid(outputs)
        outputs = outputs.reshape(-1)
        masks = masks.reshape(-1)

        tp = torch.sum(outputs * masks)
        fp = torch.sum(outputs * (1-masks))
        fn = torch.sum((1-outputs) * masks)

        recall = (tp + self.smooth) / (tp + fn + self.smooth)
        return recall.mean()
