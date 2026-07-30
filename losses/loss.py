import torch
import torch.nn as nn


class TverskyFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 2.18, smooth: float = 1e-6) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, outputs, masks):
        outputs = torch.sigmoid(outputs)
        outputs = outputs.reshape(-1)
        masks = masks.reshape(-1)

        tp = torch.sum(outputs * masks)
        fp = torch.sum(outputs * (1-masks))
        fn = torch.sum((1-outputs) * masks)

        tversky = (tp + self.smooth) / (tp + self.alpha*fp + self.beta*fn + self.smooth)
        loss = torch.pow(1 - tversky, self.gamma)
        return loss.mean()
