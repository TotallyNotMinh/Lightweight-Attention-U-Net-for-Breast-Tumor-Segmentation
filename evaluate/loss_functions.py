import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):

        super().__init__()

        self.smooth = smooth

    def forward(self, preds, targets):

        preds   = torch.sigmoid(preds)
        preds   = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (preds * targets).sum(dim=1)

        dice = (
            2. * intersection + self.smooth
        ) / (
            preds.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )

        return 1 - dice.mean()


class FocalLoss(nn.Module):

    def __init__(self, alpha=0.5, gamma=2):

        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):

        bce = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )

        pt    = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce

        return focal.mean()

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        # alpha = FP weight, beta = FN weight
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        TP = (preds * targets).sum(dim=1)
        FP = (preds * (1 - targets)).sum(dim=1)
        FN = ((1 - preds) * targets).sum(dim=1)
        tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        return 1 - tversky.mean()


dice_loss  = DiceLoss()
focal_loss = FocalLoss()
tversky_loss = TverskyLoss() #Penalize FN

def criterion(preds, targets):

    dice  = dice_loss(preds, targets)
    focal = focal_loss(preds, targets)
    tversky = tversky_loss(preds, targets)
    return 0.7 * tversky +  0.3 * focal

