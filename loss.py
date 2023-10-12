import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from chamfer_distance import ChamferDistance as chamfer_dist

class ChamferDistanceLoss(nn.Module):
    def __init__(self, reduction='mean', normfirst=False):
        super().__init__()
        self.reduction = reduction
        self.normfirst = normfirst
        self.chd = chamfer_dist()

    def forward(self, inp, tgt, mask, *args, **kwargs):
        """
        Args:
            inp: (B, C, N)
            tgt: (B, C, N)
            mask: (B, N)
            Will normalize the input before the loss
        """
        inp = inp.permute(0, 2, 1) * mask[:, :, None]
        tgt = tgt.permute(0, 2, 1) * mask[:, :, None]

        if self.normfirst:  # using cos-sim
            inp = nn.functional.normalize(inp, dim=-1, p=2)
            tgt = nn.functional.normalize(tgt, dim=-1, p=2)

        dist1, dist2, idx1, idx2 = self.chd(inp, tgt)

        if self.reduction == 'mean':
            return torch.mean(dist1) + torch.mean(dist2)
        else:
            return dist1 + dist2

class JVSLoss(nn.Module):
    def __init__(self, reduction='mean', normfirst=False):
        super().__init__()
        self.reduction = reduction
        self.normfirst = normfirst

    def forward(self, inp, tgt, mask, *args, **kwargs):
        """
        Args:
            inp: (*, C)
            tgt: (*, C)
            Will normalize the input before the loss
        """
        inp = inp.permute(0, 2, 1) * mask[:, :, None]
        tgt = tgt.permute(0, 2, 1) * mask[:, :, None]

        if self.normfirst:  # using cos-sim
            inp = nn.functional.normalize(inp, dim=-1, p=2)
            tgt = nn.functional.normalize(tgt, dim=-1, p=2)

        # similarity = 2 * inp * tgt / (inp * inp + tgt * tgt)
        similarity = 2 * inp @ tgt.transpose(-1, -2) / (inp @ inp.transpose(-1, -2) + tgt @ tgt.transpose(-1, -2))

        loss = torch.exp(-similarity)

        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss


class MultiDimCrossEntropy(nn.CrossEntropyLoss):
    def forward(self, inp, tgt, *args, **kwargs):
        """
        Args:
            inp: (*, C)
            tgt: (*, )
            Will reshape the flatten initial dimensions and then incur loss
        """
        assert inp.ndim == tgt.ndim + 1
        assert inp.shape[:-1] == tgt.shape
        res = super().forward(inp.reshape(-1, inp.size(-1)), tgt.reshape(
            (-1, )), *args, **kwargs)
        if torch.numel(res) == torch.numel(tgt):
            # Reduction was not done, so reshape back to orig shape
            res = res.reshape(tgt.shape)
        return res
