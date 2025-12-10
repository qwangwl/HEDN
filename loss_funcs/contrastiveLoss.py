# -*- encoding: utf-8 -*-
'''
file       :contrastiveLoss.py
Date       :2025/03/10 10:11:40
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''
import torch
from torch import nn
import torch.nn.functional as F

class SupConLoss(nn.Module): 
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, feat: torch.Tensor, lbl: torch.Tensor) -> torch.Tensor:
        
        lbl = lbl.to(feat.device)
        # Feature normalization
        feat = F.normalize(feat, p=2, dim=1)  # [N, D]
        
        # Compute similarity matrix
        sim = torch.mm(feat, feat.T)  # [N, N]
        
        # Build positive sample mask (exclude self)
        mask = (lbl.unsqueeze(0) == lbl.unsqueeze(1)).float()
        mask.fill_diagonal_(0)  # Remove diagonal (self-comparison)
        
        # Compute contrastive loss
        logits = sim / self.temperature

        log_prob = F.log_softmax(logits, dim=1)

        loss = -(log_prob * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        return loss.mean()
