# -*- encoding: utf-8 -*-
'''
file       :evaluate.py
Date       :2025/09/25 17:07:07
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    features, labels, _ = data_loader.dataset.get_data()
    labels = np.argmax(labels.numpy(), axis=1)
    features = features.to(device)
    y_preds = model.predict(features).cpu().numpy()
    acc = accuracy_score(labels, y_preds)
    f1 = f1_score(labels, y_preds, average='macro')

    return acc * 100., f1
    