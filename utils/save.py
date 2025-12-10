# -*- encoding: utf-8 -*-
'''
file       :save.py
Date       :2025/09/25 20:57:40
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''

import os
import pandas as pd
import numpy as np
import torch

def save_subject_results(args, subject, trainer, np_log):
    """保存单个受试者的模型和日志"""
    subject_dir = os.path.join(args.tmp_saved_path, f"target_{subject}")
    os.makedirs(subject_dir, exist_ok=True)
    
    # 保存模型
    if args.saved_model:
        torch.save(
            trainer.get_best_model_state(),
            os.path.join(subject_dir, "best.pth")
        )
        torch.save(
            trainer.get_model_state(),
            os.path.join(subject_dir, "last.pth")
        )
    
    # 保存训练日志
    np.savetxt(
        os.path.join(subject_dir, "log.csv"),
        np_log,
        delimiter=",",
        fmt="%.4f"
    )

def save_final_results(args, results):
    """保存最终的汇总结果"""
    df = pd.DataFrame(results)
    df['subject'] = df['subject'].astype(str)

    # 计算统计信息
    mean_vals = df.mean(numeric_only=True)
    std_vals = df.std(numeric_only=True)
    
    # 保存CSV文件（不包含统计信息）
    csv_path = os.path.join(args.tmp_saved_path, "results.csv")
    df.to_csv(csv_path, index=False, float_format='%.2f')

    # 保存统计信息到TXT文件
    stats_path = os.path.join(args.tmp_saved_path, "results_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        for col in mean_vals.index:
            mean = round(mean_vals[col], 2)
            std = round(std_vals[col], 2)
            f.write(f"{col}: {mean} / {std}\n")
            print(f"{col}: {mean} / {std}")


