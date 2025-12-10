# -*- encoding: utf-8 -*-
'''
file       :run_random.py
Date       :2025/11/21 15:22:31
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''
import os

dataset_name = "seed3"
for seed in [40, 41, 42, 43, 44]:
    for sess in [1, 2, 3]:
        command = (
            f"python main.py "
            f"--dataset_name {dataset_name} "
            f"--session {sess} "
            f"--seed {seed} "
            f"--tmp_saved_path logs_r/{seed} "
        )
        os.system(command)

dataset_name = "seed4"
for seed in [40, 41, 42, 43, 44]:
    for sess in [1, 2, 3]:
        command = (
            f"python main.py "
            f"--dataset_name {dataset_name} "
            f"--session {sess} "
            f"--seed {seed} "
            f"--tmp_saved_path logs_r/{seed} "
        )
        os.system(command)

# ablations = [
#     "main",
#     "abl_tda_random",
#     "abl_tda_wo_easy",
#     "abl_tda_wo_hard",
#     "abl_tda_only_adv",
#     "abl_tda_src_adv",
#     "abl_comp_wo_easy",
#     "abl_comp_wo_hard",
#     "abl_comp_wo_clusterloss",
# ]

# for ablation in ablations:
#     command = (
#         f"python cross_dataset.py "
#         f"--ablation {ablation} "
#     )
#     os.system(command)