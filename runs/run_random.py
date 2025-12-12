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
            f"python cross_subject.py "
            f"--dataset_name {dataset_name} "
            f"--session {sess} "
            f"--seed {seed} "
            f"--tmp_saved_path logs/random/{seed} "
        )
        os.system(command)

dataset_name = "seed4"
for seed in [40, 41, 42, 43, 44]:
    for sess in [1, 2, 3]:
        command = (
            f"python cross_subject.py "
            f"--dataset_name {dataset_name} "
            f"--session {sess} "
            f"--seed {seed} "
            f"--tmp_saved_path logs/random/{seed} "
        )
        os.system(command)