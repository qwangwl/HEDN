# -*- encoding: utf-8 -*-
'''
file       :run_params.py
Date       :2025/11/21 15:24:29
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''

import os

import os

src_momentum = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
tgt_momentum = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

dataset_name = "seed3"
# src_m = 0.5
# for tgt_m in tgt_momentum:
#     for sess in [1, 2, 3]:
#         command = (
#             f"python main.py "
#             f"--dataset_name {dataset_name} "
#             f"--session {sess} "
#             f"--src_momentum {src_m} "
#             f"--tgt_momentum {tgt_m} "
#             f"--tmp_saved_path logs_p/{src_m}_{tgt_m} "
#         )
#         os.system(command)

momentum = [(0, 0.1), (1, 0.1), (0.5, 0), (0.5, 0.9), (0.5, 1)]
for src_m, tgt_m in momentum:
    for sess in [3]:
        command = (
            f"python main.py "
            f"--dataset_name {dataset_name} "
            f"--session {sess} "
            f"--src_momentum {src_m} "
            f"--tgt_momentum {tgt_m} "
            f"--tmp_saved_path logs_p/{src_m}_{tgt_m} "
        )
        os.system(command)