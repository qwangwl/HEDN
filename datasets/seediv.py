# -*- encoding: utf-8 -*-
'''
file       :seediv.py
Date       :2025/12/04 15:39:10
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''

import re
import numpy as np
import scipy.io as scio
from typing import List, Tuple, Dict, Union, Optional
from datasets import SEEDRawDataset
from datasets.seed import de_extraction

class SEEDIVRawDataset(SEEDRawDataset):
    def __init__(
        self,
        root_path: str = ".\\SEED",
        channels: List[str] = None,
        window_sec: int = 1,
        step_sec: Optional[float] = None, 
        subjects: List[int] = None,
        sessions: Union[List[int], int] = [1],
        **kwargs,
    ):
        super(SEEDIVRawDataset, self).__init__(
            root_path, 
            channels, 
            window_sec, 
            step_sec,
            subjects, 
            sessions
        )
        self.window_sec = 4

    
    def _process_one_subject(self, subject_info: Dict):
        # Load .mat file containing the EEG data
        mat_data = scio.loadmat(subject_info["file_path"], verify_compressed_data_integrity=False)
        # print(mat_data.keys())
        trial_names = [k.partition('_')[0] for k in mat_data.keys() if not k.startswith('__')][0]
        trial_ids = [int(m.group(1)) for k in mat_data.keys() if not k.startswith('__') for m in [re.search(r'(\d+)$', k)] if m]
        # Initialize lists to store EEG data, group information, and labels
        data, groups, labels = [], [], []
        for trial_id in trial_ids:
            # Extract the EEG data for the current trial, transpose dimensions and select channels
            trial_data = mat_data[f"{trial_names}_eeg{trial_id}"][self.channel_indices]

            stimulus_feature = de_extraction(trial_data, 200, None, time_window=self.window_sec, overlap=0)
            # print(stimulus_feature.shape)
            stimulus_labels = self.trials_labels[subject_info["session"]-1][trial_id - 1]
            num_samples = stimulus_feature.shape[0]

            stimulus_group = np.hstack([
                np.ones((num_samples, 1), dtype=np.int16) * subject_info["subject"],
                np.ones((num_samples, 1), dtype=np.int16) * trial_id,
                np.ones((num_samples, 1), dtype=np.int16) * subject_info["session"]
            ])
            # Create group information (trial_id, subject_id, session_id)
            data.append(stimulus_feature)
            groups.append(stimulus_group)
            labels.append(np.repeat(stimulus_labels, num_samples))
        return {
            "data": np.concatenate(data),
            "labels": np.concatenate(labels),
            "groups": np.concatenate(groups)
        }

    
    def _load_trial_labels(self) -> np.ndarray:
        return np.array([
            [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
            [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
            [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
        ])
