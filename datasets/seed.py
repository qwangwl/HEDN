# -*- encoding: utf-8 -*-
'''
file       :seediv.py
Date       :2025/12/04 12:34:09
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''

from scipy import signal
import re
import numpy as np
import scipy.io as scio
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional

import math
from scipy.signal import butter, lfilter

class SEEDRawDataset(object):
    """
    SEED dataset loader class, providing feature loading and processing functions for the DEAP dataset.
    Supports filtering data by channels, labels, and subjects.

    Expected dataset directory structure:
    - root_path/
        - 1/
            - 1_20160518.mat
            - ...
        - 2/
            - 1_20161125.mat
            - ...
        - 3/
            - 1_20161126.mat
            - ...

    Parameters:
        root_path (str): Root path of the dataset (default: ".\\SEED_IV")
        channels (List[str]): List of selected EEG channels, None means all channels (default: None)
        labels (List[str]): List of selected labels (default: None)
        subjects (List[int]): List of selected subject IDs, None means all subjects (default: None)
        window_sec (int): Time window size (in seconds) for each signal segment (default: 1)
    """

    # define channels list for DEAP dataset
    CHANNELS_LIST = [
        'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4',
        'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
        'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3',
        'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ',
        'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
        'CB1', 'O1', 'OZ', 'O2', 'CB2'
    ]
    
    EEG_SAMPLING_RATE = 200  # 采样率，单位为Hz

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
        # initialize the base class
        super(SEEDRawDataset, self).__init__()
        self.root_path = Path(root_path)
        if not self.root_path.exists():
            raise FileNotFoundError(f"The specified root path does not exist: {self.root_path}")
        self.channel_indices = self._get_channel_indices(channels)
        self.window_sec = window_sec
        self.step_sec = step_sec if step_sec else window_sec
        self.num_of_subjects = 15

        self.trials_labels = self._load_trial_labels()
        meta_info = self._get_meta_info(sessions, subjects)
        self._dataset_cache = self._process_all_subjects(meta_info)
    
    def get_dataset(self):
        return self._dataset_cache

    def get_feature_dim(self):
        return self._dataset_cache["data"].shape[-1]
    
    def _process_all_subjects(self, meta_info):
        data_list = []  # List to hold all stimulus feature data
        group_list = []  # List to hold all stimulus group data
        labels_list = []  # List to hold all stimulus labels data
        
        for info in meta_info:
            stimulus = self._process_one_subject(info)
            data_list.append(stimulus["data"])
            group_list.append(stimulus["groups"])
            labels_list.append(stimulus["labels"])
        # Once all subjects are processed, convert lists to numpy arrays
        return {
            "data": np.concatenate(data_list),
            "labels": np.concatenate(labels_list),
            "groups": np.concatenate(group_list)
        }

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

            stimulus_feature = de_extraction(trial_data, 200, None, time_window=1, overlap=0)
            # print(stimulus_feature.shape)
            stimulus_labels = self.trials_labels[trial_id - 1]
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

    def _get_meta_info(self, sessions: Union[List[int], int], subjects: List[int]) -> List[Dict]:
        """get meta information of the dataset"""
        sessions = [sessions] if isinstance(sessions, int) else sessions or [1, 2, 3]
        subjects = subjects or list(range(1, 16))

        meta_info = []
        for session_id in sessions:
            session_path = self.root_path / str(session_id)
            subject_mat_files = [
                file for file in session_path.glob("*.mat") 
                if self._parse_subject_id(file) in subjects
            ]
            # print(subject_mat_files)

            for file_path in subject_mat_files:
                meta_info.append({
                    "session": session_id,
                    "subject": self._parse_subject_id(file_path),
                    "file_path": file_path
                })
        # print(meta_info)
        return meta_info
    
    @staticmethod
    def _parse_subject_id(file_path: Path) -> int:
        """parser subject ID from file name"""
        return int(file_path.stem.split("_")[0])


    
    def _segment_signal(
        self, 
        signal: np.ndarray, 
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, ...]:
        """Segment the EEG signal into overlapping windows."""
        window_points = self.window_sec * self.EEG_SAMPLING_RATE
        step_points = int(self.step_sec * self.EEG_SAMPLING_RATE)
        n_points = signal.shape[-1]

        if n_points < window_points:
            window_points = n_points

        n_slices = (n_points - window_points) // step_points + 1
        slices = np.stack([
            signal[..., i*step_points : i*step_points+window_points] 
            for i in range(n_slices)
        ], axis=1) 

        n_channels, n_slices, _ = slices.shape
        segments = slices.reshape(-1, n_channels, window_points)

        segmented_labels = None
        if labels is not None:
            segmented_labels = np.repeat(labels, n_slices, axis=0)

        return segments, segmented_labels

    def _extract_time_window(self, signal: np.ndarray, start: int, duration: int) -> np.ndarray:
        """extract a specific time window from the signal."""
        start_idx = start * self.EEG_SAMPLING_RATE
        end_idx = (start + duration) * self.EEG_SAMPLING_RATE
        return signal[..., start_idx:end_idx]

    def _get_channel_indices(self, channels: List[str]) -> np.ndarray:
        """get channel indices based on the provided channel names."""
        if not channels:
            return np.arange(len(self.CHANNELS_LIST))
        return np.where(np.isin(self.CHANNELS_LIST, channels))[0]

    def _get_label_indices(self, labels: List[str]) -> np.ndarray:
        """get label indices based on the provided label names."""
        if not labels:
            return np.arange(len(self.LABELS_LIST))
        return np.where(np.isin(self.LABELS_LIST, labels))[0]
    
    def _load_trial_labels(self) -> np.ndarray:
        # Load the label file which contains trial labels
        label_path = self.root_path / "label.mat"
        return scio.loadmat(label_path,
                            verify_compressed_data_integrity=False)['label'][0]
    
    
def de_extraction(data, sample_rate, extract_bands, time_window, overlap):
    """
    DE feature extraction
    :param data: original eeg data, input shape: (channel, filter_data)
    :param sample_rate: sample rate of eeg signal
    :param extract_bands: the frequency bands that needs to be extracted
    :param time_window: time window of one extract part
    :param overlap: overlap
    :return: de feature need to be computed
    """
    if extract_bands is None:
        extract_bands = [[0.5, 4], [4, 8], [8, 14], [14, 30], [30, 50]]
    nyq = 0.5 * sample_rate
    noverlap = int(overlap * sample_rate)
    window_size = int(time_window * sample_rate)
    if noverlap != 0:
        sample_num = (data.shape[1] - window_size) // (window_size - noverlap)
    else:
        sample_num = (data.shape[1]) // window_size
    de_data = np.zeros((sample_num, data.shape[0], len(extract_bands)))
    for b_idx, band in enumerate(extract_bands):
        b, a = signal.butter(3, [band[0]/nyq, band[1]/nyq], 'bandpass')
        band_data = signal.filtfilt(b, a, data)
        t = 0
        for i in range(sample_num):
            de_data[i,:,b_idx] = 1 / 2 * np.log2(2 * np.pi * np.e * np.var(band_data[:,t:t+window_size], axis=1, ddof=1))
            t += window_size-noverlap
    return de_data