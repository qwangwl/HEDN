# -*- encoding: utf-8 -*-
'''
file       : deep.py
Date       : 2025/02/11 20:06:22
Email      : qiang.wang@stu.xidian.edu.cn
Author     : qwangxdu
'''

import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional

import math
from scipy.signal import butter, lfilter

class DEAPDataset(object):
    """
    DEAP dataset loader class, providing feature loading and processing functions for the DEAP dataset.
    Supports filtering data by channels, labels, and subjects.

    Expected dataset directory structure:
    - root_path/
        - s01.dat
        - s02.dat
        - ...
        - s32.dat
        - label.mat

    Parameters:
        root_path (str): Root path of the dataset (default: ".\\deap")
        channels (List[str]): List of selected EEG channels, None means all channels (default: None)
        labels (List[str]): List of selected labels (default: None)
        subjects (List[int]): List of selected subject IDs, None means all subjects (default: None)
        window_sec (int): Time window size (in seconds) for each signal segment (default: 1)
    """

    # define channels list for DEAP dataset
    CHANNELS_LIST = [
        'FP1', 'AF3', 'F3',  'F7',  'FC5', 'FC1', 'C3',  'T7',  'CP5', 'CP1',
        'P3',  'P7',  'PO3', 'O1',  'Oz',  'Pz',  'FP2', 'AF4', 'Fz',  'F4',
        'F8',  'FC6', 'FC2', 'Cz',  'C4',  'T8',  'CP6', 'CP2', 'P4',  'P8',
        'PO4', 'O2',  #'hEOG','vEOG','zEMG','tEMG','GSR', 'Resp','Plet','Temp'
    ]
    
    # define labels list for DEAP dataset
    LABELS_LIST = ['valence', 'arousal', 'dominance', 'liking']
    EEG_SAMPLING_RATE = 128  # 采样率，单位为Hz
    BASELINE_DURATION = 3  # 基线持续时间，单位为秒
    STIMULUS_DURATION = 60  # 刺激持续时间，单位为秒

    def __init__(
        self,
        root_path: str = ".\\deap",
        channels: List[str] = None,
        labels: Union[str, List[str]] = None,
        window_sec: int = 1,
        step_sec: Optional[float] = None, 
        feature_name: str = None,
        **kwargs,
    ):
        # initialize the base class
        super(DEAPDataset, self).__init__()
        self.root_path = Path(root_path)
        if not self.root_path.exists():
            raise FileNotFoundError(f"The specified root path does not exist: {self.root_path}")
        self.channel_indices = self._get_channel_indices(channels)
        self.label_indices = self._get_label_indices(labels)
        self.window_sec = window_sec
        self.step_sec = step_sec if step_sec else window_sec
        self.num_of_subjects = 32

        # cache for baseline and stimulus data
        self._baseline_cache, self._stimulus_cache = \
            self._process_all_subjects()

    def get_baseline(self):
        return self._baseline_cache
    
    def get_stimulus(self):
        return self._stimulus_cache
    
    def get_dataset(self):
        changed_data = self._subtract_baseline(
            self._stimulus_cache["data"], self._stimulus_cache["groups"],  
            self._baseline_cache["data"], self._baseline_cache["groups"])
        # changed_data = self._stimulus_cache["data"]
        return {
            "data" : changed_data,
            "labels" : self._stimulus_cache["labels"],
            "groups" : self._stimulus_cache["groups"]
        }

    def get_feature_dim(self):
        return self._stimulus_cache["data"].shape[-1]

    
    def _process_all_subjects(self):
        baseline_data_list = []  # List to hold all baseline feature data
        baseline_group_list = []  # List to hold all baseline group data
        stimulus_data_list = []  # List to hold all stimulus feature data
        stimulus_group_list = []  # List to hold all stimulus group data
        stimulus_labels_list = []  # List to hold all stimulus labels data
        
        for info in self._get_meta_info():
            # Read data for each subject
            samples, labels = self._load_file(info["file_path"])

            # Process the data for this subject
            baseline, stimulus = self._process_one_subject(info["subject"], samples, labels)

            # Append the data to the respective lists
            baseline_data_list.append(baseline["data"])
            baseline_group_list.append(baseline["groups"])

            stimulus_data_list.append(stimulus["data"])
            stimulus_group_list.append(stimulus["groups"])
            stimulus_labels_list.append(stimulus["labels"])

        # Once all subjects are processed, convert lists to numpy arrays
        _baseline_cache = {
            "data": np.vstack(baseline_data_list),  # Stack all baseline feature data
            "groups": np.vstack(baseline_group_list)  # Stack all baseline group data
        }
        _stimulus_cache = {
            "data": np.vstack(stimulus_data_list),  # Stack all stimulus feature data
            "groups": np.vstack(stimulus_group_list),  # Stack all stimulus group data
            "labels": np.vstack(stimulus_labels_list)  # Concatenate all stimulus labels
        }
        return _baseline_cache, _stimulus_cache

    def _process_one_subject(
        self,
        subject_id: int,
        samples: np.ndarray,
        labels: np.ndarray,
    ):
        # select channels and labels based on user input
        samples = samples[:, self.channel_indices, :]
        labels = labels[:, self.label_indices]

        # extract baseline and stimulus data
        baseline_data = self._extract_time_window(samples, start=0, duration=self.BASELINE_DURATION)
        stimulus_data = self._extract_time_window(samples, start=self.BASELINE_DURATION, duration=self.STIMULUS_DURATION)

        # split the signal into segments
        baseline_group, baseline_data, _ = self._segment_signal(baseline_data, None)
        stimulus_group, stimulus_data, stimulus_labels = self._segment_signal(stimulus_data, labels)
        # extract features
        baseline_feature = DE()(baseline_data)
        stimulus_feature = DE()(stimulus_data)
        
        # add subject ID to the group information
        baseline_group = np.column_stack((np.full_like(baseline_group, subject_id), baseline_group))
        stimulus_group = np.column_stack((np.full_like(stimulus_group, subject_id), stimulus_group))

        # build the baseline and stimulus dictionaries
        baseline = {
            "data": baseline_feature,
            "groups": baseline_group
        }

        stimulus = {
            "data": stimulus_feature,
            "groups": stimulus_group,
            "labels": stimulus_labels
        }

        return baseline, stimulus

    def _load_file(self, file_path: Path):
        """load a single DEAP data file"""
        with open(file_path, 'rb') as f:
            pkl = pickle.load(f, encoding='latin1')
        
        samples = pkl["data"]
        labels = pkl["labels"]

        return samples, labels


    def _get_meta_info(self) -> List[Dict]:
        """get meta information of the DEAP dataset"""
        meta_info = []
        for subject in list(range(1, self.num_of_subjects + 1)):
            file_path = self.root_path / f"s{subject:02d}.dat"
            if file_path.exists():
                meta_info.append({
                    "subject": subject,
                    "file_path": file_path
                })
        if not meta_info:
            raise FileNotFoundError("No valid subject data files found in the specified root path.")
        
        return meta_info
    
    def _segment_signal(
        self, 
        signal: np.ndarray, 
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, ...]:
        """Segment the EEG signal into overlapping windows."""
        if signal.ndim != 3:
            raise ValueError("Signal must be a 3D array with shape (trials, channels, points)")
        if labels is not None and labels.ndim != 2:
            raise ValueError("Labels must be a 2D array with shape (trials, labels)")
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

        n_trials, n_slices, n_channels, _ = slices.shape
        segments = slices.reshape(-1, n_channels, window_points)
        groups = np.repeat(np.arange(1, n_trials+1), n_slices)

        segmented_labels = None
        if labels is not None:
            segmented_labels = np.repeat(labels, n_slices, axis=0)

        return groups, segments, segmented_labels
    
    def _extract_time_window(self, signal: np.ndarray, start: int, duration: int) -> np.ndarray:
        """extract a specific time window from the signal."""
        start_idx = start * self.EEG_SAMPLING_RATE
        end_idx = (start + duration) * self.EEG_SAMPLING_RATE
        return signal[..., start_idx:end_idx]
    
    def _subtract_baseline(
        self, 
        stimulus_data: np.ndarray, 
        stim_groups: np.ndarray,
        baseline_data: np.ndarray, 
        base_groups: np.ndarray
    ) -> np.ndarray:
        """baseline correction for EEG signals."""
        if stim_groups.ndim != 2 or base_groups.ndim != 2:
            raise ValueError("Groups must be 2D arrays with shape (subjects, trials)")
        corrected = np.zeros_like(stimulus_data)
        unique_subjects = np.unique(base_groups[:, 0])
        for subject in unique_subjects:
            subject_mask = base_groups[:, 0] == subject
            unique_trials = np.unique(base_groups[subject_mask, 1])
            for trial in unique_trials:
                trial_mask = (base_groups[:, 0] == subject) & (base_groups[:, 1] == trial)
                base_mean = baseline_data[trial_mask].mean(axis=0)
                stim_mask = (stim_groups[:, 0] == subject) & (stim_groups[:, 1] == trial)
                corrected[stim_mask] = stimulus_data[stim_mask] - base_mean
        return corrected

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
    
    
class DE(object):
    """
    Initializes the DE class for computing differential entropy(DE) features.

    Parameters:
    -----------
    fs : int, optional
        Sampling frequency in Hz. Default is 128 Hz.
    band : tuple, optional
        A tuple of frequency band limits. Default is (1, 4, 8, 13, 30, 50).
        These represent common EEG frequency bands: Delta, Theta, Alpha, Beta, Gamma.
    order : int, optional
        Order of the bandpass filter. Default is 3.
    """
    def __init__(
            self,
            fs: int = 128, 
            order: int = 3,
            band: tuple[int, ...] = (1, 4, 8, 13, 31, 50)):
        super(DE, self).__init__()
        self.fs = fs
        self.band = band
        self.order = order

    def __call__(self, data):
        """
        Makes the EEGFeatureExtractor callable. This method allows the class instance 
        to be called as a function to extract features from EEG data.
        """
        return self._extract_differential_entropy(data)
    
    def feature_dim(self):
        """
        Returns the number of features extracted by the DE method.
        """
        return len(self.band) - 1

    def _design_bandpass_filter(self, lowcut, highcut):
        """
        Designs a Butterworth bandpass filter.
        """
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(self.order, [low, high], btype='band')
        return b, a
    
    def _apply_bandpass_filter(self, data, lowcut, highcut):
        """
        Applies a Butterworth bandpass filter to the input signal.
        """
        b, a = self._design_bandpass_filter(lowcut, highcut)
        y = lfilter(b, a, data)
        return y
    
    def _compute_variance_entropy(self, signal):
        """
        Computes the Variance-based Differential Entropy (DE) of a signal.

        The DE is computed as the logarithm of the variance of the signal.
        """
        variance = np.var(signal, ddof=1, axis=-1)
        de = 0.5 * np.log(2 * math.pi * math.e * variance)
        return de

    def _extract_differential_entropy(self, data):
        """
        Computes the Differential Entropy (DE) for each frequency band in the given data.

        This function first applies bandpass filters to the input signal, splitting the signal 
        into the frequency bands specified in the `band` attribute. Then, it calculates the DE 
        for each band.
        """
        n_trials, n_channels, _ = data.shape
        n_bands = len(self.band) - 1
        features = np.empty((n_trials, n_channels, n_bands))

        # Apply bandpass filter for each band and compute DE
        for b in range(n_bands):
            lowcut = self.band[b]
            highcut = self.band[b + 1]
            # Apply bandpass filter
            filtered_data = self._apply_bandpass_filter(data, lowcut, highcut)
            # Compute DE for the filtered data
            de = self._compute_variance_entropy(filtered_data)
            # Store the DE features for this band
            features[:, :, b] = de

        return features