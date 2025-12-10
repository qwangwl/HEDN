# -*- encoding: utf-8 -*-

import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datasets import SEEDFeatureDataset, SEEDIVFeatureDataset, DEAPDataset, SEEDRawDataset, SEEDIVRawDataset

def get_dataset(args):

    if args.dataset_name == "seed3":
        data, one_hot_mat, Group = get_seed(args)
    elif args.dataset_name == "seed4":
        data, one_hot_mat, Group = get_seediv(args)
    elif args.dataset_name == "deap":
        data, one_hot_mat, Group = get_deap(args)
    # elif args.dataset_name == "seed_raw":
    #     data, one_hot_mat, Group = get_seed_raw(args)
    #     print(data.shape, one_hot_mat.shape, Group.shape)
    # elif args.dataset_name == "seed4_raw":
    #     data, one_hot_mat, Group = get_seed4_raw(args)
    #     print(data.shape, one_hot_mat.shape, Group.shape)
    return {
        "data": data,
        "labels": one_hot_mat, 
        "groups": Group
    }

def get_seed(args):

    num_classes = 3
    num_of_channels = 62

    SEED = SEEDFeatureDataset(args.seed3_path, sessions=args.session)
    feature_dim = num_of_channels * SEED.get_feature_dim()
    dataset = SEED.get_dataset()

    data, Label, Group = dataset["data"], dataset["labels"], dataset["groups"]
    Label += 1 # begin from 0
    data = data.reshape(-1, feature_dim)
    subject_ids = Group[:, 0] # subject ID 

    setattr(args, "feature_dim", feature_dim)
    setattr(args, "num_classes", num_classes)

    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    for i in np.unique(subject_ids):
        data[subject_ids==i] = min_max_scaler.fit_transform(data[subject_ids == i])

    one_hot_mat = np.eye(len(Label), num_classes)[Label].astype("float32")
    return data, one_hot_mat, Group

def get_seediv(args):
    num_classes = 4
    num_of_channels = 62

    SEEDIV = SEEDIVFeatureDataset(args.seed4_path, sessions=args.session)
    feature_dim = num_of_channels * SEEDIV.get_feature_dim()
    dataset = SEEDIV.get_dataset()

    data, Label, Group = dataset["data"], dataset["labels"], dataset["groups"]
    data = data.reshape(-1, feature_dim)
    subject_ids = Group[:, 0] # subject ID 

    setattr(args, "feature_dim", feature_dim)
    setattr(args, "num_classes", num_classes)

    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    for i in np.unique(subject_ids):
        data[subject_ids==i] = min_max_scaler.fit_transform(data[subject_ids == i])

    one_hot_mat = np.eye(len(Label), num_classes)[Label].astype("float32")
    return data, one_hot_mat, Group

def get_deap(args):
    """
    Get DEAP dataset with preprocessing.
    """
    if isinstance(args.emotion, str):
        num_classes = 2
    elif isinstance(args.emotion, list):
        num_classes = 4
    num_channels = 32

    params = {
        "feature_name": args.feature_name,
        "window_sec": args.window_sec, 
        "step_sec": args.step_sec,
        "labels": args.emotion,
    }

    DEAP = DEAPDataset(args.deap_path, **params)
    feature_dim = num_channels * DEAP.get_feature_dim()

    dataset = DEAP.get_dataset()

    # 对每一个受试者的每一个trial进行lds操作
    data, labels, groups = dataset["data"], dataset["labels"], dataset["groups"]

    # groups 的 shape 为 (N, 2): 第一列为 subject, 第二列为 trial
    unique_subjects = np.unique(groups[:, 0])
    unique_trials = np.unique(groups[:, 1])
    for subject in unique_subjects:
        for trial in unique_trials:
            idx = np.where((groups[:, 0] == subject) & (groups[:, 1] == trial))[0]
            data[idx] = lds(data[idx])

    data = data.reshape(-1, feature_dim)
    if num_classes == 2:
        labels = labels.reshape(-1)
        # Binary classification: > 5 -> positive class
        labels = (labels > 5).astype(int)
    elif num_classes == 4:
        # (0: Low Valence, 1: High Valence)
        v_labels = (labels[:, 0] > 5).astype(int) 
        # (0: Low Arousal, 1: High Arousal)
        a_labels = (labels[:, 1] > 5).astype(int)
        # 2. 组合为 4 个类别 (0, 1, 2, 3)
        # Class 0: LVLA (v=0, a=0) -> 0*2 + 0 = 0
        # Class 1: LVHA (v=0, a=1) -> 0*2 + 1 = 1
        # Class 2: HVLA (v=1, a=0) -> 1*2 + 0 = 2
        # Class 3: HVHA (v=1, a=1) -> 1*2 + 1 = 3
        labels = v_labels * 2 + a_labels
    labels = np.eye(len(labels), num_classes)[labels].astype("float32")

    # Set attributes for later use
    setattr(args, "feature_dim", feature_dim)
    setattr(args, "num_classes", num_classes)

    return data, labels, groups


def lds(data):
    """
    Forked from https://github.com/XJTU-EEG/LibEER/blob/main/LibEER/data_utils/preprocess.py
    Process data using a linear dynamic system approach.

    :param data: Input data array with shape (time, channel, feature)
    :return: Transformed data with shape (time, channel, feature)
    """
    [num_t, num_channel, num_feature] = data.shape
    # Flatten the channel and feature dimensions
    data = data.reshape((data.shape[0], -1))

    # Initial parameters
    prior_correlation = 0.01
    transition_matrix = 1
    noise_correlation = 0.0001
    observation_matrix = 1
    observation_correlation = 1

    # Calculate the mean for initialization
    mean = np.mean(data, axis=0)
    data = data.T  # Transpose for easier manipulation of time dimension

    num_features, num_samples = data.shape
    P = np.zeros(data.shape)
    U = np.zeros(data.shape)
    K = np.zeros(data.shape)
    V = np.zeros(data.shape)

    # Initial Kalman filter setup
    K[:, 0] = prior_correlation * observation_matrix / (
                observation_matrix * prior_correlation * observation_matrix + observation_correlation) * np.ones(
        (num_features,))
    U[:, 0] = mean + K[:, 0] * (data[:, 0] - observation_matrix * prior_correlation)
    V[:, 0] = (np.ones((num_features,)) - K[:, 0] * observation_matrix) * prior_correlation

    # Apply the Kalman filter over time
    for i in range(1, num_samples):
        P[:, i - 1] = transition_matrix * V[:, i - 1] * transition_matrix + noise_correlation
        K[:, i] = P[:, i - 1] * observation_matrix / (
                    observation_matrix * P[:, i - 1] * observation_matrix + observation_correlation)
        U[:, i] = transition_matrix * U[:, i - 1] + K[:, i] * (
                    data[:, i] - observation_matrix * transition_matrix * U[:, i - 1])
        V[:, i] = (1 - K[:, i] * observation_matrix) * P[:, i - 1]

    # Return the processed data, reshaping it to match the original input shape
    return U.T.reshape((num_t, num_channel, num_feature))



# def get_seed_raw(args):
#     num_classes = 3
#     num_of_channels = 62
#     setattr(args, "seed_raw_path", "E:\\EEG_DataSets\\SEED\\Preprocessed_EEG\\")
#     SEED = SEEDDataset(args.seed_raw_path, sessions=args.session)
#     feature_dim = num_of_channels * SEED.get_feature_dim()
#     dataset = SEED.get_dataset()

#     data, Label, Group = dataset["data"], dataset["labels"], dataset["groups"]
#     data = data.reshape(-1, feature_dim)
#     Label += 1
#     subject_ids = Group[:, 0] # subject ID 

#     setattr(args, "feature_dim", feature_dim)
#     setattr(args, "num_classes", num_classes)

#     min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
#     for i in np.unique(subject_ids):
#         data[subject_ids==i] = min_max_scaler.fit_transform(data[subject_ids == i])
#     print(Label)
#     one_hot_mat = np.eye(len(Label), num_classes)[Label].astype("float32")
#     return data, one_hot_mat, Group

# def get_seed4_raw(args):
#     num_classes = 4
#     num_of_channels = 62
#     setattr(args, "seed_raw_path", "E:\\EEG_DataSets\\SEED_IV\\eeg_raw_data\\")
#     SEEDIV = SEEDIVDataset(args.seed_raw_path, sessions=args.session)
#     feature_dim = num_of_channels * SEEDIV.get_feature_dim()
#     dataset = SEEDIV.get_dataset()

#     data, Label, Group = dataset["data"], dataset["labels"], dataset["groups"]
#     data = data.reshape(-1, feature_dim)
#     subject_ids = Group[:, 0] # subject ID 

#     setattr(args, "feature_dim", feature_dim)
#     setattr(args, "num_classes", num_classes)

#     min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
#     for i in np.unique(subject_ids):
#         data[subject_ids==i] = min_max_scaler.fit_transform(data[subject_ids == i])

#     one_hot_mat = np.eye(len(Label), num_classes)[Label].astype("float32")
#     return data, one_hot_mat, Group

