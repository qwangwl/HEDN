# HEDN  
[English](./README.md) | [简体中文](./README_zh.md)

## Overview

We propose **Hard-Easy Dual Network (HEDN)**, a novel framework for cross-subject EEG-based emotion recognition. 

![Overall Architecture](./overall.png)

---

## Project Structure

```bash
├── hedn.yaml               # Experiment configurations and hyperparameters
├── config.py              # Core configuration management
├── requirements.txt       # Python dependencies
│
├── data_utils/            # Data preprocessing and dataloader utilities
├── datasets/              # Dataset loading and organization
│
├── models/                # Model architecture definitions
├── loss_funcs/            # Custom loss functions proposed in the paper
├── trainers/              # Training logic and optimization routines
├── utils/                 # General-purpose utility functions
│
├── cross_subject.py       # Main entry point for cross-subject evaluation
├── cross_dataset.py       # Main entry point for cross-dataset evaluation
│
└── get_model_utils.py     # Helper functions for model and trainer initialization
```

---

## Datasets

We evaluate HEDN on three widely used public benchmark datasets:

- **[SEED](https://bcmi.sjtu.edu.cn/~seed/index.html)**  
  15 subjects × 3 sessions × 15 emotion-eliciting film clips (positive, neutral, negative)

- **[SEED-IV](https://bcmi.sjtu.edu.cn/~seed/index.html)**  
  Extension of SEED with 4 emotion categories: happy, sad, fear, neutral

- **[DEAP](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/)**  
  32 subjects × 40 one-minute music videos annotated with valence, arousal, dominance, and liking scores

> **⚠️ Note**: Due to data usage agreements, we cannot redistribute these datasets. Please apply for access directly from the official websites.

---

## Experimental Platform

All experiments were conducted under the following environment:

- **OS**: Windows 11  
- **CPU**: Intel Core i7-14700  
- **RAM**: 32 GB  
- **GPU**: NVIDIA RTX 5070 Ti (CUDA 11.8)  
- **Dependencies**: See `requirements.txt` for exact versions

This setup ensures reproducibility and efficient training for deep neural networks. Users with comparable hardware should expect consistent performance.

---

## Prerequisites

Ensure the following software is installed:

- Python ≥ 3.12  
- PyTorch ≥ 2.8.0  
- scikit-learn  
- NumPy  

Install all dependencies via:

```bash
pip install -r requirements.txt
```

---

## How to Use

### 1. Data Preparation

Download the SEED, SEED-IV, and DEAP datasets from their official sources. Then, update the path configurations in `hedn.yaml`:

```yaml
seed3_path: "/path/to/your/SEED/Preprocessed_EEG"
seed4_path: "/path/to/your/SEED-IV/eeg_raw_data"
deap_path: "/path/to/your/DEAP/"
```

### 2. Running Experiments

#### Cross-Subject Experiments

Run `cross_subject.py` with the appropriate arguments:

```bash
python cross_subject.py --dataset_name <dataset> [--session <int>] [--emotion <str>]
```

- `<dataset>`: `seed3`, `seed4`, or `deap`  
- `--session`: Required for SEED/SEED-IV (1, 2, or 3)  
- `--emotion`: Required for DEAP (`valence` or `arousal`)

**Examples**:
```bash
# SEED session 1
python cross_subject.py --dataset_name seed3 --session 1

# DEAP with valence labels
python cross_subject.py --dataset_name deap --emotion valence
```

#### Cross-Dataset Experiments

Currently supports `seed3 → seed4` and `seed4 → seed3`:

```bash以下是您提供的英文 README 的修订润色版本，并附上对应的中文版：

---

## English (Revised & Polished)

# HEDN  
[English](./README.md) | [简体中文](./README_zh.md)

## Overview

We propose **Hard-Easy Dual Network (HEDN)**, a novel framework for cross-subject EEG-based emotion recognition. HEDN leverages a dual-branch architecture to jointly model easy-to-transfer and hard-to-transfer patterns across subjects, thereby improving generalization and mitigating negative transfer.

![Overall Architecture](./overall.png)

---

## Project Structure

```bash
├── hedn.yaml               # Experiment configurations and hyperparameters
├── config.py              # Core configuration management
├── requirements.txt       # Python dependencies
│
├── data_utils/            # Data preprocessing and dataloader utilities
├── datasets/              # Dataset loading and organization
│
├── models/                # Model architecture definitions
├── loss_funcs/            # Custom loss functions proposed in the paper
├── trainers/              # Training logic and optimization routines
├── utils/                 # General-purpose utility functions
│
├── cross_subject.py       # Main entry point for cross-subject evaluation
├── cross_dataset.py       # Main entry point for cross-dataset evaluation
│
└── get_model_utils.py     # Helper functions for model and trainer initialization
```

---

## Datasets

We evaluate HEDN on three widely used public benchmark datasets:

- **[SEED](https://bcmi.sjtu.edu.cn/~seed/index.html)**  
  15 subjects × 3 sessions × 15 emotion-eliciting film clips (positive, neutral, negative)

- **[SEED-IV](https://bcmi.sjtu.edu.cn/~seed/index.html)**  
  Extension of SEED with 4 emotion categories: happy, sad, fear, neutral

- **[DEAP](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/)**  
  32 subjects × 40 one-minute music videos annotated with valence, arousal, dominance, and liking scores

> **⚠️ Note**: Due to data usage agreements, we cannot redistribute these datasets. Please apply for access directly from the official websites.

---

## Experimental Platform

All experiments were conducted under the following environment:

- **OS**: Windows 11  
- **CPU**: Intel Core i7-14700  
- **RAM**: 32 GB  
- **GPU**: NVIDIA RTX 5070 Ti (CUDA 11.8)  
- **Dependencies**: See `requirements.txt` for exact versions

This setup ensures reproducibility and efficient training for deep neural networks. Users with comparable hardware should expect consistent performance.

---

## Prerequisites

Ensure the following software is installed:

- Python ≥ 3.12  
- PyTorch ≥ 2.8.0  
- scikit-learn  
- NumPy  

Install all dependencies via:

```bash
pip install -r requirements.txt
```

---

## How to Use

### 1. Data Preparation

Download the SEED, SEED-IV, and DEAP datasets from their official sources. Then, update the path configurations in `hedn.yaml`:

```yaml
seed3_path: "/path/to/your/SEED/Preprocessed_EEG"
seed4_path: "/path/to/your/SEED-IV/eeg_raw_data"
deap_path: "/path/to/your/DEAP/"
```

### 2. Running Experiments

#### Cross-Subject Experiments

Run `cross_subject.py` with the appropriate arguments:

```bash
python cross_subject.py --dataset_name <dataset> [--session <int>] [--emotion <str>]
```

- `<dataset>`: `seed3`, `seed4`, or `deap`  
- `--session`: Required for SEED/SEED-IV (1, 2, or 3)  
- `--emotion`: Required for DEAP (`valence` or `arousal`)

**Examples**:
```bash
# SEED session 1
python cross_subject.py --dataset_name seed3 --session 1

# DEAP with valence labels
python cross_subject.py --dataset_name deap --emotion valence
```

#### Cross-Dataset Experiments

Currently supports `seed3 → seed4` and `seed4 → seed3`:

```bash
python cross_dataset.py
```

#### Run All Experiments

Use the provided script to reproduce all results:

```bash
python run.py
```