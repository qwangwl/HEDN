import configargparse
from utils.utils import str2bool
from datetime import datetime

def get_parser():
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", is_config_file=True, default="hedn.yaml", help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument('--num_workers', type=int, default=0)

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--max_iter', type=int, default=1000)
    parser.add_argument('--early_stop', type=int, default=0, help="Early stopping")
    parser.add_argument("--log_interval", type=int, default=1)

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    # Learning rate scheduler parameters
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=str2bool, default=False)

    # Transfer learning related
    parser.add_argument('--transfer_loss_weight', type=float, default=1)
    parser.add_argument('--transfer_loss_type', type=str, default='dann')

    # Data related
    parser.add_argument('--dataset_name', type=str, default="seed3", help="Dataset name")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of classes, varies for different datasets")
    parser.add_argument('--num_subjects', type=int, default=15, help="Number of subjects")
    parser.add_argument('--feature_dim', type=int, default=310)

    # SEED and SEED-IV Feature datasets
    parser.add_argument('--session', type=int, default=1, help="Session number for this training")
    parser.add_argument("--seed3_path", type=str, default="E:\\EEG_DataSets\\SEED\\ExtractedFeatures\\")
    parser.add_argument("--seed4_path", type=str, default="E:\\EEG_DataSets\\SEED_IV\\eeg_feature_smooth\\")
    
    # DEAP、DREAMER Signal
    parser.add_argument("--deap_path", type=str, default="E:\\EEG_DataSets\\DEAP\\")
    parser.add_argument("--emotion", default=None)
    # 特征处理参数
    parser.add_argument("--feature_name", type=str, default="de")
    parser.add_argument("--window_sec", type=int, default=1)
    parser.add_argument("--step_sec", type=int, default=None)

    # Whether to save model
    parser.add_argument('--saved_model', type=str2bool, default=False, help="Whether to save the model during training")
    parser.add_argument("--tmp_saved_path", type=str, default="logs/")
    
    # HEDN related
    parser.add_argument("--constraint_loss_weight", type=float, default=0.01, help="Constraint loss weight")
    
    parser.add_argument("--src_momentum", type=float, default=0.5, help="Momentum parameter")
    parser.add_argument("--tgt_momentum", type=float, default=0.9, help="Momentum parameter")

    # Ablation study
    parser.add_argument("--ablation", type=str, default="main", help="Ablation study type")
    # for Evaluation source numbers
    parser.add_argument("--num_sources", type=int, default=14, help="Number of source clusters")

    return parser   
