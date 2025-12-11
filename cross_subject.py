import os
import copy
os.environ["LOKY_MAX_CPU_COUNT"] = "20"
import time
import torch
import numpy as np
from config import get_parser

from utils.utils import setup_seed
from utils.evaluate import evaluate
from utils.save import save_subject_results, save_final_results

from get_model_utils import get_model_utils, get_model_utils_for_ablation
from data_utils._get_dataset import get_dataset
from data_utils.HEDNLoader import HEDNLoader


def cross_subject(args):
    """
    Main function to run the training process.
    """
    setup_seed(args.seed)
    os.makedirs(args.tmp_saved_path, exist_ok=True)
    
    # Load the dataset first in LOSO to avoid repeated loading for each train
    dataset = get_dataset(args)

    data = dataset["data"]
    labels = dataset["labels"]
    groups = dataset["groups"]

    results = []  # 用来存每一折的结果
    
    all_subjects = np.unique(groups[:, 0])

    for subject in range(1, args.num_subjects + 1):
        # if  subject >= 2:
        #     break
        print(f"开始适应受试者 {subject}")
        # 每一个受试者都重新定义种子。
        setup_seed(args.seed)
        if args.ablation == "num_sources":
            source_subjects = all_subjects[all_subjects != subject]
            selected_subjects = source_subjects[:args.num_sources]
            # print(selected_subjects)
            train_mask = np.isin(groups[:, 0], selected_subjects)
        else:
            train_mask = groups[:, 0] != subject
        # print(len(train_mask))
        # print(all_subjects)
        print(f"训练样本数: {np.sum(train_mask)}, 测试样本数: {np.sum(groups[:,0]==subject)}")
        train_dataset = {
            "data": data[train_mask],
            "labels": labels[train_mask],
            "groups": groups[train_mask]
        }

        val_dataset = {
            "data": data[groups[:,0] == subject],
            "labels": labels[groups[:,0] == subject],
            "groups": groups[groups[:,0] == subject]
        }
        train_loader, val_loader = HEDNLoader(args, train_dataset, val_dataset)()
        trainer  =  get_model_utils(args)
        # 消融实验使用不同的模型构建函数
        # trainer = get_model_utils_for_ablation(args)
        
        start_time = time.time()
        best_val_acc, np_log = trainer.train(train_loader, val_loader)
        train_time = time.time() - start_time

        # ================== Testing ==================
        # # 测试Last Epoch 准确率
        last_val_acc, last_f1_score = evaluate(trainer.model, val_loader, args.device)
        # print(trainer.model.classifier.P[:100])
        # 加载最佳模型
        trainer.model.load_state(trainer.get_best_model_state())
        # print(trainer.model.classifier.P[:100])
        val_start = time.time()
        best_val_acc, best_f1_score = evaluate(trainer.model, val_loader, args.device)
        val_time = (time.time() - val_start) * 1000

        results.append({
            "subject": subject,
            "best_val_acc": best_val_acc,
            "last_val_acc": last_val_acc,
            "best_f1_score": best_f1_score,
            "last_f1_score": last_f1_score,
            "train_time(s)": train_time,
            "val_time(ms)": val_time,
        })

        save_subject_results(args, subject, trainer, np_log)
        print(f"\n受试者 {subject} 完成:")
        print(f"  - 最佳验证准确率: {best_val_acc:.4f}")
        print(f"  - 训练时间: {train_time:.2f}s")
        print(f"  - 验证时间: {val_time:.2f}ms")

    save_final_results(args, results)
    print(f"\n{'='*60}")
    print("所有受试者训练完成！")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    if args.dataset_name == "seed3":
        setattr(args, "num_subjects", 15)
        setattr(args, "num_classes", 3)
        setattr(args, "batch_size", 96)
    elif args.dataset_name == "seed4":
        setattr(args, "num_subjects", 15)
        setattr(args, "num_classes", 4)
        setattr(args, "batch_size", 64)
    elif args.dataset_name == "deap":
        setattr(args, "num_subjects", 32)
        setattr(args, "num_sources", 31)
        setattr(args, "num_classes", 2)
        setattr(args, "batch_size", 96)
    tmp_saved_path = os.path.join(args.tmp_saved_path, 
                                  f"{args.dataset_name}",
                                  f"{args.session}_{args.emotion}")
    setattr(args, "tmp_saved_path", tmp_saved_path)
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    cross_subject(args)