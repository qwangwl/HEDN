
import torch
from models import HEDN, EASY, HARD
from trainers import HEDNTrainer, HEDNTrainer_Ablation_Comp

def get_model_utils(args):
    """
    Get model, optimizer, scheduler, and trainer.
    """
    # Model parameters
    base_params = {
        "input_dim": args.feature_dim,
        "num_classes": args.num_classes,
    }
    # print(args.num_of_s_clusters, args.num_of_t_clusters)
    pm_params = {
        "transfer_loss_type": args.transfer_loss_type,
        "max_iter": args.max_iter,
        "num_src_clusters": args.num_src_clusters,
        "num_tgt_clusters": args.num_tgt_clusters,
        "num_sources": args.num_sources,
        "src_momentum": args.src_momentum,
        "tgt_momentum": args.tgt_momentum,
    }

    combined_params = {**base_params, **pm_params}
    model = HEDN(**combined_params).cuda()

    # Optimizer
    params = model.get_parameters()
    optimizer = torch.optim.RMSprop(params, lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler default False
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    else:
        scheduler = None

    # Trainer parameters
    trainer_params = {
        "lr_scheduler": scheduler,
        "max_iter": args.max_iter,
        "transfer_loss_weight": args.transfer_loss_weight,
        "constraint_loss_weight": args.constraint_loss_weight,
        "early_stop": args.early_stop,
        "log_interval": args.log_interval,
    }

    trainer = HEDNTrainer(
        model, 
        optimizer, 
        **trainer_params
    )
    return trainer

def get_model_utils_for_ablation(args):
    """
    Get model, optimizer, scheduler, and trainer.
    """
    # Model parameters
    base_params = {
        "input_dim": args.feature_dim,
        "num_classes": args.num_classes,
    }
    # print(args.num_of_s_clusters, args.num_of_t_clusters)
    pm_params = {
        "transfer_loss_type": args.transfer_loss_type,
        "max_iter": args.max_iter,
        "num_src_clusters": args.num_src_clusters,
        "num_tgt_clusters": args.num_tgt_clusters,
        "num_sources": args.num_sources,
        "src_momentum": args.src_momentum,
        "tgt_momentum": args.tgt_momentum,
        "ablation": args.ablation,
    }
    print(args.ablation)
    combined_params = {**base_params, **pm_params}
    if "abl_comp_wo_easy" == args.ablation:
        model = HARD(**combined_params).cuda()
    elif "abl_comp_wo_hard" == args.ablation:
        model = EASY(**combined_params).cuda()
    else:
        model = HEDN(**combined_params).cuda()

    # Optimizer
    params = model.get_parameters()
    optimizer = torch.optim.RMSprop(params, lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    else:
        scheduler = None

    # Trainer parameters
    trainer_params = {
        "lr_scheduler": scheduler,
        "max_iter": args.max_iter,
        "transfer_loss_weight": args.transfer_loss_weight,
        "constraint_loss_weight": args.constraint_loss_weight,
        "early_stop": args.early_stop,
        "log_interval": args.log_interval,
    }
    if "comp" in args.ablation:
        # w/o HEDN
        # print(args.ablation)
        trainer = HEDNTrainer_Ablation_Comp(
            model, 
            optimizer, 
            ablation=args.ablation,
            **trainer_params
        )
    else:
        trainer = HEDNTrainer(
            model, 
            optimizer, 
            **trainer_params
        )
    return trainer