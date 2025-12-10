
import torch
from models import HEDN, AblationHEDN
from trainers import HEDNTrainer, AblationHEDNTrainer

def get_model_utils(args):
    """
    Get model, optimizer, scheduler, and trainer.
    """
    # Model parameters
    params = {
        "input_dim": args.feature_dim,
        "num_classes": args.num_classes,
        "transfer_loss_type": args.transfer_loss_type,
        "max_iter": args.max_iter,
        "num_src_clusters": args.num_src_clusters,
        "num_tgt_clusters": args.num_tgt_clusters,
        "num_sources": args.num_sources,
        "src_momentum": args.src_momentum,
        "tgt_momentum": args.tgt_momentum,
    }

    model = HEDN(**params).cuda()

    # Optimizer
    params = model.get_step1_parameters()
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

    params = {
        "input_dim": args.feature_dim,
        "num_classes": args.num_classes,
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
    model = AblationHEDN(**params).cuda()
    
    # Optimizer
    params = model.get_step1_parameters()
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
    trainer = AblationHEDNTrainer(
        model, 
        optimizer, 
        ablation=args.ablation,
        **trainer_params
    )
    return trainer