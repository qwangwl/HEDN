# -*- encoding: utf-8 -*-
'''
file       :HEDNTrainer.py
Date       :2025/09/21 15:08:57
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''

import torch
import numpy as np
import copy

class HEDNTrainer(object):
    def __init__(self, 
                 model, 
                 optimizer, 
                 lr_scheduler = None,
                 max_iter: int = 1000, 
                 log_interval: int = 1, 
                 early_stop: int = 0,
                 transfer_loss_weight: int = 1,
                 constraint_loss_weight: int = 0.01,
                 device: str = "cuda:0", 
                 **kwargs):
        super(HEDNTrainer, self).__init__()

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.log_interval = log_interval
        self.early_stop = early_stop
        self.max_iter = max_iter

        self.transfer_loss_weight = transfer_loss_weight
        self.constraint_loss_weight = constraint_loss_weight

        self.device = device
        self.stop = 0
        self.best_model_state = None

        # The magic optimizer is used here
        self.fe_opt = torch.optim.Adam(self.model.proto_classifier.get_parameters(), lr=1e-3)

    def get_model_state(self):
        if hasattr(self.model, "get_state"):
            return self.model.get_state()
        return self.model.state_dict()
    

    def get_best_model_state(self):
        return self.best_model_state

    def train(self, source_loaders, target_loader):
        # HEDN不通过Epoch来控制。
        stop = 0
        best_acc = 0.0
        log = []

        self.pre_training_processing(source_loaders)
        
        # torch.save(self.get_model_state(), "init.pth")

        source_iter = iter(source_loaders)
        target_iter = iter(target_loader)

        for it in range(self.max_iter):
            # if it == 100:
            #     torch.save(self.get_model_state(), "mid100.pth")

            self.model.train()
            try:
                src_data, src_label, src_cluster = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loaders)
                src_data, src_label, _ = next(source_iter)
            try:
                tgt_data, _, tgt_cluster = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                tgt_data, _, tgt_cluster = next(target_iter)

            src_data, src_label = src_data.to(
                self.device), src_label.to(self.device)
            tgt_data = tgt_data.to(self.device)

            # Phase 1: Domain Adaptation
            cls_loss, transfer_loss, cons_loss, src_clu_loss, tgt_clu_loss, easy_idx, hard_idx  = self.model(src_data, tgt_data, src_label, src_cluster, tgt_cluster)
            loss = cls_loss + self.transfer_loss_weight * transfer_loss + self.constraint_loss_weight * cons_loss
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            # Phase 2: Update Prototype Feature Extractor
            cls_loss, transfer_loss, cons_loss, src_clu_loss, tgt_clu_loss, easy_idx, hard_idx = self.model(src_data, tgt_data, src_label, src_cluster, tgt_cluster)
            self.fe_opt.zero_grad()
            loss2 = src_clu_loss + tgt_clu_loss
            loss2.backward()
            self.fe_opt.step()

            source_acc = self.test(source_loaders, mode="source")
            target_acc = self.test(target_loader, mode="target")

            best_acc = self._update_best_model(target_acc, best_acc)

            iter_losses = {
                'cls_loss': cls_loss.detach().item() if isinstance(cls_loss, torch.Tensor) else cls_loss,
                'transfer_loss': transfer_loss.detach().item() if isinstance(transfer_loss, torch.Tensor) else transfer_loss,
                'cons_loss': cons_loss.detach().item() if isinstance(cons_loss, torch.Tensor) else cons_loss,
                "src_clu_loss": src_clu_loss.detach().item() if isinstance(src_clu_loss, torch.Tensor) else src_clu_loss,
                "tgt_clu_loss": tgt_clu_loss.detach().item() if isinstance(tgt_clu_loss, torch.Tensor) else tgt_clu_loss,
            }

            log_entry = [v for _, v in iter_losses.items()] + [source_acc, target_acc, best_acc, int(easy_idx.cpu().numpy()) + 1, int(hard_idx.cpu().numpy()) + 1]
            log.append(log_entry)

            info = self._log_training_info(it, iter_losses, source_acc, target_acc, best_acc)

            if self._should_early_stop(best_acc):
                print(info)
                break
        return best_acc, np.array(log, dtype=float)
    
    @torch.no_grad()
    def test(self, dataloader, mode="target"):
        self.model.eval()
        feature, labels, _ = dataloader.dataset.get_data()
        labels = np.argmax(labels.numpy(), axis=1)
        y_preds = self.model.predict(feature.to(self.device), mode).cpu().numpy()
        acc = np.sum(y_preds == labels) / len(labels)
        return acc * 100.

    def pre_training_processing(self, source_loaders):
        dataset = source_loaders.dataset 

        src_datas = []
        src_clusters = []
        src_labels = []
        for domain_idx in range(dataset.num_domains):
            x, y, cluster = dataset.get_single_domain(domain_idx)
            src_datas.append(x.to(self.device))
            src_clusters.append(cluster)
            src_labels.append(y)
        
        self.model.on_training_start(src_datas, src_clusters, src_labels)

    def _update_best_model(self, current_acc: float, best_acc: float):
        """更新最佳模型状态"""
        self.stop += 1
        if current_acc > best_acc:
            best_acc = current_acc
            self.best_model_state = copy.deepcopy(self.get_model_state())
            self.stop = 0
            # torch.save(self.get_model_state(), "best.pth")
        return best_acc
    
    def _should_early_stop(self, best_acc: float):
        """判断是否应该早停"""
        if self.early_stop == 0:
            return False
        return (self.stop >= self.early_stop) or (100 - best_acc < 1e-3)

    def _log_training_info(self, iteration, loss_dict, source_acc, target_acc, best_acc):
        """打印训练信息"""
        info = f'Iter: [{iteration + 1:2d}/{self.max_iter}]'
        for key, value in loss_dict.items():
            info += f', {key}: {value:.4f}'
        info += f', source_acc: {source_acc:.4f}, target_acc: {target_acc:.4f}, best_acc: {best_acc:.4f}'

        if (iteration + 1) % self.log_interval == 0 or iteration == 0:
            print(info)
        return info


class HEDNTrainer_Ablation(HEDNTrainer):
    # The Pytorch implementation of Hard-Easy Dual Network (HEDN) Trainer
    def __init__(self, 
                 model, 
                 optimizer, 
                 lr_scheduler = None,
                 max_iter: int = 1000, 
                 log_interval: int = 1, 
                 early_stop: int = 0,
                 transfer_loss_weight: int = 1,
                 constraint_loss_weight: int = 0.01,
                 device: str = "cuda:0", 
                 ablation: str = "main",
                 **kwargs):

        super(HEDNTrainer_Ablation, self).__init__(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            max_iter=max_iter,
            log_interval=log_interval,
            early_stop=early_stop,
            transfer_loss_weight=transfer_loss_weight,
            constraint_loss_weight=constraint_loss_weight,
            device=device,
            **kwargs
        )
        self.ablation = ablation
        if "abl_comp_wo_hard" == self.ablation:
            params = list(self.model.feature_extractor.get_parameters()) + list(self.model.proto_classifier.get_parameters())
            self.fe_opt = torch.optim.Adam(params, lr=1e-3)

    def train(self, source_loaders, target_loader):
        # HEDN不通过Epoch来控制。
        stop = 0
        best_acc = 0.0
        log = []

        self.pre_training_processing(source_loaders)

        source_iter = iter(source_loaders)
        target_iter = iter(target_loader)

        for it in range(self.max_iter):
            self.model.train()
            try:
                src_data, src_label, src_cluster = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loaders)
                src_data, src_label, _ = next(source_iter)
            try:
                tgt_data, _, tgt_cluster = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                tgt_data, _, tgt_cluster = next(target_iter)

            src_data, src_label = src_data.to(
                self.device), src_label.to(self.device)
            tgt_data = tgt_data.to(self.device)

            # Phase 1: Domain Adaptation
            if "abl_comp_wo_easy" == self.ablation or "abl_comp_wo_clusterloss" == self.ablation:
                cls_loss, transfer_loss, cons_loss, src_clu_loss, tgt_clu_loss  = self.model(src_data, tgt_data, src_label, src_cluster, tgt_cluster)
                loss = cls_loss + self.transfer_loss_weight * transfer_loss + self.constraint_loss_weight * cons_loss
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()

            elif "abl_comp_wo_hard" == self.ablation:
                cls_loss, transfer_loss, cons_loss, src_clu_loss, tgt_clu_loss = self.model(src_data, tgt_data, src_label, src_cluster, tgt_cluster)
                self.fe_opt.zero_grad()
                loss2 = src_clu_loss + tgt_clu_loss
                loss2.backward()
                self.fe_opt.step()
            
            elif "abl_comp_two_stage" == self.ablation:
                cls_loss, transfer_loss, cons_loss, src_clu_loss, tgt_clu_loss = self.model(src_data, tgt_data, src_label, src_cluster, tgt_cluster)
                loss = cls_loss + self.transfer_loss_weight * transfer_loss + self.constraint_loss_weight * cons_loss + src_clu_loss + tgt_clu_loss
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()
            
            elif "abl_comp_wo_clusterloss_target" == self.ablation:
                cls_loss, transfer_loss, cons_loss, src_clu_loss, tgt_clu_loss  = self.model(src_data, tgt_data, src_label, src_cluster, tgt_cluster)
                loss = cls_loss + self.transfer_loss_weight * transfer_loss + self.constraint_loss_weight * cons_loss
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()

                cls_loss, transfer_loss, cons_loss, src_clu_loss, tgt_clu_loss = self.model(src_data, tgt_data, src_label, src_cluster, tgt_cluster)
                self.fe_opt.zero_grad()
                loss2 = src_clu_loss
                loss2.backward()
                self.fe_opt.step()

            elif "abl_comp_wo_clusterloss_source" == self.ablation:
                cls_loss, transfer_loss, cons_loss, src_clu_loss, tgt_clu_loss  = self.model(src_data, tgt_data, src_label, src_cluster, tgt_cluster)
                loss = cls_loss + self.transfer_loss_weight * transfer_loss + self.constraint_loss_weight * cons_loss
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()

                cls_loss, transfer_loss, cons_loss, src_clu_loss, tgt_clu_loss = self.model(src_data, tgt_data, src_label, src_cluster, tgt_cluster)
                self.fe_opt.zero_grad()
                loss2 = tgt_clu_loss
                loss2.backward()
                self.fe_opt.step()

            else:
                cls_loss, transfer_loss, cons_loss, src_clu_loss, tgt_clu_loss, easy_idx, hard_idx  = self.model(src_data, tgt_data, src_label, src_cluster, tgt_cluster)
                loss = cls_loss + self.transfer_loss_weight * transfer_loss + self.constraint_loss_weight * cons_loss
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()

                # Phase 2: Update Prototype Feature Extractor
                cls_loss, transfer_loss, cons_loss, src_clu_loss, tgt_clu_loss, easy_idx, hard_idx = self.model(src_data, tgt_data, src_label, src_cluster, tgt_cluster)
                self.fe_opt.zero_grad()
                loss2 = src_clu_loss + tgt_clu_loss
                loss2.backward()
                self.fe_opt.step()

            source_acc = self.test(source_loaders, mode="source")
            target_acc = self.test(target_loader, mode="target")

            best_acc = self._update_best_model(target_acc, best_acc)

            iter_losses = {
                'cls_loss': cls_loss.detach().item() if isinstance(cls_loss, torch.Tensor) else cls_loss,
                'transfer_loss': transfer_loss.detach().item() if isinstance(transfer_loss, torch.Tensor) else transfer_loss,
                'cons_loss': cons_loss.detach().item() if isinstance(cons_loss, torch.Tensor) else cons_loss,
                "src_clu_loss": src_clu_loss.detach().item() if isinstance(src_clu_loss, torch.Tensor) else src_clu_loss,
                "tgt_clu_loss": tgt_clu_loss.detach().item() if isinstance(tgt_clu_loss, torch.Tensor) else tgt_clu_loss,
            }

            log_entry = [v for _, v in iter_losses.items()] + [source_acc, target_acc, best_acc]
            log.append(log_entry)

            info = self._log_training_info(it, iter_losses, source_acc, target_acc, best_acc)

            if self._should_early_stop(best_acc):
                print(info)
                break
        return best_acc, np.array(log, dtype=float)