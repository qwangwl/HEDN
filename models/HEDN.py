# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loss_funcs import TransferLoss, SupConLoss

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim=310, hidden_1=64, hidden_2=64):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x

    def get_parameters(self):
        params = [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
        ]
        return params
    
class HardClassifier(nn.Module):
    def __init__(self, 
                 input_dim: int = 64,
                 hidden_dim: int = 32,
                 num_classes: int = 3,
                 ):
        super(HardClassifier, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.fc3 = nn.Linear(input_dim, num_classes)

    def forward(self, feature):
        f1 = self.fc1(feature)
        f2 = self.fc2(f1)
        return self.fc3(f2)
    
    def predict(self, feature):
        with torch.no_grad():
            logits = F.softmax(self.forward(feature), dim=1)
            y_preds = torch.argmax(logits, axis=1)
        return y_preds
    

    def get_parameters(self):
        params = [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
            {"params": self.fc3.parameters(), "lr_mult": 1}
        ]
        return params
    

class EasyNetwork(nn.Module):
    def __init__(self, 
                 input_dim=64, 
                 num_classes=3, 
                 num_src_clusters=15, 
                 num_tgt_clusters=15,
                 num_sources=14, 
                 src_momentum=0.5, 
                 tgt_momentum=0.5,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_sources = num_sources
        self.num_src_clusters = num_src_clusters
        self.num_tgt_clusters = num_tgt_clusters
        self.src_momentum = src_momentum
        self.tgt_momentum = tgt_momentum

        self.src_cluster_labels = torch.zeros(num_sources, num_src_clusters, dtype=torch.long)
        self.src_cluster_centers = torch.zeros(num_sources, num_src_clusters, input_dim)
        self.tgt_cluster_centers = torch.zeros(num_tgt_clusters, input_dim)

        # bottomneck
        self.extractor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=input_dim//2),
            nn.Linear(input_dim // 2, input_dim),
        )

    def forward(self, src_feat, src_cluster, src_idx, tgt_feat, tgt_cluster):
        src_feat = self.extractor(src_feat)
        tgt_feat = self.extractor(tgt_feat)
        self.update_source_cluster_centers(src_feat, src_cluster, src_idx)
        tgt_cluster_label = self.map_target_clusters_to_labels(tgt_feat, tgt_cluster, src_idx)
        return tgt_cluster_label[tgt_cluster]

    def update_source_cluster_centers(self, features, clusters, idx):
        new_centers = self.compute_cluster_center(features, clusters, self.num_src_clusters)
        old_centers = self.src_cluster_centers[idx]
        updated = self.src_momentum * old_centers + (1 - self.src_momentum) * new_centers
        self.src_cluster_centers[idx] = updated

    def update_target_cluster_centers(self, features, clusters):
        new_centers = self.compute_cluster_center(features, clusters, self.num_tgt_clusters)
        old_centers = self.tgt_cluster_centers
        updated = self.tgt_momentum * old_centers + (1 - self.tgt_momentum) * new_centers
        self.tgt_cluster_centers = updated
    
    def compute_cluster_center(self, features, clusters, num_clusters):
        # print(clusters, num_clusters)
        one_hot = F.one_hot(clusters.to(torch.long), num_classes=num_clusters).float()
        cluster_counts = one_hot.sum(dim=0) + 1e-6
        centers = torch.matmul(
            torch.inverse(torch.diag(cluster_counts)) + torch.eye(num_clusters),
            torch.matmul(one_hot.T, features.cpu())
        )
        return centers
    
    @torch.no_grad()
    def map_target_clusters_to_labels(self, tgt_feat, tgt_cluster, src_idx):
        self.update_target_cluster_centers(tgt_feat, tgt_cluster)
        src_cluster_center = self.src_cluster_centers[src_idx]

        # TODO Calculate similarity between target and source cluster centers
        tgt_cluster_center = F.normalize(self.tgt_cluster_centers, p=2, dim=1)
        src_cluster_center = F.normalize(src_cluster_center, p=2, dim=1)
        sim_matrix = torch.matmul(tgt_cluster_center, src_cluster_center.T)

        # TODO Get the top indices of the source clusters that are most similar to the target clusters
        src_cluster_label = self.src_cluster_labels[src_idx]

        top_indices = torch.argmax(sim_matrix, dim=1)
        tgt_cluster_label = src_cluster_label[top_indices]
        return tgt_cluster_label
    
    @torch.no_grad()
    def predict(self, tgt_feat):
        # Voting to compute target cluster label
        tgt_feat = self.extractor(tgt_feat).cpu()
        tgt_centers = F.normalize(self.tgt_cluster_centers, p=2, dim=1)  # [num_tgt_clusters, D]
        votes = torch.zeros(tgt_feat.size(0), self.num_sources, dtype=torch.long)

        # Compute nearest target cluster for each sample
        sim_tgt = torch.matmul(F.normalize(tgt_feat, p=2, dim=1), tgt_centers.T)  # [N, num_tgt_clusters]
        tgt_cluster_idx = torch.argmax(sim_tgt, dim=1)  # [N]

        for i in range(self.num_sources):
            src_centers = F.normalize(self.src_cluster_centers[i], p=2, dim=1)  # [num_src_clusters, D]
            sim_src = torch.matmul(tgt_centers, src_centers.T)  # [num_tgt_clusters, num_src_clusters]
            nn_idx = torch.argmax(sim_src, dim=1)  # [num_tgt_clusters]
            # Use source cluster labels for voting
            votes[:, i] = self.src_cluster_labels[i][nn_idx[tgt_cluster_idx]]

        tgt_cluster_label = torch.mode(votes, dim=1).values  # [N]
        return tgt_cluster_label
    
    @torch.no_grad()
    def initialize_cluster_labels(self, label_lists, cluster_lists):
        for i, (labels, clusters) in enumerate(zip(label_lists, cluster_lists)):
            true_labels = torch.argmax(labels, dim=-1)
            for cluster in np.unique(clusters):
                indices = np.where(clusters == cluster)[0]
                if len(indices) == 0:
                    self.src_cluster_labels[i, cluster] = 0
                else:
                    majority = np.bincount(true_labels[indices]).argmax()
                    self.src_cluster_labels[i, cluster] = majority

    @torch.no_grad()
    def initialize_cluster_centers(self, feat_lists, cluster_lists):
        for i, (feature, cluster) in enumerate(zip(feat_lists, cluster_lists)):
            feature = self.extractor(feature.cuda()).cpu()
            self.src_cluster_centers[i] = self.compute_cluster_center(feature, cluster, self.num_src_clusters)

    def get_parameters(self):
        params = [
            {"params": self.extractor.parameters(), "lr_mult": 1},
        ]
        return params
    

class HEDN(nn.Module):
    # The Pytorch implementation of Hard-Easy Dual Network (HEDN) for EEG-based Emotion Recognition
    def __init__(self, 
                 input_dim=310, 
                 num_classes=3, 
                 max_iter=1000, 
                 transfer_loss_type="dann", 
                 num_src_clusters=15,
                 num_tgt_clusters=15,
                 num_sources=14,
                 src_momentum=0.5,
                 tgt_momentum=0.1,
                 ablation = "main",
                 **kwargs):
        super(HEDN, self).__init__()

        self.num_classes = num_classes
        self.num_sources = num_sources
        
        self.feature_extractor = FeatureExtractor(input_dim=input_dim)
        self.hard_classifier = HardClassifier(input_dim=64, num_classes=num_classes)
        
        self.easy_network = EasyNetwork(
            input_dim=64, 
            src_momentum=src_momentum,
            tgt_momentum=tgt_momentum,
            num_classes=num_classes, 
            num_sources=num_sources,
            num_src_clusters=num_src_clusters, 
            num_tgt_clusters=num_tgt_clusters,
        )

        self.cls_loss = nn.CrossEntropyLoss()
        # cross-network consistency loss
        self.consis_loss = nn.CrossEntropyLoss()
        self.clu_loss = SupConLoss(temperature = 1)

        self.hard_advcriterion = TransferLoss(
            loss_type=transfer_loss_type,
            max_iter=max_iter,
            num_class=num_classes,
            **kwargs
        )

        self.ablation = ablation
    def hard_forward(self, src_feat, tgt_feat, src_label):

        src_logit = self.hard_classifier(src_feat)
        loss_cls = self.cls_loss(src_logit, src_label)
        loss_adv = self.hard_advcriterion(src_feat, tgt_feat)
        return loss_cls , loss_adv

    def easy_forward(self, src_feat_easy, tgt_feat, src_cluster_easy, tgt_cluster):
        src_clu_loss = self.clu_loss(
            self.easy_network.extractor(src_feat_easy),
            src_cluster_easy.squeeze(0)
        )
        tgt_clu_loss = self.clu_loss(
            self.easy_network.extractor(tgt_feat),
            tgt_cluster.squeeze(0)
        )
        return src_clu_loss, tgt_clu_loss

    def forward(self, srcs, tgt, src_labels, src_clusters, tgt_cluster):
        # [B, N, F] - > [N, B, F]
        # B: batch_size(批量大小)
        # N: 源域数量
        # F: 特征维度
        # print(srcs[0])
        srcs = srcs.permute(1, 0, 2) 
        src_labels = src_labels.permute(1, 0, 2)
        src_clusters = src_clusters.permute(1, 0)

        # TODO SRA
        hard_idx, easy_idx = self.source_assessment(srcs, src_labels, tgt)

        # TODO shared feature extractor
        tgt_feat = self.feature_extractor(tgt)
        easy_feat, easy_cluster = self.feature_extractor(srcs[easy_idx]), src_clusters[easy_idx]
        hard_feat, hard_label = self.feature_extractor(srcs[hard_idx]), src_labels[hard_idx]

        loss_cls , loss_adv = self.hard_forward(hard_feat, tgt_feat, hard_label)

        # TODO easy loss
        src_clu_loss, tgt_clu_loss = self.easy_forward(easy_feat, tgt_feat, easy_cluster, tgt_cluster)

        # Network Only Hard
        tgt_proto_pred = self.easy_network(
            easy_feat.detach(),
            easy_cluster, 
            easy_idx, 
            tgt_feat.detach(),
            tgt_cluster
        )
        tgt_logit = self.hard_classifier(tgt_feat)
        loss_consis = self.consis_loss(tgt_logit, tgt_proto_pred.to(tgt_logit.device))
        return loss_cls , loss_adv, loss_consis, src_clu_loss, tgt_clu_loss, easy_idx, hard_idx

    def predict(self, data, mode="target"):
        if mode == "source":
            return self.predict_by_hard(data)
        elif mode == "target":
            return self.predict_by_easy(data)
        
    def predict_by_easy(self, data):
        self.eval()
        with torch.no_grad():
            feat = self.feature_extractor(data)
            pred = self.easy_network.predict(feat)
        return pred
    
    def predict_by_hard(self, data):
        self.eval()
        with torch.no_grad():
            feat = self.feature_extractor(data)
            pred = self.hard_classifier.predict(feat)
        return pred
    
    @torch.no_grad()
    def source_assessment(self, srcs, src_labels, tgt):
        num_sources = srcs.size(0)
        scores = torch.zeros(num_sources, device=srcs.device)

        tgt_feat = self.feature_extractor(tgt)
        for i in range(num_sources):
            src_feat = self.feature_extractor(srcs[i])
            # Compute only the hard branch losses (no similarity loss)
            loss_cls, _ = self.hard_forward(src_feat, tgt_feat, src_labels[i])

            # Criterion: hard source = smaller loss
            scores[i] = -loss_cls

        hard_idx = torch.argmin(scores)
        easy_idx = torch.argmax(scores)

        return hard_idx, easy_idx

    @torch.no_grad
    def source_assessment_by_random(self):
        hard_idx, easy_idx = torch.randint(0, self.num_sources, (2,))
        return hard_idx, easy_idx
    
    
    def get_step1_parameters(self):
        params = [
            *self.feature_extractor.get_parameters(),
            *self.hard_classifier.get_parameters(),
        ]
        params.append(
            {"params": self.hard_advcriterion.loss_func.domain_classifier.parameters(), "lr_mult":1}
        )
        return params

    def get_step2_parameters(self):
        params = [
            *self.easy_network.get_parameters(),
        ]
        return params

    @torch.no_grad()
    def on_training_start(self, srcs, src_clusters, src_labels):
        feats = [self.feature_extractor(src).cpu() for src in srcs]
        self.easy_network.initialize_cluster_labels(src_labels, src_clusters)
        self.easy_network.initialize_cluster_centers(feats, src_clusters)


    def get_state(self):
        return {
            "model": self.state_dict(),
            "proto":{
                "src_cluster_labels": self.easy_network.src_cluster_labels.clone().detach(),
                "src_cluster_centers": self.easy_network.src_cluster_centers.clone().detach(),
                "tgt_cluster_centers": self.easy_network.tgt_cluster_centers.clone().detach(),
            }
        }
    
    def load_state(self, state):
        self.load_state_dict(state["model"])
        self.easy_network.src_cluster_labels = state["proto"]["src_cluster_labels"]
        self.easy_network.src_cluster_centers = state["proto"]["src_cluster_centers"]
        self.easy_network.tgt_cluster_centers = state["proto"]["tgt_cluster_centers"]


class AblationHEDN(HEDN):
    def __init__(self, 
                 ablation: str = "main",
                 **hedn_kwargs):
        super(AblationHEDN, self).__init__(**hedn_kwargs)

        self.ablation = ablation
        self.use_hard = True
        self.use_easy = True
        self.use_sim = True

        if self.ablation == "abl_comp_wo_easy":
            # 只训练 hard 分支（HARD 模式）
            self.use_easy = False
            self.use_sim = False
        elif self.ablation == "abl_comp_wo_hard":
            # 只训练 easy 分支（EASY 模式）
            self.use_hard = False
            self.use_sim = False

    def forward(self, srcs, tgt, src_labels, src_clusters, tgt_cluster):
        srcs = srcs.permute(1, 0, 2) 
        src_labels = src_labels.permute(1, 0, 2)
        src_clusters = src_clusters.permute(1, 0)

        if self.ablation == "abl_sra_random":
            hard_idx, easy_idx = self.source_assessment_by_random()
        elif "abl_sra_w_hard" == self.ablation:
            hard_idx, easy_idx = self.source_assessment(srcs, src_labels, tgt)
            easy_idx = hard_idx  
        elif "abl_sra_w_easy" == self.ablation:
            hard_idx, easy_idx = self.source_assessment(srcs, src_labels, tgt)
            hard_idx = easy_idx
        else:
            hard_idx, easy_idx = self.source_assessment(srcs, src_labels, tgt)

        tgt_feat = self.feature_extractor(tgt)
        easy_feat, easy_cluster = self.feature_extractor(srcs[easy_idx]), src_clusters[easy_idx]
        hard_feat, hard_label = self.feature_extractor(srcs[hard_idx]), src_labels[hard_idx]

        loss_cls, loss_adv, src_clu_loss, tgt_clu_loss, loss_consis = 0, 0, 0, 0, 0

        if self.use_hard:
            loss_cls, loss_adv = self.hard_forward(hard_feat, tgt_feat, hard_label)
        if self.use_easy:
            src_clu_loss, tgt_clu_loss = self.easy_forward(easy_feat, tgt_feat, easy_cluster, tgt_cluster)
            # update memory bank
            tgt_proto_pred = self.easy_network(
                    easy_feat.detach(),
                    easy_cluster, 
                    easy_idx, 
                    tgt_feat.detach(),
                    tgt_cluster
                )
            if self.use_sim:
                tgt_logit = self.hard_classifier(tgt_feat)
                loss_consis = self.consis_loss(tgt_logit, tgt_proto_pred.to(tgt_logit.device))
        
        return loss_cls, loss_adv, loss_consis, src_clu_loss, tgt_clu_loss, easy_idx, hard_idx
    
    def predict(self, data, mode="target"):
        if mode == "source":
            return self.predict_by_hard(data)
        elif mode == "target":
            if self.ablation == "abl_comp_wo_easy":
                return self.predict_by_hard(data)
            return self.predict_by_easy(data)
        