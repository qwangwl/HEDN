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
    
class LabelClassifier(nn.Module):
    def __init__(self, 
                 input_dim: int = 64,
                 hidden_dim: int = 32,
                 num_of_class: int = 3,
                 ):
        super(LabelClassifier, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.fc3 = nn.Linear(input_dim, num_of_class)

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
    

class ProtoClassifier(nn.Module):
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
        self.classifier = LabelClassifier(input_dim=64, num_of_class=num_classes)

        self.proto_classifier = ProtoClassifier(
            input_dim=64, 
            src_momentum=src_momentum,
            tgt_momentum=tgt_momentum,
            num_classes=num_classes, 
            num_sources=num_sources,
            num_src_clusters=num_src_clusters, 
            num_tgt_clusters=num_tgt_clusters,
        )

        self.ce_loss = nn.CrossEntropyLoss()
        self.sim_loss = nn.CrossEntropyLoss()
        self.clu_loss = SupConLoss(temperature = 1)

        self.adv_criterion = TransferLoss(
            loss_type=transfer_loss_type,
            max_iter=max_iter,
            num_class=num_classes,
            **kwargs
        )

        self.ablation = ablation

        self.count = 0
    def forward(self, srcs, tgt, src_labels, src_clusters, tgt_cluster):
        # [B, N, F] - > [N, B, F]
        # B: batch_size(批量大小)
        # N: 源域数量
        # F: 特征维度
        # print(srcs[0])
        srcs = srcs.permute(1, 0, 2) 
        src_labels = src_labels.permute(1, 0, 2)
        src_clusters = src_clusters.permute(1, 0)

        if "abl_tda_random" == self.ablation:
            hard_idx, easy_idx = self.task_assess_by_random()
        elif "abl_tda_wo_easy" == self.ablation:
            hard_idx, easy_idx = self.task_assess_by_tda(srcs, src_labels, tgt, w_src=1.0)
            easy_idx = hard_idx
        elif "abl_tda_wo_hard" == self.ablation:
            hard_idx, easy_idx = self.task_assess_by_tda(srcs, src_labels, tgt, w_src=1.0)
            hard_idx = easy_idx
        elif "abl_tda_only_adv" == self.ablation:
            hard_idx, easy_idx = self.task_assess_by_tda(srcs, src_labels, tgt, w_src=0.0)
        elif "abl_tda_src_adv" == self.ablation:
            hard_idx, easy_idx = self.task_assess_by_tda(srcs, src_labels, tgt, w_src=0.5)
        else:
            hard_idx, easy_idx = self.task_assess_by_tda(srcs, src_labels, tgt, w_src=1.0)

        # TODO Use Hardest Task to Update Classifier
        src_feat_hard = self.feature_extractor(srcs[hard_idx])
        src_logit_hard = self.classifier(src_feat_hard)
        loss_cls = self.ce_loss(src_logit_hard, src_labels[hard_idx])

        # TODO Use Hardest Task to Update Discriminator
        tgt_feat = self.feature_extractor(tgt)
        loss_adv = self.adv_criterion(src_feat_hard, tgt_feat)

        # Network Only Hard
        # TODO Use Easiest Task to Update Proto Classifier
        # Easiest Task Dont Update FeatureExtractor
        src_feat_easy = self.feature_extractor(srcs[easy_idx])
        src_cluster_easy = src_clusters[easy_idx]

        tgt_proto_pred = self.proto_classifier(
            src_feat_easy.detach(),
            src_cluster_easy, 
            easy_idx, 
            tgt_feat.detach(),
            tgt_cluster
        )

        src_clu_loss = self.clu_loss(
            self.proto_classifier.extractor(src_feat_easy),
            src_clusters[easy_idx].squeeze(0),
        )

        tgt_clu_loss = self.clu_loss(
            self.proto_classifier.extractor(tgt_feat), 
            tgt_cluster.squeeze(0), 
        )
        
        tgt_logit = self.classifier(tgt_feat)
        loss_sim = self.sim_loss(tgt_logit, tgt_proto_pred.to(tgt_logit.device))
        return loss_cls , loss_adv, loss_sim, src_clu_loss, tgt_clu_loss, easy_idx, hard_idx
    
    def predict(self, data, mode="target"):
        if mode == "source":
            return self.predict_by_hard(data)
        elif mode == "target":
            return self.predict_by_easy(data)
        
    def predict_by_easy(self, data):
        self.eval()
        with torch.no_grad():
            feat = self.feature_extractor(data)
            pred = self.proto_classifier.predict(feat)
        return pred
    
    def predict_by_hard(self, data):
        self.eval()
        with torch.no_grad():
            feat = self.feature_extractor(data)
            pred = self.classifier.predict(feat)
        return pred
    
    def get_parameters(self):
        params = [
            *self.feature_extractor.get_parameters(),
            *self.classifier.get_parameters(),
        ]
        params.append(
            {"params": self.adv_criterion.loss_func.domain_classifier.parameters(), "lr_mult":1}
        )
        return params

    def on_training_start(self, srcs, src_clusters, src_labels):
        with torch.no_grad():
            feats = [self.feature_extractor(src).cpu() for src in srcs]
        self.proto_classifier.initialize_cluster_labels(src_labels, src_clusters)
        self.proto_classifier.initialize_cluster_centers(feats, src_clusters)

    @torch.no_grad()
    def task_assess_by_tda(self, srcs, src_labels, tgt, w_src=0.5):
        src_all = srcs.reshape(-1, srcs.size(-1))
        src_lbl_all = src_labels.reshape(-1, src_labels.size(-1))
        src_feat_all = self.feature_extractor(src_all)
        src_logits_all = self.classifier(src_feat_all)
        src_loss_all = F.cross_entropy(src_logits_all, src_lbl_all, reduction='none')
        src_loss_chunks = src_loss_all.view(self.num_sources, -1)
        src_chunk_loss = src_loss_chunks.mean(dim=-1)

        tgt_feat = self.feature_extractor(tgt)
        adv_losses = torch.zeros(self.num_sources, device=tgt.device)
        for i in range(self.num_sources):
            src_feat_i = self.feature_extractor(srcs[i])
            adv_losses[i] = self.adv_criterion(src_feat_i, tgt_feat).detach()

        w_adv = 1 - w_src
        balance = w_src * src_chunk_loss + w_adv * torch.abs(adv_losses - 0.693)

        hard_idx = torch.argmax(balance)
        easy_idx = torch.argmin(balance)

        return hard_idx, easy_idx

    @torch.no_grad
    def task_assess_by_random(self):
        hard_idx, easy_idx = torch.randint(0, self.num_sources, (2,))
        return hard_idx, easy_idx
    
    def get_state(self):
        return {
            "model": self.state_dict(),
            "proto":{
                "src_cluster_labels": self.proto_classifier.src_cluster_labels.clone().detach(),
                "src_cluster_centers": self.proto_classifier.src_cluster_centers.clone().detach(),
                "tgt_cluster_centers": self.proto_classifier.tgt_cluster_centers.clone().detach(),
            }
        }
    
    def load_state(self, state):
        self.load_state_dict(state["model"])
        self.proto_classifier.src_cluster_labels = state["proto"]["src_cluster_labels"]
        self.proto_classifier.src_cluster_centers = state["proto"]["src_cluster_centers"]
        self.proto_classifier.tgt_cluster_centers = state["proto"]["tgt_cluster_centers"]


class HARD(HEDN):
    def __init__(self, 
                 input_dim=310, 
                 num_classes=3, 
                 max_iter=1000, 
                 transfer_loss_type="dann", 
                 num_src_clusters=15,
                 num_tgt_clusters=15,
                 num_sources=14,
                 src_momentum=0.5,
                 tgt_momentum=0.9,
                 **kwargs):
        super(HARD, self).__init__(
            input_dim=input_dim, 
            num_classes=num_classes, 
            max_iter=max_iter, 
            transfer_loss_type=transfer_loss_type, 
            num_src_clusters=num_src_clusters,
            num_tgt_clusters=num_tgt_clusters,
            num_sources=num_sources,
            src_momentum=src_momentum,
            tgt_momentum=tgt_momentum,
            **kwargs
        )

    def forward(self, srcs, tgt, src_labels, src_clusters, tgt_cluster):
        # [B, N, F] -> [N, B, F]
        # B: batch_size
        # N: number of source domains
        # F: feature dimension
        srcs = srcs.permute(1, 0, 2) 
        src_labels = src_labels.permute(1, 0, 2)
        src_clusters = src_clusters.permute(1, 0)

        # Task Difficulty Assessment By Source Classifier Loss
        # hard_idx, easy_idx = self.task_assess_by_src_loss(srcs, src_labels)
        hard_idx, easy_idx = self.task_assess_by_tda(srcs, src_labels, tgt, w_src=1.0)
        # TODO Use Hardest Task to Update Classifier
        src_feat_hard = self.feature_extractor(srcs[hard_idx])
        src_logit_hard = self.classifier(src_feat_hard)
        loss_cls = self.ce_loss(src_logit_hard, src_labels[hard_idx])

        # TODO Use Hardest Task to Update Discriminator
        tgt_feat = self.feature_extractor(tgt)
        loss_adv = self.adv_criterion(src_feat_hard, tgt_feat)

        return loss_cls , loss_adv, 0, 0, 0
    
    def predict(self, data, mode="target"):
        self.eval()
        with torch.no_grad():
            src_feat = self.feature_extractor(data)
            src_pred = self.classifier.predict(src_feat)
        return src_pred
    
class EASY(HEDN):
    def __init__(self, 
                 input_dim=310, 
                 num_classes=3, 
                 max_iter=1000, 
                 transfer_loss_type="dann", 
                 num_src_clusters=15,
                 num_tgt_clusters=15,
                 num_sources=14,
                 src_momentum=0.5,
                 tgt_momentum=0.9,
                 **kwargs):
        super(EASY, self).__init__(
            input_dim=input_dim, 
            num_classes=num_classes, 
            max_iter=max_iter, 
            transfer_loss_type=transfer_loss_type, 
            num_src_clusters=num_src_clusters,
            num_tgt_clusters=num_tgt_clusters,
            num_sources=num_sources,
            src_momentum=src_momentum,
            tgt_momentum=tgt_momentum,
            **kwargs
        )
    
    def forward(self, srcs, tgt, src_labels, src_clusters, tgt_cluster):
        # [B, N, F] -> [N, B, F]
        # B: batch_size
        # N: number of source domains
        # F: feature dimension
        srcs = srcs.permute(1, 0, 2) 
        src_labels = src_labels.permute(1, 0, 2)
        src_clusters = src_clusters.permute(1, 0)

        hard_idx, easy_idx = self.task_assess_by_tda(srcs, src_labels, tgt, w_src=1.0)

        tgt_feat = self.feature_extractor(tgt)
 
        # Network Only Hard
        # TODO Use Easiest Task to Update Proto Classifier
        # Easiest Task Does Not Update FeatureExtractor
        src_feat_easy = self.feature_extractor(srcs[easy_idx])
        src_cluster_easy = src_clusters[easy_idx]

        tgt_proto_pred = self.proto_classifier(
            src_feat_easy.detach(),
            src_cluster_easy, 
            easy_idx, 
            tgt_feat.detach(),
            tgt_cluster
        )

        src_clu_loss = self.clu_loss(
            self.proto_classifier.extractor(src_feat_easy),
            src_clusters[easy_idx].squeeze(0),
        )

        tgt_clu_loss = self.clu_loss(
            self.proto_classifier.extractor(tgt_feat), 
            tgt_cluster.squeeze(0), 
        )
        
        return 0, 0, 0, src_clu_loss, tgt_clu_loss
    
    def predict(self, data, mode="target"):

        if mode == "source":
            self.eval()
            with torch.no_grad():
                src_feat = self.feature_extractor(data)
                src_pred = self.classifier.predict(src_feat)
            return src_pred
        
        elif mode == "target":
            self.eval()
            with torch.no_grad():
                tgt_feat = self.feature_extractor(data)
                tgt_proto_pred = self.proto_classifier.predict(tgt_feat)
            return tgt_proto_pred