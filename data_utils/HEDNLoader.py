import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

class PMDataset(TensorDataset):
    def __init__(self, d1, d2, d3):
        super(PMDataset, self).__init__()
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3

    def __len__(self):
        return len(self.d1)
    
    def __getitem__(self, idx):
        return self.d1[idx], self.d2[idx], self.d3[idx]
    
    def get_data(self):
        return self.d1, self.d2, self.d3


class MultiSourceDataset(TensorDataset):
    def __init__(self, source_domains):
        """
        :param source_domains: List of N dictionaries, each containing:
                               - "x": Tensor (n_i, F)
                               - "y": Tensor (n_i,)
                               - "cluster": Tensor (n_i,)
        """
        self.num_domains = len(source_domains)

        # Calculate the minimum number of samples
        self.min_samples = min(domain["x"].shape[0] for domain in source_domains)

        # Preprocess data for efficient indexing
        self.x_all = torch.cat([domain["x"][:self.min_samples] for domain in source_domains], dim=0)  # (num_domains * min_samples, F)
        self.y_all = torch.cat([domain["y"][:self.min_samples] for domain in source_domains], dim=0)  # (num_domains * min_samples,)
        self.cluster_all = torch.cat([domain["cluster"][:self.min_samples] for domain in source_domains], dim=0)  # (num_domains * min_samples,)

        # (num_domains, min_samples, F)
        self.x_all = self.x_all.view(self.num_domains, self.min_samples, -1)
        self.y_all = self.y_all.view(self.num_domains, self.min_samples, -1)
        self.cluster_all = self.cluster_all.view(self.num_domains, self.min_samples)

    def __len__(self):
        return self.min_samples

    def __getitem__(self, index):
        """
        Returns:
            x: (num_domains, F)
            y: (num_domains,)
            cluster: (num_domains,)
        """
        return self.x_all[:, index], self.y_all[:, index], self.cluster_all[:, index]

    def get_single_domain(self, domain_index):
        """
        Get data for the specified domain index
        :return: (Tensor(min_samples, F), Tensor(min_samples,), Tensor(min_samples,))
        """
        if 0 <= domain_index < self.num_domains:
            return self.x_all[domain_index], self.y_all[domain_index], self.cluster_all[domain_index]
        raise IndexError(f"index need in between 0~{self.num_domains-1}")

    def get_data(self):
        """
        Get all source domain concatenated data
        :return: (Tensor(num_domains * min_samples, F)
        """
        return self.x_all.view(-1, self.x_all.shape[-1]), self.y_all.view(-1, self.y_all.shape[-1]), self.cluster_all.view(-1)
    

class HEDNLoader(object):
    def __init__(self, args, train_dataset=None, val_dataset=None):
        super(HEDNLoader, self).__init__()
        self.args = args
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        # self.best_cluster_params = {"eps": args.eps, "min_samples": args.min_samples}
        self._find_best_cluster_params(self.val_dataset)

    def __call__(self):
        train_loader = self.create_train_loader()
        val_loader = self.create_val_loader()
        return train_loader, val_loader
    
    def _find_best_cluster_params(self, datasets):

        X = datasets["data"]
        trial_ids = datasets["groups"][:, 1]

        best_nmi = -1
        eps_values = np.arange(1, 5.1, 0.5)
        min_samples_values = range(3, 7, 1)
        for eps in eps_values:
            for min_samples in min_samples_values:
                clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
                labels = clustering.labels_
                nmi = normalized_mutual_info_score(trial_ids, labels)
                if nmi > best_nmi:
                    best_nmi = nmi
                    self.best_cluster_params = {"eps": eps, "min_samples": min_samples}

    def _perform_clustering(self, X, y):
        clustering = DBSCAN(**self.best_cluster_params).fit(X)
        labels = clustering.labels_
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        mask = labels != -1
        return num_clusters, X[mask], y[mask], labels[mask]

    def create_train_loader(self):
        data = self.train_dataset["data"]
        labels = self.train_dataset["labels"]
        subjects = self.train_dataset["groups"][:, 0]

        cluster_results = {}
        min_clusters = float('inf')

        setattr(self.args, "num_sources", len(np.unique(subjects)))
        for subject_id in np.unique(subjects):
            mask = subjects == subject_id
            n, x, y, c = self._perform_clustering(data[mask], labels[mask])
            # 以trial id作为cluster labels进行训练
            # n, x, y, c = len(np.unique(self.train_dataset["groups"][mask, 1])), data[mask], labels[mask], self.train_dataset["groups"][mask, 1].astype(np.int64)
            cluster_results[subject_id] = (n, x, y, c)
            min_clusters = min(min_clusters, n)

        setattr(self.args, "num_src_clusters", min_clusters)

        source_domains = []
        for subject_id, (n, x, y, c) in cluster_results.items():
            # Filters the data to ensure each subject has the same number of clusters.
            cluster_counts = Counter(c)
            top_clusters = [cluster for cluster, _ in cluster_counts.most_common(min_clusters)]
            mask = np.isin(c, top_clusters)

            source_features = torch.from_numpy(x[mask]).type(torch.Tensor)
            source_labels = torch.from_numpy(y[mask]) 
            
            # Reorders cluster IDs to ensure consistency.
            source_cluster = c[mask]
            unique_clusters = np.unique(source_cluster)
            cluster_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_clusters)}

            source_cluster = torch.tensor([cluster_mapping[label] for label in source_cluster], dtype=torch.long)

            source_domains.append({
                "x": source_features,
                "y": source_labels,
                "cluster": source_cluster
            })
        
        source_datasets = MultiSourceDataset(source_domains)
        source_loader = DataLoader(
            dataset=source_datasets,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            drop_last=True,
        )
        return source_loader
    
    def create_val_loader(self):
        data = self.val_dataset["data"]
        labels = self.val_dataset["labels"]
        
        n, x, y, c = self._perform_clustering(data, labels)
        # print(type(c), c.dtype)
        # n, x, y, c = len(np.unique(self.val_dataset["groups"][:, 1])), data, labels, (self.val_dataset["groups"][:, 1]-1).astype(np.int64)
        
        setattr(self.args, "num_tgt_clusters", n)
        val_dataset = PMDataset(
            torch.from_numpy(x).type(torch.Tensor),
            torch.from_numpy(y),
            torch.from_numpy(c)
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            drop_last=True
        )
        return val_loader