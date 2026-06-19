import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F


class AgePruner():
    def __init__(self, ages, labels):
        self.ages = ages
        self.labels = labels
        self.unique_labels = np.unique(self.labels)
        self.easy = []
        self.hard = []
        self.moderate = []

    def V1_prune(self, r, q=0.1):
        V1_indices = []
        n_total = len(self.ages)
        # max_hard = max(1, int(n_total * q)) 

        for lbl in self.unique_labels:
            cls_mask = self.labels == lbl
            cls_ages = self.ages[cls_mask]
            cls_indices = np.where(cls_mask)[0]

            if len(cls_ages) == 0:
                continue

            n_hard_target = max(1, int(len(cls_ages) * q))

            sort_order = np.lexsort((cls_indices, cls_ages)) 
            sorted_ages = cls_ages[sort_order]
            sorted_indices = cls_indices[sort_order]
            cutoff_age = sorted_ages[n_hard_target - 1]

            hard_mask = sorted_ages < cutoff_age
            hard_indices = sorted_indices[hard_mask]

            remaining = n_hard_target - len(hard_indices)
            tie_mask = sorted_ages == cutoff_age
            tie_indices = sorted_indices[tie_mask] 
            accepted_ties = tie_indices[:remaining]
            # overflow_ties = tie_indices[remaining:]

            hard_indices = np.concatenate([hard_indices, accepted_ties])
            self.hard.extend(hard_indices.tolist())

            num_select = max(1, int(len(hard_indices) * r))
            num_select = min(num_select, len(hard_indices))
            selected = hard_indices[:num_select]  

            V1_indices.extend(selected.tolist())

        return V1_indices

    def V2_prune(self, r, q=0.1, seed=42):
        V2_indices = []

        for lbl in self.unique_labels:
            cls_mask = self.labels == lbl
            cls_ages = self.ages[cls_mask]
            cls_indices = np.where(cls_mask)[0]

            if len(cls_ages) == 0:
                continue

            n_easy_target = max(1, int(len(cls_ages) * q))

            sort_order = np.lexsort((cls_indices, -cls_ages))
            sorted_ages = cls_ages[sort_order]
            sorted_indices = cls_indices[sort_order]

            cutoff_age = sorted_ages[n_easy_target - 1]

            easy_mask = sorted_ages > cutoff_age
            easy_indices = sorted_indices[easy_mask]

            remaining = n_easy_target - len(easy_indices)
            tie_mask = sorted_ages == cutoff_age
            tie_indices = sorted_indices[tie_mask]
            accepted_ties = tie_indices[:remaining]

            easy_indices = np.concatenate([easy_indices, accepted_ties])
            self.easy.extend(easy_indices.tolist())

            n_sample = max(1, int(len(easy_indices) * r))
            n_sample = min(n_sample, len(easy_indices))
            rng = np.random.default_rng(seed)
            selected = rng.choice(easy_indices, size=n_sample, replace=False)

            V2_indices.extend(selected.tolist())

        return V2_indices
    
    def V3_prune(self, r, n_bins=10):
        V3_indices = []
        V2_indices = self.easy
        V1_indices = self.hard
        
        mask = np.ones(len(self.ages), dtype=bool)
        if V1_indices is not None:
            mask[V1_indices] = False
        if V2_indices is not None:
            mask[V2_indices] = False

        ages = self.ages[mask]
        labels = self.labels[mask]
        global_indices = np.where(mask)[0]
        # print(f'number of samples in V3: {len(ages)}')

        for lbl in self.unique_labels:
            cls_mask = (labels == lbl) 
            cls_ages = ages[cls_mask]
            cls_indices = global_indices[cls_mask]

#           print(f'class {lbl} has {len(cls_ages)} in V3')
            self.moderate.extend(cls_indices.tolist()) # for analysis

            if len(cls_ages) == 0: continue 
            elif len(cls_ages) < n_bins + 1: 
                V3_indices.extend(cls_indices) # take all if not enough for binning
                continue

            sorted_order = np.argsort(cls_ages)
            rank_bins = np.array_split(sorted_order, n_bins)

            bin_counts = [len(b) for b in rank_bins]
#            print(f'class {lbl} bin counts: {bin_counts}')

            for b, rank_idx in enumerate(rank_bins):
                bin_indices = cls_indices[rank_idx]
                if len(bin_indices) == 0:
                    continue

                n_sample = max(1, int(len(bin_indices) * r))
                n_sample = min(n_sample, len(bin_indices))
                selected = np.random.choice(
                    bin_indices,
                    size=n_sample,
                    replace=False
                )
                V3_indices.extend(selected.tolist())

        return V3_indices
    
    def prune(self, rh=0.1, re=0.1, rm=0.1, n_bins=10, q=0.1):
        '''q is quotient of easy:moderate:hard samples, default to 0.1:0.8:0.1'''
        V1_indices = self.V1_prune(rh, q) # r_hard
        V2_indices = self.V2_prune(re, q) # r_easy
        V3_indices = self.V3_prune(rm, n_bins=n_bins) # r_moderate

        print(f'number of samples in V1: {len(self.hard)} ->', end=' ')
        print(f'to V1: {len(V1_indices)}')
        print(f'number of samples in easy: {len(self.easy)} ->', end=' ')
        print(f'to V2: {len(V2_indices)}')
        print(f'number of samples in V3: {len(V3_indices)}')

        final_set = []
        for idx in V1_indices:
            final_set.append(idx)

        for idx in V2_indices:
            final_set.append(idx)

        for idx in V3_indices:
            final_set.append(idx)

        final_set = sorted(list(set(final_set)))
        # for whole set, moderate set gets sampled such that easy==hard==moderate
        # e.g. q=0.1 -> q_m = 0.8 -> sampling of rate 0.1/0.8 = 0.125
        # so for q=0.2 -> q_m = 0.6 -> sampling of rate 0.2/0.6 = 0.1667
        n_sample = int(len(self.moderate) * (q / (1 - 2*q))) 
        n_sample = min(n_sample, len(self.moderate))
        moderate_sampled = np.random.choice(self.moderate, size=n_sample, replace=False)
        print("moderate set from: ", len(self.moderate), " to ", len(moderate_sampled))
        self.moderate = moderate_sampled.tolist() # update moderate to the sampled ones for analysis
        return final_set


