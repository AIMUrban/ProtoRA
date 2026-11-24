import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = '1'
# from data import *
import torch
import torch.nn as nn
# from src.utils import *
import random
import math
from typing import Literal
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import pandas as pd


def normalize_matrix(matrix: torch.Tensor, method: str = 'zscore', eps: float = 1e-8) -> torch.Tensor:
    """
    标准化距离矩阵，支持 z-score 和 min-max 方法。

    Args:
        matrix (torch.Tensor): 输入距离矩阵 (n x n)
        method (str): 'zscore' 或 'minmax'
        eps (float): 避免除以零的小常数

    Returns:
        torch.Tensor: 标准化后的距离矩阵
    """
    if method == 'zscore':
        mean = matrix.mean()
        std = matrix.std()
        normalized = (matrix - mean) / (std + eps)
    elif method == 'minmax':
        min_val = matrix.min()
        max_val = matrix.max()
        normalized = (matrix - min_val) / (max_val - min_val + eps)
    else:
        raise ValueError("Unsupported normalization method: choose 'zscore' or 'minmax'")
    return normalized

def get_region_flow(city):
    od_data = np.load('./region_data/{}/mobility_od.npy'.format(city))

    inflow = od_data.sum(0)
    outflow = od_data.sum(1)

    return inflow+outflow





def get_hier_ranks(flow_list):
    sorted_index = np.argsort(flow_list)
    ranks = []
    index = len(sorted_index)
    selected_index = sorted_index[:index]

    while(len(selected_index) != 0):
    # for i in range(12):

        mu = np.mean(flow_list[selected_index])
        if mu == 0:
            break
        rho = flow_list[selected_index[-1]]
        f = 1 - mu / rho
        if np.isnan(f):
            break

        index = int(len(selected_index) * f)
        rank = list(selected_index[index:])
        ranks.append(rank)
        selected_index = sorted_index[:index]
    # ranks.append(sorted_index[:index])
    return ranks, sorted_index


def get_anchor_points(Z_tgt: np.ndarray, k: int = 128, method='kmeans') -> np.ndarray:
    # 1) k-means 获取簇中心
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=k, random_state=42).fit(Z_tgt)
        cluster_labels = kmeans.labels_
    else:
        spectral = SpectralClustering(n_clusters=k, random_state=42, affinity='nearest_neighbors', assign_labels='kmeans')
        cluster_labels = spectral.fit_predict(Z_tgt)

    return cluster_labels



def get_hier_anchors(s_flow_list, t_flow_list, method: Literal['rank', 'batch_mean', 'batch_random'] = 'batch_mean'):
    s_ranks, s_sorted_index = get_hier_ranks(s_flow_list)
    t_ranks, t_sorted_index = get_hier_ranks(t_flow_list)

    if method == 'rank':
        length = min(len(s_sorted_index), len(t_sorted_index))

        s_anchors = s_sorted_index[:length]
        t_anchors = t_sorted_index[:length]
        return s_anchors, t_anchors

    elif method == 'batch_mean':
        hier_levels = min(len(s_ranks), len(t_ranks))
        s_r = s_ranks[:hier_levels - 1]
        s_r.append(list(np.concatenate(s_ranks[hier_levels - 1:])))
        t_r = t_ranks[:hier_levels - 1]
        t_r.append(list(np.concatenate(t_ranks[hier_levels - 1:])))

        s_anchors = s_r
        t_anchors = t_r
        return s_anchors, t_anchors

    elif method == 'batch_random':
        s_anchors, t_anchors = [], []
        random.seed(0)
        hier_levels = min(len(s_ranks), len(t_ranks))
        s_r = s_ranks[:hier_levels - 1]
        s_r.append(list(np.concatenate(s_ranks[hier_levels - 1:])))
        t_r = t_ranks[:hier_levels - 1]
        t_r.append(list(np.concatenate(t_ranks[hier_levels - 1:])))

        for level in range(hier_levels):
            s_rank_ids = s_r[level]
            t_rank_ids = t_r[level]

            s_l = math.ceil(len(s_rank_ids) * 0.01)
            t_l = math.ceil(len(t_rank_ids) * 0.01)
            l = min(s_l, t_l)
            s_rank_sampled_ids = random.sample(list(s_rank_ids), l)
            t_rank_sampled_ids = random.sample(list(t_rank_ids), l)

            s_anchors += s_rank_sampled_ids
            t_anchors += t_rank_sampled_ids

        return s_anchors, t_anchors


