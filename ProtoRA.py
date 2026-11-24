import numpy as np
import ot
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
import pickle
import torch
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, SpectralClustering
from typing import *
import math
import random
import logging
import os
from pathlib import Path
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DPARRConfig:
    """Configuration class for centralized management of all hyperparameters and paths"""

    def __init__(self):
        self.region_emb_dir = './region_embs'
        self.region_data_dir = './region_data'
        self.k_range = [5, 10, 15, 20, 25]  # Clustering granularity
        self.hier_levels = 10  # Hierarchical levels
        self.tau = 1.0  # Softmax temperature
        self.eps = 1e-9  # Small constant for numerical stability
        self.random_seed = 42
        self.max_regions = 10000  # Maximum region limit


class InputValidator:
    """Input validation class"""

    @staticmethod
    def validate_embeddings(emb: np.ndarray, city_name: str) -> bool:
        """Validate the validity of embedding matrix"""
        if emb is None:
            raise ValueError(f"Embedding for {city_name} is None")

        if not isinstance(emb, np.ndarray):
            raise TypeError(f"Embedding for {city_name} must be numpy array")

        if emb.ndim != 2:
            raise ValueError(f"Embedding for {city_name} must be 2D array, got {emb.ndim}D")

        if emb.shape[0] == 0:
            raise ValueError(f"Embedding for {city_name} has 0 regions")

        if np.any(np.isnan(emb)) or np.any(np.isinf(emb)):
            raise ValueError(f"Embedding for {city_name} contains NaN or Inf values")

        return True

    @staticmethod
    def validate_file_path(filepath: str) -> bool:
        """Validate if file path exists"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        return True


def safe_load_pickle(filepath: str, description: str = "data") -> Any:
    """Safely load pickle file"""
    try:
        InputValidator.validate_file_path(filepath)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Successfully loaded {description} from {filepath}")
        return data
    except FileNotFoundError as e:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading {description} from {filepath}: {str(e)}")
        raise


def safe_load_npy(filepath: str, description: str = "data") -> np.ndarray:
    """Safely load numpy file"""
    try:
        InputValidator.validate_file_path(filepath)
        data = np.load(filepath)
        logger.info(f"Successfully loaded {description} from {filepath}")
        return data
    except FileNotFoundError as e:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading {description} from {filepath}: {str(e)}")
        raise


def coral_distance(X: np.ndarray, Y: np.ndarray, eps: float = 1e-9) -> float:
    """
    Calculate CORAL distance with enhanced numerical stability

    Args:
        X: Source domain feature matrix [n_samples, n_features]
        Y: Target domain feature matrix [m_samples, n_features]
        eps: Numerical stability constant

    Returns:
        CORAL distance value
    """
    try:
        # Check input
        if X.shape[0] < 2 or Y.shape[0] < 2:
            logger.warning("Too few samples for CORAL distance, using L2 distance on means")
            return np.linalg.norm(X.mean(axis=0) - Y.mean(axis=0))

        # Calculate mean
        mu_x = X.mean(axis=0)
        mu_y = Y.mean(axis=0)

        # Calculate covariance matrix with regularization
        cov_x = np.cov(X.T) + eps * np.eye(X.shape[1])
        cov_y = np.cov(Y.T) + eps * np.eye(Y.shape[1])

        # Calculate distance using Frobenius norm
        mean_dist = np.linalg.norm(mu_x - mu_y)
        cov_dist = np.linalg.norm(cov_x - cov_y, 'fro')

        return mean_dist + cov_dist

    except Exception as e:
        logger.error(f"Error computing CORAL distance: {str(e)}")
        # Fallback to simple mean distance
        return np.linalg.norm(X.mean(axis=0) - Y.mean(axis=0))


def hierarchy_path(s_city: str, t_city: str, s_emb: np.ndarray, t_emb: np.ndarray,
                   anchor_method: str = 'batch_mean', config: DPARRConfig = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hierarchical path with enhanced error handling
    """
    if config is None:
        config = DPARRConfig()

    try:
        # Load flow data
        s_flow_list = get_region_flow(s_city, config)
        t_flow_list = get_region_flow(t_city, config)

        # Get hierarchical anchors
        s_anchors, t_anchors = get_hier_anchors(s_flow_list, t_flow_list, method=anchor_method)

        s_anchor_emb, t_anchor_emb = [], []

        if anchor_method == 'rank' or anchor_method == 'batch_random':
            s_anchor_emb = s_emb[s_anchors]
            t_anchor_emb = t_emb[t_anchors]

        elif anchor_method == 'batch_mean':
            for s_rank_ids, t_rank_ids in zip(s_anchors, t_anchors):
                if len(s_rank_ids) == 0 or len(t_rank_ids) == 0:
                    logger.warning("Empty rank group encountered, skipping")
                    continue

                s_rank_emb = np.mean(s_emb[s_rank_ids], axis=0)
                t_rank_emb = np.mean(t_emb[t_rank_ids], axis=0)

                s_anchor_emb.append(s_rank_emb)
                t_anchor_emb.append(t_rank_emb)

        if len(s_anchor_emb) == 0 or len(t_anchor_emb) == 0:
            raise ValueError("No valid anchors generated from hierarchy path")

        s_anchor_emb_hier = np.array(s_anchor_emb)
        t_anchor_emb_hier = np.array(t_anchor_emb)

        # Calculate relative representation
        s_rel_hier = relative_rank_representation(s_emb, s_anchor_emb_hier, tau=config.tau)
        t_rel_hier = relative_rank_representation(t_emb, t_anchor_emb_hier, tau=config.tau)

        return s_rel_hier, t_rel_hier

    except Exception as e:
        logger.error(f"Error in hierarchy_path: {str(e)}")
        raise


def get_region_flow(city: str, config: DPARRConfig = None) -> np.ndarray:
    """Get region flow data with enhanced error handling"""
    if config is None:
        config = DPARRConfig()

    try:
        filepath = os.path.join(config.region_data_dir, city, 'mobility_od.npy')
        od_data = safe_load_npy(filepath, f"{city} mobility OD data")

        # Validate data
        if od_data.ndim != 2:
            raise ValueError(f"OD data must be 2D, got {od_data.ndim}D")

        if od_data.shape[0] != od_data.shape[1]:
            raise ValueError(f"OD data must be square matrix, got {od_data.shape}")

        inflow = od_data.sum(axis=0)
        outflow = od_data.sum(axis=1)

        flow = inflow + outflow

        # Check for outliers
        if np.any(flow < 0):
            logger.warning(f"Negative flow values detected for {city}, clipping to 0")
            flow = np.clip(flow, 0, None)

        return flow

    except Exception as e:
        logger.error(f"Error loading flow data for {city}: {str(e)}")
        raise


def get_hier_ranks(flow_list: np.ndarray, max_iterations: int = 20) -> Tuple[List, np.ndarray]:
    """
    Get hierarchical ranks with iteration limit and exception handling
    """
    try:
        if len(flow_list) == 0:
            raise ValueError("Empty flow_list")

        sorted_index = np.argsort(flow_list)
        ranks = []
        index = len(sorted_index)
        selected_index = sorted_index[:index]

        iterations = 0
        while len(selected_index) != 0 and iterations < max_iterations:
            iterations += 1

            mu = np.mean(flow_list[selected_index])
            if mu == 0 or np.isnan(mu):
                logger.warning(f"Invalid mean value at iteration {iterations}, breaking")
                break

            rho = flow_list[selected_index[-1]]
            if rho == 0 or np.isnan(rho):
                logger.warning(f"Invalid rho value at iteration {iterations}, breaking")
                break

            f = 1 - mu / rho
            if np.isnan(f) or f < 0 or f >= 1:
                logger.warning(f"Invalid f value: {f} at iteration {iterations}, breaking")
                break

            index = int(len(selected_index) * f)
            if index >= len(selected_index):
                break

            rank = list(selected_index[index:])
            if len(rank) > 0:
                ranks.append(rank)
            selected_index = sorted_index[:index]

        if len(ranks) == 0:
            logger.warning("No ranks generated, using full index as single rank")
            ranks = [list(sorted_index)]

        return ranks, sorted_index

    except Exception as e:
        logger.error(f"Error in get_hier_ranks: {str(e)}")
        raise


def get_hier_anchors(s_flow_list: np.ndarray, t_flow_list: np.ndarray,
                     method: str = 'batch_mean') -> Tuple[Any, Any]:
    """Get hierarchical anchors with enhanced error handling"""
    try:
        s_ranks, s_sorted_index = get_hier_ranks(s_flow_list)
        t_ranks, t_sorted_index = get_hier_ranks(t_flow_list)

        if method == 'rank':
            length = min(len(s_sorted_index), len(t_sorted_index))
            if length == 0:
                raise ValueError("No valid anchors for rank method")
            s_anchors = s_sorted_index[:length]
            t_anchors = t_sorted_index[:length]
            return s_anchors, t_anchors

        elif method == 'batch_mean':
            hier_levels = min(len(s_ranks), len(t_ranks))
            if hier_levels == 0:
                raise ValueError("No hierarchical levels found")

            s_r = s_ranks[:hier_levels - 1] if hier_levels > 1 else []
            if hier_levels > 0:
                remaining = list(np.concatenate(s_ranks[max(0, hier_levels - 1):]))
                if len(remaining) > 0:
                    s_r.append(remaining)

            t_r = t_ranks[:hier_levels - 1] if hier_levels > 1 else []
            if hier_levels > 0:
                remaining = list(np.concatenate(t_ranks[max(0, hier_levels - 1):]))
                if len(remaining) > 0:
                    t_r.append(remaining)

            if len(s_r) == 0 or len(t_r) == 0:
                raise ValueError("No valid anchor groups generated")

            return s_r, t_r

        elif method == 'batch_random':
            s_anchors, t_anchors = [], []
            random.seed(0)
            hier_levels = min(len(s_ranks), len(t_ranks))

            s_r = s_ranks[:hier_levels - 1] if hier_levels > 1 else []
            if hier_levels > 0:
                s_r.append(list(np.concatenate(s_ranks[max(0, hier_levels - 1):])))

            t_r = t_ranks[:hier_levels - 1] if hier_levels > 1 else []
            if hier_levels > 0:
                t_r.append(list(np.concatenate(t_ranks[max(0, hier_levels - 1):])))

            for level in range(min(len(s_r), len(t_r))):
                s_rank_ids = s_r[level]
                t_rank_ids = t_r[level]

                s_l = max(1, math.ceil(len(s_rank_ids) * 0.5))
                t_l = max(1, math.ceil(len(t_rank_ids) * 0.5))
                l = min(s_l, t_l)

                s_rank_sampled_ids = random.sample(list(s_rank_ids), l)
                t_rank_sampled_ids = random.sample(list(t_rank_ids), l)

                s_anchors += s_rank_sampled_ids
                t_anchors += t_rank_sampled_ids

            return s_anchors, t_anchors
        else:
            raise ValueError(f"Unknown anchor method: {method}")

    except Exception as e:
        logger.error(f"Error in get_hier_anchors: {str(e)}")
        raise


def structure_path(s_city: str, t_city: str, s_emb: np.ndarray, t_emb: np.ndarray,
                   k: int = 96, method: str = 'kmeans', info_flag: bool = False,
                   info_source: List[str] = ['poi'], config: DPARRConfig = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Structure path with memory optimization and enhanced error handling
    """
    if config is None:
        config = DPARRConfig()

    try:
        # Convert to tensor
        s_emb_t = torch.tensor(s_emb, dtype=torch.float32)
        t_emb_t = torch.tensor(t_emb, dtype=torch.float32)

        n_s, n_t = s_emb_t.shape[0], t_emb_t.shape[0]

        # Check scale
        if n_s > config.max_regions or n_t > config.max_regions:
            logger.warning(f"Large number of regions detected (s:{n_s}, t:{n_t}), computation may be slow")

        uW1 = torch.ones(n_s, dtype=torch.float32) / n_s
        uW2 = torch.ones(n_t, dtype=torch.float32) / n_t

        # Calculate embedding distance matrix
        s_embC = compute_distance_matrix(s_emb_t, s_emb_t)
        t_embC = compute_distance_matrix(t_emb_t, t_emb_t)

        # Load and calculate distance matrices for other modalities
        s_poiC, t_poiC = None, None
        s_mobC, t_mobC = None, None

        if 'poi' in info_source:
            try:
                s_poi = safe_load_npy(f'{config.region_data_dir}/{s_city}/poi_f_fined_del.npy',
                                      f"{s_city} POI data")
                t_poi = safe_load_npy(f'{config.region_data_dir}/{t_city}/poi_f_fined_del.npy',
                                      f"{t_city} POI data")
                s_poi_t = torch.tensor(s_poi, dtype=torch.float32)
                t_poi_t = torch.tensor(t_poi, dtype=torch.float32)
                s_poiC = compute_distance_matrix(s_poi_t, s_poi_t)
                t_poiC = compute_distance_matrix(t_poi_t, t_poi_t)
            except Exception as e:
                logger.warning(f"Failed to load POI data: {str(e)}, skipping POI modality")

        if 'mob' in info_source:
            try:
                s_mob = safe_load_npy(f'{config.region_data_dir}/{s_city}/mobility_od.npy',
                                      f"{s_city} mobility data")
                t_mob = safe_load_npy(f'{config.region_data_dir}/{t_city}/mobility_od.npy',
                                      f"{t_city} mobility data")
                s_mob_t = torch.tensor(s_mob, dtype=torch.float32)
                t_mob_t = torch.tensor(t_mob, dtype=torch.float32)
                s_mobC = compute_distance_matrix(s_mob_t, s_mob_t)
                t_mobC = compute_distance_matrix(t_mob_t, t_mob_t)
            except Exception as e:
                logger.warning(f"Failed to load mobility data: {str(e)}, skipping mobility modality")

        # Fuse structure distances
        sC, tC, weights = auto_fuse_structure_distance(
            s_embC, s_poiC, s_mobC,
            t_embC, t_poiC, t_mobC,
            eps=config.eps
        )

        # Gromov-Wasserstein optimal transport
        logger.info("Computing Gromov-Wasserstein optimal transport...")
        uT = ot.gromov.gromov_wasserstein(
            C1=sC.numpy(),
            C2=tC.numpy(),
            p=uW1.numpy(),
            q=uW2.numpy(),
            loss_fun='square_loss',
            verbose=False,
            log=False
        )
        T = uT if isinstance(uT, np.ndarray) else uT.numpy()

        # Convert back to numpy
        s_emb_np, t_emb_np = s_emb_t.numpy(), t_emb_t.numpy()

        # Calculate transported embeddings
        t2s = (T.T @ s_emb_np) / (T.T.sum(axis=1, keepdims=True) + config.eps)

        # Multi-scale clustering
        s_anchors, t_anchors = [], []
        for k_val in config.k_range:
            try:
                cluster_labels = get_anchor_points(t_emb_np, k=k_val, method=method)
                t_anchor_emb, s_anchor_emb = get_anchor_embeddings(
                    cluster_labels, t_emb_np, t2s
                )

                if len(t_anchor_emb) > 0 and len(s_anchor_emb) > 0:
                    s_anchors.append(s_anchor_emb)
                    t_anchors.append(t_anchor_emb)
                else:
                    logger.warning(f"Empty anchors for k={k_val}, skipping")
            except Exception as e:
                logger.warning(f"Failed clustering with k={k_val}: {str(e)}, skipping")
                continue

        if len(s_anchors) == 0 or len(t_anchors) == 0:
            raise ValueError("No valid anchors generated from structure path")

        s_anchor_emb = np.concatenate(s_anchors, axis=0)
        t_anchor_emb = np.concatenate(t_anchors, axis=0)

        # Calculate relative representation
        s_rel_struc = relative_rank_representation(s_emb_np, s_anchor_emb, tau=config.tau)
        t_rel_struc = relative_rank_representation(t_emb_np, t_anchor_emb, tau=config.tau)

        return s_rel_struc, t_rel_struc

    except Exception as e:
        logger.error(f"Error in structure_path: {str(e)}")
        raise


def compute_distance_matrix(X: torch.Tensor, Y: torch.Tensor,
                            chunk_size: int = 1000) -> torch.Tensor:
    """
    Compute distance matrix in chunks to avoid memory overflow
    """
    n_x, n_y = X.shape[0], Y.shape[0]

    # If matrix is not large, compute directly
    if n_x * n_y < 1e6:
        return torch.norm(X[:, None] - Y, dim=2, p=2)

    # Otherwise compute in chunks
    logger.info(f"Computing distance matrix in chunks ({n_x} x {n_y})")
    result = torch.zeros(n_x, n_y, dtype=X.dtype)

    for i in range(0, n_x, chunk_size):
        end_i = min(i + chunk_size, n_x)
        result[i:end_i] = torch.norm(X[i:end_i, None] - Y, dim=2, p=2)

    return result


def relative_rank_representation(X: np.ndarray, anchors: np.ndarray,
                                 mode: str = 'soft', tau: float = 1.0,
                                 eps: float = 1e-9) -> np.ndarray:
    """
    Relative rank representation with enhanced numerical stability
    """
    try:
        if len(anchors) == 0:
            raise ValueError("Empty anchors")

        # Calculate distances
        distances = cdist(X, anchors, metric='euclidean')

        if mode == 'soft':
            # Use softmax with enhanced numerical stability
            weights = np.exp(-distances / tau)
            # Avoid division by zero
            row_sums = np.sum(weights, axis=1, keepdims=True)
            weights = weights / (row_sums + eps)

            # Check for outliers
            if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
                logger.warning("NaN or Inf in relative representation, using uniform weights")
                weights = np.ones_like(weights) / weights.shape[1]

            return weights

        elif mode == 'rank':
            ranks = np.argsort(np.argsort(distances, axis=1), axis=1)
            one_hot = np.eye(anchors.shape[0])[ranks]
            return one_hot.reshape(X.shape[0], -1)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    except Exception as e:
        logger.error(f"Error in relative_rank_representation: {str(e)}")
        raise


def get_anchor_points(Z_tgt: np.ndarray, k: int = 128,
                      method: str = 'kmeans') -> np.ndarray:
    """
    Get anchor points with enhanced error handling
    """
    try:
        if len(Z_tgt) < k:
            logger.warning(f"Number of samples ({len(Z_tgt)}) < k ({k}), using n_clusters = {len(Z_tgt)}")
            k = len(Z_tgt)

        if method == 'kmeans':
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(Z_tgt)
        elif method == 'spectral':
            spectral = SpectralClustering(
                n_clusters=k,
                random_state=42,
                affinity='nearest_neighbors',
                assign_labels='kmeans'
            )
            cluster_labels = spectral.fit_predict(Z_tgt)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        return cluster_labels

    except Exception as e:
        logger.error(f"Error in get_anchor_points: {str(e)}")
        raise


def get_anchor_embeddings(cluster_labels: np.ndarray, t_emb: np.ndarray,
                          t2s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get anchor embeddings with enhanced error handling
    """
    try:
        t_anchor_emb, t2s_anchor_emb = [], []

        for c_id in np.unique(cluster_labels):
            if c_id == -1:  # Noise points
                continue

            mask = cluster_labels == c_id
            if not np.any(mask):
                continue

            members = t_emb[mask]
            t2s_members = t2s[mask]

            if len(members) > 0:
                t_anchor_emb.append(members.mean(axis=0))
                t2s_anchor_emb.append(t2s_members.mean(axis=0))

        if len(t_anchor_emb) == 0:
            raise ValueError("No valid cluster anchors generated")

        return np.array(t_anchor_emb), np.array(t2s_anchor_emb)

    except Exception as e:
        logger.error(f"Error in get_anchor_embeddings: {str(e)}")
        raise


def compute_entropy_score(D: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Calculate entropy score with enhanced numerical stability
    """
    try:
        if isinstance(D, torch.Tensor):
            D = D.numpy()

        # Calculate probability distribution
        P = np.exp(-D)
        row_sums = P.sum(axis=1, keepdims=True)

        # Avoid division by zero
        P = P / (row_sums + 1e-9)

        # Calculate entropy
        entropy = -np.sum(P * np.log(P + 1e-9), axis=1).mean()

        # Check validity
        if np.isnan(entropy) or np.isinf(entropy):
            logger.warning("Invalid entropy score, returning 0")
            return 0.0

        return float(entropy)

    except Exception as e:
        logger.error(f"Error computing entropy score: {str(e)}")
        return 0.0


def auto_fuse_structure_distance(s_embC: torch.Tensor, s_poiC: Optional[torch.Tensor] = None,
                                 s_mobC: Optional[torch.Tensor] = None,
                                 t_embC: Optional[torch.Tensor] = None,
                                 t_poiC: Optional[torch.Tensor] = None,
                                 t_mobC: Optional[torch.Tensor] = None,
                                 eps: float = 1e-9) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Automatically fuse structure distances with enhanced error handling and numerical stability
    """
    try:
        # Normalization function
        def z_normalize(x: torch.Tensor) -> torch.Tensor:
            return normalize_matrix(x, 'zscore', eps=eps)

        # Collect all modalities
        s_all = [('emb', z_normalize(s_embC))]
        t_all = [('emb', z_normalize(t_embC))]

        if s_poiC is not None and t_poiC is not None:
            s_all.append(('poi', z_normalize(s_poiC)))
            t_all.append(('poi', z_normalize(t_poiC)))

        if s_mobC is not None and t_mobC is not None:
            s_all.append(('mob', z_normalize(s_mobC)))
            t_all.append(('mob', z_normalize(t_mobC)))

        # Calculate entropy权重
        weights = []
        for (name, sD), (_, tD) in zip(s_all, t_all):
            e_s = compute_entropy_score(sD)
            e_t = compute_entropy_score(tD)
            diff = abs(e_s - e_t)
            weight = np.exp(-diff)
            weights.append(weight)
            logger.info(f"Modality {name}: e_s={e_s:.4f}, e_t={e_t:.4f}, weight={weight:.4f}")

        weights = np.array(weights)

        # Avoid division by zero
        weight_sum = weights.sum()
        if weight_sum < eps:
            logger.warning("Very small weight sum, using uniform weights")
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / weight_sum

        # Weighted fusion
        sC = sum(w * D for w, (_, D) in zip(weights, s_all))
        tC = sum(w * D for w, (_, D) in zip(weights, t_all))

        return sC, tC, weights

    except Exception as e:
        logger.error(f"Error in auto_fuse_structure_distance: {str(e)}")
        raise


def normalize_matrix(matrix: torch.Tensor, method: str = 'zscore',
                     eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize distance matrix with enhanced numerical stability
    """
    try:
        if method == 'zscore':
            mean = matrix.mean()
            std = matrix.std()

            # Avoid dividing by very small std
            if std < eps:
                logger.warning(f"Very small std ({std}), using mean normalization only")
                return matrix - mean

            normalized = (matrix - mean) / (std + eps)

        elif method == 'minmax':
            min_val = matrix.min()
            max_val = matrix.max()

            # Avoid division by zero
            if max_val - min_val < eps:
                logger.warning("Very small range, returning zeros")
                return torch.zeros_like(matrix)

            normalized = (matrix - min_val) / (max_val - min_val + eps)

        else:
            raise ValueError(f"Unsupported normalization method: {method}")

        # Check result
        if torch.any(torch.isnan(normalized)) or torch.any(torch.isinf(normalized)):
            logger.warning("NaN or Inf in normalized matrix, returning original")
            return matrix

        return normalized

    except Exception as e:
        logger.error(f"Error in normalize_matrix: {str(e)}")
        return matrix


def ProtoRA(s_city: str, t_city: str, seed: int, k: int = 20,
           backbone: str = 'recp',
           hier_anchor_method: str = 'batch_mean',
           info_flag: bool = False,
           info_source: List[str] = ['poi'],
           cluster_method: str = 'kmeans',
           config: DPARRConfig = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    ProtoRA main function with comprehensive error handling and validation

    Args:
        s_city: Source city name
        t_city: Target city name
        seed: Random seed
        k: Number of clusters
        backbone: Pre-trained embedding method
        hier_anchor_method: Hierarchical anchor method
        info_flag: Whether to use additional information
        info_source: Information source list
        cluster_method: Clustering method
        config: Configuration object

    Returns:
        (s_rel_emb, t_rel_emb): Aligned relative representations
    """
    if config is None:
        config = DPARRConfig()

    logger.info(f"Starting ProtoRA alignment: {s_city} -> {t_city}")
    logger.info(f"Parameters: seed={seed}, backbone={backbone}, method={hier_anchor_method}")

    try:
        # Load embeddings
        s_emb_path = f'{config.region_emb_dir}/{backbone}_emb/{s_city}_seed_{seed}.pickle'
        t_emb_path = f'{config.region_emb_dir}/{backbone}_emb/{t_city}_seed_{seed}.pickle'

        s_emb = safe_load_pickle(s_emb_path, f"{s_city} embeddings")
        t_emb = safe_load_pickle(t_emb_path, f"{t_city} embeddings")

        # Validate embeddings
        InputValidator.validate_embeddings(s_emb, s_city)
        InputValidator.validate_embeddings(t_emb, t_city)

        logger.info(f"Loaded embeddings: {s_city} {s_emb.shape}, {t_city} {t_emb.shape}")

        # Hierarchical path
        logger.info("Computing hierarchy-based path...")
        s_rel_hier, t_rel_hier = hierarchy_path(
            s_city, t_city, s_emb, t_emb, hier_anchor_method, config
        )
        logger.info(f"Hierarchy path completed: {s_rel_hier.shape}, {t_rel_hier.shape}")

        # Structure path
        logger.info("Computing structure-based path...")
        s_rel_struc, t_rel_struc = structure_path(
            s_city, t_city, s_emb, t_emb, k=k, method=cluster_method,
            info_flag=info_flag, info_source=info_source, config=config
        )
        logger.info(f"Structure path completed: {s_rel_struc.shape}, {t_rel_struc.shape}")

        # Calculate structural deviation (CORAL distance)
        d_stru = coral_distance(s_rel_struc, t_rel_struc, eps=config.eps)
        d_hier = coral_distance(s_rel_hier, t_rel_hier, eps=config.eps)

        logger.info(f"CORAL distances - Structure: {d_stru:.4f}, Hierarchy: {d_hier:.4f}")

        # Calculate adaptive weights
        alpha = np.exp(-d_stru)
        beta = np.exp(-d_hier)

        weight_sum = alpha + beta
        if weight_sum < config.eps:
            logger.warning("Very small weight sum, using equal weights")
            w_stru = 0.5
            w_hier = 0.5
        else:
            w_stru = alpha / weight_sum
            w_hier = beta / weight_sum

        logger.info(f"Adaptive weights - Structure: {w_stru:.4f}, Hierarchy: {w_hier:.4f}")

        # Channel-wise weighted merging
        s_rel_emb = np.concatenate([w_stru * s_rel_struc, w_hier * s_rel_hier], axis=1)
        t_rel_emb = np.concatenate([w_stru * t_rel_struc, w_hier * t_rel_hier], axis=1)

        logger.info(f"Final aligned embeddings: {s_rel_emb.shape}, {t_rel_emb.shape}")
        logger.info("ProtoRA alignment completed successfully")

        return s_rel_emb, t_rel_emb

    except Exception as e:
        logger.error(f"Error in ProtoRA: {str(e)}")
        raise


# Example main function
if __name__ == '__main__':
    try:
        # Configuration
        config = DPARRConfig()
        config.k_range = [5, 10, 15]  # Can be customized

        source_city, target_city = 'xa', 'cd'
        seed = 1

        # Run alignment
        s_rel_emb, t_rel_emb = ProtoRA(
            source_city, target_city, seed,
            info_flag=True,
            info_source=['poi'],
            config=config
        )

        print(f"Alignment successful!")
        print(f"Source embeddings shape: {s_rel_emb.shape}")
        print(f"Target embeddings shape: {t_rel_emb.shape}")

    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise