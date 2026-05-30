"""
sampler/active_sampler.py

Active selection of the labelled target subset D_t^l under a fixed budget.

Selection strategies (which samples to pick within each class):
  - 'random'      : stratified random sampling (baseline)
  - 'uncertainty' : select samples with highest predictive entropy
  - 'diversity'   : k-means on backbone features, select samples nearest centroids
  - 'hybrid'      : filter by top-k uncertainty, then apply diversity within that pool

Allocation strategies (how many labels to assign per class):
  - 'proportional': each class gets ratio% of its samples (default, current behaviour)
  - 'equal'       : budget split evenly across all classes
  - 'difficulty'  : harder classes (higher avg entropy) get more labels

All strategies respect per-class minimum of 1 sample.

Usage:
    from sampler.active_sampler import active_select
    labelled_idx, unlabelled_idx = active_select(
        dataset, model, ratio, strategy, device, seed,
        allocation='proportional'
    )

    dataset : ImageFolder (or Subset with .samples) -- PIL output
    model   : ResNet50DA -- used for feature/logit extraction
    ratio   : float, fraction of target to label
    strategy: str
    device  : torch.device
    seed    : int
"""

import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


# ── Minimal transform for feature extraction (no augmentation) ────────────────
_EXTRACT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def _extract_features_and_probs(dataset, model, device, batch_size=64):
    """
    Run the model over dataset in eval mode (no grad).

    Returns:
        features : np.ndarray (N, D)  -- bottleneck features
        probs    : np.ndarray (N, C)  -- softmax probabilities
        indices  : list[int]          -- original dataset indices (0..N-1)
    """
    # Wrap dataset with the plain extraction transform
    class _ExtractDataset(torch.utils.data.Dataset):
        def __init__(self, ds):
            self.ds = ds
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            img, label = self.ds[idx]
            return _EXTRACT_TRANSFORM(img), label

    loader = DataLoader(
        _ExtractDataset(dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model.eval()
    all_feats, all_probs = [], []

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            logits, feats = model(imgs)
            probs = F.softmax(logits, dim=1)
            all_feats.append(feats.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    features = np.concatenate(all_feats, axis=0)   # (N, D)
    probs    = np.concatenate(all_probs, axis=0)    # (N, C)
    indices  = list(range(len(dataset)))
    return features, probs, indices


def _entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy per sample. probs: (N, C) -> (N,)"""
    p = np.clip(probs, 1e-8, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def _kmeans_select(features: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    """
    Simple k-means: return the index (into features) of the sample
    closest to each centroid.
    Uses sklearn if available, otherwise a lightweight numpy fallback.
    """
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        km.fit(features)
        centroids = km.cluster_centers_        # (K, D)
        # For each centroid find the nearest sample
        selected = []
        for c in centroids:
            dists = np.linalg.norm(features - c, axis=1)
            selected.append(int(np.argmin(dists)))
        return np.array(selected)
    except ImportError:
        # Fallback: random init k-means (Lloyd's, 10 iters)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(features), n_clusters, replace=False)
        centroids = features[idx].copy()
        for _ in range(10):
            dists  = np.linalg.norm(features[:, None] - centroids[None], axis=2)  # (N,K)
            assign = dists.argmin(axis=1)
            for k in range(n_clusters):
                members = features[assign == k]
                if len(members):
                    centroids[k] = members.mean(axis=0)
        selected = []
        for c in centroids:
            dists = np.linalg.norm(features - c, axis=1)
            selected.append(int(np.argmin(dists)))
        return np.array(selected)


# ── Allocation: how many labels per class ──────────────────────────────────────

def _compute_class_budget(class_to_indices, n_budget, allocation, probs=None):
    """
    Return a dict {class: n_labels} respecting a total budget.

    allocation options:
      'proportional' : each class gets budget proportional to its size
      'equal'        : budget split evenly across classes
      'difficulty'   : harder classes (higher avg entropy) get more labels
    """
    classes   = sorted(class_to_indices.keys())
    n_classes = len(classes)

    if allocation == 'equal':
        base = n_budget // n_classes
        remainder = n_budget - base * n_classes
        per_class = {c: base for c in classes}
        for c in classes[:remainder]:
            per_class[c] += 1

    elif allocation == 'difficulty' and probs is not None:
        entropy = _entropy(probs)
        # average entropy per class
        avg_ent = {}
        for c, idx_list in class_to_indices.items():
            avg_ent[c] = float(np.mean(entropy[idx_list]))
        total_ent = sum(avg_ent.values())
        if total_ent == 0:
            # fallback to equal if all entropies are zero
            return _compute_class_budget(class_to_indices, n_budget, 'equal')
        # weight budget by difficulty, ensure min 1 per class
        raw = {c: max(1, int(n_budget * avg_ent[c] / total_ent)) for c in classes}
        # adjust to hit exact budget
        diff = n_budget - sum(raw.values())
        sorted_by_ent = sorted(classes, key=lambda c: -avg_ent[c])
        for i, c in enumerate(sorted_by_ent):
            if diff == 0:
                break
            if diff > 0:
                raw[c] += 1
                diff -= 1
            elif raw[c] > 1:
                raw[c] -= 1
                diff += 1
        per_class = raw

    else:  # 'proportional' (default)
        n_total = sum(len(v) for v in class_to_indices.values())
        per_class = {c: max(1, int(len(class_to_indices[c]) / n_total * n_budget))
                     for c in classes}

    # enforce min 1 per class
    for c in classes:
        per_class[c] = max(1, per_class[c])

    return per_class


# ── Public API ─────────────────────────────────────────────────────────────────

def active_select(dataset, model, ratio: float, strategy: str,
                  device, seed: int = 42, allocation: str = 'proportional'):
    """
    Select labelled_indices and unlabelled_indices from dataset.

    Parameters
    ----------
    dataset    : dataset with .samples [(path, label), ...]
    model      : ResNet50DA (used for uncertainty / diversity)
    ratio      : fraction to label  (e.g. 0.01 = 1%)
    strategy   : 'random' | 'uncertainty' | 'diversity' | 'hybrid'
    device     : torch.device
    seed       : random seed
    allocation : 'proportional' | 'equal' | 'difficulty'

    Returns
    -------
    labelled_idx   : list[int]
    unlabelled_idx : list[int]
    """
    strategy   = strategy.lower()
    allocation = allocation.lower()

    # Group indices by class
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_to_indices[label].append(idx)

    n_total  = len(dataset)
    n_budget = max(len(class_to_indices), int(n_total * ratio))

    # Random selection with any allocation (no model needed unless difficulty)
    if strategy == 'random':
        if allocation == 'difficulty' and model is not None:
            print(f"[active_sampler] Extracting features for difficulty allocation...")
            _, probs, _ = _extract_features_and_probs(dataset, model, device)
            per_class = _compute_class_budget(class_to_indices, n_budget, allocation, probs)
        else:
            per_class = _compute_class_budget(class_to_indices, n_budget, allocation)
        return _random_select_with_budget(class_to_indices, per_class, n_total, seed)

    # Model-based strategies always need features/probs
    print(f"[active_sampler] Extracting features for strategy='{strategy}'...")
    features, probs, _ = _extract_features_and_probs(dataset, model, device)
    print(f"[active_sampler] Done. features: {features.shape}, probs: {probs.shape}")

    per_class = _compute_class_budget(class_to_indices, n_budget, allocation, probs)

    if strategy == 'uncertainty':
        labelled_idx = _uncertainty_select(features, probs, class_to_indices,
                                           per_class, seed)
    elif strategy == 'diversity':
        labelled_idx = _diversity_select(features, class_to_indices,
                                         per_class, seed)
    elif strategy == 'hybrid':
        labelled_idx = _hybrid_select(features, probs, class_to_indices,
                                      per_class, seed)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. "
                         "Choose from: random, uncertainty, diversity, hybrid")

    labelled_set   = set(labelled_idx)
    unlabelled_idx = [i for i in range(n_total) if i not in labelled_set]

    print(f"[active_sampler] strategy={strategy} allocation={allocation} | "
          f"labelled: {len(labelled_idx)} ({100*len(labelled_idx)/n_total:.2f}%) | "
          f"unlabelled: {len(unlabelled_idx)}")

    return labelled_idx, unlabelled_idx


# ── Strategy implementations ───────────────────────────────────────────────────

def _random_select_with_budget(class_to_indices, per_class, n_total, seed):
    """Random selection using a pre-computed per-class budget dict."""
    rng = random.Random(seed)
    labelled, unlabelled = [], []
    for cls, idx_list in class_to_indices.items():
        shuffled = idx_list[:]
        rng.shuffle(shuffled)
        n_lab = min(per_class[cls], len(shuffled))
        labelled.extend(shuffled[:n_lab])
        unlabelled.extend(shuffled[n_lab:])
    return labelled, unlabelled


def _uncertainty_select(features, probs, class_to_indices, per_class, seed):
    """
    Select highest-entropy samples per class up to per_class[cls] budget.
    """
    entropy = _entropy(probs)
    selected = []
    for cls, idx_list in class_to_indices.items():
        cls_sorted = sorted(idx_list, key=lambda i: -entropy[i])
        selected.extend(cls_sorted[:per_class[cls]])
    return selected


def _diversity_select(features, class_to_indices, per_class, seed):
    """
    K-means per class: select samples nearest to per_class[cls] centroids.
    """
    selected = []
    for cls, idx_list in class_to_indices.items():
        n = min(per_class[cls], len(idx_list))
        if n >= len(idx_list):
            selected.extend(idx_list)
            continue
        cls_features = features[idx_list]
        chosen_local = _kmeans_select(cls_features, n, seed)
        selected.extend([idx_list[i] for i in set(chosen_local.tolist())])
        # top up with random if dedup reduced count
        if len(set(chosen_local.tolist())) < n:
            rng = random.Random(seed)
            remaining_pool = [i for i in range(len(idx_list))
                              if i not in set(chosen_local.tolist())]
            rng.shuffle(remaining_pool)
            extra = n - len(set(chosen_local.tolist()))
            selected.extend([idx_list[i] for i in remaining_pool[:extra]])
    return selected


def _hybrid_select(features, probs, class_to_indices, per_class, seed):
    """
    Per class: filter to top-2x uncertain, then apply diversity within that pool.
    """
    entropy = _entropy(probs)
    selected = []
    for cls, idx_list in class_to_indices.items():
        n = min(per_class[cls], len(idx_list))
        pool_size = min(len(idx_list), 2 * n)
        pool_local = sorted(range(len(idx_list)), key=lambda i: -entropy[idx_list[i]])[:pool_size]
        pool_global = [idx_list[i] for i in pool_local]
        if n >= len(pool_global):
            selected.extend(pool_global)
            continue
        pool_features = features[pool_global]
        chosen_local = _kmeans_select(pool_features, n, seed)
        selected.extend([pool_global[i] for i in set(chosen_local.tolist())])
    return selected
