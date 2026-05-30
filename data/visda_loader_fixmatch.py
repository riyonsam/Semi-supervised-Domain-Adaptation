"""
VisDA-2017 directory layout expected:
    <data_dir>/
        train/       <- synthetic source images, one sub-folder per class
        validation/  <- real target images, one sub-folder per class
                        (used for both training target and evaluation)

Data pipeline (FixMatch):
    - Source + labeled target: PIL -> FixMatchTransform(weak only)
      yields a single normalized tensor
    - Unlabeled target: PIL -> FixMatchTransform(weak + strong)
      yields (weak_tensor, strong_tensor) -- the FixMatch dual-view requirement
    - Test: deterministic resize + normalize
"""

import os
import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms

from utils.augment_fixmatch import get_visda_transforms
from sampler.active_sampler import active_select


# TwoViewDataset

class TwoViewDataset(Dataset):
    """
    Wraps a PIL-returning dataset and applies a FixMatchTransform.

    Labeled  (transform.strong is None): __getitem__ -> (tensor, label)
    Unlabeled (transform.strong is set): __getitem__ -> ((weak, strong), label)

    PyTorch default collate handles nested tuples, so a DataLoader over an
    unlabeled TwoViewDataset yields ((B_weak, B_strong), B_labels).
    """

    def __init__(self, dataset, transform):
        self.dataset   = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return self.transform(img), label


def _stratified_split(dataset, ratio: float, seed: int = 42):
    """
    Split dataset into (labelled_indices, unlabelled_indices) so that each
    class contributes ratio of its samples to the labelled set, with a
    minimum of 1 sample per class.
    Requires dataset to have a .samples attribute (as in ImageFolder).
    """
    rng = random.Random(seed)
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_to_indices[label].append(idx)

    labelled, unlabelled = [], []
    for cls_indices in class_to_indices.values():
        rng.shuffle(cls_indices)
        n_lab = max(1, int(len(cls_indices) * ratio))
        labelled.extend(cls_indices[:n_lab])
        unlabelled.extend(cls_indices[n_lab:])

    return labelled, unlabelled


def get_visda_dataloaders(
    data_dir: str,
    labeled_target_ratio: float = 0.01,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    img_size: int = 224,
    source_subdir: str = "train",
    target_subdir: str = "validation",
    debug: bool = False,
    debug_samples: int = 300,
    selection_strategy: str = "random",
    allocation_strategy: str = "proportional",
    model=None,
    device=None,
):
    """
    Build all four DataLoaders for SSDA-FixMatch training.

    Unlabeled loader yields ((weak_img, strong_img), label) batches.
    Labeled loaders yield (img, label) with weak augmentation only.

    Returns:
        src_loader       Source, fully labelled, weak aug
        tgt_lab_loader   Target labelled subset Dl_t, weak aug
        tgt_unlab_loader Target unlabelled pool Du_t, (weak, strong) views
        test_loader      Target test set (evaluation only)
    """
    labeled_transform, unlabeled_transform, test_transform = get_visda_transforms(img_size)

    # PIL resize only -- FixMatchTransform applies crop/flip/aug on top.
    pil_resize = transforms.Resize((img_size, img_size))

    # Base datasets (PIL output)
    source_pil = datasets.ImageFolder(
        os.path.join(data_dir, source_subdir),
        transform=pil_resize,
    )
    target_pil = datasets.ImageFolder(
        os.path.join(data_dir, target_subdir),
        transform=pil_resize,
    )

    # Debug: truncate for quick iteration
    if debug:
        rng = random.Random(seed)

        def _truncate(ds, n):
            idx = list(range(len(ds)))
            rng.shuffle(idx)
            sub = Subset(ds, idx[:n])
            sub.samples = [ds.samples[i] for i in idx[:n]]
            return sub

        source_pil = _truncate(source_pil, debug_samples)
        target_pil = _truncate(target_pil, debug_samples)
        print(f"[DEBUG MODE] Using {debug_samples} samples per split")

    # Label selection + allocation
    _dev = device if device is not None else torch.device("cpu")
    labelled_idx, unlabelled_idx = active_select(
        target_pil, model,
        ratio=labeled_target_ratio,
        strategy=selection_strategy,
        device=_dev,
        seed=seed,
        allocation=allocation_strategy,
    )
    n_total = len(target_pil)
    n_lab   = len(labelled_idx)
    print(
        f"[visda_loader] Target split ({selection_strategy}/{allocation_strategy})  ->  "
        f"labelled: {n_lab} ({100 * n_lab / n_total:.2f}%)  |  "
        f"unlabelled: {len(unlabelled_idx)}"
    )

    tgt_lab_subset    = Subset(target_pil, labelled_idx)
    tgt_unlab_subset  = Subset(target_pil, unlabelled_idx)

    # Wrap with FixMatch transforms
    src_dataset       = TwoViewDataset(source_pil,       labeled_transform)
    tgt_lab_dataset   = TwoViewDataset(tgt_lab_subset,   labeled_transform)
    tgt_unlab_dataset = TwoViewDataset(tgt_unlab_subset, unlabeled_transform)

    # Test set uses deterministic test_transform directly
    test_dataset = datasets.ImageFolder(
        os.path.join(data_dir, target_subdir),
        transform=test_transform,
    )
    if debug:
        rng2 = random.Random(seed + 1)
        idx  = list(range(len(test_dataset)))
        rng2.shuffle(idx)
        test_dataset = Subset(test_dataset, idx[:debug_samples])

    # DataLoaders
    src_loader = DataLoader(
        src_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=False,
        drop_last=True,
    )
    tgt_lab_loader = DataLoader(
        tgt_lab_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=False,
        drop_last=True,
    )
    tgt_unlab_loader = DataLoader(
        tgt_unlab_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=False,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=False,
    )

    return src_loader, tgt_lab_loader, tgt_unlab_loader, test_loader
