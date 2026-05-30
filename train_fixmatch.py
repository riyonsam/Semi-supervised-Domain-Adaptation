"""
train_fixmatch.py  -  SSDA FixMatch

Loss (paper Eq. 1 adapted for SSDA):
    L = L_x  +  lambda_u * L_u

    L_x  = CE(model(alpha(x_src)),     y_src)
         + CE(model(alpha(x_tgt_lab)), y_tgt_lab)

    L_u  = (1/B_u) * sum_b [ 1[max q_b >= tau] * H(argmax q_b, model(A(u_b))) ]
           where q_b = softmax(model(alpha(u_b)))   (weak view, no gradient)
                 A(u_b) is the STRONG augmentation  (loss on strong view)

Key components:
    - Dual augmentation: weak view for pseudo-labels, strong view for CE loss
    - Confidence threshold tau masks low-confidence pseudo-labels
    - Soft masking: mask as float weight (no re-forward on filtered samples)
    - SGD with Nesterov + cosine LR decay  (paper App. A.1)
    - EMA model weights used for evaluation (decay=0.999)
"""

import math
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

import os
import config_fixmatch as config
from data.visda_loader_fixmatch import get_visda_dataloaders
from models.resnet50_da import ResNet50DA

# Allow env-var overrides so SLURM sweep can set strategy/budget without editing config
if os.environ.get("EXP_STRATEGY"):
    config.SELECTION_STRATEGY = os.environ["EXP_STRATEGY"]
if os.environ.get("EXP_BUDGET"):
    config.LABEL_BUDGET = float(os.environ["EXP_BUDGET"])
if os.environ.get("EXP_ALLOCATION"):
    config.ALLOCATION_STRATEGY = os.environ["EXP_ALLOCATION"]

# Reproducibility
random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
torch.cuda.manual_seed_all(config.SEED)

device = config.DEVICE
print(f"Using device: {device}")

# Model (built before data loading so active sampler can use pretrained features)
model = ResNet50DA(num_classes=config.NUM_CLASSES).to(device)

# Data
src_loader, tgt_lab_loader, tgt_unlab_loader, test_loader = get_visda_dataloaders(
    data_dir=config.DATA_DIR,
    labeled_target_ratio=config.LABEL_BUDGET,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
    seed=config.SEED,
    img_size=224,
    source_subdir=config.SOURCE_SUBDIR,
    target_subdir=config.TARGET_SUBDIR,
    debug=config.DEBUG,
    debug_samples=config.DEBUG_SAMPLES,
    selection_strategy=config.SELECTION_STRATEGY,
    allocation_strategy=config.ALLOCATION_STRATEGY,
    model=model,
    device=device,
)

# Optimizer: SGD with Nesterov momentum.
# Weight decay is NOT applied to BatchNorm parameters (paper App. A.1).
def _get_optimizer(model):
    bn_params, other_params = [], []
    for name, param in model.named_parameters():
        if "bn" in name or "bias" in name:
            bn_params.append(param)
        else:
            other_params.append(param)
    return optim.SGD(
        [
            {"params": other_params, "weight_decay": config.WEIGHT_DECAY},
            {"params": bn_params,    "weight_decay": 0.0},
        ],
        lr=config.LR,
        momentum=config.MOMENTUM,
        nesterov=True,
    )

optimizer = _get_optimizer(model)

# Cosine LR decay: eta_t = eta * cos(7*pi*t / (16*T))  (paper App. A.1)
total_steps = config.EPOCHS * min(len(src_loader), len(tgt_unlab_loader))

def _cosine_schedule(step):
    return max(0.0, math.cos(math.pi * 7 * step / (16 * total_steps)))

scheduler = LambdaLR(optimizer, lr_lambda=_cosine_schedule)


class EMA:
    """
    Exponential moving average of model parameters.
    EMA weights are used for evaluation only (not for gradient computation).
    Paper App. A.1: decay = 0.999.
    """

    def __init__(self, model, decay=0.999):
        self.decay  = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        """Update EMA shadow after each optimizer step."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = (
                        self.decay * self.shadow[name]
                        + (1.0 - self.decay) * param.data
                    )

    def apply_shadow(self, model):
        """Load EMA weights into model for evaluation (saves originals)."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original weights after evaluation."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])


ema = EMA(model, decay=config.EMA_DECAY)

def _next(iterator, loader):
    """Cycle through a DataLoader indefinitely."""
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def train_one_epoch(global_step: int):
    """
    One training epoch implementing FixMatch (paper Algorithm 1).

    For each batch:
      1. L_x  = CE on labeled (source + labeled target), weak augmentation
      2. q_b  = softmax(model(u_b_weak))  -- pseudo-labels, no gradient
         mask  = 1[ max(q_b) >= tau ]     -- confidence mask
      3. L_u  = mean( mask * CE(model(u_b_strong), argmax(q_b)) )
      4. L    = L_x + lambda_u * L_u

    Returns avg total loss, avg labeled loss, avg unlabeled loss,
    avg mask ratio, and updated global_step.
    """
    model.train()

    src_iter       = iter(src_loader)
    tgt_lab_iter   = iter(tgt_lab_loader)
    tgt_unlab_iter = iter(tgt_unlab_loader)

    steps = min(len(src_loader), len(tgt_unlab_loader))
    total_sum, lx_sum, lu_sum, mask_sum = 0.0, 0.0, 0.0, 0.0
    log_every = max(1, steps // 5)
    t0 = time.time()

    for step in range(steps):

        # Labeled batches (source + labeled target)
        (src_imgs,     src_labels),     src_iter     = _next(src_iter,     src_loader)
        (tgt_lab_imgs, tgt_lab_labels), tgt_lab_iter = _next(tgt_lab_iter, tgt_lab_loader)

        src_imgs       = src_imgs.to(device)
        src_labels     = src_labels.to(device)
        tgt_lab_imgs   = tgt_lab_imgs.to(device)
        tgt_lab_labels = tgt_lab_labels.to(device)

        x_labeled = torch.cat([src_imgs, tgt_lab_imgs], dim=0)
        y_labeled = torch.cat([src_labels, tgt_lab_labels], dim=0)

        # Unlabeled batch: dual-view (weak, strong)
        unlab_batch, tgt_unlab_iter = _next(tgt_unlab_iter, tgt_unlab_loader)
        (u_weak, u_strong), _       = unlab_batch
        u_weak   = u_weak.to(device)
        u_strong = u_strong.to(device)

        # Single forward pass: labeled + both unlabeled views share BN stats
        all_inputs    = torch.cat([x_labeled, u_weak, u_strong], dim=0)
        all_logits, _ = model(all_inputs)

        n_lab  = x_labeled.size(0)
        n_unl  = u_weak.size(0)
        logits_x        = all_logits[:n_lab]
        logits_u_weak   = all_logits[n_lab : n_lab + n_unl]
        logits_u_strong = all_logits[n_lab + n_unl:]

        # Supervised loss L_x
        loss_x = F.cross_entropy(logits_x, y_labeled, reduction="mean")

        # Pseudo-labels from WEAK view (no gradient)
        with torch.no_grad():
            probs                    = torch.softmax(logits_u_weak.detach(), dim=1)
            max_probs, pseudo_labels = torch.max(probs, dim=1)
            mask = max_probs.ge(config.CONF_THRESHOLD).float()

        # Unsupervised loss L_u on STRONG view, weighted by confidence mask
        loss_u = (
            F.cross_entropy(logits_u_strong, pseudo_labels, reduction="none") * mask
        ).mean()

        # Total loss
        loss = loss_x + config.LAMBDA_U * loss_u

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema.update(model)

        total_sum += loss.item()
        lx_sum    += loss_x.item()
        lu_sum    += loss_u.item()
        mask_sum  += mask.mean().item()
        global_step += 1

        if (step + 1) % log_every == 0 or (step + 1) == steps:
            elapsed = time.time() - t0
            print(f"  step {step+1:4d}/{steps} | "
                  f"loss {total_sum/(step+1):.4f} "
                  f"(Lx={lx_sum/(step+1):.4f} Lu={lu_sum/(step+1):.4f}) | "
                  f"mask {mask_sum/(step+1)*100:.1f}% | "
                  f"{elapsed/60:.1f}min", flush=True)

    n = max(steps, 1)
    return total_sum / n, lx_sum / n, lu_sum / n, mask_sum / n, global_step

def evaluate():
    """Evaluate on test set using EMA weights. Returns (overall_acc, per_class_acc)."""
    ema.apply_shadow(model)
    model.eval()

    class_correct = torch.zeros(config.NUM_CLASSES)
    class_total   = torch.zeros(config.NUM_CLASSES)

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="  eval ", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            logits, _    = model(imgs)
            preds        = torch.argmax(logits, dim=1)
            for c in range(config.NUM_CLASSES):
                m = labels == c
                class_correct[c] += (preds[m] == labels[m]).sum().item()
                class_total[c]   += m.sum().item()

    ema.restore(model)

    per_class_acc = (class_correct / class_total.clamp(min=1)) * 100
    overall_acc   = class_correct.sum() / class_total.sum() * 100
    return overall_acc.item(), per_class_acc.tolist()


if __name__ == "__main__":
    print()
    print(f"Starting SSDA FixMatch | budget={config.LABEL_BUDGET*100:.1f}% | strategy={config.SELECTION_STRATEGY} | allocation={config.ALLOCATION_STRATEGY}")
    print()

    best_acc    = 0.0
    global_step = 0
    train_start = time.time()

    for epoch in range(1, config.EPOCHS + 1):
        epoch_start = time.time()
        print(f"\n── Epoch {epoch}/{config.EPOCHS} ──────────────────", flush=True)
        avg_loss, avg_lx, avg_lu, avg_mask, global_step = train_one_epoch(global_step)
        print(f"  Evaluating...", flush=True)
        overall_acc, per_class_acc = evaluate()

        if overall_acc > best_acc:
            best_acc = overall_acc
            torch.save(model.state_dict(), "best_model.pth")
            torch.save(ema.shadow, "best_ema_shadow.pth")

        epoch_mins = (time.time() - epoch_start) / 60
        total_mins = (time.time() - train_start) / 60
        print(
            f"Epoch [{epoch:3d}/{config.EPOCHS}] | "
            f"Loss: {avg_loss:.4f}  (Lx={avg_lx:.4f}  Lu={avg_lu:.4f}) | "
            f"Mask%: {avg_mask*100:.1f} | "
            f"Acc: {overall_acc:.2f}%  Best: {best_acc:.2f}% | "
            f"Epoch: {epoch_mins:.1f}min | Total: {total_mins:.1f}min"
        )

        if epoch % 5 == 0:
            print("  Per-class acc:", [f"{a:.1f}" for a in per_class_acc])

    print(f"Done. Best target accuracy (EMA): {best_acc:.2f}%")
