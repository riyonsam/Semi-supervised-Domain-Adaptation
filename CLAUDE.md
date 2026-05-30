# Project Context for Claude

This file gives Claude full context about this project. If a teammate pastes this into Claude alongside the project files, Claude will understand everything that has been done and why — and can assist as if it had been part of the project from the start.

---

## What this project is

Semi-Supervised Domain Adaptation (SSDA) on VisDA-2017. The goal is to classify real photos (target) using a model trained on synthetic rendered images (source), with only 1% of real photos labelled. The domain gap between synthetic and real is large — this is the core challenge.

- **Dataset**: VisDA-2017, 12 classes (aeroplane, bicycle, bus, car, horse, knife, motorcycle, person, plant, skateboard, train, truck)
- **Source**: ~152,000 synthetic images (train/)
- **Target**: ~55,000 real photos (validation/)
- **Backbone**: ResNet-50, pretrained on ImageNet
- **Framework**: PyTorch, trained on Edinburgh University Teaching Cluster (SLURM)

---

## What has been implemented and why

### Two training methods

**Baseline** (`train.py`):
- Supervised loss on source + labelled target
- Pseudo-label loss on unlabelled target (confidence ≥ 0.9)
- Adam, 30 epochs, batch size 32
- Result: **75.96%** (job 2217459, `slurm-2217459.out` in root dir)

**FixMatch** (`train_fixmatch.py`):
- Supervised loss + consistency regularisation
- Weak augmentation → pseudo-label; strong augmentation (RandAugment + Cutout) must match
- Confidence threshold 0.95, SGD + cosine LR decay, 50 epochs, batch size 16
- EMA model (decay=0.999) used for evaluation — gives more stable predictions
- Result: **86.27%** (job 2217460, `slurm-fixmatch-2217460.out` in root dir)
- FixMatch wins by **+10.31 pp** — consistency regularisation is much better than pseudo-labelling alone

### Four label selection strategies (`sampler/active_sampler.py`)

These decide *which* target samples to annotate within the budget:

- `random`: Stratified random per class. Simple baseline.
- `uncertainty`: Pick highest-entropy samples — model is most confused about these. Reasoning: informative samples near the decision boundary.
- `diversity`: K-means on ResNet features, pick spread-out samples. Reasoning: broad coverage > hitting the boundary at low budget.
- `hybrid`: Filter to uncertain samples first, then diversify within that pool.

**Why random wins**: At 1% budget, the model has barely seen target data, so entropy is unreliable as an informativeness signal. Uncertainty samples tend to be noisy/atypical. Diversity is competitive (-0.86 pp) because coverage matters more.

### Three label allocation strategies (`sampler/active_sampler.py`, `_compute_class_budget()`)

These decide *how many* labels to assign per class from the total budget:

- `proportional`: Each class gets `budget%` of its own samples. Natural default, preserves class frequency.
- `equal`: Budget divided evenly across 12 classes. Ignores class imbalance.
- `difficulty`: Classes with higher average model entropy get more labels. Focuses on hard classes.

**Why proportional wins**: VisDA is imbalanced — equal starves large classes. Difficulty allocation depends on reliable entropy estimates, which a barely-trained model cannot provide.

---

## All experiment results (complete)

### Method comparison
| Method | SLURM job | Log file | Best Acc |
|---|---|---|---|
| Baseline | 2217459 | `slurm-2217459.out` (root) | 75.96% |
| FixMatch | 2217460 | `slurm-fixmatch-2217460.out` (root) | 86.27% |

### Selection strategy sweep (budget=1%, allocation=proportional)
| Strategy | SLURM job | Log file | Best Acc |
|---|---|---|---|
| random | 2218293 | `logs/slurm-2218293.out` | 86.67% |
| uncertainty | 2218294 | `logs/slurm-2218294.out` | 82.20% |
| diversity | 2218295 | `logs/slurm-2218295.out` | 85.81% |
| hybrid | 2218296 | `logs/slurm-2218296.out` | 82.36% |

> These jobs used an older version of `active_sampler.py` with a global (not per-class) budget. Results are still valid — selection logic is unchanged.

### Budget sweep (strategy=random, allocation=proportional)
| Budget | SLURM job | Log file | Best Acc |
|---|---|---|---|
| 0.5% | 2219012 | `logs/slurm-2219012.out` | 84.25% |
| 1.0% | 2219013 | `logs/slurm-2219013.out` | 86.58% |
| 2.0% | 2219014 | `logs/slurm-2219014.out` | 87.67% |
| 5.0% | 2219015 | `logs/slurm-2219015.out` | 89.16% |

### Allocation sweep (strategy=random, budget=1%)
| Allocation | SLURM job | Log file | Node | Best Acc |
|---|---|---|---|---|
| proportional | 2219013 | `logs/slurm-2219013.out` | damnii07 | 86.58% |
| equal | 2221058 | `logs/slurm-2221058.out` | damnii07 | 83.67% |
| difficulty | 2221510 | `logs/slurm-2221510.out` | damnii10 | 84.82% |

> Jobs 2220659/2220660 failed (scratch disk on saxa/damnii09) — ignore those logs.
> Jobs 2221058/2221059 — 2221059 failed (scratch issue on damnii09), replaced by 2221510.

---

## Key design decisions and reasoning

- **Why batch size 16 (not 32)?** FixMatch loads 3 DataLoaders (source, labelled target, unlabelled target × 2 views). At batch 32 this exceeds GPU memory on Teaching cluster GPUs (~24GB GeForce RTX).
- **Why 50 epochs for FixMatch vs 30 for baseline?** FixMatch converges slower — it needs more iterations for the pseudo-label quality to improve enough for the consistency loss to be useful. The mask% (fraction of confident pseudo-labels) starts at ~40% epoch 1 and reaches ~78% by epoch 30.
- **Why EMA for evaluation?** The live model oscillates during training; the EMA model averages weights over time, giving consistently higher and more stable accuracy.
- **Why exclude `saxa` node?** `saxa` has a MIG-sliced GPU (gpu:1g.18gb) that caused OOM crashes. All `damnii` nodes have full ~24GB GPUs.
- **Why remove `--nodelist=damnii07`?** Originally pinned to damnii07 because it was the tested node. Removed to allow any damnii node — avoids queuing when damnii07 is busy.
- **Why proportional allocation reuses job 2219013?** The budget sweep at 1% is identical to proportional allocation at 1%, so no new job needed.

---

## Codebase — key files and what they do

```
config_fixmatch.py          Main config — change SELECTION_STRATEGY, ALLOCATION_STRATEGY, LABEL_BUDGET here
train_fixmatch.py           FixMatch training loop — reads EXP_STRATEGY, EXP_BUDGET, EXP_ALLOCATION env vars
data/visda_loader_fixmatch.py   Builds DataLoaders, calls active_select() to pick labelled samples
sampler/active_sampler.py   All selection + allocation logic
  - active_select()         Entry point — calls _compute_class_budget() then the strategy function
  - _compute_class_budget() Returns per-class label counts for the chosen allocation strategy
  - _uncertainty_select()   Entropy-based selection, per-class budget
  - _diversity_select()     K-means selection, per-class budget
  - _hybrid_select()        Uncertainty filter → diversity, per-class budget
  - _random_select_with_budget()  Stratified random, per-class budget
run_experiment.sh           SLURM batch script — reads STRATEGY/BUDGET/ALLOCATION env vars
submit_sweep.sh             Submits multiple jobs: modes = strategy | budget | allocation | full
plot_results.py             Reads SLURM .out log files, plots accuracy/loss curves + per-class bars
```

---

## Cluster setup

- **Cluster**: Edinburgh University Teaching Cluster (`mlp.inf.ed.ac.uk`)
- **SSH**: `ssh -fN teaching-cluster` (ControlMaster via ProxyJump through `student.ssh.inf.ed.ac.uk`)
- **Partition**: Teaching, 2-day time limit
- **Good nodes**: damnii07, damnii08, damnii10, damnii11, damnii12 (all have GeForce RTX ~24GB)
- **Avoid**: saxa (MIG GPU, OOM), damnii09 (300GB RAM, scratch issues)
- **Scratch**: Each node has local `/disk/scratch/sXXXXXXX/` — data is rsync'd there at job start
- **Data location**: `~/ml_data/VisDA/` (train/ + validation/)
- **Python env**: `~/myenv` — activate with `source ~/myenv/bin/activate`
- **Toolchain**: `. /home/htang2/toolchain-20251006/toolchain.rc` (needed before venv creation)
- **ResNet weights**: pre-downloaded to `~/.cache/torch/hub/checkpoints/` (nodes have no internet)

---

## How to run experiments

### Single job
```bash
sbatch --export=STRATEGY=random,BUDGET=0.01,ALLOCATION=proportional run_experiment.sh
```

### Sweeps
```bash
bash submit_sweep.sh strategy     # 4 jobs: selection strategies
bash submit_sweep.sh budget       # 4 jobs: label budgets
bash submit_sweep.sh allocation   # 2 new jobs: equal + difficulty
```

### Monitor
```bash
squeue -u $USER
tail -f ~/Semi-supervised-Domain-Adaptation/logs/slurm-JOBID.out
```

### Generate plots (on your Mac)
```bash
# After scp-ing logs from cluster:
PLOT_OUT=results_allocation_summary.png PLOT_OUT_PERCLASS=results_allocation_perclass.png \
    python plot_results.py logs/slurm-2219013.out logs/slurm-2221058.out logs/slurm-2221510.out
```

---

## Current project state (as of 2026-03-13)

- All 3 sweeps complete — results are final
- All logs downloaded and committed to `logs/`
- All 6 plot files generated and committed
- README fully written: intro → methods → experiment log → findings → conclusion → future development → instructions
- GitHub repo: `https://github.com/shakeswang/Semi-supervised-Domain-Adaptation` (private)
- Main branch is clean, no pending work

---

## What was NOT implemented and why

These were discussed and deliberately skipped due to time and compute constraints. Do not treat them as bugs or missing features — they are known limitations documented in the README.

**1. Multiple random seeds**: All runs use seed=42 only. Running 3 seeds across all experiments needs ~14 extra jobs (~2–3 days on cluster). Skipped due to deadline. Acknowledge as limitation in report.

**2. More epochs (100)**: Loss still declining at epoch 50 but conclusions unchanged. Doubling compute not justified.

**3. Multi-round active learning**: Real active learning iterates train→select→retrain×R. Current code does single-round upfront selection only. Significant code change required. Skipped — acknowledge as limitation.

**4. New allocation strategies (global, cluster-balanced)**: Global = class-agnostic Top-K. Cluster-balanced = MiniBatchKMeans partitioning without class labels. Both proposed by team, neither implemented. Proportional already wins clearly.

**5. Full experiment grid**: 4 selectors × 4 allocations × 4 budgets × 3 seeds = 144 runs (~2,500 GPU hours). Not feasible.

---

## How to use this file

Paste this entire file into your Claude conversation along with any project files you want help with. Claude will have full context on:
- What every file does and why it was written that way
- All experiment results and the reasoning behind each finding
- Known issues (saxa node, scratch disk failures, old code version for selection sweep)
- What was deliberately not implemented and why
- How to run, monitor, and plot any experiment
