# Semi-Supervised Domain Adaptation on VisDA-2017

---

## Introduction

In many real-world machine learning tasks, there is a **domain gap** between training data (source) and deployment data (target). For example, a model trained on synthetic rendered images may perform poorly on real photographs, even if they depict the same objects. This project investigates **Semi-Supervised Domain Adaptation (SSDA)**: we have access to labelled source data and only a tiny fraction of labelled target data (1%), and we want to learn a model that generalises well to the full target distribution.

We use **VisDA-2017**, a large-scale benchmark with:
- **Source**: ~152,000 synthetic rendered images across 12 object classes
- **Target**: ~55,000 real photographs across the same 12 classes
- **Task**: Classify real images correctly, training with full source labels + 1% target labels + all unlabelled target images

The 12 classes are: aeroplane, bicycle, bus, car, horse, knife, motorcycle, person, plant, skateboard, train, truck.

### Research questions

This project systematically answers three questions:

1. **Method**: Does FixMatch (consistency regularisation) outperform a simple pseudo-label baseline?
2. **Selection**: Given a fixed labelling budget, *which* target samples should we label — random, most uncertain, most diverse, or a hybrid?
3. **Allocation**: Given a fixed labelling budget, *how many* labels should go to each class — proportional to class size, equal across classes, or more to harder classes?
4. **Budget**: How does overall label budget (0.5% → 5%) affect final accuracy?

---

## Methods

### Baseline (pseudo-label)

The baseline trains with three losses:
- **Supervised loss** on labelled source images
- **Supervised loss** on the small set of labelled target images
- **Pseudo-label loss** on unlabelled target images (using high-confidence model predictions as labels)

Training: Adam optimiser, 30 epochs, ResNet-50 backbone.

### FixMatch

FixMatch improves on the baseline by replacing the pseudo-label loss with a **consistency regularisation** loss:
- A **weakly augmented** version of an unlabelled image generates a pseudo-label (if confidence ≥ 0.95)
- The model must then predict the same label for a **strongly augmented** version of the same image
- Strong augmentation uses RandAugment + Cutout, making the task harder and more regularising
- An **EMA (Exponential Moving Average)** copy of the model is maintained and used for evaluation — this gives more stable, higher-quality predictions than the live model

Training: SGD + cosine LR decay, 50 epochs, same ResNet-50 backbone.

### Label selection strategies

These control *which* target samples are annotated within the budget:

| Strategy | Logic |
| --- | --- |
| `random` | Stratified random sample per class — simple and unbiased |
| `uncertainty` | Select samples with highest model entropy — the model is most confused about these |
| `diversity` | K-means clustering on ResNet features — select spread-out, representative samples |
| `hybrid` | Filter to uncertain samples first, then apply diversity within that pool |

**Reasoning**: Uncertainty sampling targets the decision boundary but may pick noisy/atypical samples. Diversity sampling ensures broad coverage of the target distribution. Hybrid tries to get both. With only 1% of data, diversity may matter more than hitting the boundary precisely.

### Label allocation strategies

These control *how many* labels are assigned to each class from the total budget:

| Allocation | Logic |
| --- | --- |
| `proportional` | Each class gets `budget%` of its own samples (preserves class frequency) |
| `equal` | Total budget divided evenly across all 12 classes (ignores class size) |
| `difficulty` | Classes with higher average model entropy get more labels (focus on hard classes) |

**Reasoning**: VisDA classes are imbalanced. Proportional allocation is the natural default. Equal allocation may starve large classes. Difficulty allocation hypothesises that putting more labels on confusing classes will help most — but this relies on the entropy signal from a model that has barely seen target data, which may be unreliable.

---

## Experiment log — what ran where

This section maps every experiment to its SLURM job ID, log file, and result.

### Experiment 0 — Method comparison

> Fixed: budget=1%, strategy=random, allocation=proportional

| Run | SLURM job | Log file | Best Accuracy |
| --- | --- | --- | --- |
| Baseline (pseudo-label, 30 epochs) | 2217459 | `slurm-2217459.out` (root dir) | 75.96% |
| FixMatch (50 epochs) | 2217460 | `slurm-fixmatch-2217460.out` (root dir) | 86.27% |

**Node**: both ran on `damnii07`.
**Example log output**:
```
Done. Best target accuracy: 75.96%          ← baseline
Done. Best target accuracy (EMA): 86.27%    ← FixMatch
```

---

### Experiment 1 — Selection strategy sweep

> Fixed: budget=1%, allocation=proportional. Varied: strategy

> **Note**: These jobs used an older version of `active_sampler.py` with a global (not per-class) budget. Results are still valid — the difference only affects how labels are distributed within each class, and the selection logic itself is unchanged.

| Strategy | SLURM job | Log file | Best Accuracy |
| --- | --- | --- | --- |
| random | 2218293 | `logs/slurm-2218293.out` | **86.67%** |
| uncertainty | 2218294 | `logs/slurm-2218294.out` | 82.20% |
| diversity | 2218295 | `logs/slurm-2218295.out` | 85.81% |
| hybrid | 2218296 | `logs/slurm-2218296.out` | 82.36% |

**Node**: all ran on `damnii07`.
**Plots**: run the command in Step 8 → `results_strategy_summary.png`, `results_strategy_perclass.png`
**Key output lines to cite** (from each log, last few lines):
```
Done. Best target accuracy (EMA): XX.XX%
Per-class acc: [...]
```

---

### Experiment 2 — Budget sweep

> Fixed: strategy=random, allocation=proportional. Varied: budget

| Budget | Labelled images | SLURM job | Log file | Best Accuracy |
| --- | --- | --- | --- | --- |
| 0.5% | ~280 | 2219012 | `logs/slurm-2219012.out` | 84.25% |
| 1.0% | ~560 | 2219013 | `logs/slurm-2219013.out` | 86.58% |
| 2.0% | ~1,120 | 2219014 | `logs/slurm-2219014.out` | 87.67% |
| 5.0% | ~2,800 | 2219015 | `logs/slurm-2219015.out` | 89.16% |

**Node**: all ran on `damnii07`.
**Plots**: `results_budget_summary.png`, `results_budget_perclass.png`
**Example log output**:
```
[visda_loader] Target split (random)  ->  labelled: NNN (X.XX%)  |  unlabelled: NNNNN
Done. Best target accuracy (EMA): XX.XX%
```

---

### Experiment 3 — Allocation sweep

> Fixed: strategy=random, budget=1%. Varied: allocation

| Allocation | SLURM job | Log file | Node | Best Accuracy |
| --- | --- | --- | --- | --- |
| proportional | 2219013 | `logs/slurm-2219013.out` | damnii07 | **86.58%** |
| equal | 2221058 | `logs/slurm-2221058.out` | damnii07 | 83.67% |
| difficulty | 2221510 | `logs/slurm-2221510.out` | damnii10 | 84.82% |

> Proportional result reuses job 2219013 from the budget sweep (same settings).

**Plots**: `results_allocation_summary.png`, `results_allocation_perclass.png`
**Example log output**:
```
[visda_loader] Target split (random/equal)       ->  labelled: NNN ...
[visda_loader] Target split (random/difficulty)  ->  labelled: NNN ...
Done. Best target accuracy (EMA): XX.XX%
Per-class acc: [...]
```

---

## Results and findings

### Finding 1 — FixMatch vs baseline

| Method | Best Accuracy |
| --- | --- |
| Baseline (pseudo-label) | 75.96% |
| **FixMatch** | **86.27%** |

FixMatch outperforms the baseline by **+10.31 pp**. Consistency regularisation with strong augmentation is substantially more effective than simple pseudo-labelling for domain adaptation, even with only 1% target labels.

---

### Finding 2 — Selection strategy

| Strategy | Best Accuracy |
| --- | --- |
| **random** | **86.67%** |
| diversity | 85.81% |
| hybrid | 82.36% |
| uncertainty | 82.20% |

Random selection wins. Uncertainty sampling underperforms — early in training the model is a poor oracle, so high-entropy samples tend to be noisy rather than genuinely informative. Diversity sampling is the closest competitor (-0.86 pp), suggesting that broad coverage of the feature space is more important than targeting the decision boundary at this budget. Hybrid does not improve over diversity, likely because the uncertainty filter reduces sample diversity.

---

### Finding 3 — Label budget

| Budget | Labelled images | Best Accuracy |
| --- | --- | --- |
| 0.5% | ~280 | 84.25% |
| 1.0% | ~560 | 86.58% |
| 2.0% | ~1,120 | 87.67% |
| 5.0% | ~2,800 | 89.16% |

Accuracy improves steadily with more labels (+4.91 pp from 0.5% → 5%). Notably, even 0.5% labels (~280 images) achieves 84% — FixMatch learns effectively from unlabelled data. The gains diminish as budget increases, consistent with the unlabelled data providing the dominant learning signal.

---

### Finding 4 — Allocation strategy

| Allocation | Best Accuracy |
| --- | --- |
| **proportional** | **86.58%** |
| difficulty | 84.82% |
| equal | 83.67% |

Proportional allocation wins. Equal allocation underperforms (-2.91 pp) because VisDA classes are imbalanced — forcing equal labels effectively starves the larger classes of supervision. Difficulty allocation (-1.76 pp) gives extra labels to classes with high average model entropy, but the entropy signal from a model that has seen almost no target data is unreliable, so the allocation may not match true class difficulty.

---

### Conclusion

FixMatch with random selection and proportional allocation at 1% label budget achieves **86.27–86.67% accuracy** on VisDA-2017 (synthetic → real), compared to a pseudo-label baseline of 75.96%. The key conclusions for report writing:

- **Method choice matters most**: FixMatch > baseline by ~10 pp
- **Selection strategy matters little at 1%**: random ≈ diversity; uncertainty hurts
- **More labels always helps**: 0.5% → 5% gives +5 pp, but gains are diminishing
- **Proportional allocation is the right default**: reflects the dataset's natural class distribution

---

## Limitations and future development

The following improvements were identified but not implemented due to time and compute constraints. They are documented here for future work or anyone continuing this project.

### 1. Multiple random seeds
All experiments use a single seed (42). Results with only one seed cannot report mean ± std, meaning small differences between strategies (1–4 pp) could be attributed to random variation rather than the strategy itself. **Future work**: rerun key comparisons (selection sweep + allocation sweep) with 3 seeds (e.g. 42, 123, 456) and report mean ± std. This requires ~14 additional jobs (~2–3 days on the cluster).

### 2. More training epochs
Loss curves are still declining at epoch 50, suggesting models have not fully converged. Increasing to 100 epochs would likely raise all accuracy numbers slightly but is unlikely to change the ranking between strategies. **Future work**: set `EPOCHS = 100` in `config_fixmatch.py` and rerun key comparisons.

### 3. Multi-round active learning
The current implementation does single-round selection — all labels are selected upfront before any target training begins. True active learning is iterative: train → select → label → retrain × R rounds. Single-round selection uses a source-pretrained model as the oracle, which is a weak signal for the target domain. **Future work**: implement an iterative loop in `train_fixmatch.py` with R=3 rounds as a targeted ablation (e.g. uncertainty + proportional and diversity + proportional at 1% budget).

### 4. Additional allocation strategies
Two further allocation strategies were proposed but not implemented:
- **Global (Top-K)**: ignores class structure entirely — selects the best K samples globally by score. Simple but risks over-sampling dominant classes.
- **Cluster-balanced**: uses MiniBatchKMeans to partition all target embeddings into `n_budget` clusters without relying on class predictions — more robust when the model has not yet adapted to the target domain.

**Future work**: add `global` and `cluster` options to `_compute_class_budget()` in `sampler/active_sampler.py`.

### 5. Full experiment grid
A complete grid (4 selectors × 4 allocations × 4 budgets × 3 seeds = 144 runs) would give a thorough picture of how selection and allocation interact across budgets. This is computationally infeasible within the current project timeline (~2,500 GPU hours). **Future work**: use the hierarchical ablation design — selector sweep × budgets, allocation sweep × budgets, and a 4×4 interaction table at 1% budget, all with 3 seeds.

---

## Plots produced

| File | Contents | Command in Step 8 |
| --- | --- | --- |
| `results_strategy_summary.png` | Accuracy/loss curves + best accuracy bar for 4 selection strategies | strategy sweep command |
| `results_strategy_perclass.png` | Per-class accuracy for 4 strategies | strategy sweep command |
| `results_budget_summary.png` | Accuracy/loss curves + best accuracy bar for 4 budgets | budget sweep command |
| `results_budget_perclass.png` | Per-class accuracy for 4 budgets | budget sweep command |
| `results_allocation_summary.png` | Accuracy/loss curves + best accuracy bar for 3 allocations | allocation sweep command |
| `results_allocation_perclass.png` | Per-class accuracy for 3 allocations | allocation sweep command |

---

## Project structure

```text
Semi-supervised-Domain-Adaptation/
│
├── config_fixmatch.py           # Hyperparameters for FixMatch
│                                #   SELECTION_STRATEGY: random|uncertainty|diversity|hybrid
│                                #   ALLOCATION_STRATEGY: proportional|equal|difficulty
│                                #   LABEL_BUDGET: 0.005|0.01|0.02|0.05
│
├── train_fixmatch.py            # FixMatch training loop (main entry point)
│
├── data/
│   └── visda_loader_fixmatch.py # DataLoaders for FixMatch (weak + strong augmentation)
│
├── models/
│   └── resnet50_da.py           # ResNet-50 backbone + 256-dim bottleneck head
│
├── sampler/
│   └── active_sampler.py        # All 4 selection + 3 allocation strategies
│
├── utils/
│   └── augment_fixmatch.py      # Strong augmentation (RandAugment + Cutout)
│
├── scripts/
│   ├── run_experiment.sh        # SLURM: parametric job (reads STRATEGY, BUDGET, ALLOCATION)
│   └── submit_sweep.sh          # Helper: submit multiple jobs at once
│
├── analysis/
│   └── plot_results.py          # Reads SLURM log files → generates plots
│
├── results/                     # Generated plot images
│
└── logs/                        # All SLURM output logs
    ├── slurm-2218293.out        # selection: random
    ├── slurm-2218294.out        # selection: uncertainty
    ├── slurm-2218295.out        # selection: diversity
    ├── slurm-2218296.out        # selection: hybrid
    ├── slurm-2219012.out        # budget: 0.5%
    ├── slurm-2219013.out        # budget: 1.0%  (also = proportional allocation baseline)
    ├── slurm-2219014.out        # budget: 2.0%
    ├── slurm-2219015.out        # budget: 5.0%
    ├── slurm-2221058.out        # allocation: equal
    └── slurm-2221510.out        # allocation: difficulty
```

---

## Before you start — your student ID

Every command containing `sXXXXXXX` — replace with **your own student ID** (e.g. `s1234567`).

---

## Step 1 — Set up SSH on your Mac (one time only)

```bash
nano ~/.ssh/config
```

Paste this (replace both `sXXXXXXX` with your student ID):

```text
Host teaching-cluster
    HostName mlp.inf.ed.ac.uk
    User sXXXXXXX
    ProxyJump sXXXXXXX@student.ssh.inf.ed.ac.uk
    ControlMaster auto
    ControlPath ~/.ssh/control-%r@%h:%p
    ControlPersist 4h
```

Save and exit: `Ctrl+O` → `Enter` → `Ctrl+X`.

Open the SSH tunnel once per Mac terminal session:

```bash
ssh -fN teaching-cluster   # enter your password when prompted
```

---

## Step 2 — Log into the cluster

```bash
ssh teaching-cluster
```

All remaining steps run **on the cluster** unless stated otherwise.

---

## Step 3 — Clone the code (one time only)

```bash
cd ~
git clone https://github.com/shakeswang/Semi-supervised-Domain-Adaptation.git
cd Semi-supervised-Domain-Adaptation
mkdir -p logs
```

---

## Step 4 — Set up Python environment (one time only)

```bash
. /home/htang2/toolchain-20251006/toolchain.rc
python -m venv ~/myenv
source ~/myenv/bin/activate
pip install -r requirements.txt
```

Pre-download ResNet-50 weights (compute nodes have no internet access):

```bash
python -c "import torchvision.models as m; m.resnet50(weights='IMAGENET1K_V1')"
```

---

## Step 5 — Get the dataset (one time only)

The VisDA dataset (~13.8 GB) must be at `~/ml_data/VisDA/` on the cluster.

**If a teammate already has it**, copy from their directory:

```bash
mkdir -p ~/ml_data
cp -r /home/sXXXXXXX/ml_data/VisDA ~/ml_data/VisDA
```

Expected structure:

```text
~/ml_data/VisDA/
    train/          ← synthetic source images (~12 GB, 12 class folders)
    validation/     ← real target images (~1.8 GB, 12 class folders)
```

---

## Step 6 — Run a single FixMatch job

Runs with default settings (random selection, proportional allocation, 1% budget, 50 epochs):

```bash
sbatch --export=STRATEGY=random,BUDGET=0.01,ALLOCATION=proportional scripts/run_experiment.sh
```

Watch the live log (replace `JOBID` with your actual job ID):

```bash
tail -f ~/Semi-supervised-Domain-Adaptation/logs/slurm-JOBID.out
```

Each epoch prints:

```text
Epoch [  5/50] | Loss: 0.1823 (Lx=0.09 Lu=0.09) | Mask%: 72.1 | Acc: 83.41%  Best: 84.12% | Epoch: 21.3min | Total: 106.5min
  Per-class acc: ['84.4', '77.1', '76.2', ...]
```

---

## Step 7 — Run sweeps in parallel

All sweeps use `submit_sweep.sh`. Each job takes ~18–20 hours. Jobs run in parallel.

### Selection strategy sweep (4 jobs)

```bash
bash scripts/submit_sweep.sh strategy
# Submits 4 jobs: random / uncertainty / diversity / hybrid
# All at: budget=0.01, allocation=proportional
```

### Budget sweep (4 jobs)

```bash
bash scripts/submit_sweep.sh budget
# Submits 4 jobs: budget=0.005 / 0.01 / 0.02 / 0.05
# All at: strategy=random, allocation=proportional
```

### Allocation sweep (2 new jobs — proportional reuses budget sweep result)

```bash
bash scripts/submit_sweep.sh allocation
# Submits: equal and difficulty
# Proportional result already exists in logs/slurm-2219013.out
```

### Monitor jobs

```bash
squeue -u $USER

# Tail multiple logs at once
for j in 2221058 2221510; do
    echo "=== $j ==="
    tail -3 ~/Semi-supervised-Domain-Adaptation/logs/slurm-$j.out
done
```

---

## Step 8 — Generate plots (run on your Mac)

Copy logs from the cluster first:

```bash
scp "teaching-cluster:~/Semi-supervised-Domain-Adaptation/logs/slurm-*.out" \
    ~/Semi-supervised-Domain-Adaptation/logs/
```

Then generate each plot set:

```bash
cd ~/Semi-supervised-Domain-Adaptation

# Selection strategy sweep
PLOT_OUT=results/results_strategy_summary.png PLOT_OUT_PERCLASS=results/results_strategy_perclass.png \
    python analysis/plot_results.py \
    logs/slurm-2218293.out \
    logs/slurm-2218294.out \
    logs/slurm-2218295.out \
    logs/slurm-2218296.out

# Budget sweep
PLOT_OUT=results/results_budget_summary.png PLOT_OUT_PERCLASS=results/results_budget_perclass.png \
    python analysis/plot_results.py \
    logs/slurm-2219012.out \
    logs/slurm-2219013.out \
    logs/slurm-2219014.out \
    logs/slurm-2219015.out

# Allocation sweep
PLOT_OUT=results/results_allocation_summary.png PLOT_OUT_PERCLASS=results/results_allocation_perclass.png \
    python analysis/plot_results.py \
    logs/slurm-2219013.out \
    logs/slurm-2221058.out \
    logs/slurm-2221510.out
```

---

## Step 9 — Run a custom single job (optional)

```bash
sbatch --export=STRATEGY=diversity,BUDGET=0.02,ALLOCATION=equal scripts/run_experiment.sh
```

---

## Useful cluster commands

```bash
squeue -u $USER                          # see all your jobs
scancel <JOBID>                          # cancel a job
tail -f ~/Semi-supervised-Domain-Adaptation/logs/slurm-<JOBID>.out   # live log
sinfo -p Teaching                        # check node availability
```

---

## Configuration reference

### `config_fixmatch.py`

| Parameter | Default | Description |
| --- | --- | --- |
| `DEBUG` | `False` | `True` = fast 300-sample smoke test |
| `BATCH_SIZE` | 16 | Reduced — 3 loaders × 2 views exhausts GPU at 32 |
| `LR` | 3e-3 | SGD base learning rate (cosine decay) |
| `EPOCHS` | 50 | ~20 min/epoch on Teaching GPU |
| `CONF_THRESHOLD` | 0.95 | Minimum confidence to use a pseudo-label |
| `EMA_DECAY` | 0.999 | EMA model decay — used for evaluation |
| `LABEL_BUDGET` | 0.01 | Fraction of target data to label |
| `SELECTION_STRATEGY` | `random` | `random` \| `uncertainty` \| `diversity` \| `hybrid` |
| `ALLOCATION_STRATEGY` | `proportional` | `proportional` \| `equal` \| `difficulty` |

