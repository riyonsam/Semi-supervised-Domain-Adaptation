import os
import torch

# Paths
DATA_DIR = os.environ.get("VISDA_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "VisDA"))
NUM_CLASSES = 12

# VisDA subdirectory names.
# SOURCE_SUBDIR: synthetic source images (VisDA train/ = rendered).
# TARGET_SUBDIR: real target images -- used for BOTH train_target and test.
# (VisDA has no separate held-out test set; standard protocol evaluates on
#  the full validation set, consistent with the SSDA literature.)
SOURCE_SUBDIR = "train"
TARGET_SUBDIR = "validation"

# Debug mode (small subset for quick testing)
DEBUG         = False  # set False for full training
DEBUG_SAMPLES = 300    # max samples per split in debug mode

# Training hyperparameters  (paper App. A.1)
# SGD + Nesterov, weight decay=5e-4, no decay on BN params.
# LR is lower than the paper (0.03) since we fine-tune a pretrained ResNet-50.
BATCH_SIZE   = 16   # reduced -- FixMatch uses 3 loaders + 2 views, OOM at 32
LR           = 3e-3    # tune if needed; paper trains from scratch at 0.03
MOMENTUM     = 0.9     # SGD Nesterov momentum
WEIGHT_DECAY = 5e-4    # not applied to BN params
EPOCHS       = 50

# FixMatch loss  (paper Sec. 2)
LAMBDA_U       = 1.0   # weight for unsupervised loss lambda_u
CONF_THRESHOLD = 0.95  # confidence threshold tau

# EMA  (paper App. A.1: decay=0.999, used for evaluation)
EMA_DECAY = 0.999

# Label budget
LABEL_BUDGET  = 0.01                         # fraction of target labelled
LABEL_BUDGETS = [0.005, 0.01, 0.02, 0.05]   # grid for experiments

# Sampler -- Choices: random | uncertainty | diversity | hybrid
SELECTION_STRATEGY = "random"

# Allocation -- Choices: proportional | equal | difficulty
ALLOCATION_STRATEGY = "proportional"

# Misc
NUM_WORKERS = 0 if not torch.cuda.is_available() else 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 42
