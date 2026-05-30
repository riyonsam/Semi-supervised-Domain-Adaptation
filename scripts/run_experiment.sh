#!/bin/bash
#SBATCH --job-name=ssda_exp
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --partition=Teaching
#SBATCH --exclude=saxa

# Usage: sbatch --export=STRATEGY=uncertainty,BUDGET=0.01,ALLOCATION=equal run_experiment.sh
# STRATEGY:   random | uncertainty | diversity | hybrid
# BUDGET:     0.005 | 0.01 | 0.02 | 0.05
# ALLOCATION: proportional | equal | difficulty

STRATEGY=${STRATEGY:-random}
BUDGET=${BUDGET:-0.01}
ALLOCATION=${ALLOCATION:-proportional}

echo "=== Job ${SLURM_JOB_ID}: strategy=${STRATEGY}  budget=${BUDGET}  allocation=${ALLOCATION} ==="

# ── Environment ───────────────────────────────────────────────────────────────
. /home/htang2/toolchain-20251006/toolchain.rc
source ~/myenv/bin/activate

# ── Copy data to scratch ──────────────────────────────────────────────────────
SCRATCH=/disk/scratch/$USER
mkdir -p $SCRATCH/VisDA
echo "Copying VisDA to scratch..."
rsync -a ~/ml_data/VisDA/ $SCRATCH/VisDA/
if [ ! -d "$SCRATCH/VisDA/train" ] || [ ! -d "$SCRATCH/VisDA/validation" ]; then
    echo "ERROR: VisDA data not found in scratch after rsync. Aborting." >&2
    exit 1
fi
echo "Done. Starting training."

# ── Train ─────────────────────────────────────────────────────────────────────
cd ~/Semi-supervised-Domain-Adaptation
export VISDA_DATA_DIR=$SCRATCH/VisDA
export EXP_STRATEGY=${STRATEGY}
export EXP_BUDGET=${BUDGET}
export EXP_ALLOCATION=${ALLOCATION}

python train_fixmatch.py

# ── Save results ──────────────────────────────────────────────────────────────
OUTDIR=~/results/ssda_sweep/${STRATEGY}_b${BUDGET}
mkdir -p $OUTDIR
cp best_model.pth      $OUTDIR/best_model_${SLURM_JOB_ID}.pth     2>/dev/null || true
cp best_ema_shadow.pth $OUTDIR/best_ema_${SLURM_JOB_ID}.pth       2>/dev/null || true
cp logs/slurm-${SLURM_JOB_ID}.out $OUTDIR/train_log.out           2>/dev/null || true

echo "Job done: strategy=${STRATEGY} budget=${BUDGET} allocation=${ALLOCATION}"
