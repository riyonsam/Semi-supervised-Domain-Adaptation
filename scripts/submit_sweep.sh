#!/bin/bash
# submit_sweep.sh  –  Submit one SLURM job per combination.
#
# Usage:
#   bash submit_sweep.sh                  # full 4x4 grid (16 jobs)
#   bash submit_sweep.sh strategy         # fix budget=0.01, sweep strategies (4 jobs)
#   bash submit_sweep.sh budget           # fix strategy=random, sweep budgets (4 jobs)
#   bash submit_sweep.sh allocation       # fix strategy=random budget=0.01, sweep allocations (3 jobs)

mkdir -p logs

MODE=${1:-full}

STRATEGIES=(random uncertainty diversity hybrid)
BUDGETS=(0.005 0.01 0.02 0.05)
ALLOCATIONS=(proportional equal difficulty)

submit() {
    local s=$1 b=$2 a=${3:-proportional}
    JID=$(sbatch --export=STRATEGY=${s},BUDGET=${b},ALLOCATION=${a} scripts/run_experiment.sh | awk '{print $NF}')
    echo "Submitted job ${JID}: strategy=${s}  budget=${b}  allocation=${a}"
}

case $MODE in
  strategy)
    echo "Strategy sweep (budget=0.01, allocation=proportional)"
    for s in "${STRATEGIES[@]}"; do
        submit $s 0.01 proportional
    done
    ;;
  budget)
    echo "Budget sweep (strategy=random, allocation=proportional)"
    for b in "${BUDGETS[@]}"; do
        submit random $b proportional
    done
    ;;
  allocation)
    echo "Allocation sweep (strategy=random, budget=0.01)"
    for a in "${ALLOCATIONS[@]}"; do
        submit random 0.01 $a
    done
    ;;
  *)
    echo "Full sweep (4 strategies x 4 budgets = 16 jobs)"
    for s in "${STRATEGIES[@]}"; do
        for b in "${BUDGETS[@]}"; do
            submit $s $b proportional
        done
    done
    ;;
esac
