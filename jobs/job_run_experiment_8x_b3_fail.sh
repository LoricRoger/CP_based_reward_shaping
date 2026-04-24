#!/bin/bash
#SBATCH --job-name=Exp8x_b3_fail
#SBATCH --account=def-pesantg
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=job_run_experiment_8x_b3_fail.out
#SBATCH --error=job_run_experiment_8x_b3_fail.err

# Load environment
module load StdEnv/2023
module load python/3.11.5
module load java
module load maven

cd ~/scratch/CP_based_reward_shaping
source .venv/bin/activate

python run_experiment.py \
    --instances 8s 8medium 8hard \
    --methods q-none:b3:fail q-classic:b3:fail q-cp-etr:b3:fail cp-greedy:b3:fail \
    --seeds 20 \
    --episodes 5_000 \
    --workers 16
