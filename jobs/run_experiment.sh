#!/bin/bash
#SBATCH --job-name=RunExperiment
#SBATCH --account=def-pesantg
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8              # Nombre de cœurs CPU
#SBATCH --mem=16G                       # Mémoire vive (RAM) demandée
#SBATCH --output=run_experiment.out         # Fichier pour la sortie standard
#SBATCH --error=run_experiment.err          # Fichier pour les erreurs

# Load environment
module load StdEnv/2023
module load python/3.11.5
module load java
module load maven

cd ~/scratch/CP_based_reward_shaping
source .venv/bin/activate

# 4. Lancer ton code !
python run_experiment.py --instances 8s 8medium 8hard --methods q-none q-classic q-cp-etr cp-greedy optimal --seeds 20 --episodes 5_000 --workers 8