# Cheat Sheet — Grappe de calcul Alliance (Narval)

## Concept général

```
[Ton laptop]  →  SSH  →  [Nœud de connexion]  →  sbatch  →  [Nœuds de calcul]
                          (narval.alliancecan.ca)              (pas d'Internet !)
```

- Le **nœud de connexion** : là où tu atterris en SSH. A accès à Internet. Ne jamais y lancer de calculs lourds.
- Les **nœuds de calcul** : là où Slurm exécute tes jobs. Pas d'Internet. Accès aux mêmes fichiers via le réseau.
- **Slurm** : l'ordonnanceur — il reçoit tes scripts, les met en file d'attente, et les lance quand des ressources sont dispo.

---

## Stockage

| Espace | Chemin | Usage | Quota |
|--------|--------|-------|-------|
| **home** | `~/` ou `/home/username` | Config, scripts, code source | ~50 GB |
| **scratch** | `~/scratch` | Données, résultats, gros fichiers temporaires | ~20 TB, **non sauvegardé** |
| **project** | `~/projects/def-pesantg/username` | Données long terme, partagées avec le labo | ~1 TB |
| **$SLURM_TMPDIR** | (auto) | SSD local ultra-rapide, durée du job uniquement | ~800 GB |

> **Règle d'or** : ne jamais mettre un `.venv` dans `~/scratch` (millions de petits fichiers → ralentit toute la grappe). Utiliser `$SLURM_TMPDIR` à la place.

---

## Règles d'or

1. **Internet coupé sur les nœuds de calcul** → toute compilation Maven / pip install depuis PyPI doit se faire sur le nœud de connexion.
2. **Pas de `.venv` dans scratch** → recréer l'env Python dans `$SLURM_TMPDIR` à chaque job.
3. **Ne pas lancer de calculs lourds sur le nœud de connexion** → utiliser `salloc` pour les tests interactifs.

---

## Workflow pour ce projet (Python + Java/Maven)

### Une fois sur le nœud de connexion (avant le premier job)

```bash
# Charger les modules
module load StdEnv/2023 java maven

# Compiler Java et générer le classpath hors-ligne
cd ~/scratch/CP_based_reward_shaping
mvn -f MiniCPBP/pom.xml compile
mvn -f MiniCPBP/pom.xml dependency:build-classpath \
    -Dmdep.outputFile=MiniCPBP/target/java_classpath.txt

# Installer les dépendances Python dans un wheelhouse local (une seule fois)
module load python/3.11.5
pip download -r requirements.txt -d ~/wheelhouse/
```

### Structure d'un script Slurm (modèle)

```bash
#!/bin/bash
#SBATCH --job-name=NomDuJob
#SBATCH --account=def-pesantg
#SBATCH --time=08:00:00          # HH:MM:SS — mettre une marge
#SBATCH --cpus-per-task=16       # = nombre de workers Python
#SBATCH --mem=32G                # ~2G par worker Java
#SBATCH --output=mon_job.out
#SBATCH --error=mon_job.err

module load StdEnv/2023 python/3.11.5 java maven

cd ~/scratch/CP_based_reward_shaping

# Env Python dans le SSD temporaire (rapide, propre)
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --find-links=~/wheelhouse/ -r requirements.txt

# Lancer (workers = cpus-per-task)
python run_experiment.py --workers 16
```

---

## Commandes Slurm essentielles

### Soumettre / annuler

```bash
sbatch script.sh              # Soumettre un job
scancel <JOB_ID>              # Annuler un job
scancel -u $USER              # Annuler TOUS ses jobs
```

### Surveiller

```bash
sq                            # Ses jobs en cours (PD=pending, R=running)
squeue -u $USER               # Idem, plus verbeux
squeue -u $USER -o "%.10i %.9P %.20j %.8T %.10M %.6D %R"  # Format détaillé
```

### Suivre l'exécution en direct

```bash
tail -f mon_job.out           # Stdout en direct
tail -f mon_job.err           # Stderr en direct (tqdm va ici)
tail -f mon_job.err -n 50     # Les 50 dernières lignes puis suivi
```

### Après le job

```bash
seff <JOB_ID>                 # Rapport CPU/RAM — pour calibrer les prochains jobs
sacct -j <JOB_ID> --format=JobID,Elapsed,CPUTime,MaxRSS  # Stats détaillées
```

### Session interactive (pour déboguer)

```bash
salloc --time=00:30:00 --cpus-per-task=4 --mem=8G --account=def-pesantg
# → donne un shell sur un nœud de calcul, quitter avec exit
```

---

## Commandes utiles pour ce projet

```bash
# Voir les résultats en cache (toutes instances)
ls experiment_results/cache/ | wc -l

# Filtrer par instance
ls experiment_results/cache/ | grep "^8"
ls experiment_results/cache/ | grep "^4"

# Compter les runs réussis pour une méthode
ls experiment_results/cache/ | grep "q-cp-etr"
```

---

## Transférer des fichiers vers son laptop

```bash
# Depuis ton laptop — télécharger les résultats
scp -r username@narval.alliancecan.ca:~/scratch/CP_based_reward_shaping/experiment_results/ ./

# Juste les graphiques
scp username@narval.alliancecan.ca:~/scratch/CP_based_reward_shaping/experiment_results/*.png ./

# Avec rsync (reprend là où ça s'est arrêté, plus robuste)
rsync -avz --progress \
    username@narval.alliancecan.ca:~/scratch/CP_based_reward_shaping/experiment_results/ \
    ./experiment_results/
```

---

## Dimensionner son job

| Ressource | Règle |
|-----------|-------|
| `--cpus-per-task` | = `--workers` dans run_experiment.py |
| `--mem` | ~2G × workers (overhead JVM par worker Java) |
| `--time` | `(nb_runs × durée_1_run) / workers × 1.5` (marge de sécurité) |
| GPU | Inutile pour Q-learning tabulaire |

> **Estimer la durée** : lancer un petit job (2 seeds, 1 instance) et regarder `seff` ou la barre tqdm, puis extrapoler.

---

## Modules utiles

```bash
module load StdEnv/2023        # Environnement de base (toujours en premier)
module load python/3.11.5
module load java
module load maven

module list                    # Voir les modules chargés
module avail python            # Chercher les versions disponibles
```

---

## En cas de problème

```bash
# Voir les 20 dernières lignes d'erreur d'un job
tail -20 mon_job.err

# SSH sur le nœud de calcul en cours (visible dans squeue colonne NODELIST)
ssh <nom_du_noeud>
htop                           # Voir l'utilisation CPU/RAM en temps réel

# Vérifier l'espace disque
diskusage_report               # Quota home/scratch/project
```
