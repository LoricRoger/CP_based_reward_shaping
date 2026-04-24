# Guide de migration — Refactoring CP-based Reward Shaping

Ce document décrit toutes les modifications apportées lors du refactoring, explique pourquoi chaque changement a été fait, et indique comment vérifier que rien n'est cassé.

---

## Vue d'ensemble des modifications

| # | Fichier(s) | Type | Résumé |
|---|-----------|------|--------|
| 1 | `environment.py` | Bug | Double reset corrigé dans `FrozenLakeExtendedActions` |
| 2 | `main.py`, `q_learning_cp.py` | Refactor | Paramètre `shaping_method_name` supprimé |
| 3 | `ModeETR.java`, `ModeMS.java`, `AbstractCPMode.java` (nouveau) | Refactor | Déduplication `fillTransitions` + logique budget |
| 4 | `q_learning_cp.py` | Bug | Restauration `initial_budget` protégée par `try/finally` |
| 5 | `q_learning_cp.py` | Refactor | Classe `CurriculumBudget` extraite |
| 6a | `config.py` | Refactor | `CP_MS_SHAPING_COEFF = 0.2` ajouté |
| 6b | `FrozenLakeCPService.java` | Fix | Double `;;` supprimé |
| 6c | `main.py` | Refactor | `cmd = []` inutile supprimé, branche `else` explicite ajoutée |
| 6d | `utils.py` | Refactor | Seeds niveau module supprimées |
| 6e | `utils.py` | Fix | `action_chars` complété avec les actions no-slip (4-7) |

---

## Détail de chaque modification

---

### 1. Double reset dans `FrozenLakeExtendedActions` (`environment.py`)

**Problème** : `env.reset()` était appelé deux fois. La première valeur `obs` était ignorée, et le PRNG de l'environnement avancait d'un état supplémentaire. En plus, `_curent_state` (typo) ne mettait jamais à jour l'attribut `_current_state` réellement utilisé dans `step()`.

```python
# AVANT (bugué)
def reset(self, **kwargs):
    self.budget = self.initial_budget
    obs, info = self.env.reset(**kwargs)
    self._curent_state = obs       # typo : n'écrit pas _current_state
    return self.env.reset(**kwargs)  # deuxième reset !

# APRÈS
def reset(self, **kwargs):
    self.budget = self.initial_budget
    obs, info = self.env.reset(**kwargs)
    self._current_state = obs
    return obs, info
```

**Impact** : en mode BUDGET, `_current_state` servait de fallback si le budget était épuisé (`step()` ligne 125). Avec ce bug, la valeur était toujours celle de `__init__` (0), ce qui causait un retour incorrect au début de l'épisode au lieu de l'état courant.

**Comment vérifier** :
```bash
python -c "
import environment
env = environment.create_environment('4x4', True, None, 10, budget=2)
obs1, info1 = env.reset(seed=42)
obs2, info2 = env.reset(seed=42)
assert obs1 == obs2, 'Les deux resets avec la même seed doivent donner le même état'
print('OK')
"
```

---

### 2. Paramètre `shaping_method_name` supprimé (`main.py`, `q_learning_cp.py`)

**Problème** : `train_q_learning_with_cp_shaping` avait deux paramètres (`shaping_type` et `shaping_method_name`) qui recevaient toujours la même valeur. C'était trompeur.

**Avant** :
```python
def train_q_learning_with_cp_shaping(..., shaping_type, ..., shaping_method_name, instance_id):
    # shaping_method_name utilisé uniquement pour nommer le CSV
```

**Après** :
```python
def train_q_learning_with_cp_shaping(..., shaping_type, ..., instance_id):
    # shaping_type utilisé pour nommer le CSV aussi
```

L'appel dans `main.py` a été mis à jour en conséquence.

**Comment vérifier** : lancer un entraînement CP-ETR et vérifier que le fichier CSV est bien créé dans `results/` avec le bon nom.

---

### 3. `AbstractCPMode.java` créé — déduplication `ModeMS`/`ModeETR`

**Problème** : `fillTransitions()` et la contrainte `atmost` budget étaient copiées-collées à l'identique dans `ModeMS` et `ModeETR`. Toute correction devait être faite dans les deux fichiers.

**Solution** : nouvelle classe abstraite `AbstractCPMode` qui implémente :
- `getNbActions()` → commun
- `fillTransitions()` → commun, décomposé en deux méthodes privées `fillStochasticTransitions()` et `fillDeterministicTransitions()`
- `applyBudgetConstraint()` → helper protégé que les sous-classes appellent

`ModeETR` et `ModeMS` n'implémentent plus que `applyConstraints()`.

**Pour ajouter un nouveau mode CP** :
```java
public class ModeXYZ extends AbstractCPMode {
    public ModeXYZ(int budget) { super(budget); }

    @Override
    public void applyConstraints(Solver cp, IntVar[] action, IntVar totalReward, int goalReward, int holeReward) {
        applyBudgetConstraint(cp, action);  // si tu veux le budget
        // ... tes contraintes spécifiques
    }
}
```

Puis dans `FrozenLakeCPService.main()` :
```java
case "XYZ":
    currentMode = new ModeXYZ(budgetArg);
    break;
```

**Comment vérifier** :
```bash
cd MiniCPBP
mvn compile
# Doit compiler sans erreur
```

---

### 4. Restauration `initial_budget` protégée par `try/finally` (`q_learning_cp.py`)

**Problème** : pendant l'évaluation intermédiaire avec `budget_max`, si `evaluate_agent` levait une exception, `initial_budget` restait à `max_budget` et le curriculum était corrompu pour les épisodes suivants.

**Solution** : la logique de restauration est maintenant dans `CurriculumBudget.eval_with_full_budget()` qui utilise `try/finally`.

```python
def eval_with_full_budget(self, env, q_table, max_steps, eval_episodes):
    saved = self.wrapper.initial_budget
    self.wrapper.initial_budget = self.max_budget
    try:
        return utils.evaluate_agent(env, q_table, max_steps, eval_episodes)
    finally:
        self.wrapper.initial_budget = saved  # toujours restauré
```

---

### 5. Classe `CurriculumBudget` extraite (`q_learning_cp.py`)

**Problème** : la logique curriculum (progression du budget par paliers) était dispersée en 4 endroits dans la boucle d'entraînement avec des `if budget_wrapper is not None` répétés.

**Solution** : classe `CurriculumBudget` qui encapsule :
- `update(episode)` : met à jour le stage courant
- `current_stage` : property (valeur `initial_budget` courante)
- `current_episode_budget` : property (budget restant dans l'épisode en cours)
- `eval_with_full_budget(...)` : évaluation avec budget max + try/finally
- `print_schedule()` : affichage du planning curriculum

**Pour modifier la stratégie de curriculum** : modifier uniquement la méthode `update()` de `CurriculumBudget`. Par exemple pour une progression linéaire continue :
```python
def update(self, episode):
    # Progression linéaire au lieu de paliers discrets
    fraction = episode / self.total_episodes
    self.wrapper.initial_budget = int(fraction * self.max_budget)
```

---

### 6a. `CP_MS_SHAPING_COEFF` déplacé dans `config.py`

**Problème** : `cp_ms_coeff = 0.2` était une constante magique hardcodée dans `train_q_learning_with_cp_shaping`.

**Après** : dans `config.py` :
```python
CP_MS_SHAPING_COEFF = 0.2  # Coefficient appliqué à (marginal - 0.25) en mode CP-MS
```

Pour modifier le coefficient sans toucher au code métier, éditer uniquement `config.py`.

---

### 6b. Double `;;` supprimé (`FrozenLakeCPService.java`)

Cosmétique. N'affectait pas la compilation.

---

### 6c. Construction de `cmd` clarifiée (`main.py`)

**Avant** : trois branches `if/elif` sans `else`, précédées d'un `cmd = []` silencieux qui masquait le cas non couvert.

**Après** : les deux branches MS (cp_greedy et cp-ms) sont fusionnées, un `else` explicite avec message d'erreur et `return` a été ajouté, et `cmd` est assigné en une seule ligne.

---

### 6d. Seeds niveau module supprimées (`utils.py`)

**Problème** : `random.seed(config.seed_value)` et `np.random.seed(config.seed_value)` s'exécutaient à l'import de `utils`, ignorant la seed passée via `--seed` CLI.

**Après** : ces lignes sont supprimées. La seed est uniquement définie dans `main.py` après le parsing des arguments. Un commentaire explique pourquoi.

**Impact** : aucun changement de comportement pour les runs depuis `main.py` (la seed était de toute façon écrasée juste après). Si tu importes `utils` directement dans un script standalone, tu devras appeler `random.seed()` et `np.random.seed()` toi-même avant d'appeler `evaluate_agent`.

---

### 6e. `action_chars` complété dans `get_policy_grid_from_q_table` (`utils.py`)

**Problème** : si la politique greedy choisissait une action no-slip (4-7), la cellule de la grille affichait `'?'`.

**Après** :
```python
action_chars = {0: 'L', 1: 'D', 2: 'R', 3: 'U',
                4: 'L*', 5: 'D*', 6: 'R*', 7: 'U*'}  # * = no-slip
```

---

## Procédure de test recommandée

### Étape 1 — Vérifier la compilation Java

```bash
cd MiniCPBP
mvn compile
cd ..
```

Attendu : `BUILD SUCCESS`. Si erreur, vérifier que `AbstractCPMode.java` est bien dans le même package que `ModeETR.java` et `ModeMS.java`.

### Étape 2 — Test baseline (pas de CP)

```bash
python main.py --instance <ton_instance> --agent q --shaping none --episodes 200 --seed 42
```

Attendu : entraînement complet, fichier JSON créé dans `results/`, plots dans `plots/`.

### Étape 3 — Test CP-ETR sans budget

```bash
python main.py --instance <ton_instance> --agent q --shaping cp-etr --episodes 200 --seed 42
```

Attendu : serveur Java démarre en mode `ETR`, connexion établie, entraînement complet.

### Étape 4 — Test CP-ETR avec budget (mode BUDGET)

```bash
python main.py --instance <ton_instance> --agent q --shaping cp-etr --budget 3 --episodes 400 --seed 42
```

Attendu :
- Message `Curriculum budget: 4 paliers de 100 épisodes`
- Serveur Java démarre avec argument `ETR 3`
- Évaluations intermédiaires avec `[Budget max=3]` affichées correctement
- Entraînement complet sans crash même si une évaluation intermédiaire échoue

### Étape 5 — Test CP-MS

```bash
python main.py --instance <ton_instance> --agent q --shaping cp-ms --episodes 200 --seed 42
```

Attendu : serveur Java en mode `MS`, marginales utilisées pour le shaping.

### Étape 6 — Vérifier la grille de politique (mode BUDGET)

Après un entraînement avec `--budget`, vérifier dans les logs que la grille affiche `L*`, `D*`, etc. (et non `?`) pour les états où la politique préfère les actions no-slip.

---

## Fichiers non modifiés

Ces fichiers n'ont pas été touchés :
- `q_learning_baseline.py` (baselines vanilla)
- `heuristic_agents.py`
- `GridNav.java`
- `instances.json`
- `run_comparison.py`

---

## Points d'attention pour la suite

1. **`QUERY` en mode ETR** : la commande `QUERY <step> <action>` (marginales) existe côté Java mais n'est jamais appelée en mode ETR. Si tu retires le mode MS à terme, `handleQueryActionMarginal` dans `FrozenLakeCPService.java` pourra être supprimé.

2. **`evaluate_agent` et budget épuisé** : si la politique apprise choisit des actions no-slip (4-7) et que le budget est épuisé pendant l'évaluation, `FrozenLakeExtendedActions.step()` termine l'épisode immédiatement avec reward=0. Les métriques d'évaluation reflètent ce comportement — c'est voulu, mais à garder en tête lors de l'interprétation des résultats.

3. **Classe `FrozenLakeCPService` avec champs static** : fonctionne pour le mode single-client actuel. Si des tests unitaires plus poussés sont envisagés, migrer vers une instance encapsulée serait bénéfique (la méthode `resetStateForTests` montre que c'est déjà identifié).
