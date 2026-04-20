# CLAUDE.md — Contexte du projet pour Claude Code

Ce fichier est lu automatiquement par Claude Code au démarrage d'une session.

---

## Présentation du projet

Implémentation de **Q-learning avec reward shaping basé sur la Programmation par Contraintes (CP)**, appliqué à l'environnement **FrozenLake-v1** de Gymnasium.

L'idée centrale : un solveur CP (MiniCPBP, en Java) sert d'oracle pour guider l'apprentissage — il estime la probabilité de succès (ETR) ou les marginales d'action (MS) à chaque pas, et ces valeurs servent de signal de reward shaping côté Python.

**Direction principale du projet** : mode BUDGET (ETR + actions déterministes no-slip avec contrainte de budget). Le mode MS est secondaire et peut disparaître.

---

## Architecture client-serveur

```
Python (client)                    Java (serveur)
──────────────────                 ──────────────────────────────
main.py                            FrozenLakeCPService.java
  └─ lance le serveur Java via         └─ écoute port 12345 (TCP)
     subprocess (Maven)                └─ maintient le modèle CP
                                        └─ délègue à un CPMode
q_learning_cp.py
  └─ CPRewardClient                 ModeETR.java  (principal)
       └─ socket TCP                ModeMS.java   (secondaire)
            INIT <instance_id>      AbstractCPMode.java (base commune)
            RESET
            STEP <i> <a> <s_next>
            QUERY_ETR           →   ETR_VALUE <float>
            QUERY <i> <a>       →   REWARD <float>   (MS seulement)
```

### Protocole socket

| Commande | Sens | Réponse attendue |
|----------|------|-----------------|
| `INIT <instance_id>` | Python→Java | `OK INIT successful for <id>` |
| `RESET` | Python→Java | `OK RESET successful` |
| `STEP <i> <a> <s>` | Python→Java | `OK STEP processed` |
| `QUERY_ETR` | Python→Java | `ETR_VALUE <float>` |
| `QUERY <i> <a>` | Python→Java | `REWARD <float>` (mode MS uniquement) |
| `QUIT` | Python→Java | `OK Goodbye` |

**Note** : `QUERY` (marginales) n'est **jamais** appelée en mode ETR — uniquement en mode MS.

---

## Structure des fichiers

### Python (racine)

| Fichier | Rôle |
|---------|------|
| `main.py` | Point d'entrée, parsing args, orchestration, lifecycle Java |
| `config.py` | Hyperparamètres Q-learning, constantes de shaping |
| `environment.py` | `create_environment()` + `FrozenLakeExtendedActions` (wrapper 8 actions) |
| `q_learning_cp.py` | Agent Q-learning CP (ETR, MS) + `CPRewardClient` + `CurriculumBudget` |
| `q_learning_baseline.py` | Baselines : Q-learning vanilla, agents random |
| `utils.py` | `evaluate_agent`, `save_q_table_csv`, `plot_results`, `visualize_policy` |
| `heuristic_agents.py` | Agents heuristiques : optimal policy, cp_greedy |
| `instances.json` | Configurations des instances FrozenLake (taille, trous, goal, proba CP) |
| `run_comparison.py` | Script benchmark / comparaison de plusieurs agents |

### Java (`MiniCPBP/src/main/java/minicpbp/examples/`)

| Fichier | Rôle |
|---------|------|
| `FrozenLakeCPService.java` | Serveur TCP, dispatch des commandes, état statique du modèle CP |
| `CPMode.java` | Interface : `applyConstraints`, `getNbActions`, `fillTransitions` |
| `AbstractCPMode.java` | Classe abstraite : implémente `fillTransitions` et `getNbActions`, fournit `applyBudgetConstraint()` |
| `ModeETR.java` | Mode principal — explore toutes les trajectoires, requête `QUERY_ETR` |
| `ModeMS.java` | Mode secondaire — contraint `totalReward >= goalReward`, requête `QUERY` |
| `GridNav.java` | Utilitaires de navigation sur grille (left/right/above/below) |

---

## Modes de fonctionnement

### Mode ETR (principal)
- 4 actions (0-3) : stochastiques (glissement selon `is_slippery`)
- Reward shaping : `ETR_after - ETR_before` à chaque pas
- Requête : `QUERY_ETR` → marginal de l'état goal au dernier pas de l'horizon CP

### Mode BUDGET (extension d'ETR, direction principale)
- 8 actions : 0-3 stochastiques + 4-7 déterministes (no-slip)
- Contrainte Java `atmost(actions, {4,5,6,7}, budget)`
- Curriculum côté Python : budget augmente progressivement par paliers (`CurriculumBudget`)
- Si budget épuisé pendant un épisode : actions 4-7 interdites (masquage Q-table) + terminaison immédiate si tentée

### Mode MS (secondaire)
- Contraint `totalReward >= goalReward`
- Reward shaping : `cp_ms_coeff * (marginal(action) - 0.25)`
- `CP_MS_SHAPING_COEFF = 0.2` dans `config.py`
- Amené à disparaître — ne pas y investir d'effort

---

## Paramètres CLI

```
python main.py --instance <id> --agent <q|optimal|cp_greedy> \
               --shaping <none|classic|cp-ms|cp-etr> \
               --budget <int>   # 0 = pas de budget (mode 4 actions)
               --episodes <int>
               --seed <int>
```

- `--budget 0` : mode 4 actions (ETR standard)
- `--budget N` : mode BUDGET avec N actions no-slip max par épisode

---

## Données de configuration des instances (`instances.json`)

Chaque instance contient :
```json
{
  "size": 4,
  "holes": [5, 7, 11, 12],
  "goal": 15,
  "slippery": true,
  "max_steps": 100,
  "cp_no_slip_proba": 0.333,
  "cp_nbSteps": 20,
  "desc": ["SFFF", "FHFH", "FFFH", "HFFG"]
}
```

`cp_no_slip_proba` et `cp_nbSteps` sont utilisés **uniquement côté Java** pour construire le modèle CP.

---

## Hyperparamètres (`config.py`)

```python
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.995
EPSILON = 1.0
EPSILON_DECAY = 0.99        # non utilisé directement — eps_decay recalculé dynamiquement
EPSILON_MIN = 0.01
Q_INIT_VALUE_CP_ETR_BUDGET_NOSLIP = -0.1  # init Q pour actions no-slip
CP_MS_SHAPING_COEFF = 0.2   # coefficient shaping mode MS
EVAL_EPISODES = 100
EVAL_FREQUENCY = 100
```

---

## Conventions de code importantes

### Indices d'actions
| Indice | Direction | Type |
|--------|-----------|------|
| 0 | LEFT | stochastique |
| 1 | DOWN | stochastique |
| 2 | RIGHT | stochastique |
| 3 | UP | stochastique |
| 4 | LEFT | no-slip / déterministe |
| 5 | DOWN | no-slip / déterministe |
| 6 | RIGHT | no-slip / déterministe |
| 7 | UP | no-slip / déterministe |

Ces indices doivent être **cohérents entre Java et Python**. Toute modification de l'espace d'actions doit être répercutée dans :
- `FrozenLakeExtendedActions._modify_transition_matrix()` (`environment.py`)
- `AbstractCPMode.fillDeterministicTransitions()` (`AbstractCPMode.java`)
- `ModeETR.applyConstraints()` (tableau `noSlipActions`)
- `ModeMS.applyConstraints()` (idem)

### Indices d'états
- États numérotés row-major : état `i` = ligne `i // size`, colonne `i % size`
- État 0 = départ (haut gauche)
- État `goal` = arrivée (défini dans `instances.json`)

---

## Ajouter un nouveau mode CP

### Côté Java
1. Créer `ModeXYZ.java` qui étend `AbstractCPMode`
2. Implémenter uniquement `applyConstraints()`
3. Ajouter `case "XYZ"` dans `FrozenLakeCPService.main()`

### Côté Python
1. Ajouter `'cp-xyz'` dans les choix de `--shaping` dans `main.py`
2. Ajouter la branche de lancement Java dans `main.py`
3. Gérer le calcul du reward shaping dans `train_q_learning_with_cp_shaping` (`q_learning_cp.py`)
4. Ajouter `Q_INIT_VALUE_CP_XYZ` dans `config.py` si besoin

---

## Lancement typique

```bash
# Compiler Java (une fois ou après modification Java)
cd MiniCPBP && mvn compile && cd ..

# Entraînement ETR + budget
python main.py --instance instance_4x4_slippery --agent q --shaping cp-etr --budget 3 --episodes 500 --seed 1

# Entraînement baseline (sans CP)
python main.py --instance instance_4x4_slippery --agent q --shaping none --episodes 500 --seed 1

# Comparaison multi-agents
python run_comparison.py
```

---

## Points de fragilité connus

1. **`FrozenLakeCPService` utilise des champs `static`** — fonctionne pour single-client. La méthode `resetStateForTests()` est un workaround pour les tests unitaires.

2. **`QUERY` (marginales) vs `QUERY_ETR`** — en mode ETR, seul `QUERY_ETR` est utilisé. Si MS disparaît, `handleQueryActionMarginal` côté Java peut être retiré.

3. **`evaluate_agent` en mode BUDGET** — si la politique greedy choisit des actions 4-7 avec budget épuisé, l'épisode se termine immédiatement (reward=0). Les métriques reflètent ce comportement.

4. **Démarrage Java** — délai fixe de 10 secondes (`time.sleep(10)` dans `main.py`). Sur des machines lentes, augmenter si le serveur n'est pas prêt à temps.
