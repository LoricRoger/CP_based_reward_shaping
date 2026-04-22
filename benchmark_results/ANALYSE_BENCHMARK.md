# Analyse de performance — CP reward shaping

Date : 2026-04-22  
Branche : `feat/benchmark`  
Outil : `run_benchmark.py`

---

## 1. Ce qu'on a fait

### Outil de benchmark (`run_benchmark.py`)

Un script standalone qui reproduit la boucle Q-learning instrumentée avec `time.perf_counter()`,
sans modifier le code existant. Il mesure par épisode :

| Métrique Python | Ce que ça couvre |
|-----------------|-----------------|
| `reset_s` | `env.reset()` + `RESET` socket |
| `initial_etr_s` | Premier `QUERY_ETR` en début d'épisode |
| `env_step_s` | Cumul de tous les `env.step()` |
| `cp_step_s` | Cumul des `STEP` socket (total) |
| `cp_step_send_s` | — dont : transit aller `sendall()` |
| `cp_step_wait_s` | — dont : attente réponse Java (`fixPoint`) |
| `cp_query_etr_s` | Cumul des `QUERY_ETR` socket (total) |
| `cp_etr_send_s` | — dont : transit aller `sendall()` |
| `cp_etr_wait_s` | — dont : attente réponse Java (`vanillaBP`) |
| `bellman_s` | Mise à jour Q-table |
| `episode_total_s` | Durée totale de l'épisode |

Côté Java, des `System.nanoTime()` ont été ajoutés dans `FrozenLakeCPService.java`
pour subdiviser chaque commande en ses opérations internes :

| Commande | Sous-opérations mesurées |
|----------|--------------------------|
| `RESET` | `makeSolver`, `makeVars`, `postConstraints`, `fixPoint` |
| `STEP` | `assign`, `fixPoint` |
| `QUERY_ETR` | `vanillaBP`, `fixPoint`, `marginal` |

Le log Java est parsé automatiquement après chaque run CP et agrégé dans les résultats.

### Données collectées

- Instance : **4s** (4×4, slippery, `cp_nbSteps=110`)
- Méthodes : `q-none`, `q-classic`, `q-cp-etr`
- Épisodes : 10 000
- Seeds : **2 seeds pour q-cp-etr**, 40 seeds pour q-none et q-classic

---

## 2. Ce qu'on a trouvé

### Temps total par épisode

| Méthode | ms/épisode | Facteur vs q-none |
|---------|-----------|-------------------|
| q-none | 0.19 ms | ×1 (référence) |
| q-classic | 0.24 ms | ×1.3 |
| q-cp-etr | **48.3 ms** | **×250** |

q-cp-etr est 250× plus lent par épisode que le Q-learning standard.
Sur 10 000 épisodes : ~8 minutes vs ~2 secondes.

### Décomposition du temps pour q-cp-etr

| Opération | ms/épisode | % |
|-----------|-----------|---|
| STEP ← wait Java (`fixPoint`) | 29.3 ms | **60.7%** |
| ETR ← wait Java (`vanillaBP`) | 16.1 ms | **33.3%** |
| RESET socket | 1.5 ms | 3.1% |
| ETR initial | 0.7 ms | 1.5% |
| Réseau (send aller) | 0.2 ms | 0.5% |
| env.step() Python | 0.2 ms | 0.4% |
| Bellman update Python | 0.1 ms | 0.2% |

**94% du temps est du calcul Java pur. Python et le réseau sont négligeables.**

### Détail interne Java (par appel, pas par épisode)

**STEP** (~29 appels/épisode en moyenne) :
```
assign    : 0.0003 ms  →  négligeable
fixPoint  : 0.926 ms   →  100% du coût STEP
```

**QUERY_ETR** (~30 appels/épisode en moyenne) :
```
vanillaBP : 0.463 ms   →  100% du coût ETR
fixPoint  : 0.000 ms   →  gratuit (domaines déjà propagés par STEP précédent)
marginal  : 0.000 ms   →  gratuit
```

**RESET** (~1 appel/épisode) :
```
postConstraints : 0.71 ms   →  construction du modèle CP
fixPoint        : 0.62 ms   →  propagation initiale
makeSolver/Vars : ~0.06 ms  →  négligeable
```

### Observations clés

1. **Le `fixPoint` après `STEP` est le goulot principal.** Il est appelé à chaque step
   (~29 fois/épisode) et coûte ~0.93 ms à chaque fois → ~27 ms/épisode à lui seul.

2. **Le `fixPoint` après `vanillaBP` (dans QUERY_ETR) est gratuit.** Les domaines sont
   déjà stables depuis le `fixPoint` du STEP. Cela signifie que `fixPoint` dans STEP
   fait "doublement" le travail pour ETR.

3. **`vanillaBP` coûte 0.46 ms par appel.** Avec `BP_ITERATIONS=1` (déjà au minimum),
   il n'y a pas de levier direct dessus — sauf réduire la taille du graphe.

4. **`cp_nbSteps=110`** sur les instances 4×4. C'est l'horizon CP, soit le nombre de
   pas de temps dans le modèle de contraintes. Le graphe CP a `110 × 16 = 1 760` nœuds
   pour une instance 4×4. En 8×8 avec `cp_nbSteps=220` : `220 × 64 = 14 080` nœuds,
   soit **8× plus grand**. `fixPoint` et `vanillaBP` étant au moins linéaires en la
   taille du graphe, on peut s'attendre à un facteur ×8 ou plus sur le temps Java.

---

## 3. Limites des données actuelles

- **2 seeds seulement pour q-cp-etr** (les runs sont ~8 min chacun).
  Les std sont faibles (±0.1 ms sur le total), mais ce n'est pas suffisant pour
  affirmer avec confiance que les timings sont stables sur d'autres configurations.

- **Une seule instance testée (4s).** Les autres instances 4×4 ont la même structure
  (`cp_nbSteps=110`) mais des trous différents. Les instances 4medium/4hard pourraient
  avoir des comportements différents si les épisodes sont plus courts en moyenne
  (moins de STEP/ETR par épisode).

- **Pas de données sur 8×8.** C'est là où le problème va réellement se manifester.

### Ce qu'il faudrait faire pour avoir des données fiables

```bash
# Minimum recommandé : 5 seeds, instances 4x4 variées
python run_benchmark.py \
  --instances 4s 4medium 4hard \
  --methods q-none q-cp-etr \
  --seeds 5 \
  --episodes 10000

# Une fois les optimisations faites : tester 8x8
python run_benchmark.py \
  --instances 8s 8medium \
  --methods q-none q-cp-etr \
  --seeds 3 \
  --episodes 2000  # moins d'épisodes car beaucoup plus lent
```

---

## 4. Ce qu'il faudrait faire (par ordre de priorité)

### Priorité 1 — Réduire `cp_nbSteps` (impact immédiat, facile)

`cp_nbSteps=110` sur du 4×4 avec `max_steps=100` est probablement trop grand.
Un horizon CP plus court que la trajectoire max réelle n'est pas forcément pire
en termes de qualité de l'ETR — à vérifier empiriquement.

**Action** : faire un mini-benchmark en faisant varier `cp_nbSteps` dans `instances.json` :

| cp_nbSteps | Gain attendu sur fixPoint |
|-----------|--------------------------|
| 110 (actuel) | référence |
| 50 | ~×2 plus rapide |
| 20 | ~×5 plus rapide |
| 10 | ~×10 plus rapide |

Comparer aussi la qualité de l'agent (success rate) pour trouver le bon compromis.

### Priorité 2 — Supprimer le `fixPoint` après `STEP`, le différer au QUERY_ETR

Actuellement : `STEP` fait `assign + fixPoint`, puis `QUERY_ETR` fait `vanillaBP + fixPoint`.
Le second `fixPoint` est gratuit (0.000 ms) car le premier a déjà tout propagé.

Proposition : ne faire `assign` sans `fixPoint` dans `STEP`, et laisser `vanillaBP`
travailler sur le graphe non-propagé. Le `fixPoint` de QUERY_ETR redeviendrait non-nul
mais on économiserait le `fixPoint` de STEP (~60% du temps total).

**Risque** : `vanillaBP` sur un graphe non-propagé pourrait être plus lent ou moins précis.
À tester côté Java avec le benchmark.

### Priorité 3 — Fusionner STEP et QUERY_ETR en une seule commande

Actuellement, chaque step fait 2 allers-retours socket : `STEP` puis `QUERY_ETR`.
On pourrait ajouter une commande `STEP_AND_QUERY i a s_next` côté Java qui fait les
deux en un seul aller-retour, économisant ~0.25 ms de latence réseau par step.
Faible gain absolu sur localhost (~0.5% du temps), mais utile si le projet tourne
sur une machine distante.

### Priorité 4 — Tester sur 8×8 avant d'optimiser davantage

Les timings 4×4 donnent une baseline. Avant d'investir dans des optimisations
algorithimiques, mesurer les timings 8×8 réels pour confirmer le facteur de
ralentissement et prioriser les bons leviers.

---

## 5. Structure du benchmark pour référence

```
benchmark_results/
├── cache/          # un JSON par (instance, method, seed, episodes) — skip auto
├── java_logs/      # stdout/stderr Java par run — contient les BENCH_* lines
└── plots/
    ├── boxplot_ops.png          # distribution de chaque opération par méthode
    ├── boxplot_by_instance.png  # temps total par instance
    ├── time_evolution_*.png     # évolution temporelle (moyenne glissante)
    └── op_breakdown_*.png       # stacked bar fraction du temps par opération
```

Commandes utiles :
```bash
# Relancer uniquement les runs manquants (cache automatique)
python run_benchmark.py --instances 4s 4medium --methods q-cp-etr --seeds 5 --episodes 10000

# Régénérer les plots sans relancer
python run_benchmark.py --plots-only --instances 4s 4medium --methods q-none q-classic q-cp-etr --seeds 5 --episodes 10000

# Forcer le recalcul (après modification du code)
python run_benchmark.py --force ...
```
