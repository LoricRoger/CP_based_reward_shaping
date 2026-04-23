# Analyse de performance — CP reward shaping

Date de mise à jour : 2026-04-22
Branche : `feat/benchmark`
Outil : `run_benchmark.py`

---

## 1. Ce qu'on a fait

### Outil de benchmark (`run_benchmark.py`)

Script standalone qui reproduit la boucle Q-learning instrumentée avec `time.perf_counter()`
sans modifier le code existant. Il mesure par épisode :

| Métrique Python | Ce que ça couvre |
|-----------------|-----------------|
| `reset_s` | `env.reset()` + `RESET` socket |
| `initial_etr_s` | Premier `QUERY_ETR` en début d'épisode |
| `env_step_s` | Cumul de tous les `env.step()` |
| `cp_step_s` | Cumul des `STEP` socket — total |
| `cp_step_send_s` | — dont : transit aller `sendall()` |
| `cp_step_wait_s` | — dont : attente réponse Java (`fixPoint`) |
| `cp_query_etr_s` | Cumul des `QUERY_ETR` socket — total |
| `cp_etr_send_s` | — dont : transit aller `sendall()` |
| `cp_etr_wait_s` | — dont : attente réponse Java (`vanillaBP`) |
| `bellman_s` | Mise à jour Q-table |
| `episode_total_s` | Durée totale de l'épisode |

Côté Java, des `System.nanoTime()` ont été ajoutés dans `FrozenLakeCPService.java` :

| Commande | Sous-opérations mesurées |
|----------|--------------------------|
| `RESET` | `makeSolver`, `makeVars`, `postConstraints`, `fixPoint` |
| `STEP` | `assign`, `fixPoint` |
| `QUERY_ETR` | `vanillaBP`, `fixPoint`, `marginal` |

Le log Java est parsé automatiquement après chaque run et affiché dans le terminal.

### Données collectées

- Instances : **4s, 4medium, 4hard** (toutes 4×4, `cp_nbSteps=110`, `max_steps=100`)
- Méthodes : `q-none`, `q-cp-etr`
- Épisodes : 10 000 par run
- Seeds : **5 seeds** par (instance × méthode)

---

## 2. Résultats

### Temps total par épisode (ms)

| Instance | q-none | q-cp-etr | Facteur |
|----------|--------|----------|---------|
| 4s | 0.19 | **47.9** | ×252 |
| 4medium | 0.10 | **26.5** | ×268 |
| 4hard | 0.18 | **43.7** | ×249 |

q-cp-etr est ~250× plus lent par épisode.
Sur 10 000 épisodes : ~8 min (4s/4hard) ou ~4 min (4medium) vs ~2 s.

La différence entre instances s'explique par le nombre moyen de steps/épisode :
4medium converge plus vite (épisodes plus courts → moins d'appels Java).

### Tableau complet par opération (ms/épisode, moyenne ± std sur 5 seeds)

**Instance 4s**
```
Opération                    q-none      q-cp-etr
RESET socket             0.003±0.000   1.467±0.024   (3.1%)
QUERY_ETR initial        0.000±0.000   0.732±0.011   (1.5%)
env.step()               0.098±0.001   0.161±0.013   (0.3%)
STEP total               0.000±0.000  29.266±0.323  (61.1%)
  STEP → send            0.000±0.000   0.145±0.008   (0.3%)
  STEP ← wait fixPoint   0.000±0.000  29.121±0.315  (60.8%)
QUERY_ETR total          0.000±0.000  16.065±0.140  (33.5%)
  ETR → send             0.000±0.000   0.080±0.005   (0.2%)
  ETR ← wait vanillaBP   0.000±0.000  15.985±0.141  (33.4%)
Bellman update           0.056±0.001   0.116±0.011   (0.2%)
Total épisode            0.192±0.002  47.908±0.379
```

**Instance 4medium**
```
STEP ← wait fixPoint     0.000±0.000  15.521±0.256  (58.7%)
ETR ← wait vanillaBP     0.000±0.000   8.543±0.130  (32.3%)
Total épisode            0.099±0.002  26.456±0.398
```

**Instance 4hard**
```
STEP ← wait fixPoint     0.000±0.000  26.450±0.458  (60.5%)
ETR ← wait vanillaBP     0.000±0.000  14.623±0.237  (33.5%)
Total épisode            0.176±0.004  43.703±0.725
```

### Timings Java internes (par appel, stables sur toutes les instances)

**STEP** (~29 appels/épisode sur 4s, ~15 sur 4medium) :
```
assign    : 0.0002 ms  →  négligeable
fixPoint  : 0.90–1.00 ms  →  100% du coût STEP
```

**QUERY_ETR** (~30 appels/épisode sur 4s, ~16 sur 4medium) :
```
vanillaBP : 0.44–0.50 ms  →  100% du coût ETR
fixPoint  : 0.000 ms  →  gratuit (domaines déjà propagés par STEP)
marginal  : 0.000 ms  →  gratuit
```

**RESET** (1 appel/épisode) :
```
postConstraints : 0.65–0.71 ms  →  construction du modèle CP
fixPoint        : 0.57–0.62 ms  →  propagation initiale
makeSolver/Vars : ~0.05 ms  →  négligeable
```

### Observations clés

1. **94% du temps est du calcul Java pur.** Le réseau localhost est négligeable
   (send aller : 0.2 ms total). Python ne peut pas être optimisé davantage.

2. **La répartition est stable entre instances** : ~61% fixPoint STEP, ~33% vanillaBP ETR,
   ~3% RESET. Ce ratio ne dépend pas de la difficulté de l'instance mais de la longueur
   moyenne des épisodes.

3. **Le fixPoint après QUERY_ETR est gratuit (0.000 ms).** Les domaines CP sont déjà
   propagés par le fixPoint du STEP précédent. Cela signifie que le fixPoint de STEP
   fait "doublement" le travail — c'est un levier d'optimisation direct.

4. **`cp_nbSteps=110` pour des épisodes de max 100 steps.** Le graphe CP a
   `110 × 16 = 1 760` nœuds en 4×4. C'est le paramètre principal qui dimensionne
   le coût de fixPoint et vanillaBP.

5. **Les std sont faibles** (±1% sur le total épisode) — les timings sont très stables.
   5 seeds est suffisant pour confirmer les ordres de grandeur.

---

## 3. Ce qu'il faudrait tester ensuite (par priorité)

### A — Impact de `cp_nbSteps` (priorité haute, facile, fort impact)

C'est le paramètre le plus impactant : fixPoint et vanillaBP sont linéaires en la
taille du graphe CP (`nbSteps × nbStates`).

**Protocole** : modifier `cp_nbSteps` dans `instances.json` pour 4s, mesurer
timing ET success rate (via `run_experiment.py`) pour chaque valeur :

| cp_nbSteps | Gain timing attendu | À tester |
|-----------|--------------------|----|
| 110 (actuel) | référence | déjà mesuré |
| 50 | ~×2.2 | oui |
| 20 | ~×5.5 | oui |
| 10 | ~×11 | oui |

Questions : en dessous de quel `cp_nbSteps` la qualité de l'agent se dégrade ?
Y a-t-il un bon compromis autour de 20–30 steps ?

```bash
# Modifier cp_nbSteps dans instances.json, puis :
python run_benchmark.py --instances 4s --methods q-cp-etr --seeds 3 --episodes 2000 --force
python run_experiment.py --instances 4s --methods q-cp-etr --seeds 5 --episodes 5000 --force
```

### B — Supprimer le fixPoint dans STEP, le différer au QUERY_ETR (priorité haute, modif Java)

Observation : le fixPoint après QUERY_ETR est actuellement gratuit parce que le STEP
précédent a déjà propagé. Si on supprime le `cp.fixPoint()` dans `handleStep()` et
qu'on laisse `vanillaBP` travailler sur le graphe non-encore-propagé, on économise
~61% du temps total.

**Risque** : vanillaBP sur graphe non-propagé peut être plus lent ou moins précis.
À mesurer avec le benchmark après la modif Java.

```java
// handleStep : retirer cp.fixPoint() après assign
action[i].assign(a);
state[i].assign(sN);
// cp.fixPoint();  ← supprimer
```

### C — Mesurer sur instances 8×8 (priorité haute, données manquantes)

En 8×8 : `cp_nbSteps=220`, `nbStates=64` → graphe `220 × 64 = 14 080` nœuds
vs `110 × 16 = 1 760` en 4×4 (×8 plus grand).

fixPoint et vanillaBP étant au moins linéaires, on attend ×8 sur les temps Java,
soit ~400 ms/épisode → 10 000 épisodes ≈ **67 minutes** par run.

```bash
python run_benchmark.py --instances 8s --methods q-none q-cp-etr --seeds 3 --episodes 500 --force
```

Commencer avec peu d'épisodes pour estimer le coût avant de lancer une longue campagne.

### D — Fusionner STEP + QUERY_ETR en une commande (priorité basse, modif Java + Python)

Actuellement : 2 allers-retours socket par step (`STEP` puis `QUERY_ETR`).
Une commande `STEP_AND_QUERY i a s_next` ferait les deux en un seul échange,
économisant ~0.22 ms/step (send aller × 2). Gain relatif faible (~0.5%) en localhost,
mais utile si le serveur Java tourne sur une machine distante.

---

## 4. Structure du benchmark

```
benchmark_results/
├── ANALYSE_BENCHMARK.md   # ce fichier
├── cache/                 # un JSON par (instance, method, seed, episodes) — skip auto
├── java_logs/             # stdout/stderr Java par run (contient les BENCH_* lines)
└── plots/
    ├── boxplot_ops.png          # distribution de chaque opération par méthode
    ├── boxplot_by_instance.png  # temps total par instance
    ├── time_evolution_*.png     # évolution temporelle (moyenne glissante)
    └── op_breakdown_*.png       # fraction du temps par opération
```

### Commandes utiles

```bash
# Lancer les runs manquants (cache automatique)
python run_benchmark.py --instances 4s 4medium 4hard --methods q-none q-cp-etr --seeds 5 --episodes 10000

# Régénérer les plots sans relancer
python run_benchmark.py --plots-only --instances 4s 4medium 4hard --methods q-none q-cp-etr --seeds 5 --episodes 10000

# Forcer le recalcul après modification du code
python run_benchmark.py --force --instances 4s --methods q-cp-etr --seeds 3 --episodes 2000
```
