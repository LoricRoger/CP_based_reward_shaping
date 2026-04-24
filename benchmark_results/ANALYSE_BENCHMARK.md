# Analyse de performance — CP reward shaping

Date de mise à jour : 2026-04-24
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

- Instances 4×4 : **4s, 4medium, 4hard** (`cp_nbSteps=110`, `max_steps=100`)
  - Méthodes : `q-none`, `q-cp-etr`, `q-classic`
  - Épisodes : 10 000 par run — Seeds : **5–40 seeds**
- Instances 8×8 : **8s, 8medium, 8hard** (`cp_nbSteps=220`, `max_steps=200`)
  - Méthodes : `q-none`, `q-cp-etr`, `q-classic`
  - Épisodes : 500 par run (estimation coût) — Seeds : **3 seeds**

---

## 2. Résultats

### Temps total par épisode (ms)

**Instances 4×4** (`cp_nbSteps=110`, 10 000 épisodes, 5 seeds)

| Instance | q-none | q-cp-etr | Facteur |
|----------|--------|----------|---------|
| 4s | 0.19 | **47.9** | ×252 |
| 4medium | 0.10 | **26.5** | ×268 |
| 4hard | 0.18 | **43.7** | ×249 |

Sur 10 000 épisodes : ~8 min (4s/4hard) ou ~4 min (4medium) vs ~2 s.

**Instances 8×8** (`cp_nbSteps=220`, 500 épisodes, 3 seeds)

| Instance | q-none | q-cp-etr | Facteur | Total 10 000 ep (estimé) |
|----------|--------|----------|---------|--------------------------|
| 8s | 0.76 | **~2 183** | ×2 876 | **~6 h** |
| 8medium | 0.23 | **~1 525** | ×6 623 | **~4 h 15** |
| 8hard | 0.53 | **~1 251** | ×2 347 | **~3 h 30** |

Le passage 4×4 → 8×8 multiplie le temps Java par ~40–50× (graphe CP ×8 mais complexité
supra-linéaire confirmée : fixPoint et vanillaBP semblent superlinéaires en taille du graphe).

La différence entre instances s'explique par le nombre moyen de steps/épisode :
4medium/8medium convergent plus vite (épisodes plus courts → moins d'appels Java).

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

**Instance 8s** (500 épisodes, 3 seeds — décomposition seed 1)
```
Opération                    q-none      q-cp-etr
RESET socket             0.003±0.000   33.92±6.6    (1.5%)
QUERY_ETR initial        0.000±0.000    9.69±2.5    (0.4%)
env.step()               0.098±0.001    1.47±1.1    (0.1%)
STEP total               0.000±0.000 1719.03±947   (75.9%)
  STEP ← wait fixPoint   0.000±0.000 1718.28±947   (75.9%)
QUERY_ETR total          0.000±0.000  497.55±279   (22.0%)
  ETR ← wait vanillaBP   0.000±0.000  496.95±279   (21.9%)
Bellman update           0.056±0.001    1.49±1.1    (0.1%)
Total épisode            0.759±0.013 2264.1±...
```

La répartition fixPoint/vanillaBP reste quasi identique à 4×4 (~76%/~22%) — le ratio
est structurel, indépendant de la taille. Seuls les valeurs absolues explosent.

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

### Résultats nb_steps : tradeoff timing / success rate

Benchmark `run_nbsteps_benchmark.py` : timing sur 2 000 épisodes (4×4) / 500 épisodes (8×8), SR final sur 10 000 épisodes (4×4) / 5 000 épisodes (8×8).

**Instance 4s** (réf. 110 steps = 45.3 ms, SR=73.5%)

| cp_nbSteps | Timing (ms/ep) | Speedup vs 110 | SR final (%) |
|-----------|---------------|----------------|-------------|
| 10 | 7.4 ± 2.1 | **×6.1** | 71.6 ± 6.6 |
| 20 | 7.4 ± 1.6 | **×6.1** | 73.5 ± 4.9 |
| 30 | 11.9 ± 0.3 | ×3.8 | 67.9 ± 11.8 |
| 40 | 15.6 ± 0.7 | ×2.9 | 73.5 ± 4.2 |
| 50 | 17.7 ± 0.5 | ×2.6 | 73.1 ± 4.5 |
| 110 | 45.3 ± 0.8 | ×1.0 | 73.5 ± 5.5 |

**Instance 4medium** (réf. 110 steps = 28.9 ms, SR=38.7%)

| cp_nbSteps | Timing (ms/ep) | Speedup | SR final (%) |
|-----------|---------------|---------|-------------|
| 10 | 4.5 ± 0.2 | **×6.5** | 29.1 ± 6.9 |
| 20 | 5.4 ± 0.2 | **×5.3** | 34.1 ± 9.8 |
| 30 | 8.3 ± 0.6 | ×3.5 | 36.7 ± 6.8 |
| 40 | 10.2 ± 0.3 | ×2.8 | 36.7 ± 3.4 |
| 50 | 12.9 ± 0.2 | ×2.2 | 38.9 ± 3.7 |
| 110 | 28.9 ± 0.7 | ×1.0 | 38.7 ± 5.2 |

**Instance 4hard** (réf. 110 steps = 46.1 ms, SR=36.5%)

| cp_nbSteps | Timing (ms/ep) | Speedup | SR final (%) |
|-----------|---------------|---------|-------------|
| 10 | 5.4 ± 0.4 | **×8.5** | 29.4 ± 3.1 |
| 20 | 7.4 ± 0.3 | **×6.3** | 28.7 ± 3.4 |
| 30 | 10.7 ± 0.3 | ×4.3 | 27.9 ± 6.3 |
| 40 | 16.2 ± 1.2 | ×2.8 | 30.2 ± 5.8 |
| 50 | 18.8 ± 0.5 | ×2.5 | 31.4 ± 5.7 |
| 110 | 46.1 ± 1.0 | ×1.0 | 36.5 ± 4.8 |

**Instance 8s** (réf. 220 steps = 2 377 ms, SR=87.8%)

| cp_nbSteps | Timing (ms/ep) | Speedup vs 220 | SR final (%) |
|-----------|---------------|----------------|-------------|
| 20 | 44.8 ± 2.9 | **×53** | 26.2 ± 29.2 |
| 40 | 129.1 ± 6.5 | **×18** | 86.2 ± 6.0 |
| 60 | 277.7 ± 6.6 | ×8.6 | 85.2 ± 13.2 |
| 80 | 469.3 ± 13.7 | ×5.1 | 87.8 ± 2.6 |
| 110 | 795.2 ± 43.3 | ×3.0 | 87.8 ± 3.9 |
| 220 | 2 377 ± 340 | ×1.0 | — |

**Instance 8medium** (réf. 220 steps = 1 849 ms)

| cp_nbSteps | Timing (ms/ep) | Speedup vs 220 | SR final (%) |
|-----------|---------------|----------------|-------------|
| 20 | 27.4 ± 1.4 | **×67** | 16.8 ± 12.7 |
| 40 | 120.0 ± 12.1 | **×15** | 60.2 ± 5.7 |
| 60 | 238.8 ± 11.5 | ×7.7 | 82.6 ± 4.5 |
| 80 | 352.4 ± 25.1 | ×5.3 | 80.6 ± 7.1 |
| 110 | 627.3 ± 64.3 | ×2.9 | — |
| 220 | 1 849 ± 144 | ×1.0 | — |

**Instance 8hard** (réf. 220 steps = 1 311 ms)

| cp_nbSteps | Timing (ms/ep) | Speedup vs 220 | SR final (%) |
|-----------|---------------|----------------|-------------|
| 20 | 32.3 ± 2.1 | **×41** | 12.4 ± 3.9 |
| 40 | 115.6 ± 4.5 | **×11** | 58.6 ± 11.6 |
| 60 | 221.4 ± 12.3 | ×5.9 | 63.4 ± 7.9 |
| 80 | 373.1 ± 10.9 | ×3.5 | 64.8 ± 5.6 |
| 110 | 532.8 ± 31.6 | ×2.5 | — |
| 220 | 1 311 ± 97 | ×1.0 | — |

**Conclusions nb_steps :**

- Sur **4s et 4medium** : `cp_nbSteps=20` donne ~×6 de speedup **sans perte de SR**
  (SR identique ou dans la variance de 110). Recommandation : **utiliser 20 par défaut**.
- Sur **4hard** : la SR chute significativement sous 30 steps (29% vs 36.5% à 110).
  L'instance difficile nécessite un horizon plus long. Recommandation : **30–40 minimum**.
- Sur **8s** : `cp_nbSteps=40` donne ×18 de speedup avec SR préservée (86.2% vs 87.8% à 220).
  `cp_nbSteps=20` effondre la SR (26%). Recommandation : **40 minimum**.
- Sur **8medium** : `cp_nbSteps=60` est le seuil critique — 82.6% de SR vs 16.8% à 20 et 60.2% à 40.
  Recommandation : **60 minimum** (×7.7 de speedup).
- Sur **8hard** : SR plafonne à ~64% dès cp_nbSteps=60 (vs 58.6% à 40). Recommandation : **60**.
- Le timing n'est pas parfaitement linéaire en nb_steps (saut entre 20→30 sur 4s/4hard),
  ce qui suggère un overhead fixe par épisode indépendant du nombre de steps.
- La variance SR en 8×8 est nettement plus élevée qu'en 4×4 (std jusqu'à 29% sur 8s/20 steps),
  conséquence d'un nombre de seeds limité (5) et d'une sensibilité accrue à l'initialisation.

---

### Observations clés

1. **94–99% du temps est du calcul Java pur.** Le réseau localhost est négligeable
   (send aller : <1 ms total). Python ne peut pas être optimisé davantage.

2. **La répartition est stable entre instances ET entre tailles** : ~76% fixPoint STEP,
   ~22% vanillaBP ETR, ~2% RESET en 8×8 (vs ~61%/33%/3% en 4×4). La différence de ratio
   s'explique par l'allongement des épisodes en 8×8 (plus de steps → plus de STEP relatifs).

3. **Le fixPoint après QUERY_ETR est gratuit (0.000 ms).** Les domaines CP sont déjà
   propagés par le fixPoint du STEP précédent. Cela signifie que le fixPoint de STEP
   fait "doublement" le travail — c'est un levier d'optimisation direct.

4. **`cp_nbSteps` est le paramètre de contrôle principal.** Graphe CP en 4×4 : `110 × 16 = 1 760`
   nœuds. En 8×8 : `220 × 64 = 14 080` nœuds (×8). Le coût est supra-linéaire (~×40–50 observé).

5. **Les std sont faibles en 4×4** (±1-2% sur le total épisode). En 8×8, la std est plus
   élevée (~5-18%) car les épisodes courts/longs sont plus hétérogènes avec 500 eps seulement.

---

## 3. Ce qu'il faudrait tester ensuite (par priorité)

### A — Impact de `cp_nbSteps` ✅ FAIT (4×4 + 8×8)

Résultats disponibles dans `nbsteps_results/`. Voir tableau complet ci-dessus.

**Recommandations issues du benchmark :**
- 4×4 : `cp_nbSteps=20` sur **4s et 4medium** (×6 speedup, SR préservé) ; **30–40 minimum** sur **4hard**
- 8×8 : **40 minimum** sur **8s** ; **60 minimum** sur **8medium et 8hard**
- La structure tient : seuil minimal différent selon la difficulté de l'instance, mais identifiable clairement

### B — Déplacer fixPoint de STEP vers QUERY_ETR ❌ TESTÉ — INEFFICACE

**Hypothèse initiale** : supprimer `cp.fixPoint()` dans `handleStep()` et le déplacer
en tête de `handleQueryETR()` pour éviter les propagations inutiles entre STEP et ETR.
Gain espéré : ~76% du temps total.

**Résultats observés** (branche `feat/optim-no-fixpoint`, 3 seeds × 2000 épisodes) :

| Instance | Avant (ms/ep) | Après (ms/ep) | Δ |
|----------|--------------|--------------|---|
| 4s | 47.9 | **46.5** | -3% |
| 4medium | 26.5 | **29.7** | +12% |
| 4hard | 43.7 | **46.8** | +7% |

Timings Java internes après modification :

```
[STEP]   assign    : 0.0005 ms   (fixPoint supprimé ✓)
[ETR]    fixPoint  : ~0.96 ms    (fait le vrai travail, comme STEP avant)
         vanillaBP : ~0.50 ms    (stable)
```

**Explication** : fixPoint est sous-linéaire en nombre d'assigns accumulés, mais pas
assez pour que "1 gros fixPoint par épisode" soit moins cher que "N petits fixPoints
par step". Le graphe CP est à propagation globale — chaque assign force de toute façon
une repropagation complète au prochain fixPoint. Déplacer le fixPoint ne l'évite pas,
ça le concentre juste ailleurs.

**Conclusion : l'optimisation B n'apporte pas de gain. Branche fermée sans merge.**

### C — Mesurer sur instances 8×8 ✅ FAIT (500 épisodes bench, 5 000 perf)

Résultats : ~1 250–2 400 ms/épisode à cp_nbSteps=220 selon l'instance (×2 300–×6 600 vs q-none).
Le coût réel 4×4→8×8 est ×40–50 (supra-linéaire, pas ×8 comme attendu).

Avec `cp_nbSteps=40–60`, le timing tombe à **115–280 ms/épisode** sur 8×8, soit
~10–25 min pour 5 000 épisodes — un budget temps raisonnable pour des runs complets.

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
│                          # contient 4s/4medium/4hard (10k eps) + 8s/8medium/8hard (500 eps)
├── java_logs/             # stdout/stderr Java par run (contient les BENCH_* lines)
└── plots/
    ├── boxplot_ops.png          # distribution de chaque opération par méthode
    ├── boxplot_by_instance.png  # temps total par instance
    ├── time_evolution_*.png     # évolution temporelle (moyenne glissante)
    └── op_breakdown_*.png       # fraction du temps par opération

nbsteps_results/
├── cache/                 # bench + perf pour chaque (inst, steps, seed)
│                          # 4s/4medium/4hard × steps={10,20,30,40,50,110} (bench 2k, perf 10k)
│                          # 8s/8medium/8hard × steps={20,40,60,80,110,220} (bench 500, perf 5k)
└── plots/
    ├── timing_vs_nbsteps_*.png  # timing moyen par cp_nbSteps
    ├── sr_vs_nbsteps_*.png      # success rate final par cp_nbSteps
    └── tradeoff_*.png           # double-axe timing + SR
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
