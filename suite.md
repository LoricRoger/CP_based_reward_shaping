# Suivi du projet — CP-based Reward Shaping

Fichier de suivi global : ce qui a été fait, ce qui reste à faire, les questions ouvertes.

---

## Légende

- [x] Fait
- [ ] À faire
- [~] En cours / partiellement fait
- [?] Question ouverte / à décider

---

## Architecture & infrastructure

- [x] Premier commit — structure de base Q-learning FrozenLake (Python)
- [x] Ajout de MiniCPBP comme solveur Java
- [x] Wrapper `FrozenLakeExtendedActions` pour les 8 actions (0–3 stochastiques, 4–7 no-slip)
- [x] Protocole TCP socket Python ↔ Java (`INIT`, `RESET`, `STEP`, `QUERY_ETR`, `QUERY`, `QUIT`)
- [x] Refactorisation Java : extraction de `AbstractCPMode`, déduplication `ModeMS`/`ModeETR`
- [x] Fix bug double reset dans `FrozenLakeExtendedActions.reset()` (`environment.py`)
- [x] Fix seeds niveau module supprimées de `utils.py` (interféraient avec `--seed` CLI)
- [x] Fix `action_chars` complété pour les actions no-slip (4–7) dans la visualisation politique
- [x] Correction argmax tie-breaking en Q-learning avec 8 actions (`q_learning_standard.py`)
- [x] `FrozenLakeCPService.java` accepte `cp_nbSteps` en override via 4e argument CLI (`args[3]`)
- [x] `main.py` supporte `--cp-nbsteps-override` (propagé au serveur Java)

---

## Modes de fonctionnement

- [x] Mode MS (marginales) : `QUERY <step> <action>` → reward shaping
- [x] Mode ETR : `QUERY_ETR` → shaping par différence ETR_after − ETR_before
- [x] Mode BUDGET : 8 actions, contrainte `atmost(actions, {4..7}, budget)` côté Java, curriculum Python
- [x] Curriculum budget — stratégie `fail` : progression par paliers, terminaison si budget épuisé
- [x] Curriculum budget — stratégie `full-budget` : budget max dès le début
- [x] `CurriculumBudget` extrait en classe dédiée avec `try/finally` pour restaurer `initial_budget`
- [x] Masquage Q-table quand budget épuisé (actions 4–7 interdites en cours d'épisode)
- [ ] Mode MS : amené à disparaître — ne pas y investir d'effort

---

## Expérimentation & scripts

- [x] `main.py` : point d'entrée unique, gère le lifecycle Java, `--verbose`, `--results-dir`
- [x] `run_comparison.py` : comparaison multi-agents (résultats figés dans git)
- [x] `run_experiment.py` : benchmark parallèle sur plusieurs instances × méthodes × seeds
    - [x] Workers parallèles pour méthodes non-Java
    - [x] Workers parallèles pour méthodes Java (pool de ports dédié)
    - [x] Pre-compilation Maven unique + `java -cp` direct par worker (pas de conflit)
    - [x] Cache JSON par run (skip auto si déjà calculé)
    - [x] Courbes d'apprentissage + tableau résumé CSV
    - [x] Lignes verticales curriculum dans les courbes
    - [x] Méthodes budget avec syntaxe `base:bN:strategy`
    - [x] Agents heuristiques : `optimal`, `cp-greedy`
- [x] `run_benchmark.py` : profiler de timing de la boucle Q-learning (`feat/benchmark`)
    - [x] 11 métriques par épisode (reset, env.step, STEP socket, ETR socket, Bellman…)
    - [x] Subdivision STEP et QUERY_ETR en send/wait (isoler calcul Java vs réseau)
    - [x] Parsing des timings Java internes via `System.nanoTime()` (BENCH_* lines)
    - [x] Boxplots, évolution temporelle, décomposition par opération
    - [x] Cache + skip automatique
    - [x] Fix tolérance aux vieux caches sans les nouvelles clés
- [x] `run_nbsteps_benchmark.py` : benchmark impact de `cp_nbSteps` (`feat/benchmark`)
    - [x] Mesure timing ET success rate pour chaque valeur de `cp_nbSteps`
    - [x] Parallélisme `ThreadPoolExecutor` + pool de ports
    - [x] Plots : timing vs nb_steps, SR vs nb_steps, double-axe tradeoff
    - [x] Cache séparé pour bench (timing) et perf (SR)

---

## Analyse de performance (résultats benchmark 4×4)

- [x] Benchmark 4s / 4medium / 4hard, 5 seeds, 10 000 épisodes (`q-none` vs `q-cp-etr`)
- [x] Résultat principal : `q-cp-etr` est ~250× plus lent que `q-none` par épisode (4×4)
- [x] Répartition : 61% fixPoint (STEP), 33% vanillaBP (ETR), 3% RESET — stable entre instances
- [x] Timings Java par appel : fixPoint ≈ 0.93 ms, vanillaBP ≈ 0.47 ms
- [x] Observation clé : fixPoint après QUERY_ETR est gratuit (0.000 ms) car STEP l'a déjà fait
- [x] Benchmark 8×8 exploratoire (500 épisodes, 3 seeds) : ~1 250–2 200 ms/ep, ×2 300–6 600 vs q-none
- [x] Coût 4×4→8×8 supra-linéaire (~×40–50 observé, vs ×8 attendu linéaire)
- [x] nb_steps benchmark : cp_nbSteps=20 optimal sur 4s/4medium, 30+ requis sur 4hard
- [x] `ANALYSE_BENCHMARK.md` mis à jour avec résultats 8×8 et nb_steps

---

## Optimisations à tester (priorité décroissante)

### A — Impact de `cp_nbSteps` (priorité haute)

- [x] Lancer `run_nbsteps_benchmark.py` sur 4s/4medium/4hard avec nb_steps = 10, 20, 30, 40, 50, 110
- [x] Identifier le seuil en dessous duquel la qualité de l'agent se dégrade
- [x] Résultat : **cp_nbSteps=20 optimal sur 4s/4medium** (×6 speedup, SR identique) ; **30 minimum sur 4hard**
- [ ] Tester l'impact de `cp_nbSteps` sur instances 8×8

### B — Déplacer `fixPoint` de STEP vers QUERY_ETR ❌ Testé — inefficace

- [x] Implémenté sur `feat/optim-no-fixpoint`, mesuré sur 3 seeds × 2000 épisodes
- [x] Résultat : **0% de gain** (4s : 47.9→46.5ms, 4medium : 26.5→29.7ms, 4hard : 43.7→46.8ms)
- [x] Explication : fixPoint sous-linéaire mais pas assez — déplacer ne supprime rien, ça concentre
- [x] Branche fermée sans merge

### C — Benchmark sur instances 8×8 (priorité haute)

- [x] Mesure exploratoire 500 épisodes : **~1 250–2 200 ms/épisode** (×2 300–6 600 vs q-none)
- [x] Coût réel 4×4→8×8 : ×40–50 (supra-linéaire, pas ×8 comme attendu linéaire)
- [ ] Campagne longue 8×8 (10 000 épisodes ≈ 4–6h/run) — après réduction cp_nbSteps

### D — Fusionner STEP + QUERY_ETR en une commande (priorité basse)

- [ ] Ajouter commande `STEP_AND_QUERY i a s_next` côté Java
- [ ] Adapter `q_learning_cp.py` côté Python
- [ ] Gain attendu : ~0.22 ms/step (2 allers-retours → 1) — utile surtout si serveur distant

---

## Qualité / tests

- [x] Suite de tests Java (`tests/`) avec `resetStateForTests()` workaround pour champs static
- [x] `CHANGES.md` : guide de migration du refactoring
- [ ] Tests Python manquants (pas de pytest en place pour les agents Python)
- [?] `FrozenLakeCPService` avec champs `static` : fonctionne pour single-client, mais fragile pour tests parallèles

---

## Questions ouvertes

- [?] Quel `cp_nbSteps` minimal préserve la qualité de l'agent sur 4×4 ? (réponse attendue après run A)
- [?] Le mode MS sera-t-il retiré ? Si oui, `handleQueryActionMarginal` Java peut être supprimé
- [?] `evaluate_agent` en mode BUDGET avec budget épuisé : les métriques reflètent le comportement voulu, mais est-ce la
  bonne métrique pour comparer les modes ?
- [?] Modifier la visualisation de Gymnasium (patch `frozen_lake.py` pour `elf_img = self.elf_images[last_action%4]`) :
  fragile car dans `.venv`, perdu si `pip install` est relancé
- [?] Comment adapter le modèle CP (actuellement Markov discret) à des espaces d'états continus ou exponentiellement grands ?
- [?] Quel agent de base pour chaque niveau : DQN ? PPO ? A2C ?
- [?] Faut-il une abstraction de l'espace d'états pour construire le modèle CP sur les niveaux 3-5 ?

---

## Roadmap — Nouveaux environnements

Objectif : valider que le RS par CP scale au-delà des Q-tables vers du deep RL (DQN/PPO).

- [x] **Niveau 1 — FrozenLake (baseline)**
  Q-table suffisante, validation que le RS CP fonctionne

- [ ] **Niveau 2 — CrossingTraffic.MDP (ippc2011/2014)**
  Même paradigme grille, trafic stochastique dynamique
  Espace d'états explose → premier vrai test DQN + RS CP

- [ ] **Niveau 3 — SysAdmin.MDP (ippc2011/2014)**
  Graphe de dépendances stochastique entre machines
  Structure non-spatiale, état = vecteur binaire variable

- [ ] **Niveau 4 — SkillTeaching.MDP (ippc2014)**
  Prérequis entre compétences = contraintes naturelles pour le CP
  Test où le RS CP devrait avoir le plus de valeur ajoutée

- [ ] **Niveau 5 — Navigation.Continuous ou Reservoir.Continuous**
  Espace continu, CP classique ne s'applique plus directement
  Test limite de l'approche, potentielle contribution originale

---

## Branche courante : `feat/benchmark`

Travaux en cours sur cette branche, pas encore mergés sur `main` :

- `run_benchmark.py` et ses résultats (`benchmark_results/`)
- `run_nbsteps_benchmark.py`
- Instrumentation Java `System.nanoTime()` dans `FrozenLakeCPService.java`
- Override `cp_nbSteps` via CLI Java + `--cp-nbsteps-override` dans `main.py`
