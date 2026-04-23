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
- [x] Résultat principal : `q-cp-etr` est ~250× plus lent que `q-none` par épisode
- [x] Répartition : 61% fixPoint (STEP), 33% vanillaBP (ETR), 3% RESET — stable entre instances
- [x] Timings Java par appel : fixPoint ≈ 0.93 ms, vanillaBP ≈ 0.47 ms
- [x] Observation clé : fixPoint après QUERY_ETR est gratuit (0.000 ms) car STEP l'a déjà fait
- [x] `ANALYSE_BENCHMARK.md` écrit dans `benchmark_results/`

---

## Optimisations à tester (priorité décroissante)

### A — Impact de `cp_nbSteps` (priorité haute)

- [ ] Lancer `run_nbsteps_benchmark.py` sur instance `4s` avec nb_steps = 10, 20, 50, 110
- [ ] Identifier le seuil en dessous duquel la qualité de l'agent se dégrade
- [ ] Déterminer le bon compromis timing/qualité (cible : 20–30 steps ?)
- [ ] Relancer sur `4medium` et `4hard` si le résultat sur `4s` est concluant

### B — Supprimer `fixPoint` dans `handleStep()` (priorité haute, modif Java)

- [ ] Retirer `cp.fixPoint()` de `handleStep()` — économie potentielle de ~61% du temps total
- [ ] Vérifier que `vanillaBP` sur graphe non-propagé reste correct (précision ETR)
- [ ] Mesurer timing + SR avant/après avec `run_benchmark.py`
- [?] Risque : vanillaBP sur graphe non-propagé peut donner de mauvaises marginales

### C — Benchmark sur instances 8×8 (priorité haute, données manquantes)

- [ ] Estimer le coût sur 8×8 avec peu d'épisodes d'abord :
  ```bash
  .venv/bin/python run_benchmark.py --instances 8s --methods q-none q-cp-etr --seeds 3 --episodes 500 --force
  ```
- [ ] Si ~400 ms/épisode confirmé, planifier une campagne longue (10 000 épisodes ≈ 67 min/run)
- [ ] Tester l'impact de `cp_nbSteps` sur 8×8 si l'optimisation A est concluante

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

---

## Branche courante : `feat/benchmark`

Travaux en cours sur cette branche, pas encore mergés sur `main` :

- `run_benchmark.py` et ses résultats (`benchmark_results/`)
- `run_nbsteps_benchmark.py`
- Instrumentation Java `System.nanoTime()` dans `FrozenLakeCPService.java`
- Override `cp_nbSteps` via CLI Java + `--cp-nbsteps-override` dans `main.py`
