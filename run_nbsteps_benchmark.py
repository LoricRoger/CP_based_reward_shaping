"""
run_nbsteps_benchmark.py — Benchmark de l'impact de cp_nbSteps sur le timing et les performances.

Pour chaque valeur de cp_nbSteps, mesure :
  - Timing moyen par épisode (ms) et par opération clé (STEP, ETR, RESET)
  - Taux de succès final après entraînement (success rate)

Exécution en parallèle via ThreadPoolExecutor + pool de ports (même pattern que run_experiment.py).
Java accepte cp_nbSteps en 4e argument CLI (ajout dans FrozenLakeCPService.java).

Usage:
    python run_nbsteps_benchmark.py [options]

Options:
    --instances ID...     Instance IDs à tester          (default: 4s)
    --nb-steps N...       Valeurs de cp_nbSteps à tester (default: 10 20 50 110)
    --seeds N             Nombre de seeds, à partir de 1 (default: 3)
    --bench-episodes N    Épisodes pour le timing         (default: 2000)
    --perf-episodes N     Épisodes pour le success rate   (default: 5000)
    --workers N           Workers parallèles              (default: 1)
    --base-port P         Premier port TCP Java           (default: 12345)
    --output-dir DIR      Dossier de sortie               (default: ./nbsteps_results)
    --no-compile          Skip mvn compile
    --force               Ignorer le cache et relancer
    --plots-only          Régénérer les plots depuis le cache sans relancer

Cache:
    nbsteps_results/cache/{inst}_steps{N}_seed{S}_{episodes}eps_{type}.json
    type = "bench" (timing) ou "perf" (success rate)
"""

import argparse
import contextlib
import json
import math
import queue
import random
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Imports projet
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
import environment
import q_learning_cp

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DEFAULT_INSTANCES = ["4s", "4medium", "4hard"]
DEFAULT_NB_STEPS = [10, 20, 50, 110]
DEFAULT_SEEDS = 3
DEFAULT_BENCH_EPISODES = 2000
DEFAULT_PERF_EPISODES = 10_000
DEFAULT_WORKERS = 1
DEFAULT_BASE_PORT = 12345
DEFAULT_OUTPUT = ROOT / "nbsteps_results"

# Opérations de timing à conserver dans le résumé (sous-ensemble des 11 OPS)
KEY_OPS = [
    "reset_s",
    "cp_step_wait_s",  # fixPoint Java
    "cp_etr_wait_s",  # vanillaBP Java
    "episode_total_s",
]

KEY_OP_LABELS = {
    "reset_s": "RESET",
    "cp_step_wait_s": "STEP fixPoint",
    "cp_etr_wait_s": "ETR vanillaBP",
    "episode_total_s": "Total épisode",
}


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _cache_path(output_dir: Path, inst: str, nb_steps: int,
                seed: int, episodes: int, kind: str) -> Path:
    return output_dir / "cache" / f"{inst}_steps{nb_steps}_seed{seed}_{episodes}eps_{kind}.json"


def _load_cache(output_dir, inst, nb_steps, seed, episodes, kind):
    p = _cache_path(output_dir, inst, nb_steps, seed, episodes, kind)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        print(f"  [WARN] Cache corrompu {p.name}: {e} — will re-run")
        return None


def _save_cache(output_dir, inst, nb_steps, seed, episodes, kind, entry):
    p = _cache_path(output_dir, inst, nb_steps, seed, episodes, kind)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(entry, f, indent=2)


# ---------------------------------------------------------------------------
# Boucle instrumentée (timing uniquement)
# ---------------------------------------------------------------------------

def _run_instrumented(inst_cfg: dict, total_episodes: int, cp_client) -> list[dict]:
    """Boucle Q-learning instrumentée. Retourne une liste de dicts (un par épisode)."""
    size = inst_cfg["size"]
    slippery = inst_cfg["slippery"]
    max_steps = inst_cfg["max_steps"]
    map_name = inst_cfg.get("map_name", f"{size}x{size}")
    desc = inst_cfg.get("desc", None)

    import io
    with contextlib.redirect_stdout(io.StringIO()):
        env = environment.create_environment(
            map_name=map_name, is_slippery=slippery,
            render_mode=None, desired_max_steps=max_steps, desc=desc, budget=0
        )
    if env is None:
        raise RuntimeError("Impossible de créer l'environnement.")

    state_size = env.observation_space.n
    action_size = env.action_space.n

    q_table = np.full((state_size, action_size), config.Q_INIT_VALUE_CP_ETR)
    epsilon = config.EPSILON
    lr = config.LEARNING_RATE
    gamma = config.DISCOUNT_FACTOR
    eps_min = config.EPSILON_MIN
    eps_decay = (eps_min / epsilon) ** (1.0 / total_episodes)

    records = []
    sock = cp_client.socket

    for episode in range(total_episodes):
        t_ep = time.perf_counter()

        # RESET
        state, _ = env.reset()
        resp = cp_client.send_receive("RESET")
        if not resp.startswith("OK RESET"):
            raise RuntimeError(f"CP RESET failed: {resp}")

        etr_before = cp_client.query_etr() or 0.0
        t_reset_total = time.perf_counter() - t_ep  # reset env + RESET socket + ETR initial

        t_cp_step_wait = 0.0
        t_cp_etr_wait = 0.0

        done = False
        step_idx = 0
        final_reward = 0.0
        terminated = False

        for step_idx in range(max_steps):
            if not (0 <= state < state_size):
                break

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))

            next_state, env_reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # STEP socket — wait (fixPoint)
            cmd = f"STEP {step_idx} {action} {next_state}\n".encode("utf-8")
            sock.sendall(cmd)
            t0 = time.perf_counter()
            sock.recv(q_learning_cp.CP_BUFFER_SIZE)
            t_cp_step_wait += time.perf_counter() - t0

            # QUERY_ETR — wait (vanillaBP)
            sock.sendall(b"QUERY_ETR\n")
            t0 = time.perf_counter()
            raw = sock.recv(q_learning_cp.CP_BUFFER_SIZE).decode("utf-8").strip()
            t_cp_etr_wait += time.perf_counter() - t0

            try:
                etr_after = float(raw.split()[1]) if raw.startswith("ETR_VALUE") else None
            except (ValueError, IndexError):
                etr_after = None

            if etr_after is not None:
                reward_used = etr_after - etr_before
                if terminated and env_reward == 0.0:
                    reward_used = -etr_before
                etr_before = etr_after
            else:
                reward_used = env_reward

            best_next_q = np.max(q_table[next_state])
            td_target = reward_used + gamma * best_next_q * (1 - int(done))
            q_table[state, action] += lr * (td_target - q_table[state, action])
            state = next_state

            if done:
                final_reward = env_reward
                break

        t_ep_total = time.perf_counter() - t_ep
        epsilon = max(epsilon * eps_decay, eps_min)

        records.append({
            "episode": episode + 1,
            "n_steps": step_idx + 1,
            "success": int(final_reward == 1.0 and terminated),
            "reset_s": t_reset_total,
            "cp_step_wait_s": t_cp_step_wait,
            "cp_etr_wait_s": t_cp_etr_wait,
            "episode_total_s": t_ep_total,
        })

    env.close()
    return records


# ---------------------------------------------------------------------------
# Java : lancement / arrêt / compile
# ---------------------------------------------------------------------------

def _start_java(port: int, nb_steps_override: int, no_compile: bool, log_dir: Path):
    cp_dir = ROOT / "MiniCPBP"
    if not cp_dir.is_dir():
        cp_dir = ROOT / "java"
    pom = cp_dir / "pom.xml"

    java_args_list = ["ETR", "0", str(port), str(nb_steps_override)]

    if no_compile:
        cp_file = cp_dir / "target" / "java_classpath.txt"
        with open(cp_file) as f:
            dep_cp = f.read().strip()
        full_cp = str(cp_dir / "target" / "classes") + ":" + dep_cp
        cmd = ["java", "-cp", full_cp,
               "minicpbp.examples.FrozenLakeCPService"] + java_args_list
    else:
        mvn_exec = "mvn.cmd" if sys.platform == "win32" else "mvn"
        cmd = [mvn_exec, "-f", str(pom),
               "compile", "exec:java",
               "-Dexec.cleanupDaemonThreads=false",
               "-Dexec.mainClass=minicpbp.examples.FrozenLakeCPService",
               f"-Dexec.args={' '.join(java_args_list)}"]

    log_dir.mkdir(parents=True, exist_ok=True)
    out_log = open(log_dir / "java_stdout.log", "w")
    err_log = open(log_dir / "java_stderr.log", "w")
    proc = subprocess.Popen(cmd, stdout=out_log, stderr=err_log, text=True)
    return proc, out_log, err_log


def _stop_java(proc, out_log, err_log):
    for f in (out_log, err_log):
        try:
            f.close()
        except Exception:
            pass
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()


def _precompile_java():
    cp_dir = ROOT / "MiniCPBP"
    if not cp_dir.is_dir():
        cp_dir = ROOT / "java"
    pom = cp_dir / "pom.xml"
    mvn_exec = "mvn.cmd" if sys.platform == "win32" else "mvn"

    print("[PRE-COMPILE] Compilation Maven...")
    r = subprocess.run(
        [mvn_exec, "-f", str(pom), "compile", "-Dexec.cleanupDaemonThreads=false"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300
    )
    if r.returncode != 0:
        print("[ERROR] Maven compile failed:")
        print(r.stderr[-2000:])
        return False

    cp_file = cp_dir / "target" / "java_classpath.txt"
    r2 = subprocess.run(
        [mvn_exec, "-f", str(pom), "dependency:build-classpath",
         f"-Dmdep.outputFile={cp_file}", "-q"],
        cwd=str(cp_dir), capture_output=True, text=True, timeout=120
    )
    if r2.returncode != 0 or not cp_file.exists():
        print("[ERROR] Maven dependency:build-classpath failed.")
        return False

    print("[PRE-COMPILE] Done.\n")
    return True


# ---------------------------------------------------------------------------
# Run d'un seul (inst, nb_steps, seed) — timing
# ---------------------------------------------------------------------------

def _run_bench_one(inst_id: str, inst_cfg: dict, nb_steps: int,
                   seed: int, episodes: int, output_dir: Path,
                   port: int, no_compile: bool) -> dict | None:
    """Lance la boucle instrumentée et retourne les stats de timing."""
    random.seed(seed)
    np.random.seed(seed)

    java_proc = java_out = java_err = None
    cp_client = None
    log_dir = output_dir / "java_logs" / f"{inst_id}_steps{nb_steps}_seed{seed}_bench"

    try:
        java_proc, java_out, java_err = _start_java(port, nb_steps, no_compile, log_dir)
        time.sleep(10)
        if java_proc.poll() is not None:
            raise RuntimeError(f"Java server terminated early (exit {java_proc.poll()})")

        cp_client = q_learning_cp.CPRewardClient(port=port)
        if not cp_client.connect():
            raise RuntimeError("Impossible de se connecter au serveur Java CP.")
        resp = cp_client.send_receive(f"INIT {inst_id}")
        if not resp.startswith("OK INIT"):
            raise RuntimeError(f"CP INIT failed: {resp}")

        records = _run_instrumented(inst_cfg, episodes, cp_client)

    except Exception as exc:
        print(f"  [ERROR] bench {inst_id} steps={nb_steps} seed={seed}: {exc}")
        traceback.print_exc()
        return None
    finally:
        if cp_client:
            try:
                cp_client.send_receive("QUIT")
            except Exception:
                pass
            cp_client.close()
        if java_proc:
            _stop_java(java_proc, java_out, java_err)

    def _stats(key):
        vals = [r[key] for r in records if not math.isnan(r[key])]
        if not vals:
            return {"mean": float("nan"), "std": float("nan")}
        arr = np.array(vals)
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}

    return {
        "instance": inst_id,
        "nb_steps": nb_steps,
        "seed": seed,
        "episodes": episodes,
        "kind": "bench",
        "stats": {op: _stats(op) for op in KEY_OPS},
        "success_rate": float(np.mean([r["success"] for r in records])),
    }


# ---------------------------------------------------------------------------
# Run d'un seul (inst, nb_steps, seed) — performances (via main.py)
# ---------------------------------------------------------------------------

def _run_perf_one(inst_id: str, nb_steps: int, seed: int, episodes: int,
                  port: int, no_compile: bool,
                  print_lock: threading.Lock) -> dict | None:
    """Lance main.py et retourne le final success rate."""
    tmp_dir = tempfile.mkdtemp(prefix="fl_nbsteps_")
    try:
        cmd = [
            sys.executable, str(ROOT / "main.py"),
            "--instance", inst_id,
            "--agent", "q",
            "--shaping", "cp-etr",
            "--episodes", str(episodes),
            "--seed", str(seed),
            "--port", str(port),
            "--results-dir", tmp_dir,
            "--cp-nbsteps-override", str(nb_steps),
        ]
        if no_compile:
            cmd.append("--no-compile")

        result = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=7200
        )
        if result.returncode != 0:
            lines = (result.stdout + result.stderr).strip().splitlines()[-20:]
            with print_lock:
                tqdm.write(f"  [ERROR] perf {inst_id} steps={nb_steps} seed={seed} — exit {result.returncode}")
                for l in lines:
                    tqdm.write(f"    {l}")
            return None

        result_files = list(Path(tmp_dir).rglob("result.json")) or list(Path(tmp_dir).rglob("*_log.json"))
        if not result_files:
            with print_lock:
                tqdm.write(f"  [WARN] perf {inst_id} steps={nb_steps} seed={seed} — no result file")
            return None

        with open(result_files[0]) as f:
            raw = json.load(f)
        eval_log = raw.get("evaluation_log", [])
        if not eval_log:
            return None

        last = eval_log[-1]
        return {
            "instance": inst_id,
            "nb_steps": nb_steps,
            "seed": seed,
            "episodes": episodes,
            "kind": "perf",
            "final_sr": last.get("eval_success_rate", float("nan")),
            "eval_log": eval_log,
        }

    except subprocess.TimeoutExpired:
        with print_lock:
            tqdm.write(f"  [ERROR] perf {inst_id} steps={nb_steps} seed={seed} — timeout")
        return None
    except Exception as exc:
        with print_lock:
            tqdm.write(f"  [ERROR] perf {inst_id} steps={nb_steps} seed={seed}: {exc}")
        return None
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Allocation dynamique de ports libres
# ---------------------------------------------------------------------------

def _find_free_ports(n: int) -> list[int]:
    """
    Réserve n ports TCP libres en bindant temporairement sur port 0.
    Les sockets sont tous ouverts simultanément pour garantir que les ports
    sont distincts, puis fermés juste avant de retourner la liste.
    L'OS garantit que ces ports ne sont pas en TIME_WAIT (port 0 = port libre).
    """
    socks = []
    ports = []
    for _ in range(n):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        ports.append(s.getsockname()[1])
        socks.append(s)
    for s in socks:
        s.close()
    return ports


# ---------------------------------------------------------------------------
# Orchestration parallèle
# ---------------------------------------------------------------------------

def run_all(instances_cfg: dict, instance_ids: list[str], nb_steps_list: list[int],
            seeds: list[int], bench_episodes: int, perf_episodes: int,
            output_dir: Path, workers: int, base_port: int,
            # noqa: ARG001 — base_port ignoré, ports alloués dynamiquement
            force: bool, no_compile: bool) -> tuple[list[dict], list[dict]]:
    """
    Lance les runs de bench (timing) et de perf (success rate) en parallèle.
    Retourne (bench_entries, perf_entries).
    """
    # Pré-compilation Java une seule fois
    if not no_compile:
        if _precompile_java():
            no_compile = True
        else:
            print("[WARN] Pre-compile failed — chaque run compilera séparément.")

    # Construire la liste de tous les runs à effectuer
    bench_runs = []
    perf_runs = []

    for inst in instance_ids:
        for nb_steps in nb_steps_list:
            for seed in seeds:
                # bench (timing)
                cached = _load_cache(output_dir, inst, nb_steps, seed, bench_episodes, "bench")
                if cached is not None and not force:
                    bench_runs.append(("skip", inst, nb_steps, seed, cached))
                else:
                    bench_runs.append(("run", inst, nb_steps, seed, None))

                # perf (success rate)
                cached_p = _load_cache(output_dir, inst, nb_steps, seed, perf_episodes, "perf")
                if cached_p is not None and not force:
                    perf_runs.append(("skip", inst, nb_steps, seed, cached_p))
                else:
                    perf_runs.append(("run", inst, nb_steps, seed, None))

    total = len(bench_runs) + len(perf_runs)
    print(f"\n{'=' * 60}")
    print(f"  NB_STEPS BENCHMARK  |  {len(instance_ids)} instance(s)  |  "
          f"{len(nb_steps_list)} valeurs  |  {len(seeds)} seeds")
    print(f"  bench_episodes={bench_episodes}  perf_episodes={perf_episodes}  workers={workers}")
    print(f"{'=' * 60}\n")

    bench_entries: list[dict] = []
    perf_entries: list[dict] = []
    print_lock = threading.Lock()

    # Allouer workers ports libres dynamiquement. Bench et perf sont séquentiels
    # (d'abord tous les bench, puis tous les perf) donc un seul pool suffit.
    all_free_ports = _find_free_ports(workers)
    bench_port_pool: queue.Queue[int] = queue.Queue()
    perf_port_pool: queue.Queue[int] = queue.Queue()
    for p in all_free_ports:
        bench_port_pool.put(p)
        perf_port_pool.put(p)
    print(f"  Ports : {all_free_ports}\n")

    pbar = tqdm(total=total, desc="Progress", unit="run", dynamic_ncols=True, leave=True)

    # Précharger les runs en cache
    skipped = 0
    pending_bench = []
    pending_perf = []

    for status, inst, nb_steps, seed, data in bench_runs:
        if status == "skip":
            bench_entries.append(data)
            skipped += 1
            pbar.update(1)
        else:
            pending_bench.append((inst, nb_steps, seed))

    for status, inst, nb_steps, seed, data in perf_runs:
        if status == "skip":
            perf_entries.append(data)
            skipped += 1
            pbar.update(1)
        else:
            pending_perf.append((inst, nb_steps, seed))

    def _worker_bench(inst, nb_steps, seed):
        port = bench_port_pool.get()
        try:
            tag = f"bench {inst} steps={nb_steps} seed={seed}"
            with print_lock:
                tqdm.write(f"  [RUN]   {tag}")
            entry = _run_bench_one(inst, instances_cfg[inst], nb_steps, seed,
                                   bench_episodes, output_dir, port, no_compile)
            if entry is not None:
                _save_cache(output_dir, inst, nb_steps, seed, bench_episodes, "bench", entry)
                with print_lock:
                    total_s = entry["stats"]["episode_total_s"]["mean"] * 1000
                    tqdm.write(f"  [OK]    {tag}  → {total_s:.2f} ms/ep")
            else:
                with print_lock:
                    tqdm.write(f"  [ERROR] {tag}")
            pbar.update(1)
            return entry
        finally:
            bench_port_pool.put(port)

    def _worker_perf(inst, nb_steps, seed):
        port = perf_port_pool.get()
        try:
            tag = f"perf  {inst} steps={nb_steps} seed={seed}"
            with print_lock:
                tqdm.write(f"  [RUN]   {tag}")
            entry = _run_perf_one(inst, nb_steps, seed, perf_episodes,
                                  port, no_compile, print_lock)
            if entry is not None:
                _save_cache(output_dir, inst, nb_steps, seed, perf_episodes, "perf", entry)
                with print_lock:
                    tqdm.write(f"  [OK]    {tag}  → SR={entry['final_sr']:.3f}")
            else:
                with print_lock:
                    tqdm.write(f"  [ERROR] {tag}")
            pbar.update(1)
            return entry
        finally:
            perf_port_pool.put(port)

    def _run_phase(tasks, worker_fn, result_list, phase_name):
        """Lance une liste de tâches en parallèle avec workers workers."""
        if not tasks:
            return
        with print_lock:
            tqdm.write(f"\n  --- Phase {phase_name} ({len(tasks)} runs) ---")
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(worker_fn, inst, nb_steps, seed): (inst, nb_steps, seed)
                           for inst, nb_steps, seed in tasks}
                for future in as_completed(futures):
                    try:
                        entry = future.result()
                        if entry is not None:
                            result_list.append(entry)
                    except Exception as exc:
                        with print_lock:
                            tqdm.write(f"  [ERROR] future: {exc}")
        else:
            for inst, nb_steps, seed in tasks:
                entry = worker_fn(inst, nb_steps, seed)
                if entry is not None:
                    result_list.append(entry)

    # Phase 1 : bench (timing) — tous les runs, puis phase 2 : perf (SR)
    _run_phase(pending_bench, _worker_bench, bench_entries, "BENCH (timing)")
    _run_phase(pending_perf, _worker_perf, perf_entries, "PERF (success rate)")

    pbar.close()
    return bench_entries, perf_entries


def load_from_cache(instance_ids, nb_steps_list, seeds, bench_episodes, perf_episodes,
                    output_dir) -> tuple[list[dict], list[dict]]:
    bench_entries, perf_entries = [], []
    for inst in instance_ids:
        for nb_steps in nb_steps_list:
            for seed in seeds:
                b = _load_cache(output_dir, inst, nb_steps, seed, bench_episodes, "bench")
                if b is not None:
                    bench_entries.append(b)
                p = _load_cache(output_dir, inst, nb_steps, seed, perf_episodes, "perf")
                if p is not None:
                    perf_entries.append(p)
    print(f"Cache: {len(bench_entries)} entrées bench, {len(perf_entries)} entrées perf.\n")
    return bench_entries, perf_entries


# ---------------------------------------------------------------------------
# Tableau récapitulatif terminal
# ---------------------------------------------------------------------------

def _print_summary(bench_entries: list[dict], perf_entries: list[dict],
                   instance_ids: list[str], nb_steps_list: list[int]) -> None:
    col_w = 14

    for inst in instance_ids:
        print(f"\n{'=' * 70}")
        print(f"  Instance : {inst}")
        print(f"{'=' * 70}")

        # --- Timing ---
        print(f"\n  Timing (ms/épisode, moyenne ± std sur seeds)")
        print(f"  {'nb_steps':<12}", end="")
        for op in KEY_OPS:
            print(f"{KEY_OP_LABELS[op]:>{col_w}}", end="")
        print()
        print("  " + "-" * (12 + col_w * len(KEY_OPS)))

        for nb_steps in nb_steps_list:
            entries = [e for e in bench_entries
                       if e["instance"] == inst and e["nb_steps"] == nb_steps]
            print(f"  {nb_steps:<12}", end="")
            for op in KEY_OPS:
                if entries:
                    vals = [e["stats"][op]["mean"] * 1000 for e in entries
                            if not math.isnan(e["stats"][op]["mean"])]
                    if vals:
                        mu, sd = np.mean(vals), np.std(vals)
                        cell = f"{mu:.3f}±{sd:.3f}"
                    else:
                        cell = "N/A"
                else:
                    cell = "–"
                print(f"{cell:>{col_w}}", end="")
            print()

        # --- Succès ---
        print(f"\n  Success rate (final, moyenne ± std sur seeds)")
        print(f"  {'nb_steps':<12} {'SR mean':>{col_w}} {'SR std':>{col_w}}")
        print("  " + "-" * (12 + col_w * 2))

        for nb_steps in nb_steps_list:
            entries = [e for e in perf_entries
                       if e["instance"] == inst and e["nb_steps"] == nb_steps]
            if entries:
                srs = [e["final_sr"] for e in entries if not math.isnan(e["final_sr"])]
                if srs:
                    print(f"  {nb_steps:<12} {np.mean(srs):>{col_w}.4f} {np.std(srs):>{col_w}.4f}")
                else:
                    print(f"  {nb_steps:<12} {'N/A':>{col_w}} {'N/A':>{col_w}}")
            else:
                print(f"  {nb_steps:<12} {'–':>{col_w}} {'–':>{col_w}}")

    print()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _make_plots(bench_entries: list[dict], perf_entries: list[dict],
                instance_ids: list[str], nb_steps_list: list[int],
                output_dir: Path) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-darkgrid")

    for inst in instance_ids:
        # ------------------------------------------------------------------ #
        # Plot 1 : Timing vs nb_steps (une ligne par opération clé)
        # ------------------------------------------------------------------ #
        fig, ax = plt.subplots(figsize=(9, 5))
        op_colors = plt.cm.tab10(np.linspace(0, 0.9, len(KEY_OPS)))

        for op_idx, op in enumerate(KEY_OPS):
            means, stds, xs = [], [], []
            for nb_steps in nb_steps_list:
                entries = [e for e in bench_entries
                           if e["instance"] == inst and e["nb_steps"] == nb_steps]
                if entries:
                    vals = [e["stats"][op]["mean"] * 1000 for e in entries
                            if not math.isnan(e["stats"][op]["mean"])]
                    if vals:
                        means.append(float(np.mean(vals)))
                        stds.append(float(np.std(vals)))
                        xs.append(nb_steps)

            if xs:
                means_a = np.array(means)
                stds_a = np.array(stds)
                color = op_colors[op_idx]
                ax.plot(xs, means_a, marker="o", color=color,
                        linewidth=2, label=KEY_OP_LABELS[op])
                ax.fill_between(xs, means_a - stds_a, means_a + stds_a,
                                color=color, alpha=0.15)

        ax.set_xlabel("cp_nbSteps")
        ax.set_ylabel("ms / épisode")
        ax.set_title(f"Timing vs cp_nbSteps — {inst}", fontsize=12, fontweight="bold")
        ax.set_xticks(nb_steps_list)
        ax.legend(fontsize=9)
        plt.tight_layout()
        out = plots_dir / f"timing_vs_nbsteps_{inst}.png"
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Timing plot saved: {out}")

        # ------------------------------------------------------------------ #
        # Plot 2 : Success rate vs nb_steps
        # ------------------------------------------------------------------ #
        fig, ax = plt.subplots(figsize=(8, 5))
        xs_p, means_p, stds_p = [], [], []

        for nb_steps in nb_steps_list:
            entries = [e for e in perf_entries
                       if e["instance"] == inst and e["nb_steps"] == nb_steps]
            if entries:
                srs = [e["final_sr"] for e in entries if not math.isnan(e["final_sr"])]
                if srs:
                    xs_p.append(nb_steps)
                    means_p.append(float(np.mean(srs)))
                    stds_p.append(float(np.std(srs)))

        if xs_p:
            means_a = np.array(means_p)
            stds_a = np.array(stds_p)
            ax.plot(xs_p, means_a, marker="o", color="#9b59b6", linewidth=2, label="q-cp-etr")
            ax.fill_between(xs_p, means_a - stds_a, means_a + stds_a,
                            color="#9b59b6", alpha=0.2)
            ax.set_ylim(-0.05, 1.05)

        ax.set_xlabel("cp_nbSteps")
        ax.set_ylabel("Success rate final")
        ax.set_title(f"Performances vs cp_nbSteps — {inst}", fontsize=12, fontweight="bold")
        ax.set_xticks(nb_steps_list)
        ax.legend(fontsize=9)
        plt.tight_layout()
        out = plots_dir / f"sr_vs_nbsteps_{inst}.png"
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"SR plot saved: {out}")

        # ------------------------------------------------------------------ #
        # Plot 3 : Timing + SR sur double axe Y
        # ------------------------------------------------------------------ #
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax2 = ax1.twinx()

        # Timing (axe gauche) — Total épisode
        xs_t, means_t, stds_t = [], [], []
        for nb_steps in nb_steps_list:
            entries = [e for e in bench_entries
                       if e["instance"] == inst and e["nb_steps"] == nb_steps]
            if entries:
                vals = [e["stats"]["episode_total_s"]["mean"] * 1000 for e in entries
                        if not math.isnan(e["stats"]["episode_total_s"]["mean"])]
                if vals:
                    xs_t.append(nb_steps)
                    means_t.append(float(np.mean(vals)))
                    stds_t.append(float(np.std(vals)))

        if xs_t:
            means_a = np.array(means_t)
            stds_a = np.array(stds_t)
            ax1.plot(xs_t, means_a, marker="o", color="#e74c3c", linewidth=2,
                     label="Timing total (ms/ep)")
            ax1.fill_between(xs_t, means_a - stds_a, means_a + stds_a,
                             color="#e74c3c", alpha=0.15)
        ax1.set_xlabel("cp_nbSteps")
        ax1.set_ylabel("ms / épisode", color="#e74c3c")
        ax1.tick_params(axis="y", labelcolor="#e74c3c")

        # Success rate (axe droit)
        if xs_p:
            ax2.plot(xs_p, np.array(means_p), marker="s", color="#9b59b6",
                     linewidth=2, linestyle="--", label="Success rate")
            ax2.fill_between(xs_p,
                             np.array(means_p) - np.array(stds_p),
                             np.array(means_p) + np.array(stds_p),
                             color="#9b59b6", alpha=0.15)
        ax2.set_ylabel("Success rate", color="#9b59b6")
        ax2.tick_params(axis="y", labelcolor="#9b59b6")
        ax2.set_ylim(-0.05, 1.05)

        # Légende combinée
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
        ax1.set_xticks(nb_steps_list)

        ax1.set_title(f"Timing & Success rate vs cp_nbSteps — {inst}",
                      fontsize=12, fontweight="bold")
        plt.tight_layout()
        out = plots_dir / f"tradeoff_{inst}.png"
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Tradeoff plot saved: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--instances", nargs="+", default=DEFAULT_INSTANCES, metavar="ID")
    parser.add_argument("--nb-steps", nargs="+", type=int, default=DEFAULT_NB_STEPS,
                        metavar="N", help="Valeurs de cp_nbSteps à tester")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                        help=f"Nombre de seeds (default: {DEFAULT_SEEDS})")
    parser.add_argument("--bench-episodes", type=int, default=DEFAULT_BENCH_EPISODES,
                        help=f"Épisodes pour le timing (default: {DEFAULT_BENCH_EPISODES})")
    parser.add_argument("--perf-episodes", type=int, default=DEFAULT_PERF_EPISODES,
                        help=f"Épisodes pour le success rate (default: {DEFAULT_PERF_EPISODES})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Workers parallèles (default: {DEFAULT_WORKERS})")
    parser.add_argument("--base-port", type=int, default=DEFAULT_BASE_PORT,
                        help=f"Premier port TCP Java (default: {DEFAULT_BASE_PORT})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-compile", action="store_true",
                        help="Skip mvn compile (Java déjà compilé)")
    parser.add_argument("--force", action="store_true",
                        help="Ignorer le cache et relancer tous les runs")
    parser.add_argument("--plots-only", action="store_true",
                        help="Ne pas relancer les runs, régénérer uniquement les plots")
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = list(range(1, args.seeds + 1))
    nb_steps_list = sorted(args.nb_steps)

    # Charger instances.json
    with open(ROOT / "instances.json") as f:
        instances_cfg = json.load(f)

    for inst_id in args.instances:
        if inst_id not in instances_cfg:
            print(f"[ERROR] Instance '{inst_id}' introuvable dans instances.json.")
            sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.plots_only:
        print("--plots-only : chargement depuis le cache...")
        bench_entries, perf_entries = load_from_cache(
            args.instances, nb_steps_list, seeds,
            args.bench_episodes, args.perf_episodes, args.output_dir
        )
    else:
        bench_entries, perf_entries = run_all(
            instances_cfg=instances_cfg,
            instance_ids=args.instances,
            nb_steps_list=nb_steps_list,
            seeds=seeds,
            bench_episodes=args.bench_episodes,
            perf_episodes=args.perf_episodes,
            output_dir=args.output_dir,
            workers=args.workers,
            base_port=args.base_port,
            force=args.force,
            no_compile=args.no_compile,
        )

    if not bench_entries and not perf_entries:
        print("[WARN] Aucune donnée à afficher.")
        return

    _print_summary(bench_entries, perf_entries, args.instances, nb_steps_list)
    _make_plots(bench_entries, perf_entries, args.instances, nb_steps_list, args.output_dir)
    print("\nBenchmark cp_nbSteps terminé.")


if __name__ == "__main__":
    main()
