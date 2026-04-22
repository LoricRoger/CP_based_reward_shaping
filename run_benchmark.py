"""
run_benchmark.py — Benchmark de timing pour FrozenLake CP reward shaping.

Mesure le temps passé dans chaque opération clé à l'intérieur de la boucle
d'entraînement Q-learning, agrégé par épisode.

Opérations mesurées (méthode cp-etr) :
  reset_s          : RESET socket (début d'épisode)
  initial_etr_s    : QUERY_ETR initial (début d'épisode)
  env_step_s       : env.step() — total sur l'épisode
  cp_step_s        : STEP socket — total sur l'épisode
  cp_query_etr_s   : QUERY_ETR socket — total sur l'épisode
  bellman_s        : mise à jour Q-table — total sur l'épisode
  episode_total_s  : durée totale de l'épisode

Pour q-none et q-classic, les métriques CP sont absentes (NaN).

Usage:
    python run_benchmark.py [options]

Options:
    --instances ID...   Instance IDs à tester          (default: 4s 4hard)
    --methods M...      Méthodes à tester              (default: q-none q-classic q-cp-etr)
    --seeds N           Nombre de seeds, à partir de 1 (default: 3)
    --episodes N        Épisodes d'entraînement        (default: 500)
    --output-dir DIR    Dossier de sortie              (default: ./benchmark_results)
    --port P            Port TCP Java de base           (default: 12345)
    --no-compile        Skip mvn compile (Java déjà compilé)
    --force             Ignorer le cache et relancer
    --plots-only        Régénérer les plots sans relancer les runs
"""

import argparse
import contextlib
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
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
from q_learning_standard import shaped_reward_classic

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DEFAULT_INSTANCES = ["4s", "4hard"]
DEFAULT_METHODS   = ["q-none", "q-classic", "q-cp-etr"]
DEFAULT_SEEDS     = 3
DEFAULT_EPISODES  = 500
DEFAULT_OUTPUT    = ROOT / "benchmark_results"
DEFAULT_PORT      = 12345

JAVA_METHODS = {"q-cp-etr", "q-cp-ms"}

# Opérations mesurées (dans l'ordre affiché)
OPS = [
    "reset_s",
    "initial_etr_s",
    "env_step_s",
    "cp_step_s",
    "cp_query_etr_s",
    "bellman_s",
    "episode_total_s",
]

OP_LABELS = {
    "reset_s":         "RESET socket",
    "initial_etr_s":   "QUERY_ETR initial",
    "env_step_s":      "env.step()",
    "cp_step_s":       "STEP socket",
    "cp_query_etr_s":  "QUERY_ETR par step",
    "bellman_s":       "Bellman update",
    "episode_total_s": "Total épisode",
}

METHOD_COLORS = {
    "q-none":    "#e74c3c",
    "q-classic": "#3498db",
    "q-cp-etr":  "#9b59b6",
    "q-cp-ms":   "#2ecc71",
}


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _cache_path(output_dir: Path, inst: str, meth: str, seed: int, episodes: int) -> Path:
    return output_dir / "cache" / f"{inst}_{meth}_seed{seed}_{episodes}eps_bench.json"


def _load_cache(output_dir, inst, meth, seed, episodes):
    p = _cache_path(output_dir, inst, meth, seed, episodes)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        print(f"  [WARN] Cache corrompu {p.name}: {e} — will re-run")
        return None


def _save_cache(output_dir, inst, meth, seed, episodes, entry):
    p = _cache_path(output_dir, inst, meth, seed, episodes)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(entry, f, indent=2)


# ---------------------------------------------------------------------------
# Boucle d'entraînement instrumentée
# ---------------------------------------------------------------------------

def _run_instrumented(inst_cfg: dict, method: str, total_episodes: int,
                      cp_client=None) -> list[dict]:
    """
    Lance une session d'entraînement instrumentée et retourne une liste de dicts,
    un par épisode, avec les timings agrégés.

    inst_cfg : dict issu de instances.json
    method   : "q-none" | "q-classic" | "q-cp-etr"
    cp_client: CPRewardClient connecté (ou None pour les méthodes sans Java)
    """
    size      = inst_cfg["size"]
    holes     = inst_cfg["holes"]
    goal      = inst_cfg["goal"]
    slippery  = inst_cfg["slippery"]
    max_steps = inst_cfg["max_steps"]
    map_name  = inst_cfg.get("map_name", f"{size}x{size}")
    desc      = inst_cfg.get("desc", None)

    with open(os.devnull, "w") as _devnull, contextlib.redirect_stdout(_devnull):
        env = environment.create_environment(
            map_name=map_name, is_slippery=slippery,
            render_mode=None, desired_max_steps=max_steps, desc=desc, budget=0
        )
    if env is None:
        raise RuntimeError("Impossible de créer l'environnement.")

    state_size  = env.observation_space.n
    action_size = env.action_space.n

    # Q-table init (même logique que les agents originaux)
    if method == "q-none":
        q_table = np.full((state_size, action_size), config.Q_INIT_VALUE_NONE)
    elif method == "q-classic":
        q_table = np.full((state_size, action_size), config.Q_INIT_VALUE_CLASSIC)
    else:  # cp-etr / cp-ms
        q_table = np.full((state_size, action_size), config.Q_INIT_VALUE_CP_ETR)

    epsilon   = config.EPSILON
    lr        = config.LEARNING_RATE
    gamma     = config.DISCOUNT_FACTOR
    eps_min   = config.EPSILON_MIN
    eps_decay = (eps_min / epsilon) ** (1.0 / total_episodes)

    hole_set = set(holes)
    episode_records = []

    for episode in range(total_episodes):
        t_ep_start = time.perf_counter()

        # ---- RESET -------------------------------------------------------
        t0 = time.perf_counter()
        state, _ = env.reset()
        t_reset = time.perf_counter() - t0

        t_cp_reset = 0.0
        if cp_client is not None:
            t0 = time.perf_counter()
            resp = cp_client.send_receive("RESET")
            t_cp_reset = time.perf_counter() - t0
            if not resp.startswith("OK RESET"):
                raise RuntimeError(f"CP RESET failed: {resp}")

        # ---- ETR initial -------------------------------------------------
        t_initial_etr = 0.0
        etr_before = None
        if cp_client is not None and method == "q-cp-etr":
            t0 = time.perf_counter()
            etr_before = cp_client.query_etr()
            t_initial_etr = time.perf_counter() - t0
            if etr_before is None:
                etr_before = 0.0

        # ---- Boucle steps ------------------------------------------------
        t_env_step_total    = 0.0
        t_cp_step_total     = 0.0
        t_cp_etr_total      = 0.0
        t_bellman_total     = 0.0

        done = False
        final_reward = 0.0
        step_idx = 0
        terminated = False

        for step_idx in range(max_steps):
            if not (0 <= state < state_size):
                break

            # Action selection (epsilon-greedy, pas instrumenté — négligeable)
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))

            # env.step
            t0 = time.perf_counter()
            next_state, env_reward, terminated, truncated, _ = env.step(action)
            t_env_step_total += time.perf_counter() - t0
            done = terminated or truncated

            # STEP socket
            if cp_client is not None:
                t0 = time.perf_counter()
                cp_client.send_step(step_idx, action, next_state)
                t_cp_step_total += time.perf_counter() - t0

            # Shaping + ETR query
            if method == "q-none":
                reward_used = env_reward
            elif method == "q-classic":
                reward_used = shaped_reward_classic(
                    state, next_state, env_reward, done, hole_set, goal, size
                )
            elif method == "q-cp-etr" and cp_client is not None:
                t0 = time.perf_counter()
                etr_after = cp_client.query_etr()
                t_cp_etr_total += time.perf_counter() - t0

                if etr_after is None or etr_before is None:
                    reward_used = env_reward
                else:
                    reward_used = etr_after - etr_before
                    if terminated and env_reward == 0.0 and step_idx < max_steps - 1:
                        reward_used = -etr_before  # type: ignore[operator]
                    etr_before = etr_after
            else:
                reward_used = env_reward

            # Bellman update
            t0 = time.perf_counter()
            best_next_q = np.max(q_table[next_state])
            td_target = reward_used + gamma * best_next_q * (1 - int(done))
            q_table[state, action] += lr * (td_target - q_table[state, action])
            t_bellman_total += time.perf_counter() - t0

            state = next_state
            if done:
                final_reward = env_reward
                break

        t_ep_total = time.perf_counter() - t_ep_start
        epsilon = max(epsilon * eps_decay, eps_min)

        episode_records.append({
            "episode":          episode + 1,
            "n_steps":          step_idx + 1,
            "success":          int(final_reward == 1.0 and terminated),
            # timings (secondes)
            "reset_s":          t_reset + t_cp_reset,
            "initial_etr_s":    t_initial_etr,
            "env_step_s":       t_env_step_total,
            "cp_step_s":        t_cp_step_total,
            "cp_query_etr_s":   t_cp_etr_total,
            "bellman_s":        t_bellman_total,
            "episode_total_s":  t_ep_total,
        })

    env.close()
    return episode_records


# ---------------------------------------------------------------------------
# Lancement Java (même pattern que main.py)
# ---------------------------------------------------------------------------

def _start_java(port: int, no_compile: bool, log_dir: Path):
    cp_dir = ROOT / "MiniCPBP"
    if not cp_dir.is_dir():
        cp_dir = ROOT / "java"
    pom = cp_dir / "pom.xml"

    java_args_list = ["ETR", "0", str(port)]

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
# Run one (inst, method, seed)
# ---------------------------------------------------------------------------

def _run_one(inst_id: str, inst_cfg: dict, method: str, seed: int,
             episodes: int, output_dir: Path,
             port: int = DEFAULT_PORT, no_compile: bool = False) -> dict | None:
    """
    Lance un run complet instrumenté pour une combinaison (inst, method, seed).
    Retourne un dict avec les stats agrégées (ou None si erreur).
    """
    random.seed(seed)
    np.random.seed(seed)

    java_proc = java_out = java_err = None
    cp_client = None
    log_dir = output_dir / "java_logs" / f"{inst_id}_{method}_seed{seed}"

    try:
        if method in JAVA_METHODS:
            t_java_start = time.perf_counter()
            java_proc, java_out, java_err = _start_java(port, no_compile, log_dir)
            time.sleep(10)
            if java_proc.poll() is not None:
                raise RuntimeError(f"Java server terminated early (exit {java_proc.poll()})")

            cp_client = q_learning_cp.CPRewardClient(port=port)
            if not cp_client.connect():
                raise RuntimeError("Impossible de se connecter au serveur Java CP.")
            resp = cp_client.send_receive(f"INIT {inst_id}")
            if not resp.startswith("OK INIT"):
                raise RuntimeError(f"CP INIT failed: {resp}")
            java_startup_s = time.perf_counter() - t_java_start
        else:
            java_startup_s = 0.0

        t_total_start = time.perf_counter()
        episode_records = _run_instrumented(inst_cfg, method, episodes, cp_client)
        total_training_s = time.perf_counter() - t_total_start

    except Exception as exc:
        print(f"  [ERROR] {inst_id} / {method} / seed={seed}: {exc}")
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

    # ---- Agrégation par épisode → stats globales -------------------------
    def _stats(key):
        vals = [r[key] for r in episode_records if not math.isnan(r[key])]
        if not vals:
            return {"mean": float("nan"), "std": float("nan"),
                    "p50": float("nan"), "p90": float("nan"),
                    "p99": float("nan"), "sum": float("nan")}
        arr = np.array(vals)
        return {
            "mean": float(np.mean(arr)),
            "std":  float(np.std(arr)),
            "p50":  float(np.percentile(arr, 50)),
            "p90":  float(np.percentile(arr, 90)),
            "p99":  float(np.percentile(arr, 99)),
            "sum":  float(np.sum(arr)),
        }

    stats = {op: _stats(op) for op in OPS}

    # fraction du temps total par opération (basée sur sum)
    total_sum = stats["episode_total_s"]["sum"]
    fractions = {}
    for op in OPS:
        s = stats[op]["sum"]
        fractions[op] = s / total_sum if total_sum > 0 else float("nan")

    # évolution temporelle : moyenne glissante sur fenêtre de 50 épisodes
    window = min(50, max(1, episodes // 10))
    def _rolling_mean(key):
        vals = [r[key] for r in episode_records]
        result = []
        for i in range(len(vals)):
            w_vals = vals[max(0, i - window + 1): i + 1]
            result.append(float(np.mean(w_vals)))
        return result

    rolling = {op: _rolling_mean(op) for op in OPS}
    episodes_axis = [r["episode"] for r in episode_records]

    return {
        "instance":          inst_id,
        "method":            method,
        "seed":              seed,
        "total_episodes":    episodes,
        "total_training_s":  total_training_s,
        "java_startup_s":    java_startup_s,
        "stats":             stats,
        "fractions":         fractions,
        "rolling":           rolling,
        "episodes_axis":     episodes_axis,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _color(method: str) -> str:
    return METHOD_COLORS.get(method, "#95a5a6")


def _boxplot_ops(all_entries: list[dict], output_dir: Path) -> None:
    """
    Un boxplot par opération : distribution des moyennes par épisode,
    toutes seeds confondues, groupées par méthode.
    """
    # Collect: op → method → list of per-run mean values
    from collections import defaultdict
    data: dict[str, dict[str, list]] = {op: defaultdict(list) for op in OPS}

    for entry in all_entries:
        m = entry["method"]
        for op in OPS:
            mean_val = entry["stats"][op]["mean"]
            if not math.isnan(mean_val):
                data[op][m].append(mean_val * 1000)  # ms

    methods_present = sorted({e["method"] for e in all_entries})
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for ax_idx, op in enumerate(OPS):
        ax = axes[ax_idx]
        box_data, box_labels, box_colors = [], [], []
        for m in methods_present:
            vals = data[op][m]
            if vals:
                box_data.append(vals)
                box_labels.append(m)
                box_colors.append(_color(m))

        if box_data:
            bp = ax.boxplot(box_data, patch_artist=True, tick_labels=box_labels,
                            medianprops={"color": "black", "linewidth": 2})
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        ax.set_title(OP_LABELS[op], fontsize=10, fontweight="bold")
        ax.set_ylabel("ms / épisode")
        ax.tick_params(axis="x", labelsize=8)
        ax.set_yscale("symlog", linthresh=0.01)

    # Hide unused subplot
    for ax_idx in range(len(OPS), len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle("Distribution du temps par opération (ms/épisode)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = output_dir / "boxplot_ops.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Boxplot saved: {out}")


def _boxplot_per_instance(all_entries: list[dict], instances: list[str],
                          output_dir: Path) -> None:
    """
    Un boxplot par instance, sur episode_total_s, groupé par méthode.
    Permet de comparer facilement la vitesse globale par instance.
    """
    methods_present = sorted({e["method"] for e in all_entries})

    fig, axes = plt.subplots(1, len(instances), figsize=(5 * len(instances), 5), squeeze=False)

    for col, inst in enumerate(instances):
        ax = axes[0][col]
        entries_inst = [e for e in all_entries if e["instance"] == inst]
        box_data, box_labels, box_colors = [], [], []
        for m in methods_present:
            vals = [e["stats"]["episode_total_s"]["mean"] * 1000
                    for e in entries_inst if e["method"] == m
                    and not math.isnan(e["stats"]["episode_total_s"]["mean"])]
            if vals:
                box_data.append(vals)
                box_labels.append(m)
                box_colors.append(_color(m))

        if box_data:
            bp = ax.boxplot(box_data, patch_artist=True, tick_labels=box_labels,
                            medianprops={"color": "black", "linewidth": 2})
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        ax.set_title(f"Instance : {inst}", fontsize=11, fontweight="bold")
        ax.set_ylabel("ms / épisode (total)" if col == 0 else "")
        ax.set_yscale("symlog", linthresh=0.01)

    fig.suptitle("Temps total par épisode selon l'instance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = output_dir / "boxplot_by_instance.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Boxplot par instance saved: {out}")


def _plot_time_evolution(all_entries: list[dict], instances: list[str],
                         methods: list[str], output_dir: Path) -> None:
    """
    Évolution du temps total par épisode (moyenne glissante) au fil du training,
    une figure par instance, une courbe par méthode.
    """
    plt.style.use("seaborn-v0_8-darkgrid")

    for inst in instances:
        entries_inst = [e for e in all_entries if e["instance"] == inst]
        if not entries_inst:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        has_data = False

        for m in methods:
            entries_m = [e for e in entries_inst if e["method"] == m]
            if not entries_m:
                continue

            # Aligner les courbes rolling sur l'axe épisodes commun
            all_ep = entries_m[0]["episodes_axis"]
            matrix = np.array([e["rolling"]["episode_total_s"] for e in entries_m]) * 1000  # ms

            mean_c = np.mean(matrix, axis=0)
            std_c  = np.std(matrix, axis=0)
            color  = _color(m)

            ax.plot(all_ep, mean_c, color=color, linewidth=2, label=m)
            ax.fill_between(all_ep, mean_c - std_c, mean_c + std_c,
                            color=color, alpha=0.15)
            has_data = True

        if has_data:
            ax.set_title(f"Évolution du temps total par épisode — {inst}", fontsize=13, fontweight="bold")
            ax.set_xlabel("Épisode d'entraînement")
            ax.set_ylabel("Durée (ms, moyenne glissante)")
            ax.legend(fontsize=9)
            plt.tight_layout()
            out = output_dir / f"time_evolution_{inst}.png"
            plt.savefig(out, dpi=150)
            print(f"Time evolution saved: {out}")
        plt.close(fig)


def _plot_op_breakdown(all_entries: list[dict], instances: list[str],
                       methods: list[str], output_dir: Path) -> None:
    """
    Stacked bar chart : fraction du temps total par opération, par méthode.
    Une figure par instance.
    """
    # Opérations à inclure dans la décomposition (on exclut episode_total_s)
    ops_breakdown = [op for op in OPS if op != "episode_total_s"]
    op_colors = plt.cm.Set2(np.linspace(0, 1, len(ops_breakdown)))

    for inst in instances:
        entries_inst = [e for e in all_entries if e["instance"] == inst]
        if not entries_inst:
            continue

        methods_inst = [m for m in methods if any(e["method"] == m for e in entries_inst)]
        if not methods_inst:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(methods_inst))
        bottom = np.zeros(len(methods_inst))

        for op_idx, op in enumerate(ops_breakdown):
            fracs = []
            for m in methods_inst:
                entries_m = [e for e in entries_inst if e["method"] == m]
                if entries_m:
                    # Moyenne des fractions sur les seeds
                    frac_vals = [e["fractions"][op] for e in entries_m
                                 if not math.isnan(e["fractions"].get(op, float("nan")))]
                    fracs.append(float(np.mean(frac_vals)) if frac_vals else 0.0)
                else:
                    fracs.append(0.0)

            ax.bar(x, fracs, bottom=bottom, color=op_colors[op_idx],
                   label=OP_LABELS[op], alpha=0.85)
            bottom += np.array(fracs)

        ax.set_xticks(x)
        ax.set_xticklabels(methods_inst, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Fraction du temps total")
        ax.set_title(f"Décomposition du temps par opération — {inst}", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        out = output_dir / f"op_breakdown_{inst}.png"
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Op breakdown saved: {out}")


def _print_summary_table(all_entries: list[dict], instances: list[str],
                         methods: list[str]) -> None:
    """Affiche un tableau récapitulatif des temps moyens dans le terminal."""
    col_w = 14
    op_w  = 18

    print("\n" + "=" * (op_w + col_w * len(methods) * len(instances)))
    print(f"  BENCHMARK TIMING SUMMARY  (ms / épisode, moyenne ± std sur toutes les seeds)")
    print("=" * (op_w + col_w * len(methods) * len(instances)))

    for inst in instances:
        print(f"\n  Instance : {inst}")
        header = f"  {'Opération':<{op_w}}"
        for m in methods:
            header += f"{m:>{col_w}}"
        print(header)
        print("  " + "-" * (op_w + col_w * len(methods)))

        for op in OPS:
            row = f"  {OP_LABELS[op]:<{op_w}}"
            for m in methods:
                entries = [e for e in all_entries
                           if e["instance"] == inst and e["method"] == m]
                if entries:
                    means = [e["stats"][op]["mean"] * 1000 for e in entries
                             if not math.isnan(e["stats"][op]["mean"])]
                    if means:
                        mu = np.mean(means)
                        sd = np.std(means)
                        row += f"{f'{mu:.3f}±{sd:.3f}':>{col_w}}"
                    else:
                        row += f"{'N/A':>{col_w}}"
                else:
                    row += f"{'–':>{col_w}}"
            print(row)

    print()


def make_plots(all_entries: list[dict], instances: list[str],
               methods: list[str], output_dir: Path) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    _print_summary_table(all_entries, instances, methods)
    _boxplot_ops(all_entries, plots_dir)
    _boxplot_per_instance(all_entries, instances, plots_dir)
    _plot_time_evolution(all_entries, instances, methods, plots_dir)
    _plot_op_breakdown(all_entries, instances, methods, plots_dir)


# ---------------------------------------------------------------------------
# Orchestration principale
# ---------------------------------------------------------------------------

def run_all(instances_cfg: dict, instance_ids: list[str], methods: list[str],
            seeds: list[int], episodes: int, output_dir: Path,
            force: bool, base_port: int, no_compile: bool) -> list[dict]:

    needs_java = any(m in JAVA_METHODS for m in methods)
    if needs_java and not no_compile:
        if not _precompile_java():
            print("[WARN] Pre-compile failed — chaque run compilera séparément.")
            no_compile = False
        else:
            no_compile = True

    runs = [(inst, meth, seed)
            for inst in instance_ids
            for meth in methods
            for seed in seeds]
    total = len(runs)

    print(f"\n{'=' * 55}")
    print(f"  BENCHMARK  |  {total} runs  |  episodes={episodes}")
    print(f"{'=' * 55}\n")

    all_entries = []
    port_counter = base_port  # un seul worker pour garantir les timings réels

    pbar = tqdm(total=total, desc="Benchmark", unit="run", dynamic_ncols=True)

    for inst_id, method, seed in runs:
        tag = f"{inst_id:<10} {method:<12} seed={seed}"
        cached = _load_cache(output_dir, inst_id, method, seed, episodes)
        if cached is not None and not force:
            tqdm.write(f"  [SKIP]  {tag}  (cached)")
            all_entries.append(cached)
            pbar.update(1)
            continue

        tqdm.write(f"  [RUN]   {tag} ...")
        port = port_counter  # un seul port, runs séquentiels
        entry = _run_one(
            inst_id, instances_cfg[inst_id], method, seed,
            episodes, output_dir, port=port, no_compile=no_compile
        )
        if entry is not None:
            _save_cache(output_dir, inst_id, method, seed, episodes, entry)
            all_entries.append(entry)
            total_s = entry["total_training_s"]
            tqdm.write(f"  [OK]    {tag}  → {total_s:.1f}s total")
        else:
            tqdm.write(f"  [ERROR] {tag}")
        pbar.update(1)

    pbar.close()
    return all_entries


def load_from_cache(instance_ids, methods, seeds, episodes, output_dir) -> list[dict]:
    entries = []
    for inst in instance_ids:
        for m in methods:
            for s in seeds:
                e = _load_cache(output_dir, inst, m, s, episodes)
                if e is not None:
                    entries.append(e)
    print(f"Chargé {len(entries)} entrées depuis le cache.\n")
    return entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--instances", nargs="+", default=DEFAULT_INSTANCES, metavar="ID")
    parser.add_argument("--methods",   nargs="+", default=DEFAULT_METHODS,   metavar="M",
                        help="Méthodes : q-none  q-classic  q-cp-etr")
    parser.add_argument("--seeds",     type=int, default=DEFAULT_SEEDS,
                        help=f"Nombre de seeds (default: {DEFAULT_SEEDS})")
    parser.add_argument("--episodes",  type=int, default=DEFAULT_EPISODES,
                        help=f"Épisodes d'entraînement (default: {DEFAULT_EPISODES})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--port",       type=int,  default=DEFAULT_PORT,
                        help="Port TCP de base pour le serveur Java")
    parser.add_argument("--no-compile", action="store_true",
                        help="Skip mvn compile (Java déjà compilé)")
    parser.add_argument("--force",      action="store_true",
                        help="Ignorer le cache et relancer tous les runs")
    parser.add_argument("--plots-only", action="store_true",
                        help="Ne pas relancer les runs, régénérer uniquement les plots")
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = list(range(1, args.seeds + 1))

    # Charger instances.json
    instances_path = ROOT / "instances.json"
    with open(instances_path) as f:
        instances_cfg = json.load(f)

    for inst_id in args.instances:
        if inst_id not in instances_cfg:
            print(f"[ERROR] Instance '{inst_id}' introuvable dans instances.json.")
            sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.plots_only:
        print("--plots-only : chargement depuis le cache...")
        all_entries = load_from_cache(
            args.instances, args.methods, seeds, args.episodes, args.output_dir
        )
    else:
        all_entries = run_all(
            instances_cfg=instances_cfg,
            instance_ids=args.instances,
            methods=args.methods,
            seeds=seeds,
            episodes=args.episodes,
            output_dir=args.output_dir,
            force=args.force,
            base_port=args.port,
            no_compile=args.no_compile,
        )

    if not all_entries:
        print("[WARN] Aucune donnée à afficher.")
        return

    make_plots(all_entries, args.instances, args.methods, args.output_dir)
    print("\nBenchmark terminé.")


if __name__ == "__main__":
    main()
