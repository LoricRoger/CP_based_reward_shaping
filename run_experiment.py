"""
run_experiment.py — Lance et agrège des expériences FrozenLake CP reward-shaping
sur plusieurs instances, méthodes et seeds.

Usage:
    python run_experiment.py [options]

Options:
    --episodes N        Training episodes per run          (default: 2000)
    --seeds N           Number of seeds, starting at 1    (default: 5)
    --instances ID...   Instance IDs to benchmark         (default: 4s 4medium 4hard 8s 8medium 8hard)
    --methods M...      Methods to benchmark              (default: all five)
    --output-dir DIR    Directory for all outputs         (default: ./experiment_results)
    --workers N         Parallel workers (default: 1). Java methods utilisent un port dédié par worker.
    --base-port P       Premier port TCP pour les serveurs Java (default: 12345)
    --force             Re-run even if cache entry exists
    --plots-only        Skip running, only regenerate plots/table from cached data

Methods:
    q-none      Q-learning, no reward shaping
    q-classic   Q-learning, classic potential shaping
    q-cp-ms     Q-learning, CP-MS shaping  (Java mode: MS)
    q-cp-etr    Q-learning, CP-ETR shaping (Java mode: ETR)
    cp-greedy   CP-MS greedy agent (no Q-learning)

Cache:
    After each successful run, the script writes a compact JSON to
    experiment_results/cache/{inst}_{method}_seed{seed}_{episodes}eps.json
    containing just the eval_log, final_sr and final_dr.
    The results/ folder written by main.py is never read for skip logic.
"""

import argparse
import csv
import json
import queue
import shutil
import time
import subprocess
import sys
import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

DEFAULT_INSTANCES = ["4s", "4medium", "4hard", "8s", "8medium", "8hard"]
DEFAULT_METHODS = ["q-none", "q-classic", "q-cp-ms", "q-cp-etr", "cp-greedy", "optimal"]
DEFAULT_SEEDS = 5
DEFAULT_EPISODES = 2000
DEFAULT_OUTPUT = ROOT / "experiment_results"

# Maps our method name -> (--agent, --shaping) arguments for main.py
METHOD_ARGS: dict[str, tuple[str, str | None]] = {
    "q-none": ("q", "none"),
    "q-classic": ("q", "classic"),
    "q-cp-ms": ("q", "cp-ms"),
    "q-cp-etr": ("q", "cp-etr"),
    "cp-greedy": ("cp_greedy", None),
    "optimal": ("optimal", None),
}

# Methods that include a Q-learning training phase
Q_METHODS = {"q-none", "q-classic", "q-cp-ms", "q-cp-etr"}

# Methods that require a Java CP server (must run sequentially for now)
JAVA_METHODS = {"q-cp-ms", "q-cp-etr", "cp-greedy"}

# Methods displayed as horizontal lines (no training curve)
HLINE_METHODS = {"cp-greedy", "optimal"}

# Human-readable labels for plots and table
METHOD_LABELS = {
    "q-none": "Q-Std None",
    "q-classic": "Q-Classic",
    "q-cp-ms": "Q-CP-MS",
    "q-cp-etr": "Q-CP-ETR",
    "cp-greedy": "CP-MS Greedy",
    "optimal": "Optimal",
}

# Consistent colour palette
METHOD_COLORS = {
    "q-none": "#e74c3c",
    "q-classic": "#3498db",
    "q-cp-ms": "#2ecc71",
    "q-cp-etr": "#9b59b6",
    "cp-greedy": "#f39c12",
    "optimal": "#1abc9c",
}

# Extra colours for budget variants
_EXTRA_COLORS = [
    "#1abc9c", "#d35400", "#8e44ad", "#16a085", "#c0392b",
    "#2980b9", "#27ae60", "#7f8c8d", "#2c3e50",
]

VALID_STRATEGIES = {"fail", "full-budget"}


def _parse_method(raw: str) -> tuple[str, int, str | None]:
    """
    Parse a method string like "q-cp-etr:b3:fail" or plain "q-cp-etr".
    Returns (base, budget, strategy).
    Raises ValueError on bad input.
    """
    parts = raw.split(":")
    base = parts[0]
    if base not in METHOD_ARGS:
        raise ValueError(
            f"Unknown base method '{base}'. Valid: {sorted(METHOD_ARGS)}"
        )
    if base == "optimal":
        return base, 0, None
    if len(parts) == 1:
        return base, 0, None
    if len(parts) == 3:
        b_part, s_part = parts[1], parts[2]
        if not b_part.startswith("b"):
            raise ValueError(f"Budget part must start with 'b' (e.g. 'b3'), got '{b_part}'")
        try:
            budget = int(b_part[1:])
        except ValueError:
            raise ValueError(f"Cannot parse budget from '{b_part}'")
        if s_part not in VALID_STRATEGIES:
            raise ValueError(f"Unknown strategy '{s_part}'. Valid: {sorted(VALID_STRATEGIES)}")
        return base, budget, s_part
    raise ValueError(f"Method must be 'base' or 'base:bN:strategy', got '{raw}'")


def _method_label(raw: str) -> str:
    base, budget, strategy = _parse_method(raw)
    lbl = METHOD_LABELS.get(base, base)
    if budget:
        lbl += f" b{budget}/{strategy}"
    return lbl


def _method_color(raw: str, extra_iter) -> str:
    base, budget, _ = _parse_method(raw)
    if budget == 0 and base in METHOD_COLORS:
        return METHOD_COLORS[base]
    return next(extra_iter, "#95a5a6")


# ---------------------------------------------------------------------------
# Cache helpers  (experiment_results/cache/ — never touches results/)
# ---------------------------------------------------------------------------

def _cache_path(output_dir: Path, inst: str, meth: str, seed: int, episodes: int) -> Path:
    return output_dir / "cache" / f"{inst}_{meth}_seed{seed}_{episodes}eps.json"


def _load_cache(output_dir: Path, inst: str, meth: str, seed: int,
                episodes: int) -> dict | None:
    p = _cache_path(output_dir, inst, meth, seed, episodes)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        print(f"  [WARN] Corrupt cache entry {p.name}: {e} — will re-run")
        return None


def _save_cache(output_dir: Path, inst: str, meth: str, seed: int,
                episodes: int, entry: dict) -> None:
    p = _cache_path(output_dir, inst, meth, seed, episodes)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(entry, f, indent=2)


# ---------------------------------------------------------------------------
# Result extraction helpers
# ---------------------------------------------------------------------------

def _extract_entry(log: dict) -> dict | None:
    """Pull the fields we care about out of a main.py result JSON."""
    eval_log = log.get("evaluation_log", [])
    if not eval_log:
        return None
    last = eval_log[-1]
    return {
        "eval_log": eval_log,
        "final_sr": last.get("eval_success_rate", float("nan")),
        "final_dr": last.get("eval_avg_discounted_return", float("nan")),
    }


# ---------------------------------------------------------------------------
# Subprocess command builder
# ---------------------------------------------------------------------------

def _build_cmd(meth: str, inst: str, episodes: int, seed: int, tmp_dir: str,
               port: int = 12345, no_compile: bool = False) -> list[str]:
    base, budget, strategy = _parse_method(meth)
    agent, shaping = METHOD_ARGS[base]
    cmd = [
        sys.executable, str(ROOT / "main.py"),
        "--instance", inst,
        "--agent", agent,
        "--seed", str(seed),
        "--results-dir", tmp_dir,
    ]
    # optimal ne prend pas --episodes (il utilise config.EVAL_EPISODES)
    if base != "optimal":
        cmd += ["--episodes", str(episodes)]
    if shaping is not None:
        cmd += ["--shaping", shaping]
    if budget:
        cmd += ["--budget", str(budget)]
    if strategy:
        cmd += ["--noslip-strategy", strategy]
    if base in JAVA_METHODS:
        cmd += ["--port", str(port)]
        if no_compile:
            cmd += ["--no-compile"]
    return cmd


# ---------------------------------------------------------------------------
# Run all experiments
# ---------------------------------------------------------------------------

def _run_one(inst: str, meth: str, seed: int, episodes: int,
             output_dir: Path, tag: str, print_lock: threading.Lock,
             port: int = 12345, no_compile: bool = False,
             progress_bar: "tqdm | None" = None) -> tuple[tuple, dict | None]:
    """Execute a single (inst, meth, seed) run and return ((inst, meth, seed), entry)."""
    tmp_dir = tempfile.mkdtemp(prefix="fl_run_")

    def _done(msg: str, entry):
        with print_lock:
            tqdm.write(msg)
            if progress_bar is not None:
                progress_bar.update(1)
        return (inst, meth, seed), entry

    try:
        cmd = _build_cmd(meth, inst, episodes, seed, tmp_dir, port=port, no_compile=no_compile)
        result = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=7200,
        )
        if result.returncode != 0:
            lines = (result.stdout + result.stderr).strip().splitlines()[-20:]
            msg = f"  [ERROR] {tag} — exit code {result.returncode}\n" + "\n".join(f"    {l}" for l in lines)
            return _done(msg, None)

        result_files = list(Path(tmp_dir).rglob("*_log.json"))
        if not result_files:
            return _done(f"  [WARN]  {tag} — run OK but no result log found", None)

        with open(result_files[0]) as f:
            raw = json.load(f)

        entry = _extract_entry(raw)
        if entry is None:
            return _done(f"  [WARN]  {tag} — result log has no evaluation_log", None)

        _save_cache(output_dir, inst, meth, seed, episodes, entry)
        return _done(f"  [OK]    {tag}", entry)

    except subprocess.TimeoutExpired:
        return _done(f"  [ERROR] {tag} — timeout (>7200 s)", None)
    except Exception as exc:
        with print_lock:
            tqdm.write(f"  [ERROR] {tag} — {exc}")
            traceback.print_exc()
            if progress_bar is not None:
                progress_bar.update(1)
        return (inst, meth, seed), None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _precompile_java(print_lock: threading.Lock) -> bool:
    """Run 'mvn compile' + generate classpath file once. Returns True on success."""
    cp_dir = ROOT / "MiniCPBP"
    if not cp_dir.is_dir():
        cp_dir = ROOT / "java"
    pom = cp_dir / "pom.xml"
    if not pom.is_file():
        with print_lock:
            print("[ERROR] pom.xml not found — cannot pre-compile Java.")
        return False
    mvn_exec = "mvn.cmd" if sys.platform == "win32" else "mvn"
    cp_file = cp_dir / "target" / "java_classpath.txt"

    # Step 1: compile
    cmd_compile = [mvn_exec, "-f", str(pom), "compile", "-Dexec.cleanupDaemonThreads=false"]
    with print_lock:
        print(f"[PRE-COMPILE] {' '.join(cmd_compile)}")
    result = subprocess.run(cmd_compile, cwd=str(ROOT), capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        with print_lock:
            print("[ERROR] Maven pre-compile failed:")
            for line in (result.stdout + result.stderr).strip().splitlines()[-30:]:
                print(f"  {line}")
        return False

    # Step 2: generate classpath file for direct java invocation
    cmd_cp = [mvn_exec, "-f", str(pom),
              "dependency:build-classpath",
              f"-Dmdep.outputFile={cp_file}",
              "-q"]
    result2 = subprocess.run(cmd_cp, cwd=str(cp_dir), capture_output=True, text=True, timeout=120)
    if result2.returncode != 0 or not cp_file.exists():
        with print_lock:
            print("[ERROR] Maven dependency:build-classpath failed:")
            for line in (result2.stdout + result2.stderr).strip().splitlines()[-10:]:
                print(f"  {line}")
        return False

    with print_lock:
        print("[PRE-COMPILE] Done.\n")
    return True


def run_all(instances, methods, seeds, episodes, output_dir, force,
            workers: int = 1, base_port: int = 12345) -> dict:
    """
    Launch main.py for each (instance, method, seed) not already cached.

    workers > 1 enables parallel execution for all methods:
    - Non-Java (q-none, q-classic, optimal) : partagent les workers
    - Java (q-cp-ms, q-cp-etr, cp-greedy)  : chaque worker reçoit un port
      dédié du pool [base_port, base_port + workers - 1]
    """
    runs = [(inst, meth, seed)
            for inst in instances
            for meth in methods
            for seed in seeds]

    total = len(runs)
    w = len(str(total))
    data: dict[tuple, dict | None] = {}
    print_lock = threading.Lock()

    # Pool de ports pour les workers Java
    port_pool: queue.Queue[int] = queue.Queue()
    for i in range(workers):
        port_pool.put(base_port + i)

    # Pre-compile Java once if workers > 1 and Java methods are present
    needs_java = any(_parse_method(m)[0] in JAVA_METHODS for m in methods)
    no_compile = False
    if workers > 1 and needs_java:
        if not _precompile_java(print_lock):
            print("[WARN] Pre-compile failed; workers will each compile (may conflict).")
        else:
            no_compile = True

    start_time = time.time()

    print(f"\n{'=' * 60}")
    print(f"  Experiment  |  {total} runs  |  episodes={episodes}  |  workers={workers}")
    print(f"{'=' * 60}\n")

    # Separate skippable runs and pending runs
    skipped = 0
    pending = []
    for idx, (inst, meth, seed) in enumerate(runs, 1):
        tag = f"[{idx:{w}}/{total}] {inst:<10} {meth:<16} seed={seed}"
        cached = _load_cache(output_dir, inst, meth, seed, episodes)
        if cached is not None and not force:
            tqdm.write(f"{tag}  --> SKIP  (cached)")
            data[(inst, meth, seed)] = cached
            skipped += 1
        else:
            pending.append((inst, meth, seed, tag))

    pbar = tqdm(total=total, initial=skipped, desc="Progress",
                unit="run", dynamic_ncols=True, leave=True)

    def _worker_with_port(inst, meth, seed, tag):
        """Wrapper qui gère l'acquisition/libération du port pour les runs Java."""
        base = _parse_method(meth)[0]
        if base in JAVA_METHODS:
            port = port_pool.get()
            try:
                return _run_one(inst, meth, seed, episodes, output_dir, tag, print_lock,
                                port=port, no_compile=no_compile, progress_bar=pbar)
            finally:
                port_pool.put(port)
        else:
            return _run_one(inst, meth, seed, episodes, output_dir, tag, print_lock,
                            progress_bar=pbar)

    if pending and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_worker_with_port, inst, meth, seed, tag): (inst, meth, seed)
                for inst, meth, seed, tag in pending
            }
            for future in as_completed(futures):
                try:
                    key, entry = future.result()
                    data[key] = entry
                except Exception as exc:
                    key = futures[future]
                    with print_lock:
                        tqdm.write(f"  [ERROR] {key}: {exc}")
                        pbar.update(1)
                    data[key] = None
    else:
        for inst, meth, seed, tag in pending:
            tqdm.write(f"{tag}  --> running ...", )
            _, entry = _worker_with_port(inst, meth, seed, tag)
            data[(inst, meth, seed)] = entry

    pbar.close()

    elapsed = time.time() - start_time
    minutes, seconds = divmod(int(elapsed), 60)
    valid_entries = sum(1 for v in data.values() if v is not None)
    succeeded = valid_entries - skipped
    failed = total - valid_entries
    print(f"\nDone: {succeeded} succeeded, {skipped} skipped, {failed} failed  "
          f"| Durée totale : {minutes}m{seconds:02d}s\n")
    return data


# ---------------------------------------------------------------------------
# Load data from cache only (--plots-only)
# ---------------------------------------------------------------------------

def load_from_cache(instances, methods, seeds, episodes, output_dir) -> dict:
    data: dict[tuple, dict | None] = {}
    for inst in instances:
        for meth in methods:
            for seed in seeds:
                data[(inst, meth, seed)] = _load_cache(output_dir, inst, meth, seed, episodes)
    found = sum(1 for v in data.values() if v is not None)
    print(f"Found {found} cache entries.\n")
    return data


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def make_summary_table(data: dict, instances, methods, seeds, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[tuple, dict | None] = {}
    for inst in instances:
        for meth in methods:
            srs = [data[(inst, meth, s)]["final_sr"]
                   for s in seeds if data.get((inst, meth, s)) is not None]
            drs = [data[(inst, meth, s)]["final_dr"]
                   for s in seeds if data.get((inst, meth, s)) is not None]
            if srs:
                summary[(inst, meth)] = {
                    "sr_mean": float(np.mean(srs)),
                    "sr_std": float(np.std(srs)),
                    "dr_mean": float(np.mean(drs)),
                    "dr_std": float(np.std(drs)),
                    "n": len(srs),
                }
            else:
                summary[(inst, meth)] = None

    csv_path = output_dir / "summary_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["instance"]
            + [f"{m}_SR_mean" for m in methods]
            + [f"{m}_SR_std" for m in methods]
            + [f"{m}_DR_mean" for m in methods]
            + [f"{m}_DR_std" for m in methods]
        )
        for inst in instances:
            row = [inst]
            for m in methods:
                v = summary.get((inst, m))
                row.append(f"{v['sr_mean']:.4f}" if v else "")
            for m in methods:
                v = summary.get((inst, m))
                row.append(f"{v['sr_std']:.4f}" if v else "")
            for m in methods:
                v = summary.get((inst, m))
                row.append(f"{v['dr_mean']:.4f}" if v else "")
            for m in methods:
                v = summary.get((inst, m))
                row.append(f"{v['dr_std']:.4f}" if v else "")
            writer.writerow(row)
    print(f"Summary CSV saved to {csv_path}")

    col_w = 18
    labels = [_method_label(m) for m in methods]
    sep = "-" * (12 + col_w * len(methods) * 2)

    def _cell(v, mk, sk):
        return f"{v[mk]:.3f}±{v[sk]:.3f}" if v else "–"

    print("\n" + sep)
    header = f"{'Instance':<12}"
    for lbl in labels:
        header += f"{lbl + ' SR':>{col_w}}"
    for lbl in labels:
        header += f"{lbl + ' DR':>{col_w}}"
    print(header)
    print(sep)
    for inst in instances:
        row = f"{inst:<12}"
        for m in methods:
            row += f"{_cell(summary.get((inst, m)), 'sr_mean', 'sr_std'):>{col_w}}"
        for m in methods:
            row += f"{_cell(summary.get((inst, m)), 'dr_mean', 'dr_std'):>{col_w}}"
        print(row)
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Learning curves
# ---------------------------------------------------------------------------

def make_learning_curves(data: dict, instances, methods, seeds, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-darkgrid")

    for inst in instances:
        fig, ax = plt.subplots(figsize=(9, 5))
        has_data = False

        extra_iter = iter(_EXTRA_COLORS)
        for meth in methods:
            color = _method_color(meth, extra_iter)
            label = _method_label(meth)
            base, _, _ = _parse_method(meth)

            if base in HLINE_METHODS:
                srs = [data[(inst, meth, s)]["final_sr"]
                       for s in seeds if data.get((inst, meth, s)) is not None]
                if srs:
                    mean_sr = float(np.mean(srs))
                    ax.axhline(mean_sr, linestyle="--", color=color, linewidth=1.8,
                               label=f"{label} (SR={mean_sr:.3f})")
                    has_data = True
                continue

            if base not in Q_METHODS:
                continue

            seed_curves: dict[int, tuple] = {}
            for seed in seeds:
                entry = data.get((inst, meth, seed))
                if entry is None:
                    continue
                eps = [e["training_episode"] for e in entry["eval_log"]]
                srs = [e.get("eval_success_rate", float("nan")) for e in entry["eval_log"]]
                seed_curves[seed] = (eps, srs)

            if not seed_curves:
                continue

            all_ep_sets = [set(v[0]) for v in seed_curves.values()]
            common_eps = sorted(set.intersection(*all_ep_sets)) if all_ep_sets else []
            if not common_eps:
                common_eps = sorted(set.union(*all_ep_sets))

            matrix = []
            for eps, srs in seed_curves.values():
                lookup = dict(zip(eps, srs))
                matrix.append([lookup.get(e, float("nan")) for e in common_eps])
            mat = np.array(matrix, dtype=float)
            mean_c = np.nanmean(mat, axis=0)
            std_c = np.nanstd(mat, axis=0)

            ax.plot(common_eps, mean_c, color=color, linewidth=2, label=label)
            ax.fill_between(common_eps, mean_c - std_c, mean_c + std_c,
                            color=color, alpha=0.18)
            has_data = True

        if has_data:
            ax.set_title(f"Learning curves — {inst}", fontsize=13, fontweight="bold")
            ax.set_xlabel("Training episode")
            ax.set_ylabel("Eval success rate")
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=9)
            plt.tight_layout()
            out = output_dir / f"learning_curve_{inst}.png"
            plt.savefig(out, dpi=150)
            print(f"Learning curve saved: {out}")
        else:
            print(f"[WARN] No data for instance {inst}, skipping plot.")
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help=f"Training episodes per run (default: {DEFAULT_EPISODES})")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                        help=f"Number of seeds, starting at 1 (default: {DEFAULT_SEEDS})")
    parser.add_argument("--instances", nargs="+", default=DEFAULT_INSTANCES,
                        metavar="ID", help="Instance IDs to benchmark")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                        metavar="METHOD",
                        help=(
                            "Methods to benchmark. Base: q-none q-classic q-cp-ms q-cp-etr cp-greedy. "
                            "With budget: base:bN:strategy  e.g. q-cp-etr:b3:fail"
                        ))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Root output directory (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (default: 1 = sequential). "
                             "Java methods utilisent un port dédié par worker.")
    parser.add_argument("--base-port", type=int, default=12345,
                        help="Premier port TCP pour les serveurs Java (default: 12345). "
                             "Worker i utilise base_port + i.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if a cache entry already exists")
    parser.add_argument("--plots-only", action="store_true",
                        help="Skip running; only regenerate plots/table from cache")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = list(range(1, args.seeds + 1))

    # Validate method strings early
    try:
        for m in args.methods:
            _parse_method(m)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if not args.plots_only:
        data = run_all(
            instances=args.instances,
            methods=args.methods,
            seeds=seeds,
            episodes=args.episodes,
            output_dir=args.output_dir,
            force=args.force,
            workers=args.workers,
            base_port=args.base_port,
        )
    else:
        print("--plots-only: loading from cache …")
        data = load_from_cache(args.instances, args.methods, seeds,
                               args.episodes, args.output_dir)

    make_summary_table(data, args.instances, args.methods, seeds,
                       output_dir=args.output_dir)
    make_learning_curves(data, args.instances, args.methods, seeds,
                         output_dir=args.output_dir)
    print("All done.")


if __name__ == "__main__":
    main()
