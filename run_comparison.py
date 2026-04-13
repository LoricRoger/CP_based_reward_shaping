"""
run_comparison.py — Benchmark / regression script for the FrozenLake CP reward-shaping
project. Verifies that a Java refactoring doesn't change results, and serves as a
general benchmark tool.

Usage:
    python run_comparison.py [options]

Options:
    --episodes N        Training episodes per run          (default: 2000)
    --seeds N           Number of seeds, starting at 1    (default: 5)
    --instances ID...   Instance IDs to benchmark         (default: 4s 4medium 4hard 8s 8medium 8hard)
    --methods M...      Methods to benchmark              (default: all five)
    --results-dir DIR   Where main.py writes JSONs        (default: ./results)
    --plots-dir DIR     Output directory for plots/table  (default: ./notebook_plots)
    --force             Re-run even if result JSON exists
    --plots-only        Skip running, only regenerate plots/table from existing JSONs

Methods:
    q-none      Q-learning, no reward shaping
    q-classic   Q-learning, classic potential shaping
    q-cp-ms     Q-learning, CP-MS shaping  (Java mode: MS)
    q-cp-etr    Q-learning, CP-ETR shaping (Java mode: ETR)
    cp-greedy   CP-MS greedy agent (no Q-learning)
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

DEFAULT_INSTANCES = ["4s", "4medium", "4hard", "8s", "8medium", "8hard"]
DEFAULT_METHODS = ["q-none", "q-classic", "q-cp-ms", "q-cp-etr", "cp-greedy"]
DEFAULT_SEEDS = 5
DEFAULT_EPISODES = 2000
DEFAULT_RESULTS = ROOT / "results"
DEFAULT_PLOTS = ROOT / "notebook_plots"

# Maps our method name -> (--agent, --shaping) arguments for main.py
METHOD_ARGS: dict[str, tuple[str, str | None]] = {
    "q-none": ("q", "none"),
    "q-classic": ("q", "classic"),
    "q-cp-ms": ("q", "cp-ms"),
    "q-cp-etr": ("q", "cp-etr"),
    "cp-greedy": ("cp_greedy", None),  # no --shaping argument
}

# Methods that include a Q-learning training phase
Q_METHODS = {"q-none", "q-classic", "q-cp-ms", "q-cp-etr"}

# Human-readable labels for plots and table
METHOD_LABELS = {
    "q-none": "Q-Std None",
    "q-classic": "Q-Classic",
    "q-cp-ms": "Q-CP-MS",
    "q-cp-etr": "Q-CP-ETR",
    "cp-greedy": "CP-MS Greedy",
}

# Consistent colour palette
METHOD_COLORS = {
    "q-none": "#e74c3c",
    "q-classic": "#3498db",
    "q-cp-ms": "#2ecc71",
    "q-cp-etr": "#9b59b6",
    "cp-greedy": "#f39c12",
}


# ---------------------------------------------------------------------------
# Helpers — log file discovery
# ---------------------------------------------------------------------------

def _log_file_pattern(method: str, instance: str, episodes: int, seed: int) -> re.Pattern:
    """
    Regex that matches the JSON log file produced by main.py for a given run.

    Filename format (from main.py):
        {agent_log_name}_{instance_id}_{episodes}eps_seed{seed}_{timestamp}_log.json
    where agent_log_name is "q_{shaping}" (for agent==q) or "cp_greedy".
    """
    agent, shaping = METHOD_ARGS[method]
    prefix = f"q_{shaping}" if agent == "q" else agent
    return re.compile(
        rf"^{re.escape(prefix)}_{re.escape(instance)}_{episodes}eps"
        rf"_seed{seed}_\d{{8}}_\d{{6}}_log\.json$"
    )


def find_existing_log(method: str, instance: str, episodes: int, seed: int,
                      results_dir: Path) -> Path | None:
    """Return path of an existing result JSON for this run, or None."""
    pat = _log_file_pattern(method, instance, episodes, seed)
    try:
        for p in results_dir.iterdir():
            if p.is_file() and pat.match(p.name):
                return p
    except FileNotFoundError:
        pass
    return None


# ---------------------------------------------------------------------------
# Helpers — subprocess command builder
# ---------------------------------------------------------------------------

def build_cmd(method: str, instance: str, episodes: int, seed: int) -> list[str]:
    """Build the command to run main.py for one (method, instance, seed) combination."""
    agent, shaping = METHOD_ARGS[method]
    cmd = [
        sys.executable, str(ROOT / "main.py"),
        "--instance", instance,
        "--agent", agent,
        "--seed", str(seed),
        "--episodes", str(episodes),
    ]
    if shaping is not None:
        cmd += ["--shaping", shaping]
    return cmd


# ---------------------------------------------------------------------------
# Helpers — JSON loading
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  [WARN] Could not read {path.name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Run all experiments
# ---------------------------------------------------------------------------

def run_all(instances, methods, seeds, episodes, results_dir, force) -> dict:
    """
    Launch main.py for every (instance, method, seed) that doesn't already have
    a result JSON.  Returns:
        done_paths[(instance, method, seed)] = Path | None
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    runs = [(inst, meth, seed)
            for inst in instances
            for meth in methods
            for seed in seeds]

    total = len(runs)
    skipped = 0
    done_paths: dict[tuple, Path | None] = {}

    print(f"\n{'=' * 60}")
    print(f"  Benchmark  |  {total} runs  |  episodes={episodes}")
    print(f"{'=' * 60}\n")

    for idx, (inst, meth, seed) in enumerate(runs, 1):
        w = len(str(total))
        tag = f"[{idx:{w}}/{total}] {inst:<10} {meth:<12} seed={seed}"

        existing = find_existing_log(meth, inst, episodes, seed, results_dir)
        if existing and not force:
            print(f"{tag}  --> SKIP  ({existing.name})")
            done_paths[(inst, meth, seed)] = existing
            skipped += 1
            continue

        print(f"{tag}  --> running ...", flush=True)
        cmd = build_cmd(meth, inst, episodes, seed)
        try:
            result = subprocess.run(
                cmd,
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=7200,
            )
            if result.returncode != 0:
                print(f"  [ERROR] exit code {result.returncode}")
                for line in (result.stdout + result.stderr).strip().splitlines()[-20:]:
                    print(f"    {line}")
                done_paths[(inst, meth, seed)] = None
            else:
                fresh = find_existing_log(meth, inst, episodes, seed, results_dir)
                if fresh:
                    print(f"  [OK]    {fresh.name}")
                else:
                    print(f"  [WARN]  Run succeeded but no log found in {results_dir}")
                done_paths[(inst, meth, seed)] = fresh

        except subprocess.TimeoutExpired:
            print(f"  [ERROR] timeout (>7200 s)")
            done_paths[(inst, meth, seed)] = None
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            traceback.print_exc()
            done_paths[(inst, meth, seed)] = None

    succeeded = sum(1 for v in done_paths.values() if v is not None)
    failed = total - succeeded - skipped
    print(f"\nDone: {succeeded} succeeded, {skipped} skipped, {failed} failed\n")
    return done_paths


# ---------------------------------------------------------------------------
# Collect results from JSONs
# ---------------------------------------------------------------------------

def collect_results(done_paths: dict, instances, methods, seeds) -> dict:
    """
    Parse all result JSONs.  Returns:
        data[(inst, meth, seed)] = {
            'eval_log':  list of eval-checkpoint dicts,
            'final_sr':  float,
            'final_dr':  float,
        }
    """
    data = {}
    for (inst, meth, seed), path in done_paths.items():
        if path is None:
            continue
        log = load_json(path)
        if log is None:
            continue
        eval_log = log.get("evaluation_log", [])
        if not eval_log:
            continue
        last = eval_log[-1]
        data[(inst, meth, seed)] = {
            "eval_log": eval_log,
            "final_sr": last.get("eval_success_rate", float("nan")),
            "final_dr": last.get("eval_avg_discounted_return", float("nan")),
        }
    return data


# ---------------------------------------------------------------------------
# Summary table (Table 2 style)
# ---------------------------------------------------------------------------

def make_summary_table(data: dict, instances, methods, seeds, output_dir: Path) -> None:
    """
    Print and save a Table-2-style summary:
    mean ± std of SR and DR over seeds, per (instance × method).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate
    summary: dict[tuple, dict | None] = {}
    for inst in instances:
        for meth in methods:
            srs = [data[(inst, meth, s)]["final_sr"]
                   for s in seeds if (inst, meth, s) in data]
            drs = [data[(inst, meth, s)]["final_dr"]
                   for s in seeds if (inst, meth, s) in data]
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

    # CSV
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

    # Console table
    col_w = 18
    labels = [METHOD_LABELS[m] for m in methods]
    sep = "-" * (12 + col_w * len(methods) * 2)

    def _cell(v, key_mean, key_std):
        if v is None:
            return "–"
        return f"{v[key_mean]:.3f}±{v[key_std]:.3f}"

    print("\n" + sep)
    print(f"{'Instance':<12}", end="")
    for lbl in labels:
        print(f"{lbl + ' SR':>{col_w}}", end="")
    for lbl in labels:
        print(f"{lbl + ' DR':>{col_w}}", end="")
    print()
    print(sep)
    for inst in instances:
        print(f"{inst:<12}", end="")
        for m in methods:
            print(f"{_cell(summary.get((inst, m)), 'sr_mean', 'sr_std'):>{col_w}}", end="")
        for m in methods:
            print(f"{_cell(summary.get((inst, m)), 'dr_mean', 'dr_std'):>{col_w}}", end="")
        print()
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Learning curves (Figure 3 style)
# ---------------------------------------------------------------------------

def make_learning_curves(data: dict, instances, methods, seeds, output_dir: Path) -> None:
    """
    For each instance, plot eval success rate vs. training episode.
    - Q-learning methods: mean ± 1 std dev band over seeds.
    - cp-greedy: horizontal dashed line at mean final SR (no training curve).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-darkgrid")

    for inst in instances:
        fig, ax = plt.subplots(figsize=(9, 5))
        has_data = False

        for meth in methods:
            color = METHOD_COLORS.get(meth)
            label = METHOD_LABELS[meth]

            # ── CP-greedy: horizontal line only ──────────────────────────────
            if meth == "cp-greedy":
                srs = [data[(inst, meth, s)]["final_sr"]
                       for s in seeds if (inst, meth, s) in data]
                if srs:
                    mean_sr = float(np.mean(srs))
                    ax.axhline(mean_sr, linestyle="--", color=color, linewidth=1.8,
                               label=f"{label} (SR={mean_sr:.3f})")
                    has_data = True
                continue

            if meth not in Q_METHODS:
                continue

            # ── Q-learning: mean ± std band ───────────────────────────────────
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

            # Align curves on a common episode grid (intersection to avoid NaN gaps)
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
            ax.fill_between(common_eps,
                            mean_c - std_c,
                            mean_c + std_c,
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
                        choices=list(METHOD_ARGS.keys()),
                        help="Methods to benchmark")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS,
                        help="Directory where main.py writes result JSONs")
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS,
                        help="Output directory for plots and summary table")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if a result JSON already exists")
    parser.add_argument("--plots-only", action="store_true",
                        help="Skip running experiments; only regenerate plots/table "
                             "from existing result JSONs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = list(range(1, args.seeds + 1))

    if not args.plots_only:
        done_paths = run_all(
            instances=args.instances,
            methods=args.methods,
            seeds=seeds,
            episodes=args.episodes,
            results_dir=args.results_dir,
            force=args.force,
        )
    else:
        print("--plots-only: scanning existing results …")
        done_paths = {}
        for inst in args.instances:
            for meth in args.methods:
                for seed in seeds:
                    p = find_existing_log(meth, inst, args.episodes, seed, args.results_dir)
                    done_paths[(inst, meth, seed)] = p
        found = sum(1 for v in done_paths.values() if v)
        print(f"Found {found} existing log files.\n")

    data = collect_results(done_paths, args.instances, args.methods, seeds)

    make_summary_table(
        data, args.instances, args.methods, seeds,
        output_dir=args.plots_dir,
    )
    make_learning_curves(
        data, args.instances, args.methods, seeds,
        output_dir=args.plots_dir,
    )
    print("All done.")


if __name__ == "__main__":
    main()
