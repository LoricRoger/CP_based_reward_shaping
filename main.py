# FL/main.py
import argparse
import datetime
import json
import os
import subprocess
import time
import random
import numpy as np

import config
import environment
import utils
import q_learning_standard
import q_learning_cp
import heuristic_agents

def main():
    parser = argparse.ArgumentParser(description="Train/Evaluate FrozenLake agents.")
    parser.add_argument('--instance', type=str, required=True, help="Instance ID from instances.json")
    parser.add_argument('--agent', type=str, choices=['q', 'optimal', 'cp_greedy'], default='q', help="Agent type")
    parser.add_argument('--shaping', type=str, choices=['none', 'classic', 'cp-ms', 'cp-etr'],
                        default='none', help="Reward shaping for Q-learning")
    parser.add_argument('--budget', type=int, default=0,
                        help="Max no-slip actions per episode (0 = no budget constraint)")
    parser.add_argument('--noslip-strategy', type=str,
                        choices=['fail', 'full-budget'],
                        default='fail',
                        help=(
                            "No-slip action strategy when budget > 0 (default: fail).\n"
                            "  fail        : curriculum croissant, terminaison si budget épuisé, Q init -0.1\n"
                            "  full-budget : budget max dès le début, terminaison si dépassé, Q init 0.0"
                        ))
    parser.add_argument('--episodes', type=int, default=500, help="Training episodes for Q-learning")
    parser.add_argument('--seed', type=int, default=None, help="Random seed to override config.seed_value")
    parser.add_argument('--port', type=int, default=12345, help="TCP port for the Java CP server (default: 12345)")
    parser.add_argument('--no-compile', action='store_true',
                        help="Skip 'mvn compile' step (use when Java is already compiled)")
    parser.add_argument('--results-dir', type=str, default=None, help="Directory to store result logs")
    args = parser.parse_args()

    if args.seed is not None:
        current_seed = args.seed
        print(f"Using seed from command line: {current_seed}")
    else:
        current_seed = config.seed_value
        print(f"Using seed from config file: {current_seed}")
    random.seed(current_seed)
    np.random.seed(current_seed)

    instances_path = "instances.json"
    try:
        with open(instances_path, 'r') as f:
            instances = json.load(f)
        print(f"Loaded configurations from {instances_path}")
    except (FileNotFoundError, json.JSONDecodeError) as err:
        print(f"ERROR loading {instances_path}: {err}")
        return
    instance_id = args.instance
    if instance_id not in instances:
        print(f"ERROR: Instance '{instance_id}' not found.")
        return
    instance = instances[instance_id]
    try:
        size = instance['size']
        holes = instance['holes']
        goal = instance['goal']
        slippery = instance['slippery']
        max_steps_config = instance['max_steps']
        map_name = instance.get('map_name', f"{size}x{size}")
        desc = instance.get('desc', None)
        optimal_policy_data = instance.get('op', None)
        description = instance.get('description', '')
        print(f"Instance: '{instance_id}' ({description}), Agent: {args.agent}, Seed: {current_seed}")
        if args.agent == 'q':
            print(f"  Shaping: {args.shaping}, Episodes: {args.episodes}")
    except KeyError as e:
        print(f"ERROR: Missing key {e} in instance '{instance_id}'.")
        return
    if args.agent == 'optimal' and not optimal_policy_data:
        print(f"ERROR: Optimal policy data missing for '{instance_id}'.")
        return

    results_dir = args.results_dir if args.results_dir is not None else "results"
    plots_dir = "plots"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    agent_log_name = args.agent
    if args.agent == 'q':
        agent_log_name = f"q_{args.shaping}"
        if args.budget > 0:
            agent_log_name += f"_b{args.budget}_{args.noslip_strategy}"
    log_base_name = f"{agent_log_name}_{instance_id}_{args.episodes if args.agent == 'q' else config.EVAL_EPISODES}eps_seed{current_seed}_{timestamp}"
    log_file = os.path.join(results_dir, f"{log_base_name}_log.json")
    java_stdout_log = os.path.join(results_dir, f"{log_base_name}_java_stdout.log")
    java_stderr_log = os.path.join(results_dir, f"{log_base_name}_java_stderr.log")

    print("Creating environment...")
    env = environment.create_environment(map_name=map_name, is_slippery=slippery, render_mode=None,
                                         desired_max_steps=max_steps_config, desc=desc, budget=args.budget)
    if env is None:
        print("ERROR: Failed to create environment.")
        return
    try:
        env.reset(seed=current_seed)
        env.action_space.seed(current_seed)
        print(f"Environment reset and action space seeded with seed: {current_seed}")
    except Exception as e:
        print(f"Warning: Could not fully seed environment: {e}")

    java_process = None
    cp_client = None
    final_evaluations = None

    if args.agent == 'optimal':
        # heuristic_agents.run_optimal_policy now returns a list of dictionaries
        final_evaluations = heuristic_agents.run_optimal_policy(env, optimal_policy_data, size, max_steps_config)
        # This format is expected by utils.save_results_log.
        utils.save_results_log({'episode_log': [], 'evaluation_log': final_evaluations}, log_file)

    elif args.agent == 'cp_greedy' or (args.agent == 'q' and args.shaping in ['cp-ms', 'cp-etr']):
        project_dir = os.path.dirname(os.path.abspath(__file__))
        cp_dir = os.path.join(project_dir, 'MiniCPBP')
        if not os.path.isdir(cp_dir):
            cp_dir = os.path.join(project_dir, 'java')
        if not os.path.isdir(cp_dir):
            print("ERROR: CP directory not found.")
            env.close()
            return
        pom = os.path.join(cp_dir, 'pom.xml')
        if not os.path.isfile(pom):
            print(f"ERROR: pom.xml not found in {cp_dir}.")
            env.close()
            return

        # Determine Java mode; args passés à Java : "mode budget port"
        if args.agent == 'cp_greedy' or args.shaping == 'cp-ms':
            mode_str = "MS"
        elif args.shaping == 'cp-etr':
            mode_str = "ETR"
        else:
            print(f"ERROR: Cannot determine Java mode for agent='{args.agent}', shaping='{args.shaping}'.")
            env.close()
            return
        java_args_list = [mode_str, str(args.budget), str(args.port)]

        if args.no_compile:
            # Lancer Java directement (sans Maven) pour éviter les conflits de JVM partagée
            # en mode parallèle. Le classpath est généré une fois par mvn dependency:build-classpath.
            cp_file = os.path.join(cp_dir, 'target', 'java_classpath.txt')
            if not os.path.isfile(cp_file):
                print(f"ERROR: Classpath file not found: {cp_file}. Run pre-compile first.")
                env.close()
                return
            with open(cp_file) as f:
                dep_cp = f.read().strip()
            classes_dir = os.path.join(cp_dir, 'target', 'classes')
            full_cp = classes_dir + os.pathsep + dep_cp
            cmd = ['java', '-cp', full_cp, 'minicpbp.examples.FrozenLakeCPService'] + java_args_list
        else:
            mvn_exec = 'mvn.cmd' if os.name == 'nt' else 'mvn'
            main_class_arg = '-Dexec.mainClass=minicpbp.examples.FrozenLakeCPService'
            exec_args = ' '.join(java_args_list)
            cmd = [mvn_exec, '-f', pom, 'compile', 'exec:java', '-Dexec.cleanupDaemonThreads=false',
                   main_class_arg, f'-Dexec.args={exec_args}']

        print(f"Starting Java server for CP agent: {' '.join(cmd)}")


        try:
            with open(java_stdout_log, 'w') as out, open(java_stderr_log, 'w') as err:
                java_process = subprocess.Popen(cmd, stdout=out, stderr=err, text=True)
            print("Waiting for Java server...")
            time.sleep(10)
            if java_process.poll() is not None:
                print(f"ERROR: Java server terminated (exit {java_process.poll()}).")
                _print_logs(java_stderr_log, java_stdout_log)
                env.close()
                return
            print("Java server presumed started.")
            cp_client = q_learning_cp.CPRewardClient(port=args.port)
            if not cp_client.connect():
                print("ERROR: Failed to connect to CP server.")
                _print_logs(java_stderr_log, java_stdout_log)
                _cleanup_cp(java_process, cp_client)
                env.close()
                return
            init_resp = cp_client.send_receive(f"INIT {instance_id}")
            if not init_resp.startswith("OK INIT"):
                _cleanup_cp(java_process, cp_client)
                env.close()
                raise RuntimeError(f"CP Server INIT failed: {init_resp}")
            print("CP Server INIT successful.")
        except Exception as e:
            print(f"ERROR starting Java/CP server: {e}")
            _print_logs(java_stderr_log, java_stdout_log)
            _cleanup_cp(java_process, cp_client)
            env.close()
            return

        if args.agent == 'cp_greedy':
            final_evaluations = heuristic_agents.run_cp_ms_greedy_agent(env, cp_client, config.EVAL_EPISODES,
                                                                        max_steps_config, size, env.action_space.n,
                                                                        instance_id)
            utils.save_results_log({'episode_log': [], 'evaluation_log': final_evaluations}, log_file)
            _cleanup_cp(java_process, cp_client)

        elif args.agent == 'q':  # CP shaping for Q-learning
            noslip_strat = args.noslip_strategy
            train_fn = lambda env, episode, maxSteps: q_learning_cp.train_q_learning_with_cp_shaping(
                env, cp_client, episode, maxSteps, args.shaping, size, holes, goal, instance_id,
                noslip_strategy_name=noslip_strat)
            _run_q_learning_session(env, train_fn, args, max_steps_config, log_file, java_process, cp_client,
                                    java_stdout_log, java_stderr_log, size=size, holes=holes, goal=goal)

    elif args.agent == 'q':  # Standard Q-learning (non-CP)
        train_fn = None
        current_shaping_name = args.shaping
        if args.shaping == 'none':
            train_fn = lambda e, ep, ms: q_learning_standard.train_q_learning(e, ep, ms, size, holes, goal,
                                                                              current_shaping_name, instance_id)
        elif args.shaping == 'classic':
            train_fn = lambda e, ep, ms: q_learning_standard.train_q_learning_with_classic_shaping(e, ep, ms, size,
                                                                                                   holes, goal,
                                                                                                   current_shaping_name,
                                                                                                   instance_id)
        else:
            print(f"ERROR: Invalid shaping '{args.shaping}' for non-CP Q-agent.")
            env.close()
            return
        _run_q_learning_session(env, train_fn, args, max_steps_config, log_file, size=size, holes=holes, goal=goal)
    else:
        print(f"ERROR: Unknown agent type '{args.agent}'")

    if env:
        env.close()
        print("Environment closed.")


def _run_q_learning_session(env, train_fn, args, max_steps_config, log_file, java_process=None, cp_client=None,
                            java_stdout_log=None, java_stderr_log=None, size=None, holes=None, goal=None):
    q_table, episodes, evaluations = None, [], []
    start_time = time.time()
    try:
        print(
            f"Starting Q-learning training for {args.episodes} episodes (Agent: {args.agent}, Shaping: {args.shaping})")
        q_table, episodes, evaluations = train_fn(env, args.episodes, max_steps_config)
    except Exception as e:
        print(f"ERROR during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        end_time = time.time()
        print(f"Training finished/interrupted. Duration: {end_time - start_time:.2f}s.")
        if java_process:
            _cleanup_cp(java_process, cp_client)
    if not episodes and not evaluations:
        print("No results from Q-learning to save/plot.")
    else:
        print("Saving Q-learning results log...")
        try:
            utils.save_results_log({'episode_log': episodes, 'evaluation_log': evaluations}, log_file)
        except Exception as e:
            print(f"Error saving results log: {e}")
        print("Plotting Q-learning results...")
        try:
            utils.plot_results([log_file], "plots")
            if q_table is not None:
                policy_label = args.shaping
                if args.budget > 0:
                    policy_label += f"_b{args.budget}_{args.noslip_strategy}"
                # Derive a unique stem from the log filename (already contains timestamp)
                log_stem = os.path.splitext(os.path.basename(log_file))[0]
                log_stem = log_stem[:-4] if log_stem.endswith("_log") else log_stem
                utils.visualize_policy(
                    q_table, size, holes, goal,
                    title=f"Politique finale — {policy_label} ({args.instance})",
                    save_path=os.path.join("plots", f"policy__{log_stem}.png")
                )
        except Exception as e:
            print(f"Error plotting results: {e}")
    print(f"\nQ-learning run completed. Log: {log_file}")
    if args.shaping in ['cp-ms', 'cp-etr'] and java_process is not None:
        if java_stdout_log and os.path.exists(java_stdout_log):
            print(f"  Java stdout: {java_stdout_log}")
        if java_stderr_log and os.path.exists(java_stderr_log):
            print(f"  Java stderr: {java_stderr_log}")


def _print_logs(stderr_path: str, stdout_path: str) -> None:
    print("\n--- Reading Java Server Logs ---")
    log_printed = False
    for path in (stderr_path, stdout_path):
        if path and os.path.exists(path):
            print(f"--- Contents of {os.path.basename(path)} ---")
            try:
                with open(path, 'r') as f:
                    content = f.read()
                    print(content if content else "[File is empty]")
                    log_printed = True
            except Exception as e:
                print(f"Error reading log file {path}: {e}")
    if not log_printed:
        print("No Java server log files found or they were empty.")
    print("--- End of Java Server Logs ---\n")


def _cleanup_cp(java_proc, client) -> None:
    if client and getattr(client, 'is_connected', False):
        print("Closing CP client connection...")
        client.close()
    if java_proc and java_proc.poll() is None:
        print("Terminating Java server process...")
        java_proc.terminate()
        try:
            java_proc.communicate(timeout=10)
            print("Java server terminated.")
        except subprocess.TimeoutExpired:
            print("Java server did not terminate gracefully, killing...")
            java_proc.kill()
            java_proc.communicate()
            print("Java server killed.")
        except Exception as e:
            print(f"Error during Java process cleanup: {e}")


if __name__ == '__main__':
    main()
