# FL/utils.py
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import re
import config
from tqdm import tqdm

# Seeds are set once in main.py after parsing --seed. Do not re-seed here
# (module-level seeding would override the CLI seed).


def save_q_table(q_table, filename="results/q_table.npy"):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.save(filename, q_table)
    except Exception as e:
        print(f"Error saving Q-table to {filename}: {e}")


def load_q_table(filename="results/q_table.npy"):
    try:
        q_table = np.load(filename)
        print(f"Q-table loaded from {filename}")
        return q_table
    except FileNotFoundError:
        print(f"Error: Q-table file not found at {filename}")
        return None
    except Exception as e:
        print(f"Error loading Q-table from {filename}: {e}")
        return None


def save_q_table_csv(q_table, filename="results/q_table.csv"):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        n_actions = q_table.shape[1]
        action_names = {0: 'L', 1: 'D', 2: 'R', 3: 'U',
                        4: 'L_ns', 5: 'D_ns', 6: 'R_ns', 7: 'U_ns'}
        header = ",".join(f"Action_{i}({action_names.get(i, str(i))})" for i in range(n_actions))
        np.savetxt(filename, q_table, delimiter=",", fmt='%.6f', header=header, comments='')
        print(f"Q-table saved to {filename} (CSV format)")
    except Exception as e:
        print(f"Error saving Q-table to CSV {filename}: {e}")


def save_results_log(log_data, filename="results/training_log.json"):
    try:
        log_dir = os.path.dirname(filename)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(log_data, f, indent=4)
        print(f"Results log saved to {filename}")
    except IOError as e:
        print(f"Error writing results log to {filename}: {e}")
    except TypeError as e:
        print(f"Error serializing data for results log {filename}: {e}")
    except Exception as e:
        print(f"Unexpected error saving results log to {filename}: {e}")


def load_results_log(filename="results/training_log.json"):
    try:
        with open(filename, 'r') as f:
            log_data = json.load(f)
        print(f"Results log loaded from {filename}")
        return log_data
    except FileNotFoundError:
        print(f"Error: Results log file not found at {filename}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from {filename}: {e}")
        return None
    except Exception as e:
        print(f"Error loading results log from {filename}: {e}")
        return None


def evaluate_agent(env, q_table, max_steps, eval_episodes=config.EVAL_EPISODES):
    """
    Evaluate agent policy. Calculates success rate, avg env return (undiscounted),
    and avg discounted return.
    """
    if q_table is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    wins = 0
    total_undiscounted_return = 0.0
    total_discounted_return = 0.0
    success_steps = []
    failure_steps = []
    gamma = config.DISCOUNT_FACTOR

    if eval_episodes <= 0:
        print("WARN: eval_episodes is zero or negative.")
        return 0.0, 0.0, 0.0, 0.0, 0.0

    for episode_idx in tqdm(range(eval_episodes)):
        state, info = env.reset()
        episode_undiscounted_return = 0.0
        episode_discounted_return = 0.0
        done = False
        terminated = False
        truncated = False
        step_count = 0
        reward = 0.0  # Initialize reward for the episode scope

        while not done:
            if step_count >= max_steps:
                truncated = True
                done = True
                break
            if not (0 <= state < q_table.shape[0]):
                print(f"ERROR: Invalid state {state} during evaluation (Episode {episode_idx}).")
                terminated = True
                done = True
                reward = 0.0  # Ensure reward is non-positive on error
                break
            action = int(np.argmax(q_table[state]))
            try:
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            except Exception as err:
                print(f"ERROR during evaluation step: {err} (Episode {episode_idx}).")
                terminated = True
                done = True
                reward = 0.0  # Ensure reward is non-positive on error
                break
            episode_undiscounted_return += reward
            state = next_state
            step_count += 1
        total_undiscounted_return += episode_undiscounted_return
        if terminated and reward == 1.0:
            wins += 1
            success_steps.append(step_count)
            episode_discounted_return = (gamma ** (step_count - 1)) * 1.0
        else:
            failure_steps.append(step_count)
            episode_discounted_return = 0.0
        total_discounted_return += episode_discounted_return
    success_rate = float(wins / eval_episodes)
    avg_undiscounted_return = float(total_undiscounted_return / eval_episodes)
    avg_discounted_return = float(total_discounted_return / eval_episodes)
    avg_steps_success = float(sum(success_steps) / len(success_steps)) if success_steps else 0.0
    avg_steps_failure = float(sum(failure_steps) / len(failure_steps)) if failure_steps else 0.0
    return success_rate, avg_undiscounted_return, avg_discounted_return, avg_steps_success, avg_steps_failure


def get_policy_grid_from_q_table(q_table, size, holes, goal):
    """ Generates a grid representation of the greedy policy derived from a Q-table. """
    policy_grid_rows = []
    try:
        terminals = set(holes) | {goal}
        action_chars = {0: 'L', 1: 'D', 2: 'R', 3: 'U',
                        4: 'L*', 5: 'D*', 6: 'R*', 7: 'U*'}  # * = no-slip
        num_states = q_table.shape[0]
        if num_states != size * size:
            print(f"ERROR: Q-table size {num_states} doesn't match grid {size}x{size}.")
            return []
        for r in range(size):
            row_str = ""
            for c in range(size):
                state = r * size + c
                if state in terminals:
                    row_str += "N"
                else:
                    if state >= num_states:
                        print(f"WARN: State {state} out of Q-table bounds during policy gen.")
                        row_str += "?"
                        continue
                    best_action = np.argmax(q_table[state])
                    best_action_char = action_chars.get(best_action, '?')
                    row_str += best_action_char
            policy_grid_rows.append(row_str)
    except Exception as e:
        print(f"ERROR generating policy grid: {e}")
        return []
    return policy_grid_rows


def visualize_policy(q_table, size, holes, goal, title="Greedy Policy", save_path=None):
    """
    Visualise la politique greedy issue de la Q-table.
    - Toutes les cases : gris clair
    - Start            : bleu
    - Trous            : noir, croix blanche
    - Goal             : vert, étoile blanche
    - Flèches normales (actions 0-3) : noires
    - Flèches no-slip  (actions 4-7) : oranges
    """
    action_arrows = {0: '←', 1: '↓', 2: '→', 3: '↑',
                     4: '←', 5: '↓', 6: '→', 7: '↑'}

    fig, ax = plt.subplots(figsize=(size * 1.5, size * 1.5))
    hole_set = set(holes)

    for r in range(size):
        for c in range(size):
            state = r * size + c

            # Couleur de la case
            if state in hole_set:
                case_color = '#2d2d2d'
            elif state == goal:
                case_color = '#2ecc71'
            elif state == 0:
                case_color = '#3498db'
            else:
                case_color = '#ecf0f1'  # Gris clair pour tout le reste

            rect = plt.Rectangle([c, size - r - 1], 1, 1,
                                 facecolor=case_color, edgecolor='#bdc3c7', linewidth=1.5)
            ax.add_patch(rect)

            # Texte par dessus
            if state in hole_set:
                ax.text(c + 0.5, size - r - 0.5, 'X',
                        ha='center', va='center', fontsize=20,
                        color='white', fontweight='bold')
            elif state == goal:
                ax.text(c + 0.5, size - r - 0.5, 'G',
                        ha='center', va='center', fontsize=20,
                        color='white', fontweight='bold')
            else:
                best_action = int(np.argmax(q_table[state]))
                text = action_arrows.get(best_action, '?')
                # Couleur de la flèche selon slip ou no-slip
                if best_action >= 4:
                    arrow_color = '#e67e22'  # Orange : no-slip
                elif state == 0:
                    arrow_color = 'white'  # Blanc sur fond bleu
                else:
                    arrow_color = '#2c3e50'  # Noir : slip normal

                ax.text(c + 0.5, size - r - 0.5, text,
                        ha='center', va='center', fontsize=20,
                        color=arrow_color, fontweight='bold')

    # Légende
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    legend_elements = [
        mpatches.Patch(color='#3498db', label='État initial'),
        mpatches.Patch(color='#2ecc71', label='Goal'),
        mpatches.Patch(color='#2d2d2d', label='Trou'),
        Line2D([0], [0], marker='$←$', color='w', markerfacecolor='#2c3e50',
               markersize=12, label='Action normale (slip)'),
        Line2D([0], [0], marker='$←$', color='w', markerfacecolor='#e67e22',
               markersize=12, label='Action no-slip'),
    ]
    ax.legend(handles=legend_elements, loc='upper left',
              bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=9)

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Policy visualization saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_results(log_filepaths, output_dir="plots", window_size=50, run_tag=None):
    """
    Args:
        run_tag: string prefixed to every output filename to avoid overwrites.
                 If None, derived from the first log filepath's basename (timestamp included).
    """
    if not log_filepaths:
        print("No log files provided for plotting.")
        return
    os.makedirs(output_dir, exist_ok=True)

    if run_tag is None:
        # Use the stem of the first log file (already contains timestamp) as tag
        first_stem = os.path.splitext(os.path.basename(log_filepaths[0]))[0]
        # Strip trailing _log suffix if present
        run_tag = first_stem[:-4] if first_stem.endswith("_log") else first_stem

    def plot_path(name):
        return os.path.join(output_dir, f"{run_tag}__{name}.png")

    plt.style.use('seaborn-v0_8-darkgrid')
    all_data = []
    labels = []
    shaping_methods = []

    for filepath in log_filepaths:  # Load data
        data = load_results_log(filepath)
        if data:
            all_data.append(data)
            basename = os.path.basename(filepath)
            match = re.match(r"(\w+)_([\w-]+)_(\w+)_(\d+)eps_.*_log\.json", basename)
            label_suffix = ""
            current_shaping = "unknown"
            if match:
                agent, shaping, instance, eps = match.groups()
                label_suffix = f"{agent.upper()}-{shaping.upper()}({instance}-{eps}eps)"
                current_shaping = shaping
            else:
                label_suffix = os.path.splitext(basename)[0]
            labels.append(label_suffix)
            shaping_methods.append(current_shaping)
        else:
            print(f"Skipping invalid log file: {filepath}")
    if not all_data:
        print("No valid log data loaded.")
        return

    # Plot 1: Training Success Rate (Moving Average)
    plt.figure(figsize=(12, 6))
    for i, data in enumerate(all_data):
        if 'episode_log' in data and data['episode_log']:
            ep_log = data['episode_log']
            episodes = [item['episode'] for item in ep_log]
            success = [item['success'] for item in ep_log]
            if len(success) >= window_size:
                moving_avg_success = np.convolve(success, np.ones(window_size) / window_size, mode='valid')
                episodes_ma = episodes[window_size - 1:]
                plt.plot(episodes_ma, moving_avg_success, label=f"{labels[i]} (MA)")
            else:
                plt.plot(episodes, success, label=f"{labels[i]} (Raw)", alpha=0.5)
    plt.title(f'Training Success Rate (MA Window={window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate (MA)')
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.savefig(plot_path("train_success_rate_ma"))
    print(f"Plot saved to {plot_path('train_success_rate_ma')}")
    plt.close()

    # Plot 2: Training Environment Return (Moving Average)
    plt.figure(figsize=(12, 6))
    for i, data in enumerate(all_data):
        if 'episode_log' in data and data['episode_log']:
            ep_log = data['episode_log']
            episodes = [item['episode'] for item in ep_log]
            env_rewards = [item.get('env_reward', 0) for item in ep_log]
            if len(env_rewards) >= window_size:
                moving_avg_rewards = np.convolve(env_rewards, np.ones(window_size) / window_size, mode='valid')
                episodes_ma = episodes[window_size - 1:]
                plt.plot(episodes_ma, moving_avg_rewards, label=f"{labels[i]} (MA)")
            else:
                plt.plot(episodes, env_rewards, label=f"{labels[i]} (Raw)", alpha=0.5)
    plt.title(f'Training Environment Return (MA Window={window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Environment Return (MA)')
    plt.legend()
    plt.savefig(plot_path("train_env_return_ma"))
    print(f"Plot saved to {plot_path('train_env_return_ma')}")
    plt.close()

    # Plot 3: Evaluation Success Rate
    plt.figure(figsize=(12, 6))
    for i, data in enumerate(all_data):
        if 'evaluation_log' in data and data['evaluation_log']:
            eval_log = data['evaluation_log']
            train_episodes = [item['training_episode'] for item in eval_log]
            eval_success_rates = [item.get('eval_success_rate', 0) for item in eval_log]
            plt.plot(train_episodes, eval_success_rates, marker='o', linestyle='-', label=labels[i])
    plt.title('Evaluation Success Rate vs. Training Episode')
    plt.xlabel('Training Episode')
    plt.ylabel('Evaluation Success Rate')
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.savefig(plot_path("eval_success_rate"))
    print(f"Plot saved to {plot_path('eval_success_rate')}")
    plt.close()

    # Plot 4: Training Shaped vs Env Reward (MA)
    # Exclude non-shaping from this plot
    unique_shaping_methods = sorted(list(set(
        shaping for shaping in shaping_methods if shaping not in ['none', 'unknown', 'eval', 'optimal', 'cp_greedy'])))
    if not unique_shaping_methods:
        print("No data with shaping methods found for Shaped vs Env Reward plot.")
    else:
        num_plots = len(unique_shaping_methods)
        rows = (num_plots + 1) // 2 if num_plots > 1 else 1
        cols = 2 if num_plots > 1 else 1
        fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows), squeeze=False)
        axes = axes.flatten()
        plot_idx = 0
        for shaping_method_to_plot in unique_shaping_methods:
            ax = axes[plot_idx]
            data_found_for_method = False
            for i, data in enumerate(all_data):
                if shaping_methods[i] == shaping_method_to_plot:
                    if 'episode_log' in data and data['episode_log']:
                        ep_log = data['episode_log']
                        episodes = [item['episode'] for item in ep_log]
                        env_rewards = [item.get('env_reward', 0) for item in ep_log]
                        shaped_rewards = [item.get('shaped_reward', 0) for item in ep_log]
                        if len(env_rewards) >= window_size and len(shaped_rewards) >= window_size:
                            ma_env = np.convolve(env_rewards, np.ones(window_size) / window_size, mode='valid')
                            ma_shaped = np.convolve(shaped_rewards, np.ones(window_size) / window_size, mode='valid')
                            episodes_ma = episodes[window_size - 1:]
                            ax.plot(episodes_ma, ma_env, label=f"{labels[i].split('(')[0]} EnvR (MA)", linestyle='--')
                            ax.plot(episodes_ma, ma_shaped, label=f"{labels[i].split('(')[0]} ShapedR (MA)",
                                    linestyle='-')
                            data_found_for_method = True
            if data_found_for_method:
                ax.set_title(f'Shaped vs Env Reward (MA {window_size}) - {shaping_method_to_plot.upper()}')
                ax.set_xlabel('Episode')
                ax.set_ylabel('Reward (MA)')
                ax.legend()
                ax.grid(True)
                plot_idx += 1
        for j in range(plot_idx, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.savefig(plot_path("train_shaped_vs_env_reward_ma"))
        print(f"Plot saved to {plot_path('train_shaped_vs_env_reward_ma')}")
        plt.close(fig)

    # Plot 5: Evaluation Average Discounted Return
    plt.figure(figsize=(12, 6))
    plot_generated = False
    for i, data in enumerate(all_data):
        if 'evaluation_log' in data and data['evaluation_log']:
            eval_log = data['evaluation_log']
            if eval_log and 'eval_avg_discounted_return' in eval_log[0]:
                train_episodes = [item['training_episode'] for item in eval_log]
                eval_discounted_returns = [item.get('eval_avg_discounted_return', np.nan) for item in eval_log]
                if not np.all(np.isnan(eval_discounted_returns)):
                    plt.plot(train_episodes, eval_discounted_returns, marker='o', linestyle='-', label=labels[i])
                    plot_generated = True
    if plot_generated:
        plt.title('Evaluation Average Discounted Return vs. Training Episode')
        plt.xlabel('Training Episode')
        plt.ylabel('Average Discounted Return')
        plt.legend()
        plt.savefig(plot_path("eval_avg_discounted_return"))
        print(f"Plot saved to {plot_path('eval_avg_discounted_return')}")
    else:
        print("No data found for Evaluation Average Discounted Return plot.")
    plt.close()
