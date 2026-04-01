# FL/heuristic_agents.py
"""
Non-learning agents: Optimal policy agent and CP-MS Greedy heuristic agent.
CP-MS Greedy agent queries CP at each step (no caching).
"""
import numpy as np
import random  # For CP-MS Greedy fallback if CP queries fail
import config  # For EVAL_EPISODES and DISCOUNT_FACTOR
# CPRewardClient is passed externally (e.g., from main.py).
from q_learning_cp import CPRewardClient
import time


def _state_to_row_col(state, size):
    """Converts a state index to (row, col)."""
    return divmod(state, size)


def _char_to_action(char):
    """Converts policy character (L, D, R, U) to action index."""
    if char == 'L':
        return 0
    if char == 'D':
        return 1
    if char == 'R':
        return 2
    if char == 'U':
        return 3
    return 0  # Default for 'N' or unexpected


def run_optimal_policy(env, optimal_policy_grid, size, max_steps):
    """Runs the agent using a pre-defined optimal policy grid."""
    print("--- Running Optimal Policy Evaluation ---")
    print(f"Evaluating for {config.EVAL_EPISODES} episodes with max_steps={max_steps}...")
    if not optimal_policy_grid:
        print("ERROR: Optimal policy grid is missing or empty in instances.json.")
        return [{'training_episode': 0, 'eval_success_rate': 0.0, 'eval_avg_return': 0.0,
                 'eval_avg_discounted_return': 0.0, 'avg_steps_success': 0.0,
                 'avg_steps_failure': float(max_steps) if max_steps else 0.0}]
    if len(optimal_policy_grid) != size or any(len(row) != size for row in optimal_policy_grid):
        print(f"ERROR: Optimal policy grid dimensions mismatch. Expected {size}x{size}.")
        return [{'training_episode': 0, 'eval_success_rate': 0.0, 'eval_avg_return': 0.0,
                 'eval_avg_discounted_return': 0.0, 'avg_steps_success': 0.0,
                 'avg_steps_failure': float(max_steps) if max_steps else 0.0}]

    wins = 0
    total_undiscounted_return = 0.0
    total_discounted_return = 0.0
    success_steps_list = []
    failure_steps_list = []
    num_episodes = config.EVAL_EPISODES
    gamma = config.DISCOUNT_FACTOR
    env_reward = 0.0  # Initialize environment reward

    for episode in range(num_episodes):
        current_env_state, info = env.reset()
        episode_undiscounted_return = 0.0
        episode_discounted_return = 0.0
        terminated = False
        truncated = False
        done = False
        step_count = 0

        while not done:
            if step_count >= max_steps:
                truncated = True
                done = True
                break

            row, col = _state_to_row_col(current_env_state, size)
            if not (0 <= row < size and 0 <= col < size):
                print(f"ERROR: Optimal agent - Invalid state {current_env_state} encountered. Terminating episode.")
                terminated = True
                done = True
                env_reward = 0.0  # Ensure no reward if error
                break

            action_char = optimal_policy_grid[row][col]
            action_to_take = _char_to_action(action_char)

            try:
                next_env_state, env_reward, terminated, truncated, info = env.step(action_to_take)
                done = terminated or truncated
                episode_undiscounted_return += env_reward
                current_env_state = next_env_state
                step_count += 1
            except Exception as e:
                print(f"ERROR during env.step in optimal policy run: {e}. Terminating episode.")
                terminated = True
                done = True
                env_reward = 0.0  # Ensure no reward if error
                break

        total_undiscounted_return += episode_undiscounted_return
        if terminated and env_reward == 1.0:  # Successful termination at goal
            wins += 1
            success_steps_list.append(step_count)
            episode_discounted_return = (gamma ** (step_count - 1)) * 1.0  # Goal reward is 1.0
        else:  # Failure or truncation
            failure_steps_list.append(step_count)
            episode_discounted_return = 0.0  # No discounted reward if goal not reached
        total_discounted_return += episode_discounted_return

    success_rate = float(wins / num_episodes) if num_episodes > 0 else 0.0
    avg_undiscounted_return = float(total_undiscounted_return / num_episodes) if num_episodes > 0 else 0.0
    avg_discounted_return = float(total_discounted_return / num_episodes) if num_episodes > 0 else 0.0
    avg_steps_success = float(sum(success_steps_list) / len(success_steps_list)) if success_steps_list else 0.0
    avg_steps_failure = float(sum(failure_steps_list) / len(failure_steps_list)) if failure_steps_list else 0.0

    print("\n--- Optimal Policy Evaluation Finished ---")
    print(f"  Success Rate:              {success_rate:.4f} ({wins}/{num_episodes})")
    print(f"  Avg Return (Undiscounted): {avg_undiscounted_return:.4f}")
    print(f"  Avg Discounted Return:     {avg_discounted_return:.4f}")
    print(f"  Avg Steps (Success):       {avg_steps_success:.2f}")
    print(f"  Avg Steps (Failure/Truncated): {avg_steps_failure:.2f}")
    print("----------------------------------------")

    eval_log_entry = {'training_episode': 0, 'eval_success_rate': success_rate,
                      'eval_avg_return': avg_undiscounted_return, 'eval_avg_discounted_return': avg_discounted_return,
                      'avg_steps_success': avg_steps_success, 'avg_steps_failure': avg_steps_failure}
    return [eval_log_entry]


def run_cp_ms_greedy_agent(env, cp_client: CPRewardClient, total_episodes_for_eval, max_steps, size, action_size,
                           instance_id):
    """
    Runs a greedy agent based on CP-MS marginals for evaluation.
    Queries CP server at each step (NO CACHING).
    Assumes cp_client is already connected and initialized.
    """
    print(f"--- Running CP-MS Greedy Agent Evaluation (Instance: {instance_id}, No Cache) ---")
    num_episodes = total_episodes_for_eval
    print(f"Evaluating for {num_episodes} episodes with max_steps={max_steps}...")

    if not cp_client or not cp_client.is_connected:
        print("ERROR: CP-MS Greedy Agent requires a connected and initialized CP client.")
        return [{'training_episode': 0, 'eval_success_rate': 0.0, 'eval_avg_return': 0.0,
                 'eval_avg_discounted_return': 0.0, 'avg_steps_success': 0.0,
                 'avg_steps_failure': float(max_steps) if max_steps else 0.0}]

    wins = 0
    total_undiscounted_return = 0.0
    total_discounted_return = 0.0
    success_steps_list = []
    failure_steps_list = []
    gamma = config.DISCOUNT_FACTOR
    # No shared_action_cache here, as per non-caching version

    for episode in range(num_episodes):
        current_env_state, info = env.reset()

        reset_response = cp_client.send_receive("RESET")  # Reset CP model state for new episode
        if not reset_response.startswith("OK RESET"):
            print(f"  ERROR: CP Server RESET failed: {reset_response}. Aborting episode.")
            failure_steps_list.append(max_steps)  # Consider episode a failure using max_steps
            continue

        episode_undiscounted_return = 0.0
        episode_discounted_return = 0.0
        terminated = False
        truncated = False
        done = False
        step_count = 0
        cp_model_step = 0  # Tracks the current CP model's step count within the episode
        env_reward = 0.0  # Initialize environment reward for the episode

        while not done:
            if step_count >= max_steps:
                truncated = True
                done = True
                print(f"  Ep {episode + 1} EnvStep {step_count}: Reached max steps.")
                break

            action_to_take = -1  # Initialize to an invalid action

            # Always query CP for marginals at the current CP model step
            action_marginals = []
            all_queries_successful = True
            for act_idx in range(action_size):
                # Query using cp_model_step, which increments with each successful CP interaction
                marginal = cp_client.query_action_marginal(cp_model_step, act_idx)
                if marginal is None:
                    action_marginals.append(-1.0)  # Use a value indicating failure for np.argmax
                    all_queries_successful = False
                else:
                    action_marginals.append(marginal)

            if not action_marginals or not all_queries_successful:
                print(
                    f"    WARN: All marginal queries failed or no marginals for S{current_env_state}, CPStep{cp_model_step}. Taking random action.")
                action_to_take = env.action_space.sample()  # Fallback to random action
            else:
                action_to_take = np.argmax(action_marginals)  # Choose action with highest marginal

            try:
                next_env_state, env_reward, terminated, truncated, info = env.step(action_to_take)
                done = terminated or truncated
            except Exception as e:
                print(
                    f"ERROR during env.step for CP-MS Greedy: {e}. State: {current_env_state}, Action: {action_to_take}. Terminating episode.")
                terminated = True  # Mark as terminated due to error
                env_reward = 0.0  # Ensure no accidental success reward
                done = True
                break  # Exit step loop

            if cp_client.is_connected:  # Inform CP server about the taken step
                cp_client.send_step(cp_model_step, action_to_take, next_env_state)

            episode_undiscounted_return += env_reward
            current_env_state = next_env_state
            step_count += 1
            cp_model_step += 1  # Increment CP model step after successful environment and CP step

            if done:
                break
        # Episode finished

        total_undiscounted_return += episode_undiscounted_return
        if terminated and env_reward == 1.0:  # Successful termination at goal
            wins += 1
            success_steps_list.append(step_count)
            episode_discounted_return = (gamma ** (step_count - 1)) * 1.0  # Goal reward is 1.0
        else:  # Failure or truncation
            failure_steps_list.append(step_count)
            episode_discounted_return = 0.0  # No discounted reward if goal not reached or error
        total_discounted_return += episode_discounted_return

    success_rate = float(wins / num_episodes) if num_episodes > 0 else 0.0
    avg_undiscounted_return = float(total_undiscounted_return / num_episodes) if num_episodes > 0 else 0.0
    avg_discounted_return = float(total_discounted_return / num_episodes) if num_episodes > 0 else 0.0
    avg_steps_success = float(sum(success_steps_list) / len(success_steps_list)) if success_steps_list else 0.0
    avg_steps_failure = float(sum(failure_steps_list) / len(failure_steps_list)) if failure_steps_list else 0.0

    print("\n--- CP-MS Greedy Agent Evaluation Finished (No Cache) ---")
    print(f"  Success Rate:              {success_rate:.4f} ({wins}/{num_episodes})")
    print(f"  Avg Return (Undiscounted): {avg_undiscounted_return:.4f}")
    print(f"  Avg Discounted Return:     {avg_discounted_return:.4f}")
    print(f"  Avg Steps (Success):       {avg_steps_success:.2f}")
    print(f"  Avg Steps (Failure/Truncated): {avg_steps_failure:.2f}")
    print("----------------------------------------------------")

    eval_log_entry = {
        'training_episode': 0,
        'eval_success_rate': success_rate,
        'eval_avg_return': avg_undiscounted_return,
        'eval_avg_discounted_return': avg_discounted_return,
        'avg_steps_success': avg_steps_success,
        'avg_steps_failure': avg_steps_failure
    }
    return [eval_log_entry]
