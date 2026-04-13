# FL/q_learning_standard.py
"""
Q-learning agents: Standard (no shaping) and Classic Potential-Based Shaping.
"""
import numpy as np
import random
import config
import utils  # For evaluate_agent, get_policy_grid_from_q_table, save_q_table_csv
import datetime
import os


def _get_hyperparameters():
    """ Retrieve Q-learning hyperparameters from the configuration. """
    return (
        config.EPSILON,
        config.LEARNING_RATE,
        config.DISCOUNT_FACTOR,
        config.EPSILON_DECAY,
        config.EPSILON_MIN,
    )


def train_q_learning(env, total_episodes, max_steps, size, holes, goal, shaping_method_name, instance_id):
    """ Standard Q-learning. Logs evaluation metrics and saves Q-table CSV periodically. """
    print(f"Starting standard Q-learning: {total_episodes} episodes")
    state_size = env.observation_space.n
    action_size = env.action_space.n
    q_table = np.full((state_size, action_size), config.Q_INIT_VALUE_NONE)  # Initialize Q-table

    epsilon, lr, gamma_discount, eps_decay, eps_min = _get_hyperparameters()
    episode_log, evaluation_log = [], []

    for episode in range(total_episodes):
        state, info = env.reset()
        episode_steps = 0
        env_reward_sum = 0.0
        shaped_reward_sum = 0.0  # For standard Q-learning, shaped_reward is the same as env_reward
        final_reward = 0.0  # Tracks the environment reward from the terminal step
        done = False
        terminated = False  # Environment terminated (e.g., reached goal, fell in hole)
        truncated = False  # Episode ended due to time limit or other truncation

        for step_idx in range(max_steps):
            episode_steps += 1
            current_state_for_update = state  # Store state for Q-table update
            if not (0 <= state < state_size):  # State bounds check
                print(f"ERROR: Invalid state {state} Ep {episode + 1}/St {step_idx}.")
                break

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                max_val = np.max(q_table[state])
                best_actions = np.where(q_table[state] == max_val)[0]
                action = int(np.random.choice(best_actions))

            try:
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            except Exception as err:  # Catch potential errors during environment interaction
                print(f"ERROR: env.step exception: {err}.")
                break

            if not (0 <= next_state < state_size):  # Next state bounds check
                print(f"ERROR: Invalid next_state {next_state} Ep {episode + 1}/St {step_idx}.")
                break

            reward_used_for_update = reward  # In standard Q-learning, this is the environment reward

            # Q-table update
            best_next_action_q_value = np.max(q_table[next_state])
            # If 'done', the future reward is 0 (no further steps)
            td_target = reward_used_for_update + gamma_discount * best_next_action_q_value * (1 - int(done))
            td_error = td_target - q_table[current_state_for_update, action]
            q_table[current_state_for_update, action] += lr * td_error

            shaped_reward_sum += reward_used_for_update  # Log env reward as shaped reward for consistent logging
            env_reward_sum += reward  # Accumulate raw environment reward
            state = next_state
            if done:
                final_reward = reward  # Store the last environment reward
                break

        success = int(final_reward == 1.0 and terminated)  # Success if goal reached (env_reward=1)
        episode_log.append({
            'episode': episode + 1, 'steps': episode_steps,
            'env_reward': env_reward_sum, 'shaped_reward': shaped_reward_sum,
            'success': success
        })
        epsilon = max(epsilon * eps_decay, eps_min)  # Decay epsilon

        # Periodic evaluation
        if (episode + 1) % config.EVAL_FREQUENCY == 0 or (episode + 1) == total_episodes:
            sr, ar_undisc, ar_disc, avg_ss, avg_sf = utils.evaluate_agent(env, q_table, max_steps, config.EVAL_EPISODES)
            evaluation_log.append({
                'training_episode': episode + 1, 'eval_success_rate': sr,
                'eval_avg_return': ar_undisc, 'eval_avg_discounted_return': ar_disc,
                'avg_steps_success': avg_ss, 'avg_steps_failure': avg_sf
            })
            print(f"\n--- Evaluation at Episode {episode + 1}/{total_episodes} (Standard Q) ---")
            print(f"  Success Rate:              {sr:.2%}")
            print(f"  Avg Return (Undiscounted): {ar_undisc:.4f}")
            print(f"  Avg Discounted Return:     {ar_disc:.4f}")
            print(f"  Avg Steps (Success/Failure): {avg_ss:.1f} / {avg_sf:.1f}")
            print(f"  Current Epsilon:           {epsilon:.4f}")
            print("  Current Greedy Policy:")
            policy_grid = utils.get_policy_grid_from_q_table(q_table, size, holes, goal)
            if policy_grid:
                for row_str in policy_grid:
                    print(f"    {row_str}")
            else:
                print("    (Could not generate policy grid)")
            print("-" * (4 + size + 1))  # Dynamic separator based on grid size
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # shaping_method_name is 'none' for this function
            q_filename = os.path.join("results",
                                      f"q_{shaping_method_name}_{instance_id}_{total_episodes}eps_{episode + 1}eval_{timestamp}.csv")
            utils.save_q_table_csv(q_table, q_filename)

    print("Standard Q-learning complete.")
    return q_table, episode_log, evaluation_log


def shaped_reward_classic(state, next_state, reward, done, hole_positions, goal_position, grid_size):
    """ Classic potential shaping: R_env + gamma*Phi(s') - Phi(s). """

    def potential(s):
        if s == goal_position:
            return 0  # Potential at goal state is zero
        if s in hole_positions:
            return -grid_size * 2  # Large penalty for holes to discourage entering them
        # Calculate Manhattan distance to the goal as potential
        row, col = divmod(s, grid_size)
        goal_row, goal_col = divmod(goal_position, grid_size)
        manhattan_dist = abs(row - goal_row) + abs(col - goal_col)
        return -manhattan_dist  # Negative distance: potential increases as agent gets closer

    phi_s = potential(state)
    phi_s_prime = potential(next_state)
    gamma_potential = config.DISCOUNT_FACTOR  # Use the agent's discount factor for potential calculation
    potential_shaping_signal = gamma_potential * phi_s_prime - phi_s

    # The shaped reward is the sum of the environment reward and the potential difference
    total_shaped_reward = reward + potential_shaping_signal
    return total_shaped_reward


def train_q_learning_with_classic_shaping(env, total_episodes, max_steps, grid_size, hole_positions, goal_position,
                                          shaping_method_name, instance_id):
    """ Q-learning with Classic Potential-Based Reward Shaping. """
    print(f"Starting classic shaped training: {total_episodes} episodes")
    state_size = env.observation_space.n
    action_size = env.action_space.n
    q_table = np.full((state_size, action_size), config.Q_INIT_VALUE_CLASSIC)  # Initialize Q-table

    epsilon, lr, gamma_discount, eps_decay, eps_min = _get_hyperparameters()
    episode_log, evaluation_log = [], []
    hole_set = set(hole_positions)  # Use set for efficient 'in' checks for hole positions

    for episode in range(total_episodes):
        state, info = env.reset()
        episode_steps = 0
        env_reward_sum = 0.0
        shaped_reward_sum = 0.0  # Accumulates the shaped reward used for Q-value updates
        final_reward = 0.0  # Tracks the environment reward from the terminal step
        done = False
        terminated = False  # Environment terminated
        truncated = False  # Episode truncated

        for step_idx in range(max_steps):
            episode_steps += 1
            current_state_for_update = state  # Store state for Q-table update
            if not (0 <= state < state_size):  # State bounds check
                print(f"ERROR: Invalid state {state} Ep {episode + 1}/St {step_idx}.")
                break

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))

            try:
                next_state, base_env_reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            except Exception as err:  # Catch potential errors during environment interaction
                print(f"ERROR: env.step exception: {err}.")
                break

            if not (0 <= next_state < state_size):  # Next state bounds check
                print(f"ERROR: Invalid next_state {next_state} Ep {episode + 1}/St {step_idx}.")
                break

            # Calculate shaped reward using the classic potential-based method
            reward_used_for_update = shaped_reward_classic(
                state, next_state, base_env_reward, done, hole_set, goal_position, grid_size
            )

            # Q-Update uses the shaped reward
            best_next_action_q_value = np.max(q_table[next_state])
            td_target = reward_used_for_update + gamma_discount * best_next_action_q_value * (1 - int(done))
            td_error = td_target - q_table[current_state_for_update, action]
            q_table[current_state_for_update, action] += lr * td_error

            # Accumulate shaped reward and environment reward separately for logging
            shaped_reward_sum += reward_used_for_update
            env_reward_sum += base_env_reward

            state = next_state
            if done:
                final_reward = base_env_reward  # Store the last environment reward
                break

        success = int(final_reward == 1.0 and terminated)  # Success if goal reached
        episode_log.append({
            'episode': episode + 1, 'steps': episode_steps,
            'env_reward': env_reward_sum, 'shaped_reward': shaped_reward_sum,
            'success': success
        })
        epsilon = max(epsilon * eps_decay, eps_min)  # Decay epsilon

        # Periodic evaluation
        if (episode + 1) % config.EVAL_FREQUENCY == 0 or (episode + 1) == total_episodes:
            sr, ar_undisc, ar_disc, avg_ss, avg_sf = utils.evaluate_agent(env, q_table, max_steps, config.EVAL_EPISODES)
            evaluation_log.append({
                'training_episode': episode + 1, 'eval_success_rate': sr,
                'eval_avg_return': ar_undisc, 'eval_avg_discounted_return': ar_disc,
                'avg_steps_success': avg_ss, 'avg_steps_failure': avg_sf
            })
            print(f"\n--- Evaluation at Episode {episode + 1}/{total_episodes} (Classic Shaping) ---")
            print(f"  Success Rate:              {sr:.2%}")
            print(f"  Avg Return (Undiscounted): {ar_undisc:.4f}")
            print(f"  Avg Discounted Return:     {ar_disc:.4f}")
            print(f"  Avg Steps (Success/Failure): {avg_ss:.1f} / {avg_sf:.1f}")
            print(f"  Current Epsilon:           {epsilon:.4f}")
            print("  Current Greedy Policy:")
            policy_grid = utils.get_policy_grid_from_q_table(q_table, grid_size, hole_positions, goal_position)
            if policy_grid:
                for row_str in policy_grid:
                    print(f"    {row_str}")
            else:
                print("    (Could not generate policy grid)")
            print("-" * (4 + grid_size + 1))  # Dynamic separator
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # shaping_method_name is 'classic' for this function
            q_filename = os.path.join("results",
                                      f"q_{shaping_method_name}_{instance_id}_{total_episodes}eps_{episode + 1}eval_{timestamp}.csv")
            utils.save_q_table_csv(q_table, q_filename)

    print("Classic shaped training complete.")
    return q_table, episode_log, evaluation_log
