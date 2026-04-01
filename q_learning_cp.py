# FL/q_learning_cp.py
"""
Q-learning agents with Constraint Programming (CP) based reward shaping (CP-MS, CP-ETR).
"""
import socket
import random
import time
import numpy as np
import config
import utils  # For evaluate_agent, get_policy_grid_from_q_table, save_q_table_csv
import datetime
import os

# Constants for CP server communication
CP_SERVER_HOST = 'localhost'
CP_SERVER_PORT = 12345
CP_BUFFER_SIZE = 1024
CP_TIMEOUT = 10.0  # Timeout for CP socket operations


def _get_hyperparameters():
    """ Retrieve Q-learning hyperparameters from the configuration. """
    return (
        config.EPSILON,
        config.LEARNING_RATE,
        config.DISCOUNT_FACTOR,
        config.EPSILON_DECAY,
        config.EPSILON_MIN,
    )


class CPRewardClient:
    """ Client for interacting with the CP shaping service over a socket. """

    def __init__(self, host=CP_SERVER_HOST, port=CP_SERVER_PORT):
        self.host = host
        self.port = port
        self.socket = None
        self.is_connected = False

    def connect(self):
        """ Establishes connection with the CP server. """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(CP_TIMEOUT)
            sock.connect((self.host, self.port))
            time.sleep(0.1)  # Allow time for the server to send a welcome message
            welcome = sock.recv(CP_BUFFER_SIZE).decode('utf-8').strip()
            if welcome.startswith("OK Welcome"):
                self.socket = sock
                self.is_connected = True
                print(f"Connected to CP server at {self.host}:{self.port}")
                return True
            sock.close()
            print(f"ERROR: Unexpected welcome message from CP server: '{welcome}'")
            return False
        except ConnectionRefusedError:
            print(f"ERROR: Connection refused by CP server at {self.host}:{self.port}. Is it running?")
            return False
        except socket.timeout:
            print(f"ERROR: Timeout connecting to CP server at {self.host}:{self.port}.")
            return False
        except Exception as err:
            print(f"ERROR: Unexpected error connecting to CP server: {err}")
            return False

    def send_receive(self, command):
        """ Sends a command and receives a response from the CP server. """
        if not self.is_connected or self.socket is None:
            return "ERROR Not Connected"
        try:
            data = f"{command.strip()}\n".encode('utf-8')
            self.socket.sendall(data)
            response = self.socket.recv(CP_BUFFER_SIZE).decode('utf-8').strip()
            if not response:
                return "ERROR Empty Response"
            return response
        except socket.timeout:
            print(f"ERROR: Timeout waiting for response from CP server for command: {command}")
            return "ERROR Timeout"
        except socket.error as e:
            print(f"ERROR: Socket error during send/receive: {e}")
            self.close()
            return f"ERROR Socket Error: {e}"
        except Exception as err:
            print(f"ERROR: Unexpected error during send/receive: {err}")
            self.close()
            return f"ERROR Communication Failed: {err}"

    def query_action_marginal(self, step, action):
        """ Queries CP server for action marginal probability. """
        response = self.send_receive(f"QUERY {step} {action}")
        if response.startswith("REWARD "):
            try:
                value = float(response.split()[1])
                return max(0.0, min(1.0, value))  # Ensure value is clamped between 0 and 1
            except (ValueError, IndexError):
                print(f"ERROR: Could not parse float from REWARD response: '{response}'")
                return None
        elif response.startswith("ERROR"):
            return None
        else:
            print(f"WARN: Unexpected response format for QUERY {step} {action}: '{response}'")
            return None

    def query_etr(self):
        """ Queries CP server for Expected Total Reward (Success Probability). """
        response = self.send_receive("QUERY_ETR")
        if response.startswith("ETR_VALUE "):
            try:
                value = float(response.split()[1])
                return max(0.0, min(1.0, value))  # Ensure value is clamped between 0 and 1
            except (ValueError, IndexError):
                print(f"ERROR: Could not parse float from ETR_VALUE response: '{response}'")
                return None
        elif response.startswith("ERROR"):
            return None
        else:
            print(f"WARN: Unexpected response format for QUERY_ETR: '{response}'")
            return None

    def send_step(self, step, action, next_state):
        """ Informs CP server about an environment step. """
        response = self.send_receive(f"STEP {step} {action} {next_state}")
        if response.startswith("OK STEP"):
            return True
        elif response.startswith("ERROR"):
            return False
        else:
            print(f"WARN: Unexpected response format for STEP {step} {action} {next_state}: '{response}'")
            return False

    def close(self):
        """ Closes the socket connection. """
        if self.socket:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
                self.socket.close()
            except socket.error as e:
                print(f"WARN: Error closing socket (might be already closed): {e}")
            except Exception as e:  # Catch any other unexpected errors during close
                print(f"WARN: Unexpected error during socket close: {e}")
        self.socket = None
        self.is_connected = False


def train_q_learning_with_cp_shaping(env, cp_client, total_episodes, max_steps, shaping_type, size, holes, goal,
                                     shaping_method_name, instance_id):
    """ Trains using CP shaping. Uses optimistic Q initialization. """
    print(f"Starting CP shaped training ({shaping_type}): {total_episodes} episodes")
    state_size = env.observation_space.n
    action_size = env.action_space.n

    if shaping_type == 'cp-ms':
        q_init_val = config.Q_INIT_VALUE_CP_MS
    elif shaping_type == 'cp-etr':
        q_init_val = config.Q_INIT_VALUE_CP_ETR
    else:  # This case should ideally not be reached if parameters are validated upstream.
        print(f"Warning: Unknown CP shaping type '{shaping_type}' for Q_INIT. Defaulting to 0.05.")
        q_init_val = 0.05
    q_table = np.full((state_size, action_size), q_init_val)

    epsilon, lr, gamma_discount, eps_decay, eps_min = _get_hyperparameters()
    episode_log, evaluation_log = [], []

    if not cp_client or not cp_client.is_connected:
        print("ERROR: CP client not connected. Aborting training.")
        return q_table, episode_log, evaluation_log

    cp_ms_coeff = 0.2  # Coefficient for CP-MS shaping reward component
    total_steps_processed = 0

    for episode in range(total_episodes):
        state, info = env.reset()
        reset_response = cp_client.send_receive("RESET")
        if not reset_response.startswith("OK RESET"):
            print(f"ERROR: CP Server RESET failed: {reset_response}. Aborting training.")
            break

        etr_before = None
        etr_after = None  # Initialize for clarity within the loop
        if shaping_type == 'cp-etr':
            etr_before = cp_client.query_etr()
            if etr_before is None:  # Handle potential failure of initial ETR query
                print(f"WARN: Failed initial ETR query Ep {episode + 1}. Setting to 0.")
                etr_before = 0.0

        episode_steps = 0
        env_reward_sum = 0.0
        shaped_reward_sum = 0.0
        final_reward = 0.0  # Tracks the environment reward from the terminal step
        done = False
        terminated = False  # Environment terminated (e.g., reached goal, fell in hole)
        truncated = False  # Episode ended due to time limit or other truncation

        for step_idx in range(max_steps):
            total_steps_processed += 1
            episode_steps += 1
            current_state_debug = state  # Store state for Q-table update
            current_step_marginals = {}  # For CP-MS shaping

            if shaping_type == 'cp-ms':
                # Query marginals for all actions from the current state for CP-MS
                for a_query in range(action_size):
                    marginal = cp_client.query_action_marginal(step_idx, a_query)
                    if marginal is not None:
                        current_step_marginals[a_query] = marginal
                    else:
                        # If a marginal query fails, use a neutral value (e.g., 0)
                        print(
                            f"WARN: Failed marginal query Ep {episode + 1}/St {step_idx} for S{current_state_debug}, A{a_query}. Using 0.")
                        current_step_marginals[a_query] = 0.0

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                if not (0 <= state < state_size):  # State bounds check
                    print(f"ERROR: Invalid state {state} Ep {episode + 1}/St {step_idx}. Stopping episode.")
                    break
                action = int(np.argmax(q_table[state]))

            raw_cp_shaping_reward = 0.0
            reward_used_for_update = 0.0

            try:
                next_state, env_reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            except Exception as err:  # Catch potential errors during environment interaction
                print(f"ERROR: env.step exception: {err}. Stopping episode.")
                break

            # Inform CP server about the taken step, regardless of shaping type
            step_ok = cp_client.send_step(step_idx, action, next_state)
            if not step_ok:
                print(f"WARN: CP Server failed STEP {step_idx} processing. CP state may be inconsistent.")
                if shaping_type == 'cp-etr':
                    etr_after = None  # Invalidate ETR if CP step failed

            # Calculate shaping reward based on the method
            if shaping_type == 'cp-ms':
                action_marginal = current_step_marginals.get(action, 0.0)
                raw_cp_shaping_reward = cp_ms_coeff * (action_marginal - 0.25)  # 0.25 is baseline for 4 actions
                reward_used_for_update = env_reward + raw_cp_shaping_reward
            elif shaping_type == 'cp-etr' and step_ok:  # Only use ETR if CP step was acknowledged
                etr_after = cp_client.query_etr()
                if etr_after is None or etr_before is None:
                    # If ETR query fails, fall back to environment reward
                    print(f"WARN: Failed ETR query after step {step_idx}. Using env_reward for update.")
                    raw_cp_shaping_reward = 0.0
                    reward_used_for_update = env_reward
                else:
                    raw_cp_shaping_reward = etr_after - etr_before
                    reward_used_for_update = raw_cp_shaping_reward
                    # Override for falling into a hole (terminal, env_reward=0, but not goal)
                    if terminated and env_reward == 0.0 and step_idx < max_steps - 1:
                        reward_used_for_update = 0.0 - (etr_before if etr_before is not None else 0.0)
                    etr_before = etr_after  # Update etr_before for the next step's PBRS calculation
            else:  # Fallback: No shaping or CP step failed for ETR
                raw_cp_shaping_reward = 0.0
                reward_used_for_update = env_reward

            shaped_reward_sum += reward_used_for_update

            # Q-table update
            if not (0 <= next_state < state_size):  # Next state bounds check
                print(f"ERROR: Invalid next_state {next_state} Ep {episode + 1}/St {step_idx}. Stopping episode.")
                break
            best_next_action_q_value = np.max(q_table[next_state])
            # If 'done', the future reward is 0 (no further steps)
            td_target = reward_used_for_update + gamma_discount * best_next_action_q_value * (1 - int(done))
            td_error = td_target - q_table[current_state_debug, action]
            q_table[current_state_debug, action] += lr * td_error

            env_reward_sum += env_reward
            state = next_state
            if done:
                final_reward = env_reward  # Store the last environment reward
                break  # End of episode

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
            print(f"\n--- Evaluation at Episode {episode + 1}/{total_episodes} ({shaping_type}) ---")
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
            # shaping_method_name already contains shaping_type (e.g. 'cp-ms')
            q_filename = os.path.join("results",
                                      f"q_{shaping_method_name}_{instance_id}_{total_episodes}eps_{episode + 1}eval_{timestamp}.csv")
            utils.save_q_table_csv(q_table, q_filename)

    print(f"CP shaped ({shaping_type}) training complete. Total steps: {total_steps_processed}")
    return q_table, episode_log, evaluation_log
