# FL/q_learning_cp_w_cache.py
"""
Q-learning agents with Constraint Programming (CP) based reward shaping (CP-MS, CP-ETR).
Includes caching for shaped rewards, ignoring cp_model_step in cache keys as a simplifying assumption.
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
                return max(0.0, min(1.0, value))  # Clamp value between 0 and 1
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
                return max(0.0, min(1.0, value))  # Clamp value between 0 and 1
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
            except socket.error:  # Socket might already be closed or in error state
                pass
            except Exception:  # Catch any other unexpected errors during close
                pass
        self.socket = None
        self.is_connected = False


def train_q_learning_with_cp_shaping(env, cp_client, total_episodes, max_steps, shaping_type, size, holes, goal,
                                     shaping_method_name, instance_id):
    """
    Trains using CP shaping (CP-MS or CP-ETR) with shared caching for shaped rewards.
    Simplifying assumption: cache keys do NOT include cp_model_step.
    """
    print(f"Starting CP shaped training ({shaping_type} with simplified caching): {total_episodes} episodes")
    state_size = env.observation_space.n
    action_size = env.action_space.n

    if shaping_type == 'cp-ms':
        q_init_val = config.Q_INIT_VALUE_CP_MS
    elif shaping_type == 'cp-etr':
        q_init_val = config.Q_INIT_VALUE_CP_ETR
    else:
        print(f"Warning: Unknown CP shaping type '{shaping_type}' for Q_INIT. Defaulting to 0.0.")
        q_init_val = 0.0
    q_table = np.full((state_size, action_size), q_init_val)

    epsilon, lr, gamma_discount, eps_decay, eps_min = _get_hyperparameters()
    episode_log, evaluation_log = [], []

    if not cp_client or not cp_client.is_connected:
        print("ERROR: CP client not connected. Aborting training.")
        return q_table, episode_log, evaluation_log

    # For CP-MS:
    # Cache 1: Stores raw marginals for (state) -> {action_idx: marginal_value, ...}
    cp_ms_raw_marginals_cache = {}
    # Cache 2: Stores the final shaped reward for (state, action) -> shaped_reward_value
    cp_ms_shaped_reward_cache = {}

    # For CP-ETR:
    # Cache 1: Stores raw ETR values for (state) -> etr_value
    cp_etr_value_cache = {}  # Stores ETR(s)
    # Cache 2: Stores the final shaped reward for (state, next_state) -> shaped_reward_value
    cp_etr_shaped_reward_cache = {}

    cp_ms_coeff = 0.2
    baseline_prob = 1.0 / action_size

    total_steps_processed = 0
    # Cache hit counters (optional, for debugging/analysis)
    hits_ms_marginals = 0
    hits_ms_shaped_reward = 0
    hits_etr_value = 0
    hits_etr_shaped_reward = 0
    queries_ms = 0
    queries_etr = 0

    for episode in range(total_episodes):
        current_env_state, info = env.reset()
        reset_response = cp_client.send_receive("RESET")  # CP model plan resets each episode
        if not reset_response.startswith("OK RESET"):
            print(f"ERROR: CP Server RESET failed: {reset_response}. Aborting training.")
            break

        cp_model_step = 0  # This tracks CP's internal plan step for the *current* episode

        etr_s_t = None  # For ETR: ETR(s_t)
        etr_s_t_plus_1 = None  # For ETR: ETR(s_{t+1})

        episode_steps_count = 0
        env_reward_sum_episode = 0.0
        shaped_reward_sum_episode_log = 0.0
        final_env_reward_episode = 0.0
        done = False
        terminated = False
        truncated = False

        for _ in range(max_steps):
            total_steps_processed += 1
            episode_steps_count += 1

            state_for_q_learning = current_env_state  # s_t

            if random.random() < epsilon:
                action_taken = env.action_space.sample()
            else:
                action_taken = int(np.argmax(q_table[state_for_q_learning]))

            final_shaped_reward_signal = 0.0  # This is F_t, the shaping potential
            reward_for_q_update = 0.0  # This is r'_t or R'_t used in Q-update

            if shaping_type == 'cp-ms':
                cache_key_ms_shaped = (state_for_q_learning, action_taken)
                if cache_key_ms_shaped in cp_ms_shaped_reward_cache:
                    final_shaped_reward_signal = cp_ms_shaped_reward_cache[cache_key_ms_shaped]
                    hits_ms_shaped_reward += 1
                else:
                    # Need P(action_taken | state_for_q_learning)
                    # First, get/cache all marginals for state_for_q_learning
                    all_marginals_for_s_t = None
                    if state_for_q_learning in cp_ms_raw_marginals_cache:
                        all_marginals_for_s_t = cp_ms_raw_marginals_cache[state_for_q_learning]
                        hits_ms_marginals += 1
                    else:
                        all_marginals_for_s_t = {}
                        for ac_idx in range(action_size):
                            # cp_model_step IS used for the query itself to CP server
                            marginal = cp_client.query_action_marginal(cp_model_step, ac_idx)
                            queries_ms += 1
                            all_marginals_for_s_t[ac_idx] = marginal if marginal is not None else 0.0
                        cp_ms_raw_marginals_cache[state_for_q_learning] = all_marginals_for_s_t

                    marginal_for_action_taken = all_marginals_for_s_t.get(action_taken, 0.0)
                    final_shaped_reward_signal = cp_ms_coeff * (marginal_for_action_taken - baseline_prob)
                    cp_ms_shaped_reward_cache[cache_key_ms_shaped] = final_shaped_reward_signal
                # For CP-MS, Q-update uses: env_reward + F_t
                # env_reward will be obtained after env.step()

            elif shaping_type == 'cp-etr':
                # ETR(s_t) is needed before the step
                if state_for_q_learning in cp_etr_value_cache:
                    etr_s_t = cp_etr_value_cache[state_for_q_learning]
                    hits_etr_value += 1
                else:
                    etr_val = cp_client.query_etr()  # cp_model_step is implicit in server's state
                    queries_etr += 1
                    etr_s_t = etr_val if etr_val is not None else 0.0
                    cp_etr_value_cache[state_for_q_learning] = etr_s_t

            next_env_state, env_reward_step, terminated, truncated, info = env.step(action_taken)
            done = terminated or truncated

            # Inform CP Server (ALWAYS), cp_model_step is used here to advance the CP model's internal state
            step_communicated_to_cp = cp_client.send_step(cp_model_step, action_taken, next_env_state)

            if shaping_type == 'cp-ms':
                reward_for_q_update = env_reward_step + final_shaped_reward_signal

            elif shaping_type == 'cp-etr':
                cache_key_etr_shaped = (state_for_q_learning, next_env_state)  # Key uses s_t, s_{t+1}
                if cache_key_etr_shaped in cp_etr_shaped_reward_cache:
                    final_shaped_reward_signal = cp_etr_shaped_reward_cache[cache_key_etr_shaped]
                    hits_etr_shaped_reward += 1
                elif step_communicated_to_cp:  # Only query/calculate if step was acknowledged by CP
                    # ETR(s_{t+1}) is needed after the step (CP server state is now at s_{t+1})
                    if next_env_state in cp_etr_value_cache:
                        etr_s_t_plus_1 = cp_etr_value_cache[next_env_state]
                        hits_etr_value += 1  # Count this as a hit for the individual ETR value
                    else:
                        etr_val_after = cp_client.query_etr()  # cp_model_step+1 is implicit in server
                        queries_etr += 1
                        etr_s_t_plus_1 = etr_val_after if etr_val_after is not None else 0.0
                        cp_etr_value_cache[next_env_state] = etr_s_t_plus_1

                    # etr_s_t should have been fetched before the step
                    if etr_s_t is not None and etr_s_t_plus_1 is not None:
                        final_shaped_reward_signal = gamma_discount * etr_s_t_plus_1 - etr_s_t
                        # Hole override: if fell in hole (terminated, env_reward 0, not max_steps)
                        if terminated and env_reward_step == 0.0 and episode_steps_count < max_steps - 1:
                            final_shaped_reward_signal = 0.0 - (etr_s_t if etr_s_t is not None else 0.0)
                        cp_etr_shaped_reward_cache[cache_key_etr_shaped] = final_shaped_reward_signal
                    else:  # Fallback if any ETR query failed for the fresh calculation
                        final_shaped_reward_signal = 0.0  # No shaping if ETR values are missing
                else:  # Fallback if step communication to CP server failed
                    final_shaped_reward_signal = 0.0

                # For CP-ETR, Q-update uses F_t directly
                reward_for_q_update = final_shaped_reward_signal

            best_next_q = np.max(q_table[next_env_state])
            td_target = reward_for_q_update + gamma_discount * best_next_q * (1 - int(done))
            q_table[state_for_q_learning, action_taken] += lr * (
                        td_target - q_table[state_for_q_learning, action_taken])

            env_reward_sum_episode += env_reward_step
            shaped_reward_sum_episode_log += reward_for_q_update  # Log the reward value used for Q-update

            current_env_state = next_env_state
            cp_model_step += 1  # Advance CP model's internal step counter for next query

            if done:
                final_env_reward_episode = env_reward_step
                break
        # End step loop

        success_episode = int(final_env_reward_episode == 1.0 and terminated)
        episode_log.append({
            'episode': episode + 1, 'steps': episode_steps_count,
            'env_reward': env_reward_sum_episode, 'shaped_reward': shaped_reward_sum_episode_log,
            'success': success_episode
        })
        epsilon = max(epsilon * eps_decay, eps_min)

        if (episode + 1) % config.EVAL_FREQUENCY == 0 or (episode + 1) == total_episodes:
            sr, ar_undisc, ar_disc, avg_ss, avg_sf = utils.evaluate_agent(env, q_table, max_steps, config.EVAL_EPISODES)
            evaluation_log.append({
                'training_episode': episode + 1, 'eval_success_rate': sr,
                'eval_avg_return': ar_undisc, 'eval_avg_discounted_return': ar_disc,
                'avg_steps_success': avg_ss, 'avg_steps_failure': avg_sf
            })
            print(f"\n--- Evaluation at Episode {episode + 1}/{total_episodes} ({shaping_method_name} w/ Cache) ---")
            print(f"  Success Rate:              {sr:.2%}")
            print(f"  Avg Return (Undiscounted): {ar_undisc:.4f}")
            print(f"  Avg Discounted Return:     {ar_disc:.4f}")  # Added missing line
            print(f"  Avg Steps (Success/Failure): {avg_ss:.1f} / {avg_sf:.1f}")  # Added missing line
            print(f"  Current Epsilon:           {epsilon:.4f}")  # Added missing line

            if shaping_type == 'cp-ms':
                print(
                    f"  CP-MS Cache Hits: RawMarginals={hits_ms_marginals}, ShapedReward={hits_ms_shaped_reward} (Total MS Queries: {queries_ms})")
            elif shaping_type == 'cp-etr':
                print(
                    f"  CP-ETR Cache Hits: ETRValue={hits_etr_value}, ShapedReward={hits_etr_shaped_reward} (Total ETR Queries: {queries_etr})")
            policy_grid = utils.get_policy_grid_from_q_table(q_table, size, holes, goal)
            if policy_grid:
                print("  Current Greedy Policy:")
                for row_str in policy_grid:
                    print(f"    {row_str}")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            q_filename = os.path.join("results",
                                      f"q_{shaping_method_name}_cached_{instance_id}_{total_episodes}eps_{episode + 1}eval_{timestamp}.csv")
            utils.save_q_table_csv(q_table, q_filename)

    print(
        f"CP shaped ({shaping_type} with simplified caching) training complete. Total steps processed: {total_steps_processed}")
    # Print final cache hit summary
    if shaping_type == 'cp-ms':
        print(
            f"Final CP-MS Cache Stats: RawMarginals Hits={hits_ms_marginals} (Queries: {queries_ms}), ShapedReward Hits={hits_ms_shaped_reward}")
    elif shaping_type == 'cp-etr':
        print(
            f"Final CP-ETR Cache Stats: ETRValue Hits={hits_etr_value} (Queries: {queries_etr}), ShapedReward Hits={hits_etr_shaped_reward}")

    return q_table, episode_log, evaluation_log
