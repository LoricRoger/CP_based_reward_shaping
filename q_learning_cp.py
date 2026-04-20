# FL/q_learning_cp.py
"""
Q-learning agents with Constraint Programming (CP) based reward shaping (CP-MS, CP-ETR).

No-slip strategy variants (--noslip-strategy):

  fail             Budget épuisé → terminaison immédiate (reward=0). Curriculum croissant.
                   Pénalité : aucune. Init Q no-slip : -0.1 (pessimiste).

  full-budget      Budget max dès le début, terminaison si dépassé. Pas de curriculum.
                   Init Q no-slip : 0.0 (neutre).
"""
import socket
import random
import time
import logging
import numpy as np
import config
import utils
import datetime
import os
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CP server communication constants
# ---------------------------------------------------------------------------
CP_SERVER_HOST = 'localhost'
CP_SERVER_PORT = 12345
CP_BUFFER_SIZE = 1024
CP_TIMEOUT = 10.0


def _get_hyperparameters():
    return (
        config.EPSILON,
        config.LEARNING_RATE,
        config.DISCOUNT_FACTOR,
        config.EPSILON_MIN,
    )


# ---------------------------------------------------------------------------
# CPRewardClient — socket interface with the Java CP server
# ---------------------------------------------------------------------------

class CPRewardClient:
    """Client for interacting with the CP shaping service over a socket."""

    def __init__(self, host=CP_SERVER_HOST, port=CP_SERVER_PORT):
        self.host = host
        self.port = port
        self.socket = None
        self.is_connected = False

    def connect(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(CP_TIMEOUT)
            sock.connect((self.host, self.port))
            time.sleep(0.1)
            welcome = sock.recv(CP_BUFFER_SIZE).decode('utf-8').strip()
            if welcome.startswith("OK Welcome"):
                self.socket = sock
                self.is_connected = True
                print(f"Connected to CP server at {self.host}:{self.port}")
                return True
            sock.close()
            print(f"ERROR: Unexpected welcome from CP server: '{welcome}'")
            return False
        except ConnectionRefusedError:
            print(f"ERROR: Connection refused at {self.host}:{self.port}. Is the server running?")
            return False
        except socket.timeout:
            print(f"ERROR: Timeout connecting to {self.host}:{self.port}.")
            return False
        except Exception as err:
            print(f"ERROR: Unexpected error connecting to CP server: {err}")
            return False

    def send_receive(self, command):
        if not self.is_connected or self.socket is None:
            return "ERROR Not Connected"
        try:
            self.socket.sendall(f"{command.strip()}\n".encode('utf-8'))
            response = self.socket.recv(CP_BUFFER_SIZE).decode('utf-8').strip()
            return response if response else "ERROR Empty Response"
        except socket.timeout:
            print(f"ERROR: Timeout waiting for response to: {command}")
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
        """Queries marginal probability for an action (MS mode only, not used by ETR)."""
        response = self.send_receive(f"QUERY {step} {action}")
        if response.startswith("REWARD "):
            try:
                return max(0.0, min(1.0, float(response.split()[1])))
            except (ValueError, IndexError):
                print(f"ERROR: Could not parse REWARD response: '{response}'")
                return None
        elif response.startswith("ERROR"):
            return None
        print(f"WARN: Unexpected QUERY response: '{response}'")
        return None

    def query_etr(self):
        """Queries Expected Total Reward (success probability) from the CP model."""
        response = self.send_receive("QUERY_ETR")
        if response.startswith("ETR_VALUE "):
            try:
                return max(0.0, min(1.0, float(response.split()[1])))
            except (ValueError, IndexError):
                print(f"ERROR: Could not parse ETR_VALUE response: '{response}'")
                return None
        elif response.startswith("ERROR"):
            return None
        print(f"WARN: Unexpected QUERY_ETR response: '{response}'")
        return None

    def send_step(self, step, action, next_state):
        response = self.send_receive(f"STEP {step} {action} {next_state}")
        if response.startswith("OK STEP"):
            return True
        if not response.startswith("ERROR"):
            print(f"WARN: Unexpected STEP response: '{response}'")
        return False

    def close(self):
        if self.socket:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
                self.socket.close()
            except socket.error as e:
                print(f"WARN: Error closing socket: {e}")
            except Exception as e:
                print(f"WARN: Unexpected error closing socket: {e}")
        self.socket = None
        self.is_connected = False


# ---------------------------------------------------------------------------
# Helper: walk wrapper chain to find FrozenLakeExtendedActions
# ---------------------------------------------------------------------------

def _get_budget_wrapper(env):
    wrapper = env
    while wrapper is not None:
        if hasattr(wrapper, 'budget'):
            return wrapper
        wrapper = getattr(wrapper, 'env', None)
    return None


# ---------------------------------------------------------------------------
# NoslipStrategy — abstract base + 6 concrete variants
# ---------------------------------------------------------------------------

class NoslipStrategy:
    """
    Base class for no-slip action strategies.

    Responsibilities:
      - Q-table initialization for no-slip actions (columns 4-7)
      - Budget scheduling (how initial_budget evolves per episode)
      - Deciding what happens when budget is exhausted
      - Optional per-step shaping penalty for no-slip use
    """

    NAME = "base"

    def __init__(self, budget_wrapper, total_episodes):
        self.wrapper = budget_wrapper
        self.max_budget = budget_wrapper.initial_budget
        self.total_episodes = total_episodes

    # -- Q-table init ---------------------------------------------------------

    def init_q_noslip(self, q_table):
        """Override to set initial Q values for no-slip actions (columns 4-7)."""
        q_table[:, 4:] = 0.0

    # -- Per-episode budget ---------------------------------------------------

    def update(self, episode):
        """Called at the start of each episode to set initial_budget."""
        self.wrapper.initial_budget = self.max_budget  # default: full budget always

    # -- Action masking when budget is exhausted ------------------------------

    def select_action(self, state, q_table, epsilon, env, budget_exhausted):
        """
        Epsilon-greedy action selection.
        budget_exhausted=True means the episode's budget is gone.
        Default: allow no-slip actions freely (budget exhaustion never triggered here).
        """
        if random.random() < epsilon:
            return env.action_space.sample()
        return int(np.argmax(q_table[state]))

    # -- What happens when the env returns from a budget-exhausted no-slip? ---

    def handle_budget_exhausted_step(self, env, action):
        """
        Called only when action >= 4 AND budget is exhausted.
        Returns (next_state, env_reward, terminated, truncated, done).
        Default: terminaison immédiate (même comportement que FrozenLakeExtendedActions).
        L'environnement gère déjà cela si action >= 4 et budget <= 0,
        donc cette méthode peut être un no-op pour la stratégie 'fail'.
        """
        pass  # env.step() handles it already

    # -- Per-step shaping penalty ---------------------------------------------

    def noslip_step_penalty(self, action):
        """Returns an immediate reward penalty for using a no-slip action. Default: 0."""
        return 0.0

    # -- Evaluation with full budget ------------------------------------------

    def eval_with_full_budget(self, env, q_table, max_steps, eval_episodes):
        saved = self.wrapper.initial_budget
        self.wrapper.initial_budget = self.max_budget
        try:
            return utils.evaluate_agent(env, q_table, max_steps, eval_episodes)
        finally:
            self.wrapper.initial_budget = saved

    # -- Logging --------------------------------------------------------------

    def describe(self):
        return f"Strategy '{self.NAME}' | max_budget={self.max_budget}"


# ---- Stratégie 1 : fail (comportement actuel) -------------------------------

class FailStrategy(NoslipStrategy):
    """
    Curriculum croissant : budget 0 → max par paliers égaux.
    Budget épuisé → terminaison immédiate (géré par FrozenLakeExtendedActions).
    Init Q no-slip : pessimiste (-0.1).
    Pénalité : aucune.
    """

    NAME = "fail"

    def __init__(self, budget_wrapper, total_episodes):
        super().__init__(budget_wrapper, total_episodes)
        self.stage_size = total_episodes // (self.max_budget + 1)

    def init_q_noslip(self, q_table):
        q_table[:, 4:] = config.Q_INIT_VALUE_CP_ETR_BUDGET_NOSLIP

    def update(self, episode):
        stage = min(episode // self.stage_size, self.max_budget)
        self.wrapper.initial_budget = stage

    def select_action(self, state, q_table, epsilon, env, budget_exhausted):
        if random.random() < epsilon:
            return random.randint(0, 3) if budget_exhausted else env.action_space.sample()
        if budget_exhausted:
            masked = q_table[state].copy()
            masked[4:] = -np.inf
            return int(np.argmax(masked))
        return int(np.argmax(q_table[state]))

    def describe(self):
        return (f"Strategy '{self.NAME}' | max_budget={self.max_budget} | "
                f"curriculum: {self.max_budget + 1} stages × {self.stage_size} eps | "
                f"Q_init_noslip={config.Q_INIT_VALUE_CP_ETR_BUDGET_NOSLIP}")


# ---- Stratégie 2 : full-budget ----------------------------------------------

class FullBudgetStrategy(NoslipStrategy):
    """
    Budget max dès le début. Pas de curriculum.
    Budget épuisé → terminaison immédiate.
    Init Q : 0.0 (neutre — l'agent découvre librement l'utilité des no-slip).
    """

    NAME = "full-budget"

    def describe(self):
        return (f"Strategy '{self.NAME}' | max_budget={self.max_budget} | "
                f"no curriculum | budget exhausted → fail | Q_init_noslip=0.0")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

NOSLIP_STRATEGIES = {
    'fail':        FailStrategy,
    'full-budget': FullBudgetStrategy,
}


def make_noslip_strategy(name, budget_wrapper, total_episodes):
    cls = NOSLIP_STRATEGIES.get(name)
    if cls is None:
        raise ValueError(f"Unknown no-slip strategy '{name}'. "
                         f"Available: {list(NOSLIP_STRATEGIES.keys())}")
    return cls(budget_wrapper, total_episodes)


# ---------------------------------------------------------------------------
# Verbose helpers
# ---------------------------------------------------------------------------

def _log(msg: str, verbose: int, min_level: int = 1, logger: logging.Logger = None):
    if verbose >= min_level:
        tqdm.write(msg)
    if logger:
        logger.info(msg)


def _maybe_save_qtable(q_table, shaping_type, noslip_strategy_name, instance_id,
                       total_episodes, episode, verbose: int, run_dir: str):
    """Sauvegarde la Q-table CSV uniquement si verbose >= 2."""
    if verbose < 2:
        return
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    qtables_dir = os.path.join(run_dir, "qtables")
    os.makedirs(qtables_dir, exist_ok=True)
    q_filename = os.path.join(
        qtables_dir,
        f"q_{shaping_type}_{noslip_strategy_name}_{instance_id}_{total_episodes}eps_{episode}eval_{timestamp}.csv"
    )
    utils.save_q_table_csv(q_table, q_filename)


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_q_learning_with_cp_shaping(env, cp_client, total_episodes, max_steps, shaping_type, size, holes, goal,
                                     instance_id, noslip_strategy_name='fail',
                                     verbose: int = 0, run_dir: str = "results",
                                     logger: logging.Logger = None):
    """
    Trains a Q-learning agent with CP-based reward shaping (ETR or MS).

    Args:
        noslip_strategy_name: one of 'fail', 'full-budget'.
                              Only used when action_size == 8 (budget mode).
        verbose: 0=silent, 1=tqdm+log file, 2=+Q-table CSVs
        run_dir: base directory for this run's outputs
        logger:  file logger (None if verbose=0)
    """
    _log(f"Starting CP shaped training ({shaping_type}): {total_episodes} episodes",
         verbose, logger=logger)
    state_size = env.observation_space.n
    action_size = env.action_space.n

    # Q-table initialization
    if shaping_type == 'cp-ms':
        q_init_val = config.Q_INIT_VALUE_CP_MS
    elif shaping_type == 'cp-etr':
        q_init_val = config.Q_INIT_VALUE_CP_ETR
    else:
        _log(f"Warning: Unknown CP shaping type '{shaping_type}'. Defaulting Q_init to 0.0.",
             verbose, logger=logger)
        q_init_val = 0.0
    q_table = np.full((state_size, action_size), q_init_val)

    # Strategy setup (only in 8-action / budget mode)
    budget_wrapper = _get_budget_wrapper(env) if action_size == 8 else None
    strategy = None
    if budget_wrapper is not None:
        strategy = make_noslip_strategy(noslip_strategy_name, budget_wrapper, total_episodes)
        strategy.init_q_noslip(q_table)
        _log(f"  No-slip strategy: {strategy.describe()}", verbose, logger=logger)
    else:
        _log("  No-slip strategy: N/A (4-action mode)", verbose, logger=logger)

    epsilon, lr, gamma_discount, eps_min = _get_hyperparameters()
    eps_decay = (eps_min / epsilon) ** (1.0 / total_episodes)
    episode_log, evaluation_log = [], []

    if not cp_client or not cp_client.is_connected:
        tqdm.write("ERROR: CP client not connected. Aborting training.")
        return q_table, episode_log, evaluation_log

    total_steps_processed = 0
    global_noslip_by_state = {}

    episode_iter = tqdm(range(total_episodes), desc="Training", leave=False, dynamic_ncols=True) \
        if verbose >= 1 else range(total_episodes)

    for episode in episode_iter:
        # Update budget for this episode
        if strategy is not None:
            strategy.update(episode)

        state, _ = env.reset()
        reset_response = cp_client.send_receive("RESET")
        if not reset_response.startswith("OK RESET"):
            tqdm.write(f"ERROR: CP Server RESET failed: {reset_response}. Aborting.")
            break

        # Initial ETR query (ETR mode only)
        etr_before = None
        if shaping_type == 'cp-etr':
            etr_before = cp_client.query_etr()
            if etr_before is None:
                _log(f"WARN: Failed initial ETR query Ep {episode + 1}. Setting to 0.",
                     verbose, logger=logger)
                etr_before = 0.0

        episode_steps = 0
        env_reward_sum = 0.0
        shaped_reward_sum = 0.0
        final_reward = 0.0
        done = False
        terminated = False
        noslip_count = 0

        for step_idx in range(max_steps):
            total_steps_processed += 1
            episode_steps += 1
            current_state = state

            # MS mode: pre-fetch marginals for all actions
            current_step_marginals = {}
            if shaping_type == 'cp-ms':
                for a_query in range(action_size):
                    marginal = cp_client.query_action_marginal(step_idx, a_query)
                    if marginal is not None:
                        current_step_marginals[a_query] = marginal
                    else:
                        _log(f"WARN: Failed marginal query Ep {episode + 1}/St {step_idx} "
                             f"S{current_state}/A{a_query}. Using 0.", verbose, logger=logger)
                        current_step_marginals[a_query] = 0.0

            # Compute budget state
            current_budget = strategy.wrapper.budget if strategy is not None else None
            budget_exhausted = current_budget is not None and current_budget <= 0

            # Action selection (delegated to strategy)
            if not (0 <= state < state_size):
                tqdm.write(f"ERROR: Invalid state {state} Ep {episode + 1}/St {step_idx}. Stopping.")
                break

            if strategy is not None:
                action = strategy.select_action(state, q_table, epsilon, env, budget_exhausted)
            else:
                action = (env.action_space.sample() if random.random() < epsilon
                          else int(np.argmax(q_table[state])))

            env_action = action

            if action >= 4:
                noslip_count += 1
                global_noslip_by_state[state] = global_noslip_by_state.get(state, 0) + 1

            # Environment step
            try:
                next_state, env_reward, terminated, truncated, _ = env.step(env_action)
                done = terminated or truncated
            except Exception as err:
                tqdm.write(f"ERROR: env.step exception: {err}. Stopping episode.")
                break

            # Inform CP server of the actual action + next state
            step_ok = cp_client.send_step(step_idx, env_action, next_state)
            if not step_ok:
                _log(f"WARN: CP Server STEP {step_idx} failed. CP state may be inconsistent.",
                     verbose, logger=logger)
                break

            # Per-step no-slip penalty (strategy-dependent)
            noslip_penalty = strategy.noslip_step_penalty(action) if strategy is not None else 0.0

            # Shaped reward computation
            if shaping_type == 'cp-ms':
                action_marginal = current_step_marginals.get(env_action, 0.0)
                cp_shaping = config.CP_MS_SHAPING_COEFF * (action_marginal - 0.25)
                reward_used_for_update = env_reward + cp_shaping + noslip_penalty
            elif shaping_type == 'cp-etr':
                etr_after = cp_client.query_etr()
                if etr_after is None or etr_before is None:
                    _log(f"WARN: Failed ETR query after step {step_idx}. Using env_reward.",
                         verbose, logger=logger)
                    reward_used_for_update = env_reward + noslip_penalty
                else:
                    cp_shaping = etr_after - etr_before
                    reward_used_for_update = cp_shaping + noslip_penalty
                    # Negative signal on hole: wipe out accumulated ETR progress
                    if terminated and env_reward == 0.0 and step_idx < max_steps - 1:
                        reward_used_for_update = -etr_before + noslip_penalty
                    etr_before = etr_after
            else:
                reward_used_for_update = env_reward + noslip_penalty

            shaped_reward_sum += reward_used_for_update

            if not (0 <= next_state < state_size):
                tqdm.write(f"ERROR: Invalid next_state {next_state}. Stopping episode.")
                break

            # Bellman update
            best_next_q = np.max(q_table[next_state])
            td_target = reward_used_for_update + gamma_discount * best_next_q * (1 - int(done))
            q_table[current_state, action] += lr * (td_target - q_table[current_state, action])

            env_reward_sum += env_reward
            state = next_state
            if done:
                final_reward = env_reward
                break

        success = int(final_reward == 1.0 and terminated)
        episode_log.append({
            'episode': episode + 1, 'steps': episode_steps,
            'env_reward': env_reward_sum, 'shaped_reward': shaped_reward_sum,
            'success': success,
            'noslip_used': noslip_count
        })
        epsilon = max(epsilon * eps_decay, eps_min)

        # Periodic evaluation
        if (episode + 1) % config.EVAL_FREQUENCY == 0 or (episode + 1) == total_episodes:
            sr, ar_undisc, ar_disc, avg_ss, avg_sf = utils.evaluate_agent(
                env, q_table, max_steps, config.EVAL_EPISODES)
            evaluation_log.append({
                'training_episode': episode + 1, 'eval_success_rate': sr,
                'eval_avg_return': ar_undisc, 'eval_avg_discounted_return': ar_disc,
                'avg_steps_success': avg_ss, 'avg_steps_failure': avg_sf
            })

            _log(f"\n--- Eval Ep {episode + 1}/{total_episodes} [{shaping_type}] ---", verbose, logger=logger)
            _log(f"  Success Rate:              {sr:.2%}", verbose, logger=logger)
            _log(f"  Avg Return (Undiscounted): {ar_undisc:.4f}", verbose, logger=logger)
            _log(f"  Avg Discounted Return:     {ar_disc:.4f}", verbose, logger=logger)
            _log(f"  Avg Steps (Succ/Fail):     {avg_ss:.1f} / {avg_sf:.1f}", verbose, logger=logger)
            _log(f"  Epsilon:                   {epsilon:.4f}", verbose, logger=logger)

            if strategy is not None:
                current_stage_budget = strategy.wrapper.initial_budget
                _log(f"  Budget (episode initial):  {current_stage_budget}/{strategy.max_budget}",
                     verbose, logger=logger)
                top5 = sorted(global_noslip_by_state.items(), key=lambda x: -x[1])[:5]
                _log(f"  No-slip top-5 états:       {top5}", verbose, logger=logger)
                recent = episode_log[-config.EVAL_FREQUENCY:]
                succ_ns = [e['noslip_used'] for e in recent if e['success'] == 1]
                fail_ns = [e['noslip_used'] for e in recent if e['success'] == 0]
                _log(f"  No-slip moy. (succès): {np.mean(succ_ns):.2f}" if succ_ns
                     else "  No-slip moy. (succès): N/A", verbose, logger=logger)
                _log(f"  No-slip moy. (échec):  {np.mean(fail_ns):.2f}" if fail_ns
                     else "  No-slip moy. (échec):  N/A", verbose, logger=logger)

                if current_stage_budget < strategy.max_budget:
                    sr_full, _, ar_disc_full, _, _ = strategy.eval_with_full_budget(
                        env, q_table, max_steps, config.EVAL_EPISODES)
                    _log(f"  [Budget max={strategy.max_budget}] "
                         f"SR={sr_full:.2%} | DiscReturn={ar_disc_full:.4f}", verbose, logger=logger)

            _log("  Greedy Policy:", verbose, logger=logger)
            policy_grid = utils.get_policy_grid_from_q_table(q_table, size, holes, goal)
            for row_str in (policy_grid or ["(Could not generate)"]):
                _log(f"    {row_str}", verbose, logger=logger)
            _log("-" * (size + 5), verbose, logger=logger)
            _maybe_save_qtable(q_table, shaping_type, noslip_strategy_name, instance_id,
                               total_episodes, episode + 1, verbose, run_dir)

    _log(f"\nTraining complete ({shaping_type}, strategy={noslip_strategy_name}). "
         f"Total steps: {total_steps_processed}", verbose, logger=logger)
    if action_size == 8:
        top10 = sorted(global_noslip_by_state.items(), key=lambda x: -x[1])[:10]
        _log(f"No-slip global top-10: {top10}", verbose, logger=logger)
    return q_table, episode_log, evaluation_log
