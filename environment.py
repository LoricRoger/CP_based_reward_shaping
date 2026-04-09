# Filename: environment.py
import random
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit

import config


def create_environment(map_name, is_slippery, render_mode, desired_max_steps=None, desc=None, budget=0):
    """
    Creates and returns the FrozenLake environment using map_name or desc,
    ensuring the desired max_episode_steps is applied correctly.

    Args:
        map_name (str): The standard map name ('4x4' or '8x8'), used if desc is None.
        is_slippery (bool): Whether the environment has slippery ice.
        render_mode (str or None): Rendering mode (e.g., 'human' for visualization).
        desired_max_steps (int, optional): If provided, ensures the environment
                                           uses this value as the step limit.
                                           Defaults to None (uses environment default).
        desc (list of str, optional): Custom map description. If provided, this is used
                                      instead of map_name. Defaults to None.
        budget (int): If provided, sets the budget for the environment. The number of non slippery actions available
                                 Defaults to 0.

    Returns:
        gym.Env: The FrozenLake environment instance, optionally wrapped.
    """
    if desc:
        print(
            f"Creating Gym environment: using custom `desc`, is_slippery={is_slippery}, render_mode='{render_mode}, with budget={budget}'")
    else:
        print(
            f"Creating Gym environment: map_name='{map_name}', is_slippery={is_slippery}, render_mode='{render_mode}, with budget={budget}'")

    try:
        if desc:
            env_maybe_wrapped = gym.make("FrozenLake-v1",
                                         desc=desc,
                                         is_slippery=is_slippery,
                                         render_mode=render_mode)
        else:
            env_maybe_wrapped = gym.make("FrozenLake-v1",
                                         map_name=map_name,
                                         is_slippery=is_slippery,
                                         render_mode=render_mode)

        env_to_wrap = env_maybe_wrapped
        default_limit = env_maybe_wrapped.spec.max_episode_steps if env_maybe_wrapped.spec else 'N/A'
        print(f"  Default env.spec.max_episode_steps from registration: {default_limit}")

        if isinstance(env_maybe_wrapped, TimeLimit):
            env_to_wrap = env_maybe_wrapped.env
            print(f"  Detected implicit TimeLimit wrapper added by gym.make.")

        if desired_max_steps is not None:
            env = TimeLimit(env_to_wrap, max_episode_steps=desired_max_steps)
            final_limit = env.spec.max_episode_steps if env.spec else 'N/A (after wrap)'
            print(f"  Applied TimeLimit wrapper. Effective max_episode_steps: {final_limit}")
        else:
            env = env_maybe_wrapped
            print(f"  Using environment default limit: {default_limit}")

        if env.spec is None and hasattr(env_to_wrap, 'spec') and env_to_wrap.spec is not None:
            env.spec = env_to_wrap.spec
            if desired_max_steps is not None:
                env.spec.max_episode_steps = desired_max_steps

        if budget > 0:
            env = FrozenLakeExtendedActions(env, budget=budget)

        return env

    except Exception as e:
        print(f"ERROR creating environment: {e}")
        import traceback
        traceback.print_exc()
        return None


class FrozenLakeExtendedActions(gym.Wrapper):
    """
    Wrapper qui double l'espace d'actions de FrozenLake :
      - Actions 0-3 : comportement original (glissement selon is_slippery)
      - Actions 4-7 : mêmes directions, mais toujours déterministes (sans glissement)

    La table de transitions étendue est calculée une seule fois à l'initialisation.
    Le budget représente le nombre d'actions déterministes disponibles par épisode.
    Si le budget est épuisé, les actions 4-7 se comportent comme 0-3 (glissantes).
    """

    def __init__(self, env: gym.Env, budget):
        super().__init__(env)
        self.action_space = spaces.Discrete(8)
        self.initial_budget = budget
        self.budget = budget
        self._modify_transition_matrix()

    def reset(self, **kwargs):
        self.budget = self.initial_budget
        return self.env.reset(**kwargs)

    def step(self, action: int):
        if action < 4:
            return self.env.step(action)

        direction = action - 4
        if self.budget > 0:
            self.budget -= 1
            lake_env = self._get_lake_env()
            lake_env.lastaction = direction
            return self.env.step(action)
        else:
            return self.env.step(direction)

    def _modify_transition_matrix(self):
        """
        Calcule et injecte les transitions déterministes (actions 4-7) dans la
        table P du FrozenLakeEnv. Appelé une seule fois à l'initialisation.
        """
        lake_env = self._get_lake_env()

        for state in range(lake_env.observation_space.n):
            col, row = state % lake_env.ncol, state // lake_env.ncol

            for action in range(4):
                new_col, new_row = col, row
                if action == 0:  # LEFT
                    new_col = max(col - 1, 0)
                elif action == 1:  # DOWN
                    new_row = min(row + 1, lake_env.nrow - 1)
                elif action == 2:  # RIGHT
                    new_col = min(col + 1, lake_env.ncol - 1)
                elif action == 3:  # UP
                    new_row = max(row - 1, 0)

                new_state = new_row * lake_env.ncol + new_col

                # Chercher reward et terminated pour cette destination
                target_reward, target_terminated = 0.0, False
                for _, s_next, r, t in lake_env.P[state][action]:
                    if s_next == new_state:
                        target_reward = r
                        target_terminated = t
                        break

                lake_env.P[state][action + 4] = [(1.0, new_state, target_reward, target_terminated)]

    def _get_lake_env(self):
        """Remonte la chaîne de wrappers jusqu'au FrozenLakeEnv (qui expose .P)."""
        env = self.env
        while env is not None:
            if hasattr(env, "P"):
                return env
            env = getattr(env, "env", None)
        raise RuntimeError("FrozenLakeEnv introuvable dans la chaîne de wrappers.")  
    

def visualize_agent(env, q_table, num_episodes=5):
    """
    Visualize the agent's behavior over several episodes.
    (Assumes env was created with render_mode='human')

    Fonctionne avec une Q-table de 4 ou 8 colonnes.

    Args:
        env: The FrozenLake environment instance.
        q_table: The trained Q-table.
        num_episodes: The number of episodes to visualize.
    """
    if env.render_mode != "human":
        print("Error: Visualization requires environment created with render_mode='human'")
        return
    if q_table is None:
        print("Error: Cannot visualize without a valid Q-table.")
        return

    print(f"\n--- Starting visualization for {num_episodes} episodes ---")
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        print(f"Visualization Episode {episode + 1}")
        env.render()
        time.sleep(0.5)

        step = 0
        vis_max_steps = env.spec.max_episode_steps if env.spec else 200

        while not done and not truncated and step < vis_max_steps:
            action = int(np.argmax(q_table[state]))
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state
            done = terminated
            env.render()
            time.sleep(0.1)
            step += 1
            if truncated:
                print("  (Visualization truncated by step limit)")
                break

        print(f"  Episode finished. Success: {reward == 1.0 and terminated}. Steps: {step}")
        time.sleep(1)

    print("--- Visualization finished ---")