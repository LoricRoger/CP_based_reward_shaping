# Filename: environment.py
import random
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit  # Import TimeLimit to check its type
import time
import config


# random.seed(config.seed_value)
# np.random.seed(config.seed_value)

# MODIFIED create_environment function to handle 'desc' argument
def create_environment(map_name, is_slippery, render_mode, desired_max_steps=None, desc=None):
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

    Returns:
        gym.Env: The FrozenLake environment instance.
    """
    if desc:
        print(f"Creating Gym environment: using custom `desc`, is_slippery={is_slippery}, render_mode='{render_mode}'")
    else:
        print(
            f"Creating Gym environment: map_name='{map_name}', is_slippery={is_slippery}, render_mode='{render_mode}'")

    try:
        # Create the base environment using desc if provided, otherwise map_name
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

        # --- Check for and handle existing TimeLimit wrapper ---
        env_to_wrap = env_maybe_wrapped
        default_limit = env_maybe_wrapped.spec.max_episode_steps if env_maybe_wrapped.spec else 'N/A'
        print(f"  Default env.spec.max_episode_steps from registration: {default_limit}")

        if isinstance(env_maybe_wrapped, TimeLimit):
            env_to_wrap = env_maybe_wrapped.env
            print(f"  Detected implicit TimeLimit wrapper added by gym.make.")
        # -----------------------------------------------------

        # Apply the desired TimeLimit wrapper
        if desired_max_steps is not None:
            env = TimeLimit(env_to_wrap, max_episode_steps=desired_max_steps)
            final_limit = env.spec.max_episode_steps if env.spec else 'N/A (after wrap)'
            print(f"  Applied TimeLimit wrapper. Effective max_episode_steps: {final_limit}")
        else:
            env = env_maybe_wrapped
            print(f"  Using environment default limit: {default_limit}")

        # Ensure the final environment has a spec if possible
        if env.spec is None and hasattr(env_to_wrap, 'spec') and env_to_wrap.spec is not None:
            env.spec = env_to_wrap.spec
            if desired_max_steps is not None:
                env.spec.max_episode_steps = desired_max_steps

        return env
    except Exception as e:
        print(f"ERROR creating environment: {e}")
        import traceback
        traceback.print_exc()
        return None


# Keep the visualize_agent function as is
def visualize_agent(env, q_table, num_episodes=5):
    """
    Visualize the agent's behavior over several episodes.
    (Assumes env was created with render_mode='human')

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
        env.render()  # Render initial state
        time.sleep(0.5)  # Pause briefly

        step = 0
        vis_max_steps = env.spec.max_episode_steps if env.spec else 200  # Use spec if available

        while not done and not truncated and step < vis_max_steps:
            action = np.argmax(q_table[state])  # Exploit policy
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state
            done = terminated
            env.render()
            time.sleep(0.1)  # Slow down visualization
            step += 1
            if truncated:
                print("  (Visualization truncated by step limit)")
                break

        print(f"  Episode finished. Success: {reward == 1.0 and terminated}. Steps: {step}")
        time.sleep(1)  # Pause after episode end

    print("--- Visualization finished ---")
