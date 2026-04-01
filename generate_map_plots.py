import json
import os
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv  # For type checking

# Project-specific environment creation function
import environment as fl_env


def get_map_desc_from_env(instance_id, instance_config):
    """
    Creates a temporary environment using fl_env.create_environment
    to extract its map description (desc).
    """
    map_name = instance_config.get("map_name")
    desc_from_config = instance_config.get("desc")
    is_slippery = instance_config.get("slippery", False)
    max_steps = instance_config.get("max_steps", 100)

    temp_env_outer = fl_env.create_environment(
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode=None,
        desired_max_steps=max_steps,
        desc=desc_from_config
    )

    if temp_env_outer is None:
        print(f"Failed to create temporary environment for instance {instance_id}.")
        return None

    # Traverse the wrapper stack to get to the base FrozenLakeEnv
    actual_env = temp_env_outer
    while hasattr(actual_env, 'env') and not isinstance(actual_env, FrozenLakeEnv):
        actual_env = actual_env.env

    # Final check if we reached a FrozenLakeEnv
    if not isinstance(actual_env, FrozenLakeEnv) or not hasattr(actual_env, 'desc'):
        print(
            f"Could not access base FrozenLakeEnv or its 'desc' for {instance_id}. Actual env type: {type(actual_env)}")
        temp_env_outer.close()
        return None

    map_description_raw = actual_env.desc
    temp_env_outer.close()

    # Convert map_description_raw (which is often list of list of bytes) to list of strings
    map_description_list_of_strings = []
    try:
        if isinstance(map_description_raw, list) and \
                all(isinstance(row, list) for row in map_description_raw) and \
                map_description_raw:  # Ensure not empty
            if all(isinstance(c, bytes) for c in map_description_raw[0]):  # List of lists of bytes
                map_description_list_of_strings = ["".join(c.decode('utf-8') for c in row) for row in
                                                   map_description_raw]
            elif all(isinstance(c, str) for c in map_description_raw[0]):  # List of lists of chars
                map_description_list_of_strings = ["".join(row) for row in map_description_raw]
        elif isinstance(map_description_raw, list) and \
                all(isinstance(row, str) for row in map_description_raw):  # Already list of strings
            map_description_list_of_strings = map_description_raw
        elif isinstance(map_description_raw, np.ndarray):  # Numpy array of chars/bytes
            # Convert to list of lists of native Python types first
            list_of_lists = map_description_raw.tolist()
            if list_of_lists and isinstance(list_of_lists[0][0], bytes):
                map_description_list_of_strings = ["".join(c.decode('utf-8') for c in row) for row in list_of_lists]
            elif list_of_lists and isinstance(list_of_lists[0][0], str):
                map_description_list_of_strings = ["".join(row) for row in list_of_lists]
        else:
            print(f"Unrecognized format for map_description_raw for {instance_id}: {type(map_description_raw)}")
            return None

    except Exception as e:
        print(f"Error converting map_description_raw for {instance_id}: {e}. Raw desc: {map_description_raw}")
        return None

    if not map_description_list_of_strings:
        print(
            f"Extracted 'desc' for {instance_id} but conversion resulted in empty list. Raw desc was: {map_description_raw}")
        return None

    return map_description_list_of_strings


def plot_frozen_lake_map(desc, title, filename, plot_dir="notebook_plots/maps"):
    os.makedirs(plot_dir, exist_ok=True)
    nrow = len(desc)
    ncol = len(desc[0])
    cmap_dict = {'S': 0, 'F': 1, 'H': 2, 'G': 3}
    for r_idx, row_str in enumerate(desc):
        for c_idx, char_val in enumerate(row_str):
            if char_val not in cmap_dict:
                print(
                    f"Error: Character '{char_val}' at row {r_idx}, col {c_idx} in map '{title}' is not recognized (S,F,H,G).")
                return
    num_desc = np.array([[cmap_dict[c] for c in row] for row in desc])
    fig, ax = plt.subplots(figsize=(ncol * 0.6, nrow * 0.6))
    colors = ['lightblue', 'lightgrey', 'dimgray', 'lightgreen']
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors)
    ax.imshow(num_desc, cmap=cmap, interpolation='nearest')
    for r in range(nrow):
        for c in range(ncol):
            ax.text(c, r, desc[r][c], ha="center", va="center", color="black", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title(title, fontsize=11)
    ax.set_xticks(np.arange(-.5, ncol, 1), minor=True)
    ax.set_yticks(np.arange(-.5, nrow, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    ax.set_frame_on(True)
    plt.tight_layout()
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"Saved map: {filepath}")


if __name__ == "__main__":
    output_directory = "notebook_plots/maps"
    os.makedirs(output_directory, exist_ok=True)
    try:
        with open("instances.json", 'r') as f:
            all_configs = json.load(f)
    except FileNotFoundError:
        print("Error: instances.json not found.")
        exit()

    instance_ids_to_plot = ["4s", "4medium", "4hard", "8s", "8medium", "8hard", "6dw"]
    for instance_id in instance_ids_to_plot:
        if instance_id in all_configs:
            config = all_configs[instance_id]
            map_desc_list = get_map_desc_from_env(instance_id, config)
            if map_desc_list:
                plot_title = config.get("description", f"FrozenLake {instance_id}")
                file_name = f"{instance_id}_map.png"
                plot_frozen_lake_map(map_desc_list, plot_title, file_name, plot_dir=output_directory)
            else:
                print(f"Skipping plot for instance {instance_id} due to map description extraction issue.")
        else:
            print(f"Warning: Instance ID '{instance_id}' not found in instances.json. Skipping plot.")
    print("Map generation complete.")
