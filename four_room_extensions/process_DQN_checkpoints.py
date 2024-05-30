import zipfile
import os
from collections import OrderedDict


def unzip_models():
    path = os.path.join(os.getcwd(), 'DQN_models')
    for file in os.listdir(path):
        if file.endswith('.zip'):
            zip_path = os.path.join(path, file)

            file_name = file.split('.')[0]
            os.makedirs(os.path.join(path, file_name), exist_ok=True)

            unzip_path = os.path.join(path, file_name)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)


def list_checkpoint_files():
    path = os.path.join(os.getcwd(), 'DQN_models')
    for folder in os.listdir(path):
        fodler_path = os.path.join(path, folder)
        if os.path.isdir(fodler_path):
            print([file for file in os.listdir(fodler_path)])
            break
        break


def get_checkpoint_performance(path, episode_length=None, best_policy=False):
    """Select policies based on the length of the episode"""
    # use path when you run this script from another folder
    with open(os.path.join(path, 'four_room_extensions', 'DQN_models', 'performance_per_model.txt'), 'r') as f:
        checkpoints = []
        for line in f.readlines():
            timestep = line.split(' - ')[0]
            timestep = timestep.replace('_', '')
            num_steps = float(line[line.find(':')+2: line.find(',')])
            checkpoints.append((timestep, num_steps))
    # sort checkpoints by number of steps
    checkpoints.sort(key=lambda x: x[1])
    # keep only unique checkpoints
    # TODO: maybe I want to keep the best out of the duplicates???
    seen = set()
    checkpoints = [(a, num_steps) for a, num_steps in checkpoints if not (num_steps in seen or seen.add(num_steps))]

    if best_policy:
        # select the best 16 checkpoints (16 is based on my observations - maximum 55 steps per episode)
        checkpoints = checkpoints[:16]
        max_steps = checkpoints[-1][1]
        # adjust the percentage of the episode length
        episode_length = [i * max_steps / 100 for i in episode_length]

    if episode_length is None:
        selected_policies = checkpoints
    else:
        selected_policies = []
        for percent in episode_length:
            selected_policies.append(min(checkpoints, key=lambda x: abs(x[1] - percent)))
        print(f"policies [100%, 75%, 50%, 25%, 0%]: {selected_policies}")
    # return a list of timesteps
    return [timestep for timestep, b in selected_policies]
