import zipfile
import os

import gymnasium as gym
from gymnasium.core import Wrapper

from four_room.env import FourRoomsEnv
from four_room.wrappers import gym_wrapper

def unzip_models():
    path = os.path.join(os.getcwd(), 'four_room_extensions', 'DQN_models')
    for file in os.listdir(path):
        if file.endswith('.zip'):
            zip_path = os.path.join(path, file)

            file_name = file.split('.')[0]
            os.makedirs(os.path.join(path, file_name), exist_ok=True)

            unzip_path = os.path.join(path, file_name)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)


def list_files():
    path = os.path.join(os.getcwd(), 'four_room_extensions', 'DQN_models')
    for folder in os.listdir(path):
        fodler_path = os.path.join(path, folder)
        if os.path.isdir(fodler_path):
            print([file for file in os.listdir(fodler_path)])
            break
        break

def get_checkpoint_performance(path, episode_length=None, best_policy=False):
    """Select DQN policies based on the length of the episode

    Args:
        path (str): path to the performance_per_model.txt file
        episode_length (list, optional): list of episode lengths (between 0 and 100). Defaults to None. Ex: [0, 25, 50, 75, 100]
        best_policy (bool, optional): Selects the best 16 policies to get the policy from. Defaults to False.

    Returns:
        list of the selected policies with distribution based on the episode length
    """

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

    # select the best 16 checkpoints (16 is based on my observations - maximum 55 steps per episode)
    if best_policy:
        checkpoints = checkpoints[:16]
        max_steps = checkpoints[-1][1]
        # adjust the percentage of the episode length
        episode_length = [i * max_steps / 100 for i in episode_length]

    if episode_length is None:
        selected_policies = checkpoints
    else:
        selected_policies = []
        for episode_length_goal in episode_length:
            selected_policies.append(min(checkpoints, key=lambda x: abs(x[1] - episode_length_goal)))   # select the closest number of steps to the percentage
        print(f"policies {best_policy,sorted(reverse=True)}: {selected_policies}")
    # return a list of timesteps
    return [timestep for timestep, b in selected_policies]


def create_env(config):
    gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)
    env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1',
                               agent_pos=config['agent positions'],
                               goal_pos=config['goal positions'],
                               doors_pos=config['topologies'],
                               agent_dir=config['agent directions'],
                               render_mode="rgb_array"))
    return env


class ObservationFlattenerWrapper(Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs.flatten(), reward, terminated, truncated, info
    
    def reset(self):
        obs, info = self.env.reset()
        return obs.flatten(), info