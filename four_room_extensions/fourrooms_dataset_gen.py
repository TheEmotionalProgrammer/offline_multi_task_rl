import gymnasium as gym
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from four_room.env import FourRoomsEnv
from four_room.shortest_path import find_all_action_values
from four_room.utils import obs_to_state
from four_room.wrappers import gym_wrapper
gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)


num_steps = 1000

def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


def get_random_dataset():
    env = wrap_env(gym_wrapper(gym.make(('MiniGrid-FourRooms-v1'))))
    observation, info = env.reset()

    dataset = {'observations':[], 'next_observations':[], 'actions':[], 'rewards':[],
                'terminals':[], 'timeouts':[], 'infos':[]}
    for i in range(num_steps):
        action = env.action_space.sample()
        last_observation = observation
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

        dataset['observations'].append(np.array(last_observation).flatten())
        dataset['next_observations'].append(np.array(observation).flatten())
        dataset['actions'].append(np.array([action]))
        dataset['rewards'].append(reward)
        dataset['terminals'].append(terminated)
        dataset['timeouts'].append(truncated)
        dataset['infos'].append(info)

    for key in dataset:
        dataset[key] = np.array(dataset[key])
    return dataset

def get_expert_dataset():
    env = wrap_env(gym_wrapper(gym.make(('MiniGrid-FourRooms-v1'))))
    observation, info = env.reset()

    dataset = {'observations':[], 'next_observations':[], 'actions':[], 'rewards':[],
                'terminals':[], 'timeouts':[], 'infos':[]}
    for i in range(num_steps):

        state = obs_to_state(observation)
        q_values = find_all_action_values(state[:2], state[2], state[3:5], state[5:], 0.99)
        action = np.argmax(q_values)

        last_observation = observation
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

        dataset['observations'].append(np.array(last_observation).flatten())
        dataset['next_observations'].append(np.array(observation).flatten())
        dataset['actions'].append(np.array([action]))
        dataset['rewards'].append(reward)
        dataset['terminals'].append(terminated)
        dataset['timeouts'].append(truncated)
        dataset['infos'].append(info)

    for key in dataset:
        dataset[key] = np.array(dataset[key])
    return dataset, env


def get_expert_dataset_iql(batch_size=256):
    dataset, env = get_expert_dataset()

    observations_tensor = torch.tensor(dataset['observations'], dtype=torch.float32)
    next_observations_tensor = torch.tensor(dataset['next_observations'], dtype=torch.float32)
    actions_tensor = torch.tensor(dataset['actions'], dtype=torch.long)
    rewards_tensor = torch.tensor(dataset['rewards'], dtype=torch.float32)
    terminals_tensor = torch.tensor(dataset['terminals'], dtype=torch.bool)
    # timeouts_tensor = torch.tensor(dataset['timeouts'], dtype=torch.bool)
    # infos_tensor = torch.tensor(dataset['infos'], dtype=torch.float32)

    tensordata = TensorDataset(observations_tensor,
                               actions_tensor,
                               rewards_tensor,
                               next_observations_tensor,
                               terminals_tensor)

    dataloader = DataLoader(tensordata, batch_size=batch_size, shuffle=True)

    return dataloader, env
