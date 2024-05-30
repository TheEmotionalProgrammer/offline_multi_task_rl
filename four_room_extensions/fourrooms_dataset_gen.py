import os
from pathlib import Path

import gymnasium as gym
from typing import Any, Dict, List, Union
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from stable_baselines3.dqn.dqn import DQN

from four_room.env import FourRoomsEnv
from four_room.shortest_path import find_all_action_values
from four_room.utils import obs_to_state
from four_room.wrappers import gym_wrapper

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import dill
import imageio

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)


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


def get_random_dataset(num_steps: int = 1000):
    """
    This function generates a dataset using a random policy for the FourRooms environment.
    """
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


def get_expert_dataset(num_steps: int = 1000):
    """
    This function generates a dataset using the expert policy for the FourRooms environment.
    """
    env = wrap_env(gym_wrapper(gym.make(('MiniGrid-FourRooms-v1'))))
    observation, info = env.reset()

    dataset = {'observations': [], 'next_observations': [], 'actions': [], 'rewards': [],
               'terminals': [], 'timeouts': [], 'infos': []}
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
    return dataset


def get_dataset_from_config(config, policy=0, render=False, render_name="") -> tuple[Dict[str, Any], gym.Env, int, int]:
    '''
    Generates a dataset from the tasks specified in config. Size of returned dataset thus depends on amount of tasks
    specified in config as well as on the quality of the policy used to generate the dataset. If step_limit=True is
    used as argument the generation of data samples is stopped after num_steps steps. If all task in config are
    completed before num_steps a smaller dataset is returned. The policy argument takes an int, where 0=expert,
    1=random.
    '''
    gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)
    env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1',
                               agent_pos=config['agent positions'],
                               goal_pos=config['goal positions'],
                               doors_pos=config['topologies'],
                               agent_dir=config['agent directions'],
                               render_mode="rgb_array"))
    
    tasks_finished = 0
    tasks_failed = 0

    dataset = {'observations': [], 'next_observations': [], 'actions': [], 'rewards': [],
               'terminals': [], 'timeouts': [], 'infos': []}

    imgs = []

    first_observations = set()
    # with Display(visible=False) as disp:    # TODO why?
    for _ in range(len(config["topologies"])):
        observation, _ = env.reset()
        done = False
        while not done:
            imgs.append(env.render()) if render else None
            if policy == 0:
                state = obs_to_state(observation)
                q_values = find_all_action_values(state[:2], state[2], state[3:5], state[5:], 0.99)
                action = np.argmax(q_values)
            elif policy == 1:
                action = env.action_space.sample()
            elif policy == 2: #suboptimal policy, with a 70% chance of going in the right direction
                state = obs_to_state(observation)
                q_values = find_all_action_values(state[:2], state[2], state[3:5], state[5:], 0.99)
                action = np.argmax(q_values)
                if np.random.rand() < 0.15: # 15% chance of going in a random direction
                    action = env.action_space.sample()
            else:
                # implement default behaviour or return error, for now just uses random policy
                action = env.action_space.sample()

            last_observation = observation
            observation, reward, terminated, truncated, info = env.step(action)

            dataset['observations'].append(np.array(last_observation).flatten())
            dataset['next_observations'].append(np.array(observation).flatten())
            dataset['actions'].append(np.array([action]))
            dataset['rewards'].append(reward)
            dataset['terminals'].append(terminated)
            dataset['timeouts'].append(truncated)
            dataset['infos'].append(info)

            if terminated:
                tasks_finished += 1
            if truncated:
                tasks_failed += 1
            done = terminated or truncated

    for key in dataset:
        dataset[key] = np.array(dataset[key])

    render_name = f"{render_name}" if render_name else f'rendered_episode_{"random" if policy else "expert"}'
    imageio.mimsave(f'rendered_episodes/{render_name}.gif', [np.array(img) for i, img in enumerate(imgs) if i%1 == 0], duration=200) if render else None

    return dataset, env, tasks_finished, tasks_failed


def get_config_isidoro(path):
    with open(path, 'rb') as file:
        train_config = dill.load(file)
    file.close()
    return train_config


def get_config(config_data: str):
    with open(f'../four_room/configs/fourrooms_{config_data}_config.pl', 'rb') as file:
        train_config = dill.load(file)
    file.close()
    return train_config


def get_expert_dataset_from_config(config, render=False, render_name=""):
    return get_dataset_from_config(config, policy=0, render=render, render_name=render_name)


def get_random_dataset_from_config(config, render=False, render_name=""):
    return get_dataset_from_config(config, policy=1, render=render, render_name=render_name)


def get_suboptimal_dataset_from_config(config, render=False, render_name=""):
    return get_dataset_from_config(config, policy=2, render=render, render_name=render_name)


def get_mixed_dataset_from_config(config, train_env, checkpoints):
    return load_dqn_models(config, train_env, checkpoints)


def load_dqn_models(config, train_env, checkpoints_list):
    parent_dir = Path(os.getcwd()).parents[0]
    checkpoints_path = os.path.join(parent_dir, 'four_room_extensions', 'DQN_models')
    datasets = {'observations': [], 'next_observations': [], 'actions': [], 'rewards': [], 'terminals': [], 'timeouts': [], 'infos': []}
    finished = 0
    failed = 0
    start = 0
    configurations_per_policy = len(config["topologies"]) // len(checkpoints_list)
    for checkpoint in os.listdir(checkpoints_path):
        time_step = checkpoint[checkpoint.find('_')+1: checkpoint.find('.')]
        if time_step in checkpoints_list and checkpoint.endswith('.zip'):
            model = DQN.load(os.path.join(checkpoints_path, checkpoint), env=train_env)
            # print(f"configuration {start}")

            dataset = {'observations': [], 'next_observations': [], 'actions': [], 'rewards': [],
                       'terminals': [], 'timeouts': [], 'infos': []}
            tasks_finished = 0
            tasks_failed = 0

            for i in range(start, min(start + configurations_per_policy, len(config["topologies"]))):
                state, _ = train_env.reset()
                done = False
                while not done:
                    last_observation = state
                    action, _ = model.predict(state)
                    state, reward, terminated, truncated, info = train_env.step(action)

                    dataset['observations'].append(np.array(last_observation).flatten())
                    dataset['next_observations'].append(np.array(state).flatten())
                    dataset['actions'].append(np.array([action]))
                    dataset['rewards'].append(reward)
                    dataset['terminals'].append(terminated)
                    dataset['timeouts'].append(truncated)
                    dataset['infos'].append(info)

                    if terminated:
                        tasks_finished += 1
                    if truncated:
                        tasks_failed += 1
                    done = terminated or truncated

            start += configurations_per_policy
            finished += tasks_finished
            failed += tasks_failed
            print(f"Data policy: {time_step}, Num_transitions: {len(dataset['observations'])}")
            for key in dataset:
                dataset[key] = np.array(dataset[key])
                datasets[key].extend(dataset[key])

    for key in datasets:
        datasets[key] = np.array(datasets[key])

    return datasets, finished, failed
