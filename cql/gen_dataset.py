import gymnasium as gym
import dill
import numpy as np
import imageio
from typing import Dict, Any
from stable_baselines3.dqn.dqn import DQN
import pickle

import sys
sys.path.append("C:/Users/aukes/Documents/Code/MSc Computer Science/CS4210-B Intelligent Decision Making Project/offline_multi_task_rl/")

from four_room.env import FourRoomsEnv
from four_room.wrappers import gym_wrapper
import os

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

def get_config(path):
    with open(path, 'rb') as file:
        train_config = dill.load(file)
    file.close()
    return train_config

def get_mixed_dataset_from_config(config, models=[300000, 350000, 390000, 450000, 470000], render=False, render_name="") -> tuple[Dict[str, Any], gym.Env]:
    '''
    Generates a dataset from multiple policies on the tasks specified in config. Size of returned dataset thus 
    depends on amount of tasks specified in config as well as on the quality of the policies used to generate the 
    dataset. If step_limit=True is used as argument the generation of data samples is stopped after num_steps steps. 
    If all task in config are completed before num_steps a smaller dataset is returned. The integers in models array
    should point to a pretrained model (a.k.a. policy) that can be loaded from the DQN_models folder.
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

    model = DQN.load(f"four_room_extensions/DQN_models/DQN_{models[0]}.zip")
    last_used_model = 0

    for i in range(len(config["topologies"])):

        model_index = i // (len(config["topologies"]) // len(models))
        if model_index != last_used_model:
            model = DQN.load(f"four_room_extensions/DQN_models/DQN_{models[model_index]}.zip")
            last_used_model = model_index
            print("last used model changed to: ", last_used_model)
        
        observation, _ = env.reset()
        done = False
        while not done:
            imgs.append(env.render()) if render else None
            
            action, _ = model.predict(observation)

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

    render_name_extension = '_'.join(map(str, models))
    render_name = f"{render_name}" if render_name else f'rendered_episode_{render_name_extension}'
    imageio.mimsave(f'rendered_episodes/{render_name}.gif', [np.array(img) for i, img in enumerate(imgs) if i%1 == 0], duration=200) if render else None

    return dataset, env


config = get_config("four_room/configs/fourrooms_train_config.pl")
models = [300000, 350000, 390000, 450000, 470000]
dataset, env = get_mixed_dataset_from_config(config, models)

dataset_file_name = "four_room_extensions/datasets/dataset_from_models_" + '_'.join(map(str, models)) + ".pkl"
with open(dataset_file_name, 'wb') as f:
  pickle.dump(dataset, f)