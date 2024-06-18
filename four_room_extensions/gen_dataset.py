from pathlib import Path
import gymnasium as gym
import dill
import numpy as np
import imageio
from typing import Dict, Any
from stable_baselines3.dqn.dqn import DQN
import pickle

import sys
sys.path.append(r'C:\Users\shaya\Documents\TU_projects\random\offline_multi_task_rl')

from four_room.shortest_path import find_all_action_values
from four_room.utils import obs_to_state
from utils import get_DQN_checkpoints

from four_room.env import FourRoomsEnv
from four_room.wrappers import gym_wrapper
import os

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

def get_config(path):
    with open(path, 'rb') as file:
        train_config = dill.load(file)
    file.close()
    return train_config

def get_mixed_dataset_from_config(config, models_paths: list[str], render=False, render_name="") -> tuple[Dict[str, Any], gym.Env]:
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

    for idx, model_path in enumerate(model_paths):
        if model_path == "real_expert":
            model = None
            model_name = "real_expert"
        else:
            model_name = model_path.split("/")[-2:]
            model_name = model_name[0] + "_" + model_name[1]
            model_name = model_name.replace(".zip", "")
            model = DQN.load(model_path)
            
        imgs = []
        for i in range(len(config["topologies"])):
            observation, _ = env.reset()
            done = False
            while not done:
                imgs.append(env.render()) if render else None
                
                if model_path == "real_expert":
                    state = obs_to_state(observation)
                    q_values = find_all_action_values(state[:2], state[2], state[3:5], state[5:], 0.99)
                    action = np.argmax(q_values)
                else:
                    action, _ = model.predict(observation)

                last_observation = observation
                observation, reward, terminated, truncated, info = env.step(action)

                dataset['observations'].append(last_observation.flatten())
                dataset['next_observations'].append(observation.flatten())
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
        print(f"progress: {idx+1}/{len(model_paths)}")
        Path("rendered_episodes/").mkdir(parents=True, exist_ok=True)
        render_name_ext = f"{render_name}__rendered_episode_{model_name}" if render_name else f'rendered_episode_{model_name}'
        imageio.mimsave(f'rendered_episodes/{render_name_ext}.gif', [np.array(img) for i, img in enumerate(imgs) if i%1 == 0], duration=200) if render else None

    for key in dataset:
        dataset[key] = np.array(dataset[key])

    return dataset, env, tasks_failed, tasks_finished

if __name__ == "__main__":
    config = get_config("four_room/configs/fourrooms_train_config.pl")
    
    # desired_episode_length = [  # TODO Add different trained models here here (0 is best model, 100 is worst model)
    #     [50],
    #     [25],
    #     [0],
    #     [0, 75],
    #     [0, 50],
    #     [25, 50],
    #     [0, 25],
    #     [0, 50, 100],
    #     [0, 25, 50],
    #     [0, 25, 50, 75],
    #     [0, 25, 50, 75, 100],
    # ]
    best_policy = False # TODO select from top 16 or not?

    # for episode_length in desired_episode_length:
    #     DQN_models_path = os.path.join("four_room_extensions", "DQN_models", "performance_per_model.txt")
    #     models = get_DQN_checkpoints(DQN_models_path, episode_length, best_policy=best_policy)    # TODO, change to True so selection can be any model and not just top 16


    dqn_dir = os.path.abspath(os.path.join(__file__, "..", "..", "models"))
    model_paths = [
        # "real_expert",
        dqn_dir + "/DQN_models/DQN_500000",
        dqn_dir + "/dqn_models_seed_1/DQN_700000_steps",
        dqn_dir + "/dqn_models_seed_2/DQN_700000_steps",
        # dqn_dir + "/dqn_models_seed_3/DQN_700000_steps",
        # dqn_dir + "/dqn_models_seed_4/DQN_700000_steps",
        # dqn_dir + "/DQN_models/DQN_390000",
        dqn_dir + "/DQN_models/DQN_330000",
        dqn_dir + "/DQN_models/DQN_110000",
    ]
    dataset_name_extension = "3_dqn_experts,330k,110k_0,0,0,50,75"
    dataset, env, tasks_failed, tasks_finished = get_mixed_dataset_from_config(config, model_paths, render=False)
    print(f"Tasks finished: {tasks_finished}, tasks failed: {tasks_failed}")

    dataset_path = "datasets/dataset_from_models_"
    Path(dataset_path).mkdir(parents=True, exist_ok=True)
    file_name_best_policy_extension = "_best_policies_only" if best_policy else ""
    dataset_file_name = dataset_path + "/" + dataset_name_extension + file_name_best_policy_extension + ".pkl"
    with open(dataset_file_name, 'wb') as f:
        pickle.dump(dataset, f)

        # To read the pickles (we should automate all this stuff tho):
        # with open(dataset_file_name, 'rb') as f:
        #     data = pickle.load(f)