import gymnasium as gym
import dill
from env import FourRoomsEnv
from wrappers import gym_wrapper

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

with open('configs/fourrooms_train_config.pl', 'rb') as file:
    train_config = dill.load(file)

env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1', 
                                agent_pos=train_config['agent positions'], 
                                goal_pos=train_config['goal positions'], 
                                doors_pos=train_config['topologies'], 
                                agent_dir=train_config['agent directions']))

obs, _ = env.reset()
print(obs.shape)