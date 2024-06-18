# from math import trunc
# from os import truncate
# from venv import create
# import gymnasium as gym
# import dill
# from four_room.env import FourRoomsEnv
# from four_room.wrappers import gym_wrapper
# from d3rlpy.algos import DiscreteDecisionTransformerConfig
# from d3rlpy.metrics import EnvironmentEvaluator
# from d3rlpy.datasets import MDPDataset

# gym.register('MiniGrid-FourRooms-v0', FourRoomsEnv)
# with open('four_room/configs/fourrooms_train_config.pl', 'rb') as file:
#     train_config = dill.load(file)
# env = gym_wrapper(gym.make('MiniGrid-FourRooms-v0',
#                            agent_pos=train_config['agent positions'], 
#                            goal_pos=train_config['goal positions'], 
#                            doors_pos=train_config['topologies'], 
#                            agent_dir=train_config['agent directions']))

# # Generate dataset
# dataset = MDPDataset

# dt = DiscreteDecisionTransformerConfig().create(device="cpu")

# # calculate metrics with training dataset
# td_error_evaluator = TDErrorEvaluator(episodes=dataset.episodes)

# # set environment in scorer function
# env_evaluator = EnvironmentEvaluator(env)

# # offline training
# dt.fit(dataset,
#        n_steps=100000,
#        n_steps_per_epoch=1000,
#        eval_env=env,
#        eval_target_return=0,
#        evaluators={
#            "td_error": td_error_evaluator,
#            "environment": env_evaluator,
#            },
#        )

# # # wrap as stateful actor for interaction
# # actor = dt.as_stateful_wrapper(target_return=0)

# # # interaction 
# # observation, reward = env.reset(), 0.0
# # while True:
# #     action = actor.predict(observation, reward)
# #     observation, reward, terminated, truncated, info = env.step(action)
# #     if terminated or truncated:
# #         break
    
# # # reset history
# # actor.reset()  


    
