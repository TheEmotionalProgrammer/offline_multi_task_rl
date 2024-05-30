import argparse
import os
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import wandb
from four_room.env import FourRoomsEnv
from four_room.wrappers import gym_wrapper
import four_room_extensions
from agent import IQL
from four_room_extensions.fourrooms_dataset_gen import get_expert_dataset_from_config, get_random_dataset_from_config, get_mixed_dataset_from_config
from four_room_extensions.process_DQN_checkpoints import get_checkpoint_performance
from four_room_extensions.sac_n_discrete import ReplayBuffer
import time


def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="IQL", help="Run name, default: SAC")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes, default: 100")
    parser.add_argument("--num_updates_per_episode", type=int, default=50, help="Number of updates per episode, default: 100")
    parser.add_argument("--buffer_size", type=int, default=100_000, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size, default: 256")
    parser.add_argument("--hidden_size", type=int, default=256, help="")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for regularization")
    parser.add_argument("--temperature", type=float, default=100, help="")  # 3 ?
    parser.add_argument("--expectile", type=float, default=0.8, help="")  # in the paper it is 0.95
    parser.add_argument("--tau", type=float, default=0.01, help="")
    parser.add_argument("--save_every", type=int, default=10, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--eval_every", type=int, default=1, help="")

    args = parser.parse_args()
    return args


def create_env(eval_config):
    gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)
    env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1',
                               agent_pos=eval_config['agent positions'],
                               goal_pos=eval_config['goal positions'],
                               doors_pos=eval_config['topologies'],
                               agent_dir=eval_config['agent directions'],
                               render_mode="rgb_array"))
    return env


def evaluate(policy, env, n_configs):
    """
    Makes an evaluation run with the current policy
    """
    reward_batch = []
    tasks_finished = 0
    tasks_failed = 0

    num_steps_list = []
    for i in range(n_configs):
        state = env.reset()
        rewards = 0
        num_steps = 0
        while True:
            action = policy.get_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            num_steps += 1
            rewards += reward
            if terminated:
                tasks_finished += 1
            if truncated:
                tasks_failed += 1
            if terminated or truncated:
                break
        reward_batch.append(rewards)
        num_steps_list.append(num_steps)
    return np.mean(reward_batch), tasks_finished, tasks_failed, np.mean(num_steps_list)


def train(config, policy="expert"):
    train_config = four_room_extensions.fourrooms_dataset_gen.get_config(config_data="train")
    if policy == "expert":
        dataset, train_env, tasks_finished, tasks_failed = get_expert_dataset_from_config(train_config)
    elif policy == "random":
        dataset, train_env, tasks_finished, tasks_failed = get_random_dataset_from_config(train_config)
    elif policy == "mixed":
        parent_dir = Path(os.getcwd()).parents[0]
        checkpoints = get_checkpoint_performance(parent_dir, episode_length=[0, 25, 50, 75, 100], best_policy=True)
        train_env = create_env(train_config)
        dataset, tasks_finished, tasks_failed = get_mixed_dataset_from_config(train_config, train_env, checkpoints)
    print("Train terminated: " + str(tasks_finished))
    print("Train truncated: " + str(tasks_failed))

    # # Seeds
    # np.random.seed(config.seed)
    # random.seed(config.seed)
    # torch.manual_seed(config.seed)
    # env.seed(config.seed)
    # env.action_space.seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print("Device: ", device)

    observations = train_env.observation_space.shape[0] * train_env.observation_space.shape[1] * train_env.observation_space.shape[2]

    # Initialize the replay buffer
    buffer = ReplayBuffer(observations, train_env.action_space.n, config.buffer_size, device)

    buffer.load_d4rl_dataset(dataset)

    batches = 0

    # Evaluation env during training
    eval_env = create_env(train_config)

    # Test env for reachable states
    test_reachable_config = four_room_extensions.fourrooms_dataset_gen.get_config(config_data="test_100")
    test_reachable_env = create_env(test_reachable_config)

    # Test env for unreachable states
    test_unreachable_config = four_room_extensions.fourrooms_dataset_gen.get_config(config_data="test_0")
    test_unreachable_env = create_env(test_unreachable_config)

    rewards_reachable = []
    rewards_unreachable = []

    with wandb.init(project="IQL-offline", name=config.run_name, config=config):

        agent = IQL(state_size=observations,
                    action_size=train_env.action_space.n,
                    device=device,
                    learning_rate=config.learning_rate,
                    hidden_size=config.hidden_size,
                    tau=config.tau,
                    temperature=config.temperature,
                    expectile=config.expectile,
                    weight_decay=config.weight_decay)

        wandb.watch(agent, log="gradients", log_freq=10)
        eval_reward, _, _, num_steps = evaluate(agent, eval_env, n_configs=len(train_config["topologies"]))
        wandb.log({"Eval Reward": eval_reward, "Episode": 0, "Avg num steps to goal: evaluation": num_steps}, step=batches)
        start_time = time.time()
        for i in range(1, config.episodes + 1):
            start_episode_time = time.time()
            for _ in range(config.num_updates_per_episode):
                states, actions, rewards, next_states, dones = buffer.sample(config.batch_size)

                dones = dones.clone().detach().to(device, dtype=torch.bool)
                actions = torch.tensor([actions[i][0] for i in range(len(actions))]).unsqueeze(dim=0).to(device)

                policy_loss, critic1_loss, critic2_loss, value_loss = agent.learn((states, actions, rewards, next_states, dones))
                batches += 1

            if i % config.eval_every == 0:
                eval_reward, terminated_eval, truncated_eval, num_steps_eval = evaluate(agent, eval_env, n_configs=len(train_config["topologies"]))
                wandb.log({"Eval Reward": eval_reward, "Episode": i, "Avg num steps to goal: evaluation": num_steps_eval}, step=batches)
                print("Episode: {} | Reward: {} | Polciy Loss: {} | Batches: {} | terminated: {} | truncated {} | num_steps: {}"
                      .format(i, eval_reward, policy_loss, batches, terminated_eval, truncated_eval, num_steps_eval))

                # Test Reachable
                reachable_reward, terminated_reachable, truncated_reachable, num_steps_reachable = evaluate(agent, test_reachable_env, n_configs=len(train_config["topologies"]))
                wandb.log({"Test Cumulative Reachable Reward": reachable_reward, "Episode": i, "Num steps to goal: reachable": num_steps_reachable})
                print("Test Reachable: {} | Reward: {} | Steps to goal: {} Terminated: {} | Truncated: {}"
                      .format(i, reachable_reward, num_steps_reachable, terminated_reachable, truncated_reachable))
                rewards_reachable.append(reachable_reward)

                # Test Unreachable
                unreachable_reward, terminated_unreachable, truncated_unreachable, num_steps_unreachable = evaluate(agent, test_unreachable_env, n_configs=len(train_config["topologies"]))
                wandb.log({"Test Cumulative Unreachable Reward": unreachable_reward, "Episode": i, "Num steps to goal: unreachable": num_steps_unreachable})
                print("Test Unreachable: {} | Reward: {} | Steps to goal: {} Terminated: {} | Truncated: {}"
                      .format(i, unreachable_reward, num_steps_unreachable, terminated_unreachable, truncated_unreachable))
                rewards_unreachable.append(unreachable_reward)

            wandb.log({
                "Policy Loss": policy_loss,
                "Value Loss": value_loss,
                "Critic 1 Loss": critic1_loss,
                "Critic 2 Loss": critic2_loss
            })

            print("--- %s seconds ---" % (time.time() - start_episode_time))

            # if i % config.episodes == 0:
            #     save_model(agent, filename=f"model_{config.episodes}_{config.num_updates_per_episode}.pth")

        print("Avg Reachable Reward: ", np.mean(rewards_reachable))
        print("Avg Unreachable Reward: ", np.mean(rewards_unreachable))

        print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    config = get_config()
    train(config, policy="mixed")
