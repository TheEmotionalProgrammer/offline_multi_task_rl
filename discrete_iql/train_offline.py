# import d4rl
import argparse
import random
from collections import deque
import gymnasium as gym
import numpy as np
import torch
import wandb
from four_room.env import FourRoomsEnv
from four_room.wrappers import gym_wrapper
import four_room_extensions
from agent import IQL
from four_room_extensions.fourrooms_dataset_gen import get_expert_dataset, get_expert_dataset_from_config
from four_room_extensions.sac_n_discrete import ReplayBuffer
from utils import save_model, load_model


def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="IQL", help="Run name, default: SAC")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes, default: 100")
    parser.add_argument("--num_updates_per_episode", type=int, default=500, help="Number of updates per episode, default: 100")
    parser.add_argument("--buffer_size", type=int, default=100_000, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size, default: 256")
    parser.add_argument("--hidden_size", type=int, default=256, help="")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for regularization")
    parser.add_argument("--temperature", type=float, default=100, help="")  # 3 ?
    parser.add_argument("--expectile", type=float, default=0.8, help="")  # in the paper it is 0.95
    parser.add_argument("--tau", type=float, default=5e-3, help="")
    # parser.add_argument("--save_every", type=int, default=10, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--eval_every", type=int, default=1, help="")

    args = parser.parse_args()
    return args


def evaluate(policy, eval_config, train=True, reachable=True):
    """
    Makes an evaluation run with the current policy
    """
    reward_batch = []
    tasks_finished = 0
    tasks_failed = 0

    gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)
    env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1',
                               agent_pos=eval_config['agent positions'],
                               goal_pos=eval_config['goal positions'],
                               doors_pos=eval_config['topologies'],
                               agent_dir=eval_config['agent directions'],
                               render_mode="rgb_array"))

    eval_runs = 5 if train else 40

    num_steps_list = []
    for i in range(eval_runs):
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
            # Truncated gives reward
            if terminated or truncated:
                break
        reward_batch.append(rewards)
        num_steps_list.append(num_steps)
        if not train:
            if reachable:
                wandb.log({"Test Reachable Reward": np.mean(reward_batch), "Episode": i + 1, "Num steps to goal: reachable": num_steps})
                print("Test Run: {} | Test Reachable Reward: {} | Num steps to goal: {}".format(i + 1, np.mean(reward_batch), num_steps))
            else:
                wandb.log({"Test Unreachable Reward": np.mean(reward_batch), "Episode": i + 1, "Num steps to goal: unreachable": num_steps})
                print("Test Run: {} | Test Unreachable Reward: {} | Num steps to goal: {}".format(i + 1, np.mean(reward_batch), num_steps))
    return np.mean(reward_batch), tasks_finished, tasks_failed, np.mean(num_steps_list)


def train(config):
    # np.random.seed(config.seed)
    # random.seed(config.seed)
    # torch.manual_seed(config.seed)

    # Load the dataset
    train_config = four_room_extensions.fourrooms_dataset_gen.get_config(config_data="train")
    dataset, env, tasks_finished, tasks_failed = get_expert_dataset_from_config(train_config)
    print("Train terminated: " + str(tasks_finished))
    print("Train truncated: " + str(tasks_failed))

    # env.seed(config.seed)
    # env.action_space.seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print("Device: ", device)

    observations = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]

    # Initialize the replay buffer
    buffer = ReplayBuffer(observations, env.action_space.n, config.buffer_size, device)

    buffer.load_d4rl_dataset(dataset)

    batches = 0

    with wandb.init(project="IQL-offline", name=config.run_name, config=config):

        agent = IQL(state_size=observations,
                    action_size=env.action_space.n,
                    device=device,
                    learning_rate=config.learning_rate,
                    hidden_size=config.hidden_size,
                    tau=config.tau,
                    temperature=config.temperature,
                    expectile=config.expectile,
                    weight_decay=config.weight_decay)

        wandb.watch(agent, log="gradients", log_freq=10)
        eval_reward, _, _, num_steps = evaluate(agent, train_config, train=True)
        wandb.log({"Eval Reward": eval_reward, "Episode": 0, "Avg num steps to goal: evaluation": num_steps}, step=batches)
        for i in range(1, config.episodes + 1):
            for _ in range(config.num_updates_per_episode):
                states, actions, rewards, next_states, dones = buffer.sample(config.batch_size)
                dones = dones.clone().detach().to(device, dtype=torch.bool)
                actions = torch.tensor([actions[i][0] for i in range(len(actions))]).unsqueeze(dim=0).to(device)

                policy_loss, critic1_loss, critic2_loss, value_loss = agent.learn((states, actions, rewards,
                                                                                   next_states, dones))
                batches += 1

            if i % config.eval_every == 0:
                eval_reward, terminated, truncated, num_steps = evaluate(agent, train_config, train=True)
                wandb.log({"Eval Reward": eval_reward, "Episode": i, "Avg num steps to goal: evaluation": num_steps}, step=batches)

                print("Episode: {} | Reward: {} | Polciy Loss: {} | Batches: {} | terminated: {} | truncated {} "
                      "| num_steps: {}".format(i, eval_reward, policy_loss, batches, terminated, truncated, num_steps))

            wandb.log({
                "Policy Loss": policy_loss,
                "Value Loss": value_loss,
                "Critic 1 Loss": critic1_loss,
                "Critic 2 Loss": critic2_loss
            })

            if i % config.episodes == 0:
                save_model(agent, filename=f"model_{config.episodes}_{config.num_updates_per_episode}.pth")


def test_loaded_model(config, save_path="./trained_models/", filename="model.pth"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    observations = 324
    actions = 3

    with wandb.init(project="IQL-offline", name=config.run_name, config=config):

        agent = IQL(state_size=observations,
                    action_size=actions,
                    device=device,
                    learning_rate=config.learning_rate,
                    hidden_size=config.hidden_size,
                    tau=config.tau,
                    temperature=config.temperature,
                    expectile=config.expectile,
                    weight_decay=config.weight_decay)

        load_model(agent, save_path, filename)

        # Testing
        test_reachable_config = four_room_extensions.fourrooms_dataset_gen.get_config(config_data="test_100")
        _, test_terminated, test_truncated, _ = evaluate(agent, test_reachable_config, train=False, reachable=True)
        print("Terminated reachable: " + str(test_terminated) + " | Truncated reachable: " + str(test_truncated))

        test_unreachable_config = four_room_extensions.fourrooms_dataset_gen.get_config(config_data="test_0")
        _, test_terminated, test_truncated, _ = evaluate(agent, test_unreachable_config, train=False, reachable=False)
        print("Terminated unreachable: " + str(test_terminated) + " | Truncated unreachable: " + str(test_truncated))


if __name__ == "__main__":
    config = get_config()
    train(config)
    # test_loaded_model(config, filename="model_10_500.pth")
