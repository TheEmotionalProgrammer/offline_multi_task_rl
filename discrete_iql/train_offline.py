# import d4rl
import argparse
import random
from collections import deque

import numpy as np
import torch
import wandb

from agent import IQL
from discrete_iql.networks import Actor
from four_room_extensions.fourrooms_dataset_gen import get_expert_dataset
from four_room_extensions.sac_n_discrete import ReplayBuffer
from utils import save

save_model = 0


def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="IQL", help="Run name, default: SAC")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes, default: 100")
    parser.add_argument("--num_updates_per_episode", type=int, default=1000, help="Number of updates per episode, default: 100")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--save_every", type=int, default=10, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--buffer_size", type=int, default=100_000, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size, default: 256")
    parser.add_argument("--hidden_size", type=int, default=256, help="")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="")
    parser.add_argument("--temperature", type=float, default=3, help="")
    parser.add_argument("--expectile", type=float, default=0.7, help="")
    parser.add_argument("--tau", type=float, default=5e-3, help="")
    parser.add_argument("--eval_every", type=int, default=1, help="")

    args = parser.parse_args()
    return args


def evaluate(env, policy, eval_runs=5):
    """
    Makes an evaluation run with the current policy
    """
    reward_batch = []
    for i in range(eval_runs):
        state = env.reset()

        rewards = 0
        while True:
            action = policy.get_action(state, eval=True)

            state, reward, terminated, truncated, _ = env.step(action)

            # # Check negative reward
            # if reward < 0:
            #     print(action, reward, state, terminated, truncated)

            rewards += reward

            # Truncated gives reward
            if terminated or truncated:
                break
        reward_batch.append(rewards)
    return np.mean(reward_batch)


def train(config):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Load the dataset
    dataset, env = get_expert_dataset()

    env.action_space.seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    observations = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]

    # Initialize the replay buffer
    buffer = ReplayBuffer(observations, env.action_space.n, config.buffer_size, device)

    buffer.load_d4rl_dataset(dataset)

    batches = 0
    average10 = deque(maxlen=10)

    with wandb.init(project="IQL-offline", name=config.run_name, config=config):

        agent = IQL(state_size=observations,
                    action_size=env.action_space.n,
                    device=device)
        # learning_rate = config.learning_rate,
        # hidden_size = config.hidden_size,
        # tau = config.tau,
        # temperature = config.temperature,
        # expectile = config.expectile,

        wandb.watch(agent, log="gradients", log_freq=10)
        eval_reward = evaluate(env, agent)
        wandb.log({"Test Reward": eval_reward, "Episode": 0, "Batches": batches}, step=batches)
        for i in range(1, config.episodes + 1):
            for _ in range(config.num_updates_per_episode):
                states, actions, rewards, next_states, dones = buffer.sample(config.batch_size)
                dones = torch.tensor(dones, dtype=torch.bool).to(device)
                actions = torch.tensor([actions[i][0] for i in range(len(actions))]).unsqueeze(dim=0).to(device)

                policy_loss, critic1_loss, critic2_loss, value_loss = agent.learn((states, actions, rewards,
                                                                                   next_states, dones))
                batches += 1

            if i % config.eval_every == 0:
                eval_reward = evaluate(env, agent)
                wandb.log({"Test Reward": eval_reward, "Episode": i, "Batches": batches}, step=batches)

                average10.append(eval_reward)
                print("Episode: {} | Reward: {} | Polciy Loss: {} | Batches: {}".format(i, eval_reward, policy_loss,
                                                                                        batches))

            wandb.log({
                "Average10": np.mean(average10),
                "Policy Loss": policy_loss,
                "Value Loss": value_loss,
                "Critic 1 Loss": critic1_loss,
                "Critic 2 Loss": critic2_loss})

            # works when developed mode on! (windows)
            # if i % config.save_every == 0:
            #     save(config, save_name="IQL", model=agent.actor_local, wandb=wandb, ep=config.episodes)


def test_iql(config, state_size, action_size, hidden_size, device):
    # TODO: use test config
    model = Actor(state_size, action_size, hidden_size).to(device)
    model.load_state_dict(torch.load(f"IQL_{config.episodes}.pth"))


if __name__ == "__main__":
    config = get_config()
    train(config)
