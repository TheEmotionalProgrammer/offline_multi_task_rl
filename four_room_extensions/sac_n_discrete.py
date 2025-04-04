# Inspired by:
# 1. paper for SAC-N: https://arxiv.org/abs/2110.01548
# 2. implementation: https://github.com/snu-mllab/EDAC
import math
import os
import random
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

#import d4rl
import gymnasium as gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import wandb
from torch.distributions import Normal
from tqdm import trange
from torch.functional import F

from .fourrooms_dataset_gen import get_expert_dataset, get_random_dataset, get_config, get_dataset_from_config, get_expert_dataset_from_config, get_random_dataset_from_config
from four_room.env import FourRoomsEnv
from four_room.wrappers import gym_wrapper
gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)


@dataclass
class TrainConfig:
    # wandb params
    project: str = "CORL"
    group: str = "SAC-N"
    name: str = "SAC-N"
    # model params
    hidden_dim: int = 256
    num_critics: int = 10
    gamma: float = 0.99
    tau: float = 5e-3
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    max_action: float = 1.0
    # training params
    buffer_size: int = 1_000_000
    env_name: str = "MiniGrid-FourRooms-v1"
    batch_size: int = 256
    num_epochs: int = 10
    num_updates_on_epoch: int = 1000
    normalize_reward: bool = False
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 1
    # general params
    checkpoints_path: Optional[str] = None
    deterministic_torch: bool = False
    train_seed: int = 10
    eval_seed: int = 42
    log_every: int = 100
    device: str = "cpu"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


# general utils
TensorBatch = List[torch.Tensor]


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


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


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        raise NotImplementedError


# SAC Actor & Critic implementation
class VectorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # default pytorch init for nn.Linear module
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [ensemble_size, batch_size, input_size]
        # weight: [ensemble_size, input_size, out_size]
        # out: [ensemble_size, batch_size, out_size]
        return x @ self.weight + self.bias


class Actor(nn.Module):
    def __init__(
        self, state_dim: int, num_actions: int, hidden_dim: int
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, num_actions),
            nn.Softmax(),
        )
        # with separate layers works better than with Linear(hidden_dim, 2 * action_dim)
        #self.mu = nn.Linear(hidden_dim, action_dim)
        #self.log_sigma = nn.Linear(hidden_dim, action_dim)

        # init as in the EDAC paper
        for layer in self.trunk[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        self.num_actions = num_actions
        #adjusted for dicrete actions
        self.action_dim = 1
        #self.max_action = max_action

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        need_log_prob: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden = self.trunk(state)
        #mu, log_sigma = self.mu(hidden), self.log_sigma(hidden)

        # clipping params from EDAC paper, not as in SAC paper (-20, 2)
        #log_sigma = torch.clip(log_sigma, -5, 2)
        #policy_dist = Normal(mu, torch.exp(log_sigma))

        # This outputs the logits of the distribution over actions
        policy_probs = self.policy(hidden)
        policy_dist = torch.distributions.categorical.Categorical(probs=policy_probs)

        if deterministic:
            action = torch.argmax(policy_probs) # TODO was policiy_logits before, fixed (?)
        else:
            action = policy_dist.sample()
            

        # log_prob = None
        # if need_log_prob:
        #     # change of variables formula (SAC paper, appendix C, eq 21)
        #     log_prob = policy_dist.log_prob(action).sum(axis=-1)
        #     log_prob = log_prob - torch.log(1 - tanh_action.pow(2) + 1e-6).sum(axis=-1)

        return action, policy_probs

    @torch.no_grad()
    def act(self, state: Any, device: str) -> np.ndarray:
        deterministic = not self.training
        #print(state)

        #There is an empty dict at the end of the first state tensor in caroline's original code for some reason, like ([state],{}), so I removed it otherwise nothing works
        if isinstance(state[-1], dict):
            state = state[:-1]
        observation = np.array(state)

        # Add a new axis to create a batch dimension
        state = torch.tensor(observation, device=device, dtype=torch.float32).flatten() # observation used instead of state
        action = self(state, deterministic=deterministic)[0].cpu().numpy()
        return action


class VectorizedCritic(nn.Module):
    def __init__(
        self, state_dim: int, num_actions: int, hidden_dim: int, num_critics: int
    ):
        super().__init__()
        # change critic to map state -> q val for each action
        self.critic = nn.Sequential(
            VectorizedLinear(state_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, num_actions, num_critics),
        )
        # init as in the EDAC paper
        for layer in self.critic[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.critic[-1].weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.critic[-1].bias, -3e-3, 3e-3)

        self.num_critics = num_critics

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # [batch_size, state_dim + action_dim]
        #state_action = torch.cat([state, action], dim=-1)
        # [num_critics, batch_size, state_dim + action_dim]
        state = state.unsqueeze(0).repeat_interleave(
            self.num_critics, dim=0
        )
        # [num_actions, num_critics, batch_size]
        q_values = self.critic(state).squeeze(-1)
        return q_values


class SACN:
    def __init__(
        self,
        actor: Actor,
        actor_optimizer: torch.optim.Optimizer,
        critic: VectorizedCritic,
        critic_optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_learning_rate: float = 1e-4,
        device: str = "cpu",
        bc: bool = True, #set to True to use behavioral cloning regularization
    ):
        self.device = device

        self.actor = actor
        self.critic = critic
        with torch.no_grad():
            self.target_critic = deepcopy(self.critic)

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        self.tau = tau
        self.gamma = gamma

        # adaptive alpha setup
        self.target_entropy = -float(self.actor.action_dim)
        self.log_alpha = torch.tensor(
            [0.0], dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_learning_rate)
        self.alpha = self.log_alpha.exp().detach()

        self.bc = bc

    def _alpha_loss(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action, action_probs = self.actor(state)

        # adjusted for discrete
        loss = (action_probs * (-self.log_alpha * (torch.log(action_probs) + self.target_entropy))).sum(-1).mean()

        return loss

    def _actor_loss(self, state: torch.Tensor) -> Tuple[torch.Tensor, float, float, torch.Tensor, float]:
        action, action_probs = self.actor(state, need_log_prob=True)
        q_value_dist = self.critic(state)
        assert q_value_dist.shape[0] == self.critic.num_critics
        q_value_min = q_value_dist.min(0).values
        # needed for logging
        q_value_std = q_value_dist.std(0).mean().item()
        batch_entropy = -torch.distributions.Categorical(probs = action_probs).entropy().mean().item()

        # lambda added
        lmbda = 2.5/q_value_dist.abs().mean().detach() #regularization term

        loss = (action_probs * (self.alpha * torch.log(action_probs) - q_value_min)).sum(-1).mean()

        return loss, batch_entropy, q_value_std, action_probs, lmbda

    def _critic_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_action, next_action_probs = self.actor(
                next_state, need_log_prob=True
            )
            q_next = self.target_critic(next_state).min(0).values
            q_next = (next_action_probs * (q_next - self.alpha * torch.log(next_action_probs))).sum(1)
            assert q_next.unsqueeze(-1).shape == done.shape == reward.shape
            q_target = reward + self.gamma * (1 - done) * q_next.unsqueeze(-1)

        num_actions = self.critic(state).shape[-1]
        q_values = (torch.nn.functional.one_hot(action.squeeze().to(torch.long),
                        num_classes=num_actions) * self.critic(state)).sum(-1)

        # [ensemble_size, batch_size] - [1, batch_size]
        loss = ((q_values - q_target.view(1, -1)) ** 2).mean(dim=1).sum(dim=0)

        return loss

    def update(self, batch: TensorBatch) -> Dict[str, float]:
        state, action, reward, next_state, done = [arr.to(self.device) for arr in batch]
        # Usually updates are done in the following order: critic -> actor -> alpha
        # But we found that EDAC paper uses reverse (which gives better results)

        # Alpha update
        alpha_loss = self._alpha_loss(state)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().detach()

        # Actor update
        actor_loss, actor_batch_entropy, q_policy_std, action_probs, lmbda = self._actor_loss(state)
        #print(action.shape)

        log_probs = torch.log(action_probs.gather(1, action.long().reshape(1,256)).squeeze(-1)) 
        #log_probs = torch.log(action_probs.gather(0, action.long()).squeeze(-1))
        #log_probs = torch.log(action_probs)

        #print(log_probs.mean())
        #SAC + BEHAVIORAL CLONING
        #print(torch.log(action_probs).mean())
        actor_loss = actor_loss if not self.bc else actor_loss - log_probs.mean()# FIXME, TODO chosen_act gives problems with backpropagation as it is stochastic. Either use reparametrization trick or use probs

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Critic update
        critic_loss = self._critic_loss(state, action, reward, next_state, done)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #  Target networks soft update
        with torch.no_grad():
            soft_update(self.target_critic, self.critic, tau=self.tau)
            # for logging, Q-ensemble std estimate with the random actions:
            random_actions = torch.randint(low=0, high=self.actor.num_actions-1, size=action.shape).squeeze()
            random_actions = torch.nn.functional.one_hot(random_actions,num_classes=self.actor.num_actions)
            q_random_std = (random_actions * self.critic(state)).sum(-1).std(0).mean().item()

        update_info = {
            "alpha_loss": alpha_loss.item(),
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "batch_entropy": actor_batch_entropy,
            "alpha": self.alpha.item(),
            "q_policy_std": q_policy_std,
            "q_random_std": q_random_std,
            "train_reward": reward.mean().item(),
            
        }
        return update_info

    def state_dict(self) -> Dict[str, Any]:
        state = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "log_alpha": self.log_alpha.item(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "critic_optim": self.critic_optimizer.state_dict(),
            "alpha_optim": self.alpha_optimizer.state_dict(),
        }
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_critic.load_state_dict(state_dict["target_critic"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optim"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optim"])
        self.alpha_optimizer.load_state_dict(state_dict["alpha_optim"])
        self.log_alpha.data[0] = state_dict["log_alpha"]
        self.alpha = self.log_alpha.exp().detach()

def eval_actor(
    env: gym.Env, actor: Actor, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    #env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset(seed=seed)
        state = state.flatten()
        done = False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, terminated, truncated, info = env.step(action)
            state = state.flatten()
            done = terminated or truncated
            episode_reward += reward
        episode_rewards.append(episode_reward)

    print("Evaluated actor.")
    print("Episode rewards: ", episode_rewards)
    print("Tot reward:", sum(episode_rewards))

    actor.train()
    return np.array(episode_rewards)


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


@pyrallis.wrap()
def train(config: TrainConfig, traindatachoice: str = "fourrooms_train_config.pl", evaldatachoice: str = "fourrooms_test_0_config.pl"):
    set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)
    wandb_init(asdict(config))

    # data, evaluation, env setup
    #eval_env = wrap_env(gym_wrapper(gym.make(config.env_name)))
    eval_env = wrap_env(get_expert_dataset_from_config(get_config("four_room/configs/" + evaldatachoice))[1])
    num_actions = eval_env.action_space.n
    state_dim=1
    for dim in eval_env.observation_space.shape:
        state_dim *= dim
    #state_dim = eval_env.observation_space.shape[0]

    #Discrete actions
    action_dim = 1

    # Load whatever dataset you wish to use here, in d4rl format

    if traindatachoice == 'random':
        train_dataset = get_random_dataset()
    elif traindatachoice == 'expert':
        train_dataset = get_expert_dataset()
    else:
        train_dataset = get_expert_dataset_from_config(get_config("four_room/configs/" + traindatachoice))[0]

    if config.normalize_reward:
        modify_reward(train_dataset, config.env_name)

    buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=config.buffer_size,
        device=config.device,
    )
    buffer.load_d4rl_dataset(train_dataset)

    # Actor & Critic setup
    actor = Actor(state_dim, num_actions, config.hidden_dim)
    actor.to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate)
    critic = VectorizedCritic(
        state_dim, num_actions, config.hidden_dim, config.num_critics
    )
    critic.to(config.device)
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=config.critic_learning_rate
    )

    trainer = SACN(
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic=critic,
        critic_optimizer=critic_optimizer,
        gamma=config.gamma,
        tau=config.tau,
        alpha_learning_rate=config.alpha_learning_rate,
        device=config.device,
    )
    # saving config to the checkpoint
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    total_updates = 0.0
    for epoch in trange(config.num_epochs, desc="Training"):
        # training
        for _ in trange(config.num_updates_on_epoch, desc="Epoch", leave=False):
            batch = buffer.sample(config.batch_size)
            update_info = trainer.update(batch)

            if total_updates % config.log_every == 0:
                wandb.log({"epoch": epoch, **update_info})

            total_updates += 1

        # evaluation
        if epoch % config.eval_every == 0 or epoch == config.num_epochs - 1:
            eval_returns = eval_actor(
                env=eval_env,
                actor=actor,
                n_episodes=config.eval_episodes,
                seed=config.eval_seed,
                device=config.device,
            )
            eval_log = {
                "epoch": epoch,
                "eval/reward_mean": np.mean(eval_returns).item(),
                "eval/reward_std": np.std(eval_returns).item(),
                
            }
            if hasattr(eval_env, "get_normalized_score"):
                normalized_score = eval_env.get_normalized_score(eval_returns) * 100.0
                eval_log["eval/normalized_score_mean"] = np.mean(normalized_score)
                eval_log["eval/normalized_score_std"] = np.std(normalized_score)

            wandb.log(eval_log)

            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"{epoch}.pt"),
                )

    wandb.finish()


if __name__ == "__main__":
    train()
