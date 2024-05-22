import torch
import os


def save_model(agent, save_path="./trained_models/", filename="final_model.pth"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save({
        'actor_state_dict': agent.actor_local.state_dict(),
        'actor_optimizer_state_dict': agent.actor_optimizer.state_dict()
    }, os.path.join(save_path, filename))


def load_model(agent, save_path="./trained_models/", filename="model.pth"):
    model = torch.load(os.path.join(save_path, filename))
    agent.actor_local.load_state_dict(model['actor_state_dict'])
    agent.actor_optimizer.load_state_dict(model['actor_optimizer_state_dict'])


def collect_random(env, dataset, num_samples=200):
    state = env.reset()
    for _ in range(num_samples):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        dataset.add(state, action, reward, next_state, terminated)
        state = next_state
        if terminated:
            state = env.reset()
