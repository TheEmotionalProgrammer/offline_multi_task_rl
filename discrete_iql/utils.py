import torch


def save(args, save_name, model, wandb, ep):
    import os
    save_dir = './trained_models/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = save_dir + args.run_name + "_" + str(ep) + ".pth"
    torch.save(model.state_dict(), save_path)
    wandb.save(save_path)


def collect_random(env, dataset, num_samples=200):
    state = env.reset()
    for _ in range(num_samples):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        dataset.add(state, action, reward, next_state, terminated)
        state = next_state
        if terminated:
            state = env.reset()
