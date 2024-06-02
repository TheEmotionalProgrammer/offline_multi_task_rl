# Offline Multi-Task RL

1. general_papers -> Papers that we need to read to have a general understanding of the research field
2. implementation_papers -> Papers related to our work, e.g. to the environment, experiments etc.
3. methods_papers -> The 4 methods that we have to implement

### Using mixed dataset
Basically, you can create a dataset as you're used to with the following:
```
checkpoints = get_DQN_checkpoints(os.path.join("..", "four_room_extensions", "DQN_models", "performance_per_model.txt"), episode_length, best_policy=best_policy)
mixed_dataset, finished, failed = get_mixed_policy_dataset(train_config, train_env, checkpoints)
```

With `mixed_dataset` containing:
```
    observations=mixed_dataset.get("observations"),
    actions=mixed_dataset.get("actions"),
    rewards=mixed_dataset.get("rewards"),
    terminals=mixed_dataset.get("terminals"),
```


