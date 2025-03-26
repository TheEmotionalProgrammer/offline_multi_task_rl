# Offline Multi-Task RL

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


