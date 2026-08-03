from gym.envs.registration import register

register(
    id="pusht",
    entry_point="env.pusht.pusht_wrapper:PushTWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)

# Alias used by AdaJEPA's released bases (env.name="pushobj"); same wrapper/env,
# shape set per-episode via update_env. Mirrors adajepa/env/__init__.py.
register(
    id="pushobj",
    entry_point="env.pusht.pusht_wrapper:PushTWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)
