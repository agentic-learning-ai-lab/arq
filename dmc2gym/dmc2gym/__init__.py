import functools
import gymnasium as gym
from gymnasium.envs.registration import register


def make(
    domain_name,
    task_name,
    discrete=False,
    seed=1,
    visualize_reward=True,
    from_pixels=False,
    height=84,
    width=84,
    camera_id=0,
    frame_skip=1,
    episode_length=1000,
    environment_kwargs=None,
    time_limit=None,
    channels_first=True,
    action_noise=False,
    action_noise_type="normal",
    action_noise_level=0.0,
):
    env_id = "dmc_%s_%s_%s_%s-v1" % (domain_name, task_name, seed, discrete)

    if from_pixels:
        assert (
            not visualize_reward
        ), "cannot use visualize reward when learning from pixels"

    # shorten episode length
    max_episode_steps = (episode_length + frame_skip - 1) // frame_skip

    if not env_id in gym.envs.registry.keys():
        print(env_id)
        task_kwargs = {}
        if seed is not None:
            task_kwargs["random"] = seed
        if time_limit is not None:
            task_kwargs["time_limit"] = time_limit
        if discrete:
            entry_point = "dmc2gym.wrappers:DiscretizedDMCWrapper"
        else:
            entry_point = "dmc2gym.wrappers:DMCWrapper"
        register(
            id=env_id,
            entry_point=entry_point,
            kwargs=dict(
                domain_name=domain_name,
                task_name=task_name,
                task_kwargs=task_kwargs,
                environment_kwargs=environment_kwargs,
                visualize_reward=visualize_reward,
                from_pixels=from_pixels,
                height=height,
                width=width,
                camera_id=camera_id,
                frame_skip=frame_skip,
                channels_first=channels_first,
                action_noise=action_noise,
                action_noise_type=action_noise_type,
                action_noise_level=action_noise_level,
            ),
            max_episode_steps=max_episode_steps,
        )
    return gym.make(env_id)


def vector_make(
    domain_name,
    task_name,
    num_envs,
    seeds,
    visualize_reward=True,
    from_pixels=False,
    height=84,
    width=84,
    camera_id=0,
    frame_skip=1,
    episode_length=1000,
    environment_kwargs=None,
    time_limit=None,
    channels_first=True,
    action_noise=False,
    action_noise_type="normal",
    action_noise_level=0.0,
):
    assert (
        len(seeds) == num_envs or len(seeds) == 1
    ), "seeds must be either of length 1 or equal to num_envs"
    if len(seeds) == 1:
        seeds = seeds * num_envs
    ids = []
    for i in range(num_envs):
        env_id = "dmc_%s_%s_%s-v1" % (domain_name, task_name, seeds[i])
        ids.append(env_id)

        if from_pixels:
            assert (
                not visualize_reward
            ), "cannot use visualize reward when learning from pixels"

        # shorten episode length
        max_episode_steps = (episode_length + frame_skip - 1) // frame_skip

        if not env_id in gym.envs.registry.keys():
            task_kwargs = {}
            if seeds[i] is not None:
                task_kwargs["random"] = seeds[i]
            if time_limit is not None:
                task_kwargs["time_limit"] = time_limit
            register(
                id=env_id,
                entry_point="lambda_ac.third_party.dmc2gym.dmc2gym.wrappers:DMCWrapper",
                kwargs=dict(
                    domain_name=domain_name,
                    task_name=task_name,
                    task_kwargs=task_kwargs,
                    environment_kwargs=environment_kwargs,
                    visualize_reward=visualize_reward,
                    from_pixels=from_pixels,
                    height=height,
                    width=width,
                    camera_id=camera_id,
                    frame_skip=frame_skip,
                    channels_first=channels_first,
                    action_noise=action_noise,
                    action_noise_type=action_noise_type,
                    action_noise_level=action_noise_level,
                ),
                max_episode_steps=max_episode_steps,
            )
    return gym.vector.AsyncVectorEnv(
        [functools.partial(lambda x: gym.make(x), x=id) for id in ids]
    )
