from gym.envs.registration import register


register(
    id='FetchPlan-v0',
    entry_point='gym_fetch.envs:FetchPlanEnv',
    max_episode_steps=400
)

register(
    id='FetchPlanTest-v0',
    entry_point='gym_fetch.envs:FetchPlanTestEnv',
    max_episode_steps=400
)


