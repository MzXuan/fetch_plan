from gym.envs.registration import register

register(
    id='FetchPlan-v0',
    entry_point='gym_fetch.envs:FetchPlanEnv',
)
