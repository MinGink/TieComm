from gym.envs.registration import register



register(
    id='TrafficJunction-v0',
    entry_point='traffic_junction.traffic_junction_env:TrafficJunctionEnv',
)