from .environment import Environment

from gym import register
register(
    id='RocketSim-v0',
    entry_point='custom_envs.rocketsim.environment:Environment',)