from .node import Node
from .tile_map import TileMap
from .environment import Environment

from gym import register
register(
    id='SimpleTrapEnv-v0',
    entry_point='custom_envs.simple_trap_env.environment:Environment',)