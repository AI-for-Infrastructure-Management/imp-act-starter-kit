from functools import partial

from envs.MultiAgentEnv import MultiAgentEnv
from envs.pymarl_ma import PymarlMARoadEnv

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
REGISTRY["road_marl"] = partial(env_fn, env=PymarlMARoadEnv)