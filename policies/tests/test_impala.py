from policies import ImpalaPolicy
from utils import ImpalaEnvWrapper
import torch
import procgen
import gym


@torch.no_grad()
def run_test():
    env = ImpalaEnvWrapper(gym.make("procgen-fruitbot-v0"))
    net = ImpalaPolicy(env.observation_space.shape, env.action_space.n)
    net.reset()

    obs, _ = env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        action = net.get_action(obs)
        obs, rew, terminated, truncated, _ = env.step(action.item())


if __name__ == "__main__":
    run_test()
