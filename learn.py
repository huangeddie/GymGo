import gym_go
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
import gym

env = gym.make("go-v0",size=11)
env = DummyVecEnv([lambda: env])

rl_algo = PPO2("MlpPolicy", env=env)
rl_algo.learn(10000)