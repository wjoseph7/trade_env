from custom_env import TradeEnv
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from typing import Dict
import pickle
from time import time

def train_PPO(config: Dict, iterations: int) -> None:
    """
    Summary:
        Trains PPO on the TradeEnv environment.
    Args:
        config (Dict): dictionary containing training data and environment
                       parameters
        iterations (int): number of iterations we want to train for
    Returns:
        None
    """
    config = PPOConfig().environment(TradeEnv, env_config=config).framework("torch").resources(num_gpus=1)
    algo = config.build()

    for _ in range(iterations):
        tick = time()
        print(algo.train())
        tock = time()
        print(f"took {(tock-tick)/60} minutes to execute one training iteration")

if __name__ == '__main__':
    df = pickle.load(open('../df_2013-02-21_2023-02-21_15_SPY.pickle', 'rb'))
    config = {'tick_size' : 15, 'num_ticks' : 2, 'df' : df, 'debug' : False, 'seed' : 0, 'reward_scale' : 10000}
    train_PPO(config, 10000)
