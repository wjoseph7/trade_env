import gym
from typing import Dict, Tuple
import numpy as np
import pickle
from pprint import pprint
from collections import OrderedDict

class TradeEnv(gym.Env):
    """
    Summary:
        This extends this generic gym Env to create a trade environment.
        We process intraday ticker data and the agent either does nothing, 
        buys, holds, or sells at each tick.
    """

    def __init__(self, config: Dict):
        """
        Summary:
            Constructor for TrandEnv instance. Sets the discrete action space size.
            The size is two. There are two trade modes, pre-entry and pre-exit.
            In the pre-entry mode 0 = do nothing and 1 = buy. In the pre-exit mode
            0 = hold and 1 = sell.

            The size of the observation space is 7*num_ticks. It contains the following information:
            {trade mode : 0 for pre-entry, 1 for pre-exit;
             high, low, open, close, volume weighted, volume, and time data for each tick,
             purchase price if in pre exit mode (otherwise -1)}

            Note all price data is normalized to the previous day's close.
            
            Note times are normalized so that 930 = 0 and 4pm = 1.

            The reward is the log appreciation if holding and zero otherwise:
                = log( current close price / price )

        Args:
            config (Dict): Dictionary with configuration values

        Returns:
            A TradeEnv object.
        """

        num_tick_features = 2 + 7 # trade mode, purchase price, and 7 ticker features
                                  # we must stack the previous two features in with the ticker to use an RNN

        self.state_columns = ['n_o', 'n_c', 'n_h', 'n_l', 'n_v', 'n_vw', 'n_t']
        self.logging_columns = ['datetime']

        np.random.seed(seed=config['seed'])
        
        self.tick_size = config['tick_size']
        self.num_ticks = config['num_ticks']
        self.reward_scale = config['reward_scale']
        
        self.df = config['df'].dropna()
        self.debug = config['debug']

        self.days = list(set(self.df['day']))
        self.days.sort()

        discrete_space = gym.spaces.Discrete
        box_space = gym.spaces.Box
        tuple_space = gym.spaces.Tuple
        dict_space = gym.spaces.Dict
        
        self.action_space = discrete_space(2) # 0 = nothing / hold, 1 = buy / sell

        observation_space = OrderedDict()
        observation_space['ticker_data'] = box_space(low=0.0, high=2.0, shape=(self.num_ticks*len(self.state_columns),), dtype=np.float32)
        observation_space['purchase_price'] = box_space(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)
        observation_space['mode'] = discrete_space(2)
        
        self.observation_space = dict_space(observation_space)

        self.rewards = []

    def reset(self) -> Tuple[np.ndarray, Dict]:
        """
        Summary:
            Resets the environment.
          
            Samples uniformly from the list of available trading days.

            Sets mode to pre-entry and last price to -1.
            Sets mode to pre-entry and purchase price to -1.
        
        Args:
            None

        Returns:
            Tuple[np.ndarray, Dict]: Initial state and info
        """
        if self.rewards != []:
            appreciation = np.sum(self.rewards)
            appreciation = np.exp(appreciation) - 1
            appreciation *= 100
            print(f"Assets changed by {appreciation}% over the last episode\n\n")

        self.rewards = []

        
        self.mode = 0
        self.last_price = -1
        self.purchase_price = -1
        
        sample = np.random.randint(low=0, high=len(self.days))

        day = self.days[sample]

        self.df_day = self.df[self.df['day']==day].reset_index()

        if len(self.df_day.index) <= self.num_ticks + 1: # ensure there is enough data in this day
            self.reset()

        self.n = self.num_ticks - 1

        state = self.make_state_and_info()

        return state

    def make_state_and_info(self) -> Tuple[np.ndarray, Dict]:
        """
        Summary:
            Creates a np.ndarray for the state. We start at n and move include num_ticks preceding tickers.
            Also creates info dict which contains the corresponding portion of the dataframe.

        Args:
            None: all info obtained through self

        Returns:
            Tuple[np.ndarray, Dict]: the state and info dict
        """
        df = self.df_day
        n = self.n
        num_ticks = self.num_ticks
        
        state = []
        tick_substate = []
        for tick in reversed(range(num_ticks)):
            for col in self.state_columns:
                x = df.loc[n - tick , col]

                tick_substate.append(x)

            state += tick_substate

            tick_substate = []

        dict_state= OrderedDict()
        dict_state['ticker_data'] = np.array(state, dtype=np.float32)
        dict_state['purchase_price'] = np.array([self.purchase_price], dtype=np.float32)
        dict_state['mode'] = self.mode

        info = {'df' : df.loc[n-num_ticks+1:n, self.state_columns + self.logging_columns]}

        if self.debug:
            pprint(dict_state)
            print(info['df'])
            print('\n\n')

        self.info = info

        return dict_state


        
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Summary:
            This steps the environment and returns the usual state, reward, done, info tuple.
            An nonzero action will flip the mode. In a nonzero mode the log appreciation from one state to the next
            will be returned.

            We need to end the sim on ticker early because we must exit at end of day.
            
        Args:
            action (int): a 0 or a 1. 
                          In pre-entry mode 0 = do nothing and 1 = make a buy.
                          In pre-exit mode 0 = hold and 1 = sell.

        Returns:
            Tuple[np.ndarray, float, bool, Dict]: state, reward, done, info
            
        """
        df = self.df_day
        n = self.n
        done = False

        if n == len(self.df_day.index) - 2: # we need to reserve the last tick to close out 
            done = True

        self.mode = self.mode ^ action # if the action is 1 flip the mode
        
        if self.mode: # set the last price if we are in pre-exit
            self.last_price = df.loc[n, 'n_c']

        else: # ensure the last price is negative 1 if not applicable
            self.last_price = -1

        if self.mode and self.purchase_price == -1: # set the purchase price if we are in pre-exit and it has not already been set
            self.purchase_price = df.loc[n, 'n_c']

        elif not self.mode:
            self.purchase_price = -1 # ensure the purchase price is -1 if not applicable

        potential_exit = df.loc[n+1, 'n_c']

        reward = self.mode * np.log( potential_exit / abs(self.last_price)) # 0 if in pre entry mode, log of appreciation if otherwise

        self.rewards.append(reward)

        self.n += 1

        state = self.make_state_and_info()

        reward *= self.reward_scale # scaling reward to make learning easier

        return state, reward, done, self.info
    
def manual_step(config: Dict, max_iterations: int):
    """
    Summary:
        This allows the user to manually step through the TradeEnv, making 
        specified decisions for a specified number of iterations.

        Used for debugging / testing.
    Args:
        config (Dict): configuration parameters for TradeEnv
        max_iterations (int): max number of iterations to step for.
    Returns:
        None
    """
    trade_env = TradeEnv(config)
    trade_env.reset()


    for n in range(max_iterations):

        action = input('action = ')
        action = int(action)
        state, reward, done, info = trade_env.step(action)
        print(f'reward = {reward}')
        print(f'done = {done}\n\n')
        if done:
            trade_env.reset()

def automatic_step(config: Dict, max_iterations: int) -> None:
    """
    Summary:
        This automatically steps through the TradeEnv, making random decisions
        for a specified number of iterations.

        Used for debugging / testing.
    Args:
        config (Dict): configuration parameters for TradeEnv
        max_iterations (int): max number of iterations to step for.
    Returns:
        None
    """
    trade_env = TradeEnv(config)
    trade_env.reset()


    for n in range(max_iterations):

        action = np.random.randint(low=0, high=2)
        state, reward, done, info = trade_env.step(action)
        print(f'reward = {reward}')
        print(f'done = {done}\n\n')
        if done:
            trade_env.reset()

if __name__ == '__main__':

    df = pickle.load(open('../df_2013-02-21_2023-02-21_15_SPY.pickle', 'rb'))
    config = {'tick_size' : 15, 'num_ticks' : 2, 'df' : df, 'debug' : True, 'seed' : 0, 'reward_scale' : 10000}

    automatic_step(config, 10000)
    #manual_step(config, 100)
