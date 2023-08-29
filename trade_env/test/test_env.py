from trade_env.custom_env import *
from unittest import TestCase
import unittest

eps = 1e-6
log_eps = 1e-3


def reset_test(test_case: TestCase, tradeEnv: TradeEnv) -> None:
    """
    Summary:
        This is a helper function to test TradeEnv's reset. We test the 
        following.

        - The rewards list is reset to be empty
        - The trade mode is reset to zero (we are not going long)
        - The last_price is reset to -1, NA filler
        - The purchase_price is reset to -1, NA filler
        - The length of our day df is greater than the number of ticks in the 
            state space
        - Our index is the number of ticks in the state space - 1

    Args:
        test_case (TestCase): TestCase from the unittest framework
        tradeEnv (TradeEnv): an instance of the TradeEnv class
    Returns:
        None
    """
    test_case.assertEqual(len(tradeEnv.rewards), 0)
    test_case.assertEqual(tradeEnv.mode, 0)
    test_case.assertEqual(tradeEnv.last_price, -1)
    test_case.assertEqual(tradeEnv.purchase_price, -1)

    test_case.assertGreater(len(tradeEnv.df_day.index), 2)
    test_case.assertEqual(tradeEnv.n, 2-1)

    # ensure all data is from the same day
    test_case.assertEqual(len(set(tradeEnv.df_day['day'])), 1) 

class TestTradeEnv(TestCase):
    """
    Summary:
        Set of unit tests for the TradeEnv class
    """

    def setUp(self):
        """
        Summary:
            First loads in the aggregates dataframe and stores the 
            configurations for creating a TradeEnv instance. Then 
            creates tradeEnv.
        """

        self.df = pickle.load(open('../../../df_2013-02-21_2023-02-21_15_SPY.pickle', 'rb'))
        self.config = {'tick_size' : 15, 'num_ticks' : 2, 'df' : self.df, 'debug' : True, 'seed' : 0, 'reward_scale' : 10000}
        tradeEnv = TradeEnv(self.config)

        self.tradeEnv = tradeEnv

    def test_reset(self):
        """
        Summary:
            This is a unit test for TradeEnv's reset method. We create a new
            environment, reset it, and use the reset_test helper function to
            ensure the environment parameters are appropriately reset.

            This is done twice in a row to ensure we're not just testing the 
            special case right after the environment is instantiated.
        """

        tradeEnv = self.tradeEnv
        tradeEnv.reset()

        reset_test(self, tradeEnv)

        for n in range(len(tradeEnv.df_day.index)-tradeEnv.num_ticks):
            action = np.random.randint(low=0, high=2)
            state, reward, done, info = tradeEnv.step(action)
            
        self.assertTrue(done)
        tradeEnv.reset()

        reset_test(self, tradeEnv)

    def test_make_state_and_info(self):
        """
        Summary:
            This is a simple test where we check the following:
                - mode is 0 after reset when we're not in a trade and 1 when we
                    are
                - purchase price is set to -1 initially and then the normalized
                    close when we enter a trade
                - the ticker data is a flattened array of size 14 (7 ticker
                    data points, 2 instances)

        """        
        tradeEnv = self.tradeEnv
        state = tradeEnv.reset()

        ticker_data = state['ticker_data']
        purchase_price = state['purchase_price']
        mode = state['mode']

        self.assertEqual(mode, 0)
        self.assertEqual(purchase_price, -1)
        self.assertEqual(ticker_data.shape, (14,))

        purchase_close = tradeEnv.info['df'].iloc[1, 1]

        tradeEnv.step(1)
        state, _, _, _ = tradeEnv.step(0)

        ticker_data = state['ticker_data']
        purchase_price = state['purchase_price']
        mode = state['mode']
        
        self.assertEqual(mode, 1)
        self.assertLess(abs(purchase_price - purchase_close), eps)
        self.assertLess(abs(purchase_close - 0.9973760255090177), eps)
        self.assertEqual(ticker_data.shape, (14,))

        #ToDo: add more testing for info



    def test_step(self):
        """
        Summary:
            This tests the step method which is the most complex. I break it
            into the following cases:

            1. Test intraday non investment behavior
                - mode will be 0
                - purchase price will be -1
                - Reward will be 0
                - Done will be False
                - last_price is -1
            2. Test intraday long behavior
                - mode will be 1
                - purchase price will be last normalized close
                - Reward will be log (current n_c / purchase_price)
                - Done will be False
                - last_price is the current n_c
            3. Test end of day non investment behavior
                - mode will be 0
                - purchase price will be -1
                - Reward will be 0
                - Done will be True
                - last_price is -1
            4. Test end of day long behavior
                The env should reject the investment and return the same data
                as for (3) except that the last price is the current n_c
        """
        tradeEnv = self.tradeEnv
        state = tradeEnv.reset()

        for n in range(4): # (1)

            state, reward, done, _ = tradeEnv.step(0)

            ticker_data = state['ticker_data']
            purchase_price = state['purchase_price']
            mode = state['mode']

            last_price = tradeEnv.last_price


            self.assertEqual(mode, 0)
            self.assertEqual(purchase_price, -1)
            self.assertEqual(reward, 0)
            self.assertFalse(done)
            self.assertEqual(last_price, -1)

        manual_purchase = ticker_data[8]

        state, reward, done, _ = tradeEnv.step(1)

        for n in range(4): # (2)
            state, reward, done, _ = tradeEnv.step(0)

            ticker_data = state['ticker_data']
            purchase_price = state['purchase_price']
            mode = state['mode']
            

            last_price = tradeEnv.last_price
            manual_last_price = ticker_data[1]

            n = tradeEnv.n
            # just n, not n+1 since it's already been incremented
            potential_exit = tradeEnv.df_day.loc[n, 'n_c'] 

            self.assertEqual(mode, 1)
            self.assertEqual(purchase_price, manual_purchase)

            self.assertLess(abs(reward - 10000*np.log( potential_exit / manual_last_price)), log_eps)
            self.assertFalse(done)
            self.assertLess(abs(last_price - manual_last_price), eps)

        #ToDo: Note this set of tests is not yet fully exhaustive"""
        

if __name__ == '__main__':
    unittest.main()