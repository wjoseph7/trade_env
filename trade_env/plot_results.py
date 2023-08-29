from typing import List, Dict
from tqdm import tqdm
from pprint import pprint
from functools import reduce
import operator
import numpy as np
from matplotlib import pyplot as plt
import pickle

def read_appreciation_stats(fp: str) -> List:
    """
    Summary:
        Reads the daily returns statistics in sequence from the rllib logs.

        Just looks for the lines that have the % symbol in them and uses the
        position of the number (which is consistent) to get the data
    Args:
        fp (str): filepath to logfile with daily returns
    Returns:
        List: list containing daily returns in sequence
    """
    fp = open(fp, 'r')
    lines = fp.readlines()

    appreciations = [line for line in lines if '%' in line]
    
    appreciations = [float(line.split('%')[0].split(' ')[-1]) for line in appreciations]

    return appreciations

def compute_moving_average(fp: str, n_days: int) -> List:
    """
    Summary:
        Uses the read_appreciation_stat function to read in the daily returns
        from the logfile. Then computes the n, day moving average in a for loop.

        This takes a while since the log file has over a million datapoints so
        I added tqdm to the for loop so I could track the progress.
    Args:
        fp (str): filepath to logfile with daily returns
        n_days (int): number of days to include in moving average
    Returns:
        List: list containing the sequential moving average data
    """

    appreciations = read_appreciation_stats(fp)

    moving_average = []
    for n in tqdm(range(len(appreciations) - n_days + 1)):
        mean = np.mean(appreciations[n: n + n_days])
        
        moving_average.append(mean)

    return moving_average

def plot_moving_average(fp: str, n_days: int) -> None:
    """
    Summary:
        Plots moving average of daily returns.

        Calls compute moving average helper function and then creates and saves
        plot with appropriate labels.
    Args:
        fp (str): filepath to logfile with daily returns
        n_days (int): number of days to include in moving average
    Returns:
        None
    """

    moving_average = compute_moving_average(fp, n_days)

    fp = fp.split('/')[-1]

    x_axis = list(range(len(moving_average)))
    x_axis = [x + n_days for x in x_axis]

    plt.plot(x_axis, moving_average)
    plt.xlabel('Training Days')
    plt.ylabel(f'Daily Returns ({n_days}-day Moving Avg)')
    plt.savefig(f'ppo_mean_daily_return_{n_days}_moving_average.png')

def split_training_data_consecutively(data: List, splits: int) -> List:
    """
    Summary:
        This splits a list of data up into splits consecutive parts
        of equal length
    Args:
        data (List): the data to split
        splits (int): the number of splits to make
    Returns:
        List: of lists which contain the data for each split
    """
    split_index = len(data) // splits
    split_data = [data[n*split_index:(n+1)*split_index] for n in range(splits)]

    return split_data

def boxplot(split_stats: List[List], splits: int, outliers: bool) -> None:
    """
    Summary:
        Creates multiple box plots so we can better examine the returns over
        training time by examining changes in contiguous slices.
    Args:
        split_stats(List[List]): a list of splits number of contiguous
            training returns
        splits (int): Number of splits to examine
        outliers (bool): whether to include outliers in the boxplot
    Returns:
        None
    """
    assert len(split_stats) == splits

    fig, ax = plt.subplots()

    ax.set_title(f'Daily Returns Over {str(splits)} Consecutive Training Periods')
    ax.boxplot(split_stats, showfliers=outliers)

    fig_title = 'ppo_training_daily_returns_boxplot'
    if outliers:
        fig_title += '_with_outliers.png'
    else:
        fig_title += '_without_outliers.png'

    plt.savefig(fig_title)

def compute_stats(data: List) -> Dict:
    """
    Summary:
        Returns dictionary with mean, std, and approx annual expectd return
    Args:
        data (List): the data we want the statistics of
    Returns:
        Dict: dict containings mean, std, and annual expected return
    """
    mean = np.mean(data)
    annual = approx_annual_return(mean)
    return {'mean' : mean, 'std' : np.std(data), 'approx_annual_ROI' : annual}

def approx_annual_return(arithmetic_mean: float) -> float:
    """
    Summary:
        This approximates expected annual return given arithmetic mean daily
        returns.

        Really, we should be using the geometric mean. But computing this is 
        numerically unstable for large datasets and this gives us a good 
        approximate number.

        We just raise the daily appreciation to the number of trading days in
        the typical year (252)
    Args:
        arithmetic_mean (float): Arithmetic mean of daily appreciation
    Returns:
        float: approximation for the expected annual appreciation
        
    """
    return (((arithmetic_mean / 100) + 1)**252 - 1) * 100

def compute_stats_and_plots(fp_logs: str, splits: int, fp_baseline: str, years: int) -> List[Dict]:
    """
    Summary:
        Splits training data into splits # of contiguous slices to compare 
        changes in performance over training.

        Saves boxplots with and without outliers and returns list
        of dictionaries containing mean and std.

        Also, plots the approx annual return for each training interval.
    Args:
        fp_logs (str): The path to the log file
        splits (int): Number of splits to examine
        fp_baseline (str): filepath to the baseline data
        years (int): # years to compute geo average for baseline
    Returns:
        List[Dict]: index of the list corresponds to the box plot. Dict
            contains meand and std of daily returns
    """
    stats = read_appreciation_stats(fp_logs)

    split_stats = split_training_data_consecutively(stats, splits)

    boxplot(split_stats, 8, outliers=True)
    boxplot(split_stats, 8, outliers=False)

    stat_list = [compute_stats(data) for data in split_stats]

    plot_approx_annual(fp_baseline, years, stat_list)

    return stat_list

def plot_approx_annual(fp_baseline: str, years: int, stat_list: List[Dict]) -> None:
    """
    Summary:
        This plots the approx annual returns over the training intervals
        It also plots the geometric average S&P return over the same period
    Args:
        fp_baseline (str): filepath to the baseline data
        years (int): # years to compute geo average for baseline
        stat_list (List[Dict]): list containing dictionary of statistics
    Returns:
        None
    """
    x = list(range(1, len(stat_list)+1))
    y = [stats['approx_annual_ROI'] for stats in stat_list]

    baseline = get_market_appreciation(fp_baseline, years)
    print(f"baseline average for period {baseline}")

    fig, ax = plt.subplots()

    ax.scatter(x, y, label='DRL Training Performance')
    ax.set_xlabel('Training Interval')
    ax.set_ylabel('Approx Annual Returns')

    ax.axhline(baseline, label='S&P avg Performance', linestyle='dashed', color='red')
    plt.legend()

    plt.savefig('approx_annual_returns_v_training_interval.png')

def get_market_appreciation(fp: str, years: int) -> float:
    """
    Summary:
        Approximates geometric average of S&P performance over a given
        number of years by dividing the last close by the first close of the
        dataset and raising that to the power of 1/n
    Args:
        fp (str): filepath to the baseline data
        years (int): # years to compute geo average for baseline
    Returns:
        float: geometric average of performance over years in terms of % ROI
    """
    df = pickle.load(open(fp, 'rb'))
    
    approx_appreciation = df.loc[df.index[-1], 'c'] / df.loc[df.index[0], 'c']

    geo_avg = np.power(approx_appreciation, 1/years)

    geo_avg -= 1.0
    geo_avg *= 100

    return geo_avg
    

if __name__ == '__main__':
  
    stats = compute_stats_and_plots(fp_logs='../logs/ppo_results.txt',
                                    splits=8,
                                    fp_baseline='../df_2013-02-21_2023-02-21_15_SPY.pickle',
                                    years=10)
    pprint(stats)

    
