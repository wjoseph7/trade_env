import requests
from typing import Dict
import pickle
import time

from datetime import datetime, timedelta, timezone

KEY = 'urfPdgfiolULd80nU9_cDEhY39uxl6qz'

def get_agg_url(start_date: str, end_date: str, multiplier: int, ticker: str) -> str:
    """
    Summary:
        Formats url for a polygon agg REST API query
    Args:
        start_date (str): start date for query in YYYY-MM-DD HH:MM:SS format
        end_date (str): end date for query in YYYY-MM-DD HH:MM:SS format
        multiplier (int): The number of minutes we want each high, low, open, 
            close, volume aggregate to be.
        ticker (str): The ticker we want data on 
    Returns:
        str: the url for the polygon REST agg call 
    """
    poly_url =  f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/ \
        {multiplier}/minute/{start_date}/{end_date}?adjusted=true&sort=asc&limit= \
        50000&apiKey={KEY}"

    return poly_url.replace(' ', '')

def convert_ms_timestamp_to_time_str(ms_timestamp: int) -> str:
    """
    Summary:
        This converts a machine timestamp to a human readable (but still 
        sortable) str format. The datetime library does most of the heavy
        lifting. Only catches are adjusting the timezone to be EST rather than
        Zulu and adjusting for daylight savings
    Args:
        ms_timestamp (int): the millisecond int timestamp we wish convert to 
            a human interpretable string format
    Returns:
        str: timestamp in string format YYYY-MM-DD HH:MM:SS
    """
    format_ = '%Y-%m-%d %H:%M:%S'

    dt = datetime.fromtimestamp(int(ms_timestamp) // 1000)
    dt -= timedelta(hours=5) # adjust for eastern standard time

    
    time_str = dt.strftime(format_)

    month = time_str.split('-')[1]
    day = time_str.split('-')[2].replace(' ', '')

    date = month + '-' + day

    # approximately adjust for daylight savings
    if date >= '03-12' and date < '11-05':
        dt += timedelta(hours=1)

    
    time_str = dt.strftime(format_)

    return time_str


def get_data(start_date: str, end_date: str, multiplier: int, ticker: str="SPY") -> None:
    """
    Summary:
        This function makes a call to polygon's aggregate API for a particular
        ticker in order to get
            high
            low
            open
            close
            volume

        information. Polygon splits the data into chunks of 50k so we need a 
        while loop to search the response for a 'next_url' and query it if it
        exists, appending the data to a list.

        The data is then dumped into a pickle file.
    Args:
        start_date (str): start date for query in YYYY-MM-DD HH:MM:SS format
        end_date (str): end date for query in YYYY-MM-DD HH:MM:SS format
        multiplier (int): The number of minutes we want each high, low, open, 
            close, volume aggregate to be.
        ticker (str): The ticker we want data on (SPY is default)
    Returns:
        None 
    """

    url = get_agg_url(start_date, end_date, multiplier, ticker)
    print(url)

    save_str = f'{start_date}_{end_date}_{multiplier}_{ticker}.pickle'
    
    response = requests.get(url).json()

    keys = response.keys()

    data = [response['results']]

    while 'next_url' in keys:

        print(response['queryCount'])

        url = response['next_url'] + f'&apiKey={KEY}'
        print(url)
        response = requests.get(url).json()
        keys = response.keys()

        data.append(response['results'])
        latest_time = response['results'][-1]['t']
        latest_time = convert_ms_timestamp_to_time_str(latest_time)
        print(latest_time)
        time.sleep(2)
        
    pickle.dump(data, open(save_str, 'wb'))
    


if __name__ == '__main__':
    get_data(start_date='2013-02-21', end_date='2023-02-21', multiplier=15)
