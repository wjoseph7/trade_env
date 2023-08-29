import pickle
import pandas as pd
from get_data import convert_ms_timestamp_to_time_str
from pprint import pprint
from tqdm import tqdm
from datetime import datetime, timedelta

def normalize_time(time: str) -> float:
    """
    Summary:
        This simply normalizes the timestep to something more neural-network 
        friendly. We replace the time str with a float that is 0 at market open
        and 1 at market close.
    Args:
        time (str): time str in YYYY-MM-DD HH:MM:SS format
    Returns:
        float: The time str converted to a float that is 0 at market open and 1
            at market close.
    """
    format_ = '%Y-%m-%d %H:%M:%S'
    total_minutes = 6.5*60
    
    dt = datetime.strptime(time, format_)

    dt = dt - timedelta(hours=9, minutes=30)

    time = dt.strftime(format_).split(' ')[-1]

    hours = int(time.split(':')[0])
    minutes = int(time.split(':')[1])

    minutes_passed = hours*60 + minutes

    normalized_time = minutes_passed / total_minutes

    return normalized_time

def define_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary:
        This function adds the following time columns to our dataframe
            datetime: YYYY-MM-DD HH:MM:SS str
            day: YYYY-MM-DD str
            time: HH:MM:SS str
        
        It then removes all data that is not intraday. Later a column is 
        created based on the time column which is scaled 0-1 for deep learning
    Args:
        df (pd.DataFrame): dataframe of aggregate data we want to add human 
            interpretable time columns to
    Returns:
        pd.DataFrame: df with the added time columns
    """

    df['datetime'] = df['t'].apply(lambda x : convert_ms_timestamp_to_time_str(x))

    df = df.sort_values('datetime')

    df['day'] = df['datetime'].apply(lambda x : x.split(' ')[0])
    df['time'] = df['datetime'].apply(lambda x : x.split(' ')[1])
    

    df['intraday'] = df['time'].apply(lambda x : x >= '09:30:00' and x <= '16:00:00')
    df = df[df['intraday']].reset_index()

    return df

def normalize_price_with_previous_close(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary:
        This function normalizes all of the price values:
            high
            low
            open
            close
        
        by dividing by the previous day's close. This helps the neural network
        understand price movement relative to the previous day's close rather
        than going through the work of trying to understand absolute measures.
    Args:
        df (pd.DataFrame): dataframe who's price data we want to normalize
    Returns:
        pd.DataFrame: df with the normalized price data
    """

    price_normalization_columns = ['vw', 'o', 'c', 'h', 'l']

    for col in price_normalization_columns:
        df['n_' + col] = df[col]

    days = list(set(df['day']))
    days.sort()

    for n in tqdm(range(len(days)-1)):
        day = days[n]
        next_day = days[n+1]

        df_day = df[df['day']==day]
        df_next_day = df[df['day']==next_day]

        close = df_day.loc[df_day.index[-1], 'c']


        for d in df_next_day.index:
            for col in price_normalization_columns:
                df.loc[d, 'n_' + col] /= close

    df = df[df['day']!=days[0]] # remove first day since it can't be normalized

    return df

def preprocess_data(fp: str) -> None:
    """
    Summary:
        This unpacks the data we downloaded from the agg REST API call to
        polygon and converts the data to a data frame. Then we create new time
        columns including a normalized one which is used as a training feature.
        The price is then normalized by dividing by the previous day's close. 
        Finally, we normalize volume by dividing with the max value and save
        the df to a new pickle file.
    Args:
        fp (str): fp to pickle file from get_data REST call
    Returns:
        None
    """

    data = pickle.load(open(fp, 'rb'))

    new_data = []
    for datum in data:
        new_data += datum

    df = pd.DataFrame.from_dict(new_data)

    df = define_time_columns(df)

    df = normalize_price_with_previous_close(df)    

    # normalize volume
    max_vol = max(list(df['v']))
    df['n_v'] = df['v'].apply(lambda x : x / max_vol)

    # normalize time
    df['n_t'] = df['datetime'].apply(lambda x : normalize_time(x))

    print(df)

    pickle.dump(df, open('df_' + fp, 'wb'))

        
if __name__ == '__main__':
    preprocess_data(fp='2013-02-21_2023-02-21_15_SPY.pickle')
