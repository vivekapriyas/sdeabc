from openpyxl.utils.datetime import from_excel, CALENDAR_MAC_1904
import pandas as pd
import numpy as np


def col_from_excel(df, col, round_freq = "S"):

    """Convert a column of a DataFrame from Excel1904 dates to DateTime.
    Input:

        df: DataFrame with an index in Excel1904 format

        round_freq: Frequency to round to, for example S for nearest second.
    
    Output:

        Index converted to DateTimeIndex format
    """

    return pd.DatetimeIndex([from_excel(x, epoch = CALENDAR_MAC_1904) for x in df[col]]).round("S")


def load(fpath = 'Kode\wind.dat2', nullspeeds = False):
    """"
    reads in data from forcing.dat2
    converts cartesian wind components to direction and speed
    if nullspeeds = True, one occurence of zero wind is returned 
    returns dataframe with data from 2008-2014 
    """
    df = pd.read_table(fpath, delimiter = ' ', header = 0)
    df = df.assign(Date = col_from_excel(df, 'Date'))
    df = df.loc[df.Date.dt.year <= 2014] 

    df['w'] = np.sqrt(df.u**2 + df.v**2)
    df['theta'] = df.apply(lambda x: np.arctan2(x['v'], x['u']), axis = 1)

    if not nullspeeds:
        df = df.loc[df.w != 0]
    return df

