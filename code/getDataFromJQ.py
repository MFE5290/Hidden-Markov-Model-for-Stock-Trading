# import datetime
# import numpy as np
# import pandas as pd
# import pathlib
from jqdatasdk import *

auth('18210006280', 'Programming6280')

data_dir = '../data/'

# data_path = pathlib.Path.cwd() / ".." / "data" / "CSI300.csv"

df = get_price('000300.XSHG', start_date='2005-04-08',
               end_date='2021-12-31', frequency='daily', panel=False)
print(df.head())
df.to_csv(data_dir + 'CSI300.csv')
