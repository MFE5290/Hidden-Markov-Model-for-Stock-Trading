from jqdatasdk import *

import numpy as np
import pandas as pd

pd.options.display.max_rows = 100
pd.options.display.max_columns = 100

auth('18210006280', 'Programming6280')

data_dir = '../data/'

# df = get_price('000300.XSHG', start_date='2005-04-08',
#                end_date='2021-12-31', frequency='daily', panel=False)

# df.to_csv(data_dir + 'CSI 300.csv', index_label='date')

# df = get_price('000905.XSHG', start_date='2007-01-15',
#                end_date='2021-12-31', frequency='daily', panel=False)

# df.to_csv(data_dir + 'CSI 905.csv', index_label='date')

# df = get_price('000012.XSHG', start_date='2003-01-02',
#                end_date='2021-12-31', frequency='daily', panel=False)

# df.to_csv(data_dir + 'CSI012.csv', index_label='date')

# df = get_price('000979.XSHG', start_date='2005-01-02',
#                end_date='2021-12-31', frequency='daily', panel=False)

# df.to_csv(data_dir + 'CSI979.csv', index_label='date')

df = get_price('000032.XSHG', start_date='2009-01-09',
               end_date='2021-12-31', frequency='daily', panel=False)

df.to_csv(data_dir + 'CSI032.csv', index_label='date')

df = get_price('000033.XSHG', start_date='2009-01-09',
               end_date='2021-12-31', frequency='daily', panel=False)

df.to_csv(data_dir + 'CSI033.csv', index_label='date')

df = get_price('000034.XSHG', start_date='2009-01-09',
               end_date='2021-12-31', frequency='daily', panel=False)

df.to_csv(data_dir + 'CSI034.csv', index_label='date')

df = get_price('000036.XSHG', start_date='2009-01-09',
               end_date='2021-12-31', frequency='daily', panel=False)

df.to_csv(data_dir + 'CSI036.csv', index_label='date')

df = get_price('000037.XSHG', start_date='2009-01-09',
               end_date='2021-12-31', frequency='daily', panel=False)

df.to_csv(data_dir + 'CSI037.csv', index_label='date')


df = get_price('000038.XSHG', start_date='2009-01-09',
               end_date='2021-12-31', frequency='daily', panel=False)

df.to_csv(data_dir + 'CSI038.csv', index_label='date')


df = get_price('000039.XSHG', start_date='2009-01-09',
               end_date='2021-12-31', frequency='daily', panel=False)

df.to_csv(data_dir + 'CSI039.csv', index_label='date')

