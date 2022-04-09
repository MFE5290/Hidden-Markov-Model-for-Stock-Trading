from jqdatasdk import *

auth('18210006280', 'Programming6280')

data_dir = '../data/'

df = get_price('000300.XSHG', start_date='2005-04-08',
               end_date='2021-12-31', frequency='daily', panel=False)

df.to_csv(data_dir + 'CSI300.csv', index_label='date')
