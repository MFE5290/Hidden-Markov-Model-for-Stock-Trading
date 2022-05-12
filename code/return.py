import numpy as np
import pandas as pd

import pathlib
from matplotlib import pyplot as plt

# %matplotlib inline
import quantstats as qs


def simple_test(dataset1, dataset2):
    ret = []
    cumulative_ret = [1]
    period_length = dataset1.shape[0]

    for i in range(period_length):
        signal = np.array(dataset1.iloc[i, :])
        signal = (signal + 1) / 2
        rate = np.array(dataset2.iloc[i, :])
        position_sum = np.sum(signal) + 1e-9
        w = signal / position_sum
        r = np.dot(w.T, rate)
        ret.append(r)
        cumulative_ret.append(cumulative_ret[-1] * (1 + r))

    return cumulative_ret


indexList = ['CSI300', 'CSI905', 'CSI012', 'CSI032', 'CSI033',
             'CSI034', 'CSI036', 'CSI037', 'CSI038', 'CSI039']

indexList = ['CSI300', 'CSI012']


# indexList = ['CSI300', 'CSI905', 'CSI012']

df_data_path = pathlib.Path.cwd() / ".." / "data"

dataset1 = pd.DataFrame()
dataset2 = pd.DataFrame()
# print(dataset1)
for index in indexList:
    dataset = pd.read_csv(df_data_path / ('test' + index + '.csv'), header=0,
                          index_col="date", parse_dates=True)
    # print(dataset.shape)
    # dataset1 = dataset1.append(dataset['signal'])
    dataset1 = pd.merge(dataset1, dataset['signal'], how='outer',
                        left_index=True, right_index=True, suffixes=('', index))
    dataset2 = pd.merge(dataset2, dataset['return'], how='outer',
                        left_index=True, right_index=True, suffixes=('', index))


# print(dataset1)
print(dataset2)

cumulative_ret = simple_test(dataset1, dataset2)

dataset1['return'] = cumulative_ret[1:]

# print(cumulative_ret)
# plt.plot(cumulative_ret)

# plt.figure()
qs.reports.basic(dataset1['return'], benchmark=dataset2['return'], rf=0.0, grayscale=False, figsize=(8, 5), display=True, compounded=True)
# basic(returns, benchmark=None, rf=0.0, grayscale=False, figsize=(8, 5), display=True, compounded=True, periods_per_year=252, match_dates=False)

# plt.show()
