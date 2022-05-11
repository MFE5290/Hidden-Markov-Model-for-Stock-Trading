#!/usr/bin/env python
# coding: utf-8

# ### Libraries
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from matplotlib import cm, pyplot as plt
from hmmlearn.hmm import GaussianHMM
import scipy
import datetime
import json
import seaborn as sns
import joblib
import pathlib

sns.set()

# ### Basic functions for the analysis
# Modelling, feature engineering, plotting.

warnings.filterwarnings("ignore")


# Brute force modelling
def get_best_hmm_model(X, max_states, max_iter=10000):
    """

    :param X: stock data
    :param max_states: the number of hidden states
    :param max_iter: numbers of model iterations
    :return: the optimal HMM
    """
    best_score = -(10 ** 10)
    best_state = 0

    for state in range(1, max_states + 1):
        hmm_model = GaussianHMM(n_components=state, random_state=100,
                                covariance_type="full", n_iter=max_iter).fit(X)
        if hmm_model.score(X) > best_score:
            best_score = hmm_model.score(X)
            best_state = state

    best_model = GaussianHMM(n_components=best_state, random_state=100,
                             covariance_type="full", n_iter=max_iter).fit(X)
    return best_model


# Normalized st. deviation
def std_normalized(vals):
    return np.std(vals) / np.mean(vals)


# Ratio of diff between last price and mean value to last price
def ma_ratio(vals):
    return (vals[-1] - np.mean(vals)) / vals[-1]


# z-score for volumes and price
def values_deviation(vals):
    return (vals[-1] - np.mean(vals)) / np.std(vals)


# General plots of hidden states
def plot_hidden_states(hmm_model, data, X, column_price):
    # plt.figure(figsize=(15, 15))
    fig, axs = plt.subplots(hmm_model.n_components, 3, figsize=(15, 15))
    colours = cm.prism(np.linspace(0, 1, hmm_model.n_components))
    hidden_states = hmm_model.predict(X)

    for i, (ax, colour) in enumerate(zip(axs, colours)):
        mask = hidden_states == i
        ax[0].plot(data.index, data[column_price], c='grey')
        ax[0].plot(data.index[mask], data[column_price][mask], '.', c=colour)
        ax[0].set_title("{0}th hidden state".format(i))
        ax[0].grid(True)

        ax[1].hist(data["future_return"][mask], bins=30)
        ax[1].set_xlim([-0.1, 0.1])
        ax[1].set_title("future return distribution at {0}th hidden state".format(i))
        ax[1].grid(True)

        ax[2].plot(data["future_return"][mask].cumsum(), c=colour)
        ax[2].set_title("cumulative future return at {0}th hidden state".format(i))
        ax[2].grid(True)

    plt.tight_layout()


def mean_confidence_interval(vals, confidence):
    a = 1.0 * np.array(vals)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m - h, m, m + h


def compare_hidden_states(hmm_model, cols_features, conf_interval, iters=1000):
    # plt.figure(figsize=(15, 15))
    fig, axs = plt.subplots(len(cols_features), hmm_model.n_components, figsize=(15, 15))
    colours = cm.prism(np.linspace(0, 1, hmm_model.n_components))

    for i in range(0, hmm_model.n_components):
        mc_df = pd.DataFrame()

        # Samples generation
        for j in range(0, iters):
            row = np.transpose(hmm_model._generate_sample_from_state(i))
            mc_df = mc_df.append(pd.DataFrame(row).T)
        mc_df.columns = cols_features

        for k in range(0, len(mc_df.columns)):
            axs[k][i].hist(mc_df[cols_features[k]], color=colours[i])
            axs[k][i].set_title(cols_features[k] + " (state " + str(i) + "): " +
                                str(np.round(mean_confidence_interval(mc_df[cols_features[k]], conf_interval), 3)))
            axs[k][i].grid(True)

    plt.tight_layout()


pd.options.display.max_rows = 30
pd.options.display.max_columns = 30
PLOT_SHOW = True
# PLOT_SHOW = False
# ### load data and plot
df_data_path = '/Users/wyb/PycharmProjects/pythonProject/HMM_project/CSI300.csv'
# start_date_string = '2014-04-01'
asset = 'CSI 300 Index'
column_close = 'close'
column_high = 'high'
column_low = 'low'
column_volume = 'volume'

dataset = pd.read_csv(df_data_path, index_col=0, parse_dates=True)
# dataset = dataset.shift(1)

# Feature params
future_period = 2
std_period = 10
ma_period = 10
price_deviation_period = 10
volume_deviation_period = 10
hold_period = 10
short_period = 3

# 计算日收益率
dataset['return'] = dataset[column_close].pct_change()

# 计算持仓时间平均收益率
dataset['hold_return'] = dataset['return'].rolling(hold_period).mean()

# 计算5日平均收益率
dataset['5_days_return'] = dataset['return'].rolling(short_period).mean()

# 计算5日和持仓期的平均成交量之比
dataset['volume_ratio'] = dataset[column_volume].rolling(short_period).mean() / \
                          dataset[column_volume].rolling(hold_period).mean()

# 计算持仓时间长度内夏普比率（暂取无风险利率为0）
dataset['Sharpe'] = dataset['return'].rolling(hold_period).mean() / \
                    dataset['return'].rolling(hold_period).std()  # *np.sqrt(252)

cols_features = ['hold_return', '5_days_return', 'volume_ratio', 'Sharpe']
dataset["future_return"] = dataset[column_close].pct_change(future_period).shift(-future_period)


# 划分训练数据、测试数据
dataset = dataset.replace([np.inf, -np.inf], np.nan)
dataset = dataset.dropna()
df = dataset.copy()
train_ind = int(np.where(dataset.index == '2014-01-02')[0])
train_set = pd.DataFrame(dataset[cols_features].values[:train_ind])
test_set = pd.DataFrame(dataset[cols_features].values[train_ind:])
df = df.iloc[train_ind:, :]


# 设置调仓周期
adjustment_period = 2
a = []
for i in range(0, train_set.shape[0], adjustment_period):
    a.append(i)
train_set = train_set.iloc[a]


# Model
model = get_best_hmm_model(X=train_set, max_states=3, max_iter=1000)
print("Best model with {0} states ".format(str(model.n_components)))
print('The number of Hidden States', model.n_components)
print('Mean matrix')
print(model.means_)
print('Covariance matrix')
print(model.covars_)
print('Transition matrix')
print(model.transmat_)

# Back-test
# 不定长回测
# output = []
# for i in range(0, test_set.shape[0], 2):
#     index_list = []
#     for j in range(0, i+1, 2):
#         index_list.append(j)
#     test_data = test_set.iloc[index_list]
#     output.append(model.predict(test_data)[-1])

# 定长回测
fix_length = 1000
output = []
for i in range(0, test_set.shape[0], 2):
    index_list = []
    for j in range(0, fix_length, 2):
        index_list.append(i+train_ind-j)
    index_list.reverse()
    test_data = dataset[cols_features].values[index_list]
    output.append(model.predict(test_data)[-1])

# 讲状态信号调整为调仓信号
# 具体调仓信号需要根据各状态因子值大小顺序判断
# state: 1 [0, 1, 2]->[up,down,plat]
# state: 2 [0, 1, 2]->[up,plat,down]
# state: 3 [0, 1, 2]->[down,up,plat]
# state: 4 [0, 1, 2]->[plat,up,down]
# state: 5 [0, 1, 2]->[down,plat,up]
# state: 6 [0, 1, 2]->[plat,down,up]
signal = []
for i in range(len(output)):
    if output[i] == 0:
        signal.append(1)
    elif output[i] == 1:
        signal.append(-1)
    else:
        signal.append(0)
    if len(signal) < test_set.shape[0]:
        signal.append(0)

# 若连续多天开仓，则后续信号为0，持有，不重复开仓
for i in range(2, len(signal), 2):
    if signal[i-2] == 1 and signal[i] == 1:
        signal[i] == 0

# 输出预测信号，导入回测模块
df['o_signal'] = signal
df['signal'] = df['o_signal'].shift(1)
df = df.dropna()
# print(df)
df.to_csv('backtest.csv')




