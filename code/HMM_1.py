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
df_data_path = pathlib.Path.cwd() / "CSI300.csv"
# start_date_string = '2014-04-01'
asset = 'CSI 300 Index'
column_close = 'close'
column_high = 'high'
column_low = 'low'
column_volume = 'volume'

dataset = pd.read_csv(df_data_path, index_col=0, parse_dates=True)
# dataset = dataset.shift(1)

# ### Let's generate the features and look at them

# Feature params
future_period = 1
std_period = 10
ma_period = 10
price_deviation_period = 10
volume_deviation_period = 10

hold_period = 10  # 持仓周期
# 计算日收益率
dataset['return'] = dataset[column_close].pct_change()

# 计算持仓时间平均收益率
dataset['hold_return'] = dataset['return'].rolling(hold_period).mean()

# 计算5日平均收益率
dataset['5_days_return'] = dataset['return'].rolling(5).mean()

# 计算5日和持仓期的平均成交量之比
dataset['volume_ratio'] = dataset[column_volume].rolling(5).mean() / \
                          dataset[column_volume].rolling(hold_period).mean()

# 计算持仓时间长度内夏普比率（暂取无风险利率为0）
dataset['Sharpe'] = dataset['return'].rolling(hold_period).mean() / \
                    dataset['return'].rolling(hold_period).std()  # *np.sqrt(252)

cols_features = ['hold_return', '5_days_return', 'volume_ratio', 'Sharpe']
# dataset["future_return"] = dataset[column_close].pct_change(future_period).shift(-future_period)

# 划分训练数据、测试数据
dataset = dataset.replace([np.inf, -np.inf], np.nan)
dataset = dataset.dropna()
df = dataset.copy()
train_ind = int(np.where(dataset.index == '2014-01-02')[0])
train_set = pd.DataFrame(dataset[cols_features].values[:train_ind])
test_set = dataset[cols_features].values[train_ind:]
df = df.iloc[train_ind:, :]

# 设置调仓周期
adjustment_period = 2
a = []
for i in range(0, train_set.shape[0], adjustment_period):
    a.append(i)
train_set = train_set.iloc[a]


### Modeling

model = get_best_hmm_model(X=train_set, max_states=3, max_iter=1000)
print("Best model with {0} states ".format(str(model.n_components)))
print('The number of Hidden States', model.n_components)
print('Mean matrix')
print(model.means_)
print('Covariance matrix')
print(model.covars_)
print('Transition matrix')
print(model.transmat_)

### Lets look at state and the next market movement


# plot_hidden_states(model, dataset[:train_ind].reset_index(), train_set, column_close)
# plt.show()

### Back_test

output = list(model.predict(test_set))
signal = []
for i in range(len(output)):
    if i%2 == 1:
        signal.append(0)
    else:
        if output[i] == 0:
            signal.append(0)
            print(0)
        elif output[i] == 1:
            signal.append(-1)
        else:
            signal.append(1)
            print(2)
for i in range(2, len(signal), 2):
    if signal[i-2] == 1 and signal[i] == 1:
        signal[i] == 0

df['signal'] = signal
df = df.shift(1)
# print(df)
df = df.dropna()
df.to_csv('backtest.csv')



