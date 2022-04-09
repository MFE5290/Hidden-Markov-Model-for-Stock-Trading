#!/usr/bin/env python
# coding: utf-8

# ### Libraries

import warnings
# import quandl
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
from hmmlearn.hmm import GaussianHMM
import scipy
import datetime
import json
import seaborn as sns
import joblib
import pathlib

from plotting import plot_in_sample_hidden_states
from plotting import plot_hidden_states
from plotting import hist_plot

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


def obtain_prices_df(csv_filepath, start_date, end_date):
    """
    Obtain the prices DataFrame from the CSV file,
    filter by start date and end date.
    """
    df = pd.read_csv(
        csv_filepath, header=0,
        names=["date", "open", "close", "high", "low", "volume", "money"],
        index_col="date", parse_dates=True)
    df = df[start_date.strftime("%Y-%m-%d"):end_date.strftime("%Y-%m-%d")]
    df.dropna(inplace=True)
    return df

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
df_data_path = pathlib.Path.cwd() / ".." / "data" / "CSI300.csv"
start_date = datetime.datetime(2005, 4, 8)
end_date = datetime.datetime(2021, 12, 31)
dataset = obtain_prices_df(df_data_path, start_date, end_date)


# dataset = pd.read_csv(df_data_path, index_col=0, parse_dates=True)
# dataset = dataset.shift(1)
# print(dataset.columns)

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 1, 1)
ax.plot(dataset["close"])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set_title('Close Price CSI300', fontsize=30)

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 1, 1)
ax.plot(dataset["volume"])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set_title('Volume CSI300', fontsize=30)
# plt.show()


# ### Let's generate the features and look at them

# Feature params
future_period = 1
std_period = 10
ma_period = 10
price_deviation_period = 10
volume_deviation_period = 10

hold_period = 10  # 持仓周期
# 计算日收益率
dataset['return'] = dataset["close"].pct_change()

# 计算持仓时间平均收益率
dataset['hold_return'] = dataset['return'].rolling(hold_period).mean()

# 计算持仓前5日平均收益率
dataset['5_days_return'] = dataset['return'].rolling(5).mean().shift(hold_period - 5)

# # 计算持仓前5日和持仓期的平均成交量之比
# dataset['volume_ratio'] = dataset[column_volume].rolling(
#     5).mean().shift(hold_period-5) / dataset[column_volume].rolling(hold_period).mean()

# 计算持仓后5日和持仓期的平均成交量之比，结果应该跟上一个没太大差别
dataset['volume_ratio'] = dataset["volume"].rolling(
    5).mean() / dataset["volume"].rolling(hold_period).mean()

# 计算持仓时间长度内夏普比率（暂取无风险利率为0）
dataset['Sharpe'] = dataset['return'].rolling(hold_period).mean(
) / dataset['return'].rolling(hold_period).std()  # *np.sqrt(252)

# 计算未来一个周期的收益
dataset["future_return"] = dataset["close"].pct_change(future_period).shift(-future_period)


hist_plot(dataset['hold_return'], str(hold_period) + '_days_hold_return')
hist_plot(dataset['5_days_return'], '5_days_return')
hist_plot(dataset['volume_ratio'], 'volume_ratio_5_' + str(hold_period))
hist_plot(dataset['Sharpe'], str(hold_period) + '_days_Sharpe_ratio')

print(dataset.head())

# Create features  /////这部分需要重新构造
# cols_features = ['last_return', 'std_normalized', 'ma_ratio', 'price_deviation', 'volume_deviation']
cols_features = ['hold_return', '5_days_return', 'volume_ratio', 'Sharpe']  #
# dataset['last_return'] = dataset[column_close].pct_change()
# dataset['std_normalized'] = dataset[column_close].rolling(std_period).apply(std_normalized)
# dataset['ma_ratio'] = dataset[column_close].rolling(ma_period).apply(ma_ratio)
# dataset['price_deviation'] = dataset[column_close].rolling(
#     price_deviation_period).apply(values_deviation)
# dataset['volume_deviation'] = dataset[column_volume].rolling(
#     volume_deviation_period).apply(values_deviation)


dataset = dataset.replace([np.inf, -np.inf], np.nan)
dataset = dataset.dropna()

# 这部分取训练样本的时候应该间隔一个持仓周期取
a = []
for i in range(0, dataset.shape[0], 1):
    a.append(i)
dataset = dataset.iloc[a]
print("dataset:\n", dataset)
# print()

# Split the data on sets
# train_ind = int(np.where(dataset.index == '2014-01-02')[0])
train_ind = int(dataset.shape[0] * 1)
print(train_ind)
train_set = dataset[cols_features][:train_ind]
# test_set = dataset[cols_features].values[train_ind:]

print("train_set：\n", train_set)


# Plot features
fig, axs = plt.subplots(len(cols_features), 1, figsize=(15, 15))
colours = cm.rainbow(np.linspace(0, 1, len(cols_features)))
for i in range(0, len(cols_features)):
    axs[i].plot(dataset.reset_index()[cols_features[i]], color=colours[i])
    axs[i].set_title(cols_features[i], fontsize=20)
    axs[i].grid(True)

# plt.tight_layout()

# ### Modeling

model = get_best_hmm_model(X=train_set, max_states=2, max_iter=1000000)
# print(model)
print("Best model with {0} states ".format(str(model.n_components)))
print('The number of Hidden States', model.n_components)
print('Mean matrix:\n', model.means_)
print('Covariance matrix:\n', model.covars_)
print('Transition matrix:\n', model.transmat_)


# ### Lets look at state and the next market movement


plot_hidden_states(model, dataset[:train_ind].reset_index(), train_set, "close")


hidden_states = model.predict(train_set)

plot_in_sample_hidden_states(model, dataset[:train_ind].reset_index(), hidden_states, "close")

# ### Feature distribution depending on market state


# compare_hidden_states(hmm_model=model, cols_features=cols_features, conf_interval=0.95)


# ### Save our model


# joblib.dump(model, '../data/'+'quandl_' + asset.replace('/', '_') + '_final_model.pkl')

if PLOT_SHOW:
    plt.show()
