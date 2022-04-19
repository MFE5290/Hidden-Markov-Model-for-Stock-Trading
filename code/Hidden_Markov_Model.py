#!/usr/bin/env python
# coding: utf-8

# ### Libraries

import warnings
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
warnings.filterwarnings("ignore")


# ### Basic functions for the analysis
# Modelling, feature engineering, plotting.


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

def model_selection(X, max_states, max_iter=10000):
    """
    :param X: Feature matrix
    :param max_states: the max number of hidden states
    :param max_iter: the max numbers of model iterations
    :return: aic bic caic best_state
    """

    # to store Akaike information criterion (AIC)value
    aic_vect = np.empty([0, 1])
    # to store Bayesian information criterion (BIC) value
    bic_vect = np.empty([0, 1])
    # to store the Bozdogan Consistent Akaike Information Criterion (CAIC)
    caic_vect = np.empty([0, 1])

    for state in range(2, max_states + 1):
        num_params = state**2 + 2 * state - 1
        hmm_model = GaussianHMM(n_components=state, random_state=100,
                                covariance_type="full", n_iter=max_iter).fit(X)
        aic_vect = np.vstack((aic_vect, -2 * hmm_model.score(X) + 2 * num_params))
        bic_vect = np.vstack((bic_vect, -2 * hmm_model.score(X) + num_params * np.log(X.shape[0])))
        caic_vect = np.vstack((caic_vect, -2 * hmm_model.score(X) +
                               num_params * (np.log(X.shape[0]) + 1)))
        best_state = np.argmin(bic_vect) + 2
    return aic_vect, bic_vect, caic_vect, best_state

# Brute force modelling
def get_best_hmm_model(X, best_state, max_iter=100000):
    """
    :param X: stock data
    :param max_states: the number of hidden states
    :param max_iter: numbers of model iterations
    :return: the optimal HMM
    """
    best_model = GaussianHMM(n_components=best_state, random_state=100,
                             covariance_type="full", n_iter=max_iter).fit(X)
    return best_model

def get_expected_return(hmm_model, train_set, train_features):
    hidden_states = model.predict(train_features)
    ave_return = np.zeros(model.n_components)
    for i in range(0, model.n_components):
        mask = hidden_states == i
        ave_return[i] = train_set['return'][mask].mean()
    # print("ave_return:", ave_return)
    print("current state:", hidden_states[-1])
    prob = model.transmat_[hidden_states[-1]]
    # print(prob)
    expected_return = sum(prob*ave_return)
    print("expected_return:", expected_return)
    return hidden_states, expected_return

# Normalized st. deviation
def std_normalized(vals):
    return np.std(vals) / np.mean(vals)

# Ratio of diff between last price and mean value to last price
def ma_ratio(vals):
    return (vals[-1] - np.mean(vals)) / vals[-1]

# z-score for volumes and price
def values_deviation(vals):
    return (vals[-1] - np.mean(vals)) / np.std(vals)


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
            axs[k][i].set_title(cols_features[k] + " (state " + str(i) + "): \
                " + str(np.round(mean_confidence_interval(mc_df[cols_features[k]], conf_interval), 3)))
            axs[k][i].grid(True)

    plt.tight_layout()


pd.options.display.max_rows = 30
pd.options.display.max_columns = 30
PLOT_SHOW = True    # 显示绘图结果
# PLOT_SHOW = False   # 不显示绘图结果

# ### load data and plot
df_data_path = pathlib.Path.cwd() / ".." / "data" / "CSI300.csv"
start_date = datetime.datetime(2005, 4, 8)
end_date = datetime.datetime(2021, 12, 31)
dataset = obtain_prices_df(df_data_path, start_date, end_date)

if (0 == 1):
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


# Feature params
future_period = 1
long_period = 10     # long period
short_period = 5    # short period

# 计算日收益率
dataset['return'] = dataset["close"].pct_change()

# 计算长周期平均收益率
dataset['long_period_return'] = dataset['return'].rolling(long_period).mean()

# 计算短周期平均收益率
# dataset['short_period_return'] = dataset['return'].rolling(
#     short_period).mean().shift(long_period - short_period)

dataset['short_period_return'] = dataset['return'].rolling(short_period).mean()

# # 计算持仓前5日和持仓期的平均成交量之比
# dataset['volume_ratio'] = dataset[column_volume].rolling(
#     5).mean().shift(long_period-5) / dataset[column_volume].rolling(long_period).mean()

# 计算短期持仓和长期平均成交量之比
dataset['volume_ratio'] = dataset["volume"].rolling(
    short_period).mean() / dataset["volume"].rolling(long_period).mean()

# 计算长周期内夏普比率（暂取无风险利率为0）
dataset['Sharpe'] = dataset['return'].rolling(long_period).mean(
) / dataset['return'].rolling(long_period).std()        # *np.sqrt(252)

# 计算未来一个周期的收益
dataset["future_return"] = dataset["close"].pct_change(future_period).shift(-future_period)

# hist plot
hist_plot(dataset['long_period_return'], str(long_period) + '_days_return')
hist_plot(dataset['short_period_return'], str(short_period) + '_days_return')
hist_plot(dataset['volume_ratio'], 'volume_ratio_of' + str(short_period) + str(long_period))
hist_plot(dataset['Sharpe'], str(long_period) + '_days_Sharpe_ratio')

# Create features
cols_features = ['long_period_return', 'short_period_return', 'volume_ratio', 'Sharpe']  #
dataset = dataset.replace([np.inf, -np.inf], np.nan)
dataset = dataset.dropna()
dataset1 = dataset.copy()  # for back test

# 选取训练样本，从第2000开始往前每间隔一个adjustment_period取样
adjustment_period = 1
train_end_ind = 2000
train_index = []
for i in range(train_end_ind, 0, -adjustment_period):
    train_index.append(i)
train_set = dataset.iloc[train_index]
train_set = train_set.sort_index()
train_features = train_set[cols_features]

print("train_set:\n", train_set)

test_index = []
for i in range(train_end_ind + adjustment_period, dataset.shape[0], adjustment_period):
    test_index.append(i)
test_set = dataset.iloc[test_index]
test_set = test_set.sort_index()
test_features = test_set[cols_features]

print("test_set：\n", test_set)


back_test_set = dataset[cols_features]


# ### Plot features
# fig, axs = plt.subplots(len(cols_features), 1, figsize=(15, 15))
# colours = cm.rainbow(np.linspace(0, 1, len(cols_features)))
# for i in range(0, len(cols_features)):
#     axs[i].plot(dataset.reset_index()[cols_features[i]], color=colours[i])
#     axs[i].set_title(cols_features[i], fontsize=20)
#     axs[i].grid(True)

# ----------------------------------------------------------------------------------------------------------
# ### get the best states number
# aic_matrix = np.empty([7, 0])
# bic_matrix = np.empty([7, 0])
# best_states_vector = np.empty([0])
# for i in range(0, 10):
#     print(i)
#     train_set_i = dataset[cols_features][i * 100:1000 + i * 100]
#     aic_vect, bic_vect, caic_vect,best_state = model_selection(X=train_set_i, max_states=8, max_iter=10000)
#     aic_matrix = np.hstack((aic_matrix, aic_vect))
#     bic_matrix = np.hstack((bic_matrix, bic_vect))
#     best_states_vector = np.hstack((best_states_vector, best_state))


# fig, axs = plt.subplots(1, 1, figsize=(15, 15))
# axs.plot(bic_matrix[0], label='2-states', alpha=0.9)
# axs.plot(bic_matrix[1], label='3-states', alpha=0.9)
# axs.plot(bic_matrix[2], label='4-states', alpha=0.9)
# axs.plot(bic_matrix[3], label='5-states', alpha=0.9)
# axs.plot(bic_matrix[4], label='6-states', alpha=0.9)
# axs.plot(bic_matrix[5], label='7-states', alpha=0.9)
# axs.plot(bic_matrix[6], label='8-states', alpha=0.9)
# axs.legend(loc='best')
# plt.grid(linestyle='-.')

# print("best_states_vector", best_states_vector)

# ----------------------------------------------------------------------------------------------------------
model = get_best_hmm_model(train_features, best_state=4, max_iter=10000)

print("Best model with {0} states ".format(str(model.n_components)))
print('Mean matrix:\n', model.means_)
print('Covariance matrix:\n', model.covars_)
print('Transition matrix:\n', model.transmat_)
# ### Modeling
# for i in range(2000, dataset.shape[0]):
signal = []
for i in range(2000, dataset.shape[0]-1):
    print(i)
    adjustment_period = 1
    train_end_ind = i
    train_index = []
    for j in range(train_end_ind, 0, -adjustment_period):
        train_index.append(j)

    train_set = dataset.iloc[train_index]
    train_set = train_set.sort_index()
    train_features = train_set[cols_features]

    # model = get_best_hmm_model(train_features, best_state=5, max_iter=10000)
    hidden_states, expected_return = get_expected_return(model, train_set, train_features)
    # if (hidden_states[-1]== 0 or hidden_states[-1] == 2):
    #     signal.append(1)
    # else:
    #     signal.append(-1)
    if (expected_return > 0.000):           #预期收益大于0.2%，持仓状态
        signal.append(1)
    # elif(expected_return > 0.0):        
    #     signal.append(0)
    else:
        signal.append(-1)                    

test_set["signal"] = signal
test_set.to_csv('test_set.csv')


plot_hidden_states(model, train_set, train_features, "close")

plot_in_sample_hidden_states(model, train_set, train_features, "close")


# ### Feature distribution depending on market state

# compare_hidden_states(hmm_model=model, cols_features=cols_features, conf_interval=0.95)


# Back_test
output = list(model.predict(back_test_set))
df = dataset1
# print(df)
cumulative_ret = [1]
daily_ret = []
for i in range(0, df.shape[0] - adjustment_period, adjustment_period):
    open_price = df.iloc[i + 1, 0]
    close_price = df.iloc[i + adjustment_period, 1]
    if output[int(i / adjustment_period)] == 2:
        daily_ret.append(close_price / open_price - 1)
        cumulative_ret.append(cumulative_ret[-1] * close_price / open_price)
    elif output[int(i / adjustment_period)] == 1:
        daily_ret.append(0)
        cumulative_ret.append(cumulative_ret[-1])
    else:
        daily_ret.append(open_price / close_price - 1)
        cumulative_ret.append(cumulative_ret[-1] * close_price / open_price)

annualized_ret = cumulative_ret[-1]**(252 / df.shape[0])
print(annualized_ret)
print(np.mean(daily_ret) / np.std(daily_ret) / (adjustment_period**0.5))
benchmark = (df.iloc[-1, 1] / df.iloc[0, 1])**(252 / df.shape[0])
print(benchmark)

# fig, axs = plt.subplots(1, 1, figsize=(15, 15))
# axs.plot(cumulative_ret)


if PLOT_SHOW:
    plt.show()


