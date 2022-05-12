#!/usr/bin/env python
# -*- coding: utf-8 -*-

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


def obtain_prices_df(csv_filepath, start_date, end_date):
    """
    Obtain the prices DataFrame from the CSV file,
    filter by start date and end date.
    """
    df = pd.read_csv(
        csv_filepath, header=0,
        names=["date", "open", "close", "high", "low", "volume"],
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

def get_best_hmm_model(X, best_state, max_iter=10000):
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
    #获取转移概率
    prob = model.transmat_[hidden_states[-1]]
    # print(prob)
    #获取期望收益
    expected_return = sum(prob * ave_return)
    return hidden_states, expected_return


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

def compute_features(dataset, long_period, short_period):
    # 计算日收益率
    dataset['return'] = dataset["close"].pct_change()

    # 计算长周期平均收益率
    dataset['long_period_return'] = dataset['return'].rolling(long_period).mean()

    # 计算短周期平均收益率
    dataset['short_period_return'] = dataset['return'].rolling(short_period).mean()

    # 计算短期平均成交量和长期平均成交量之比
    dataset['volume_ratio'] = dataset["volume"].rolling(
        short_period).mean() / dataset["volume"].rolling(long_period).mean()

    # 计算长周期内夏普比率，取无风险利率为0
    dataset['Sharpe'] = dataset['return'].rolling(long_period).mean(
    ) / dataset['return'].rolling(long_period).std()        # *np.sqrt(252)

    # 计算指数加权平均回报
    # spanNum = 5
    # dataset['ewma'] = dataset['return'].ewm(span=spanNum, min_periods=1).mean()
    # dataset['ewma_2'] = dataset['return'].ewm(span=spanNum, min_periods=1).std()

    # 计算未来一个周期的收益
    dataset["future_return"] = dataset["close"].pct_change(future_period).shift(-future_period)
    return dataset


pd.options.display.max_rows = 30
pd.options.display.max_columns = 30
PLOT_SHOW = True    # 显示绘图结果
# PLOT_SHOW = False   # 不显示绘图结果


# load data and plot
df_data_path = pathlib.Path.cwd() / ".." / "data"
start_date = datetime.datetime(2010, 1, 1)
end_date = datetime.datetime(2021, 12, 31)

# Feature params
future_period = 1
long_period = 7     # long period
short_period = 3    # short period

indexList = ['CSI300', 'CSI905', 'CSI012', 'CSI033', 'CSI036', 'CSI037']

# indexList = ['CSI300']            


for index in indexList:
    # 读取指数从2010年—2021年的历史数据
    dataset = obtain_prices_df(df_data_path / (index + '.csv'), start_date, end_date)
    dataset = compute_features(dataset, long_period, short_period)

    # Create features
    cols_features = ['long_period_return', 'short_period_return',
                     'volume_ratio', 'Sharpe']  #
    # cols_features = ['ewma','ewma_2']  #
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset.dropna()

    # 选取训练样本，从第1000开始往前每间隔一个adjustment_period取样
    adjustment_period = 1
    train_end_ind = 1500
    train_index = []
    for i in range(train_end_ind, 0, -adjustment_period):
        train_index.append(i)
    train_set = dataset.iloc[train_index]
    train_set = train_set.sort_index()
    train_features = train_set[cols_features]

    # hist plot
    hist_plot(train_set['long_period_return'], str(long_period) + '_days_return')
    hist_plot(train_set['short_period_return'], str(short_period) + '_days_return')
    hist_plot(train_set['volume_ratio'], 'volume_ratio_of' + str(short_period) + str(long_period))
    hist_plot(train_set['Sharpe'], str(long_period) + '_days_Sharpe_ratio')

    # print("train_set:\n", train_set)

    test_index = []
    for i in range(train_end_ind + adjustment_period, dataset.shape[0], adjustment_period):
        test_index.append(i)
    test_set = dataset.iloc[test_index]
    test_set = test_set.sort_index()
    test_features = test_set[cols_features]

    # print("test_set：\n", test_set)

    # ### Plot features
    # fig, axs = plt.subplots(len(cols_features), 1, figsize=(15, 15))
    # colours = cm.rainbow(np.linspace(0, 1, len(cols_features)))
    # for i in range(0, len(cols_features)):
    #     axs[i].plot(dataset.reset_index()[cols_features[i]], color=colours[i])
    #     axs[i].set_title(cols_features[i], fontsize=20)
    #     axs[i].grid(True)

    # # ------------------------------------------------------------------------------------------
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

    plot_hidden_states(model, train_set, train_features, "close")
    # plt.savefig("../figure/hidden_states1.png", dpi=400, bbox_inches='tight')

    plot_in_sample_hidden_states(model, train_set, train_features, "close")
    # plt.savefig("../figure/hidden_states2.png", dpi=400, bbox_inches='tight')

    # print("Best model with {0} states ".format(str(model.n_components)))
    # print('Mean matrix:\n', model.means_)
    # print('Covariance matrix:\n', model.covars_)
    # print('Transition matrix:\n', model.transmat_)

    # ### 滚动预测
    signal = []
    for i in range(train_end_ind, dataset.shape[0] - adjustment_period, adjustment_period):
        # print(dataset.iloc[i:, :].index[0].date())
        train_end_ind = i
        train_index = []
        for j in range(train_end_ind, -1, -adjustment_period):
            train_index.append(j)

        train_set = dataset.iloc[train_index]
        train_set = train_set.sort_index()
        train_features = train_set[cols_features]

        model = get_best_hmm_model(train_features, best_state=4, max_iter=10000)
        hidden_states, expected_return = get_expected_return(model, train_set, train_features)
        print(dataset.iloc[i:, :].index[0].date(), "current state: {}".format(hidden_states[-1]) +
              ", expected_return:{:.4f}".format(expected_return))
        threshold = train_set['return'].mean()
        # print(threshold)

        if ((expected_return > 0.0)
                & (expected_return > 0.1 * threshold)
                ):  # 期望收益大于0.0且大于历史平均收益的0.1倍，买入
            signal.append(1)
        elif(expected_return < 0.0):  # 期望收益小于0.0卖出
            signal.append(-1)
        else:
            signal.append(0)  # 其他状态保持持仓状态不变

    test_set["signal"] = signal
    test_set.to_csv(df_data_path / ('test' + index + '.csv'))


if PLOT_SHOW:
    plt.show()
