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
    best_score = -(10 ** 10)
    best_state = 0

    for state in range(1, max_states + 1):
        hmm_model = GaussianHMM(n_components=state, random_state=100,
                                covariance_type="diag", n_iter=max_iter).fit(X)
        if hmm_model.score(X) > best_score:
            best_score = hmm_model.score(X)
            best_state = state

    best_model = GaussianHMM(n_components=best_state, random_state=100,
                             covariance_type="diag", n_iter=max_iter).fit(X)
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
        ax[1].set_title("future return distrbution at {0}th hidden state".format(i))
        ax[1].grid(True)

        ax[2].plot(data["future_return"][mask].cumsum(), c=colour)
        ax[2].set_title("cummulative future return at {0}th hidden state".format(i))
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


# ### load data and plot
df_data_path = pathlib.Path.cwd() / ".." / "data" / "CSI300.csv"
# start_date_string = '2014-04-01'
asset = 'CSI 300 Index'
column_close = 'close'
column_high = 'high'
column_low = 'low'
column_volume = 'volume'

# quandl.ApiConfig.api_key = '3Rt-DUmq8LCQ9AkPi22g'
# dataset = quandl.get(asset, collapse='daily',
#                      trim_start=start_date_string)

dataset = pd.read_csv(df_data_path, index_col='date', parse_dates=True)
dataset = dataset.shift(1)
print(dataset.columns)

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 1, 1)
ax.plot(dataset[column_close])
ax.set_title(asset)

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 1, 1)
ax.plot(dataset[column_volume])
ax.set_title(asset)
# plt.show()


# ### Let's generate the features and look at them

# Feature params
future_period = 1
std_period = 10
ma_period = 10
price_deviation_period = 10
volume_deviation_period = 10

# Create features  /////这部分需要重新构造
cols_features = ['last_return', 'std_normalized', 'ma_ratio', 'price_deviation', 'volume_deviation']
dataset['last_return'] = dataset[column_close].pct_change()
dataset['std_normalized'] = dataset[column_close].rolling(std_period).apply(std_normalized)
dataset['ma_ratio'] = dataset[column_close].rolling(ma_period).apply(ma_ratio)
dataset['price_deviation'] = dataset[column_close].rolling(
    price_deviation_period).apply(values_deviation)
dataset['volume_deviation'] = dataset[column_volume].rolling(
    volume_deviation_period).apply(values_deviation)

dataset["future_return"] = dataset[column_close].pct_change(future_period).shift(-future_period)

dataset = dataset.replace([np.inf, -np.inf], np.nan)
dataset = dataset.dropna()

# Split the data on sets
train_ind = int(np.where(dataset.index == '2018-01-02')[0])
train_set = dataset[cols_features].values[:train_ind]
test_set = dataset[cols_features].values[train_ind:]

# Plot features
# plt.figure(figsize=(20, 10))
fig, axs = plt.subplots(len(cols_features), 1, figsize=(15, 15))
colours = cm.rainbow(np.linspace(0, 1, len(cols_features)))
for i in range(0, len(cols_features)):
    axs[i].plot(dataset.reset_index()[cols_features[i]], color=colours[i])
    axs[i].set_title(cols_features[i])
    axs[i].grid(True)

# plt.tight_layout()

# ### Modeling


model = get_best_hmm_model(X=train_set, max_states=5, max_iter=1000000)
# print(model)
print("Best model with {0} states ".format(str(model.n_components)))

print('The number of Hidden States', model.n_components)
print('Mean matrix')
print(model.means_)
print('Covariance matrix')
print(model.covars_)
print('Transition matrix')
print(model.transmat_)


# ### Lets look at state and the next market movement


plot_hidden_states(model, dataset[:train_ind].reset_index(), train_set, column_close)


# ### Feature distribution depending on market state


compare_hidden_states(hmm_model=model, cols_features=cols_features, conf_interval=0.95)


# ### Save our model


joblib.dump(model, 'quandl_' + asset.replace('/', '_') + '_final_model.pkl')

plt.show()
