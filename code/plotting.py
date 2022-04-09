
import numpy as np
import pandas as pd

from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator

from hmmlearn.hmm import GaussianHMM


def plot_in_sample_hidden_states(hmm_model, df, hidden_states, column_price):
    """
    Plot the adjusted closing prices masked by 
    the in-sample hidden states as a mechanism
    to understand the market regimes.
    """
    # Predict the hidden states array
    # hidden_states = hmm_model.predict(features)
    # Create the correctly formatted plot
    fig, axs = plt.subplots(figsize=(15, 15), sharex=True, sharey=True)
    colours = cm.rainbow(
        np.linspace(0, 1, hmm_model.n_components)
    )
    for i, colour in enumerate(colours):
        mask = hidden_states == i
        axs.plot(df.index, df[column_price], c='grey')
        axs.plot_date(
            df.index[mask],
            df[column_price][mask],
            ".", linestyle='none',
            c=colour
        )
        # axs.set_title("Hidden State #%s" % i)
        axs.xaxis.set_major_locator(YearLocator())
        axs.xaxis.set_minor_locator(MonthLocator())
        axs.grid(True)
    # plt.show()

# General plots of hidden states
def plot_hidden_states(hmm_model, data, X, column_price):
    # plt.figure(figsize=(15, 15))
    fig, axs = plt.subplots(hmm_model.n_components, 3, figsize=(15, 15))
    colours = cm.rainbow(
        np.linspace(0, 1, hmm_model.n_components))
    hidden_states = hmm_model.predict(X)

    for i, (ax, colour) in enumerate(zip(axs, colours)):
        mask = hidden_states == i
        ax[0].plot(data.index, data[column_price], c='grey')
        ax[0].plot(data.index[mask], data[column_price][mask], '.', c=colour)
        ax[0].set_title("{0}th hidden state".format(i))
        ax[0].xaxis.set_major_locator(YearLocator())
        ax[0].xaxis.set_minor_locator(MonthLocator())
        ax[0].grid(True)

        ax[1].hist(data["future_return"][mask], bins=30)
        ax[1].set_xlim([-0.1, 0.1])
        ax[1].set_title("future return distribution at {0}th hidden state".format(i))
        ax[1].grid(True)

        ax[2].plot(data["future_return"][mask].cumsum(), c=colour)
        ax[2].set_title("cumulative future return at {0}th hidden state".format(i))
        ax[2].grid(True)

    plt.tight_layout()


def hist_plot(data, title):
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(data, bins=30, color='gray', edgecolor='blue')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.set_title(title, fontsize=30)
