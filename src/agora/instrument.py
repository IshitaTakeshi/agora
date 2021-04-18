
import numpy as np
import pandas as pd
import seaborn as sns
import random
import re
import sys
from pandas_datareader import data as web
from datetime import datetime, date

import utils


def download_data(ticker, start, end):
    try:
        return web.DataReader(ticker, data_source='yahoo',
                              start=start, end=end)
    except ValueError:
        raise ValueError(
            "Invalid ticker symbol specified or else there "
            "was not an internet connection available.")


def calculate_return_statistics(closing_prices):
    #######################################################################
    #           P_t - P_{t-1}                                              #
    #  [2] R_t = -------------   ,   change_t = log(P_t) - log(P_{t - 1}) #
    #             P_{t-1}                                                 #
    #                                                                         #
    #      [*] Expected Annual Return  = R_t * 252                          #
    #      [*] Annualized Return (APR) = AVG(SUM(R_t per Year))           #
    #                                                                     #
    #######################################################################
    returns = closing_prices.pct_change().dropna()
    log_returns = closing_prices.apply(
        lambda x: np.log(x) - np.log(x.shift(1))).dropna()

    # [3] [*] For the expected return, we simply take the mean value of the calculated daily returns.
    #     [*] Multiply the average daily return by the length of the time series in order to
    #         obtain the expected return over the entire period.
    cummulative_return = returns.iloc[::-1].sum().values[0]

    expected_daily_return = returns.mean().values[0]
    expected_total_return = expected_daily_return * len(returns)
    expected_annual_return = expected_daily_return * 252
    APR = returns.resample('Y').sum().mean().values[0]
    APY = ((1 + cummulative_return) ** (252 / len(returns)) - 1)

    return returns, expected_annual_return


def calculate_risk_statistics(returns):
    # [1] Retrieve Closing prices

    ##############################################################
    #                 ____________________                       #
    #                |              _                              #
    #                |  SUM( R_t - R_t)                             #
    #  [2] σ_t   = \ |  -----------------   , Var_t = σ_t ^ 2    #
    #               \|  # trading days                           #
    #                                ___                              #
    #      [*] Annual Std = σ_t * \|252                              #
    #      [*] Annual Var = σ_t^2 * 252                          #
    #                                                            #
    ##############################################################
    # standrd deviation
    daily_std = returns.std().values[0]
    total_std = daily_std * \
        np.sqrt(len(returns))
    annual_std = daily_std * np.sqrt(252)

    # variance
    daily_var = daily_std ** 2
    total_var = total_std ** 2
    annual_var = annual_std ** 2

    return annual_std


class Instrument():
    def __init__(self, ticker, start, end):
        self.start = start
        self.end = end
        self.data = download_data(ticker, self.start, self.end)

        closing_prices = self.data['Adj Close'].to_frame()
        self.returns, self.expected_annual_return = calculate_return_statistics(closing_prices)
        self.annual_std = calculate_risk_statistics(self.returns)

    @property
    def n_trading_dates(self):
        return len(self.data)
