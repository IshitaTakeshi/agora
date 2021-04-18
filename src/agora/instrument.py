
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


class Instrument():
    def __init__(self, ticker, start, end):
        self.id = -1

        self.start = start
        self.end = end
        self.data = download_data(ticker, self.start, self.end)

        self.returns, self.expected_annual_return = self.calculate_return_statistics()
        self.annual_std = self.calculate_risk_statistics()

    @property
    def n_trading_dates(self):
        return len(self.data)

    def calculate_return_statistics(self):
        statistics = []

        # [1] Occasionally, values of zero are obtained as an asset price. In all likelihood, this
        #      value is rubbish and cannot be trusted, as it implies that the asset has no value.
        #     In these cases, we replace the reported asset price by the mean of all asset prices.
        closing_prices = self.data['Adj Close'].to_frame()
        # closing_prices[closing_prices == 0] = closing_prices.mean()

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

    def calculate_risk_statistics(self):
        # [1] Retrieve Closing prices

        returns = self.returns

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
