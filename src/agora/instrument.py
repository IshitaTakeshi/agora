
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
    def __init__(self, ticker, date_range):
        self.id = -1

        if date_range is not None:
            self.date_range = date_range
        else:
            # If there was no specified time interval, presume that the
            # user intends to download historical price data from the
            # past year. Notice that the end of the time interval is
            # today, while the start is one year in the past.
            end = datetime.datetime.now().strftime("%Y-%m-%d")
            start = (datetime.datetime.now() -
                     datetime.timedelta(days=365)).strftime("%Y-%m-%d")
            self.date_range = {"start": start, "end": end}

        self.data = download_data(ticker, date_range['start'], date_range['end'])

        self.return_statistics = {}
        self.risk_statistics = {}
        self.risk_analysis_statistics = {}

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

        self.return_statistics = [
                returns, log_returns, expected_daily_return,
                expected_total_return, expected_annual_return, APR, APY
        ]
        return self.return_statistics

    def calculate_risk_statistics(self):
        statistics = {}

        # [1] Retrieve Closing prices

        returns = self.return_statistics[0]

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
        statistics["daily_std"] = returns.std().values[0]
        statistics["total_std"] = statistics["daily_std"] * \
            np.sqrt(len(returns))
        statistics["annual_std"] = statistics["daily_std"] * np.sqrt(252)

        # variance
        statistics["daily_var"] = statistics["daily_std"] ** 2
        statistics["total_var"] = statistics["total_std"] ** 2
        statistics["annual_var"] = statistics["annual_std"] ** 2

        # statistics["annual_std"] = (returns.resample('Y').std() * np.sqrt(252)).mean().values[0]

        self.risk_statistics = statistics
        return self.risk_statistics

    def calculate_statistics(self):
        returns = self.calculate_return_statistics()
        risks = self.calculate_risk_statistics()
        return returns, risks
