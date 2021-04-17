import numpy as np
import pandas as pd
import seaborn as sns
import sys
from pandas_datareader import data as web
from datetime import datetime, date

import utils
from instrument import Instrument
from portfolio import Portfolio

tickers_data = pd.DataFrame(pd.read_csv('data/tickers.csv'))
stocks_tickers = list(tickers_data.dropna(subset=['IPOyear'])['Symbol'])


def get_ticker_historical_data(ticker, start, end):
    '''
    function:
        This function retrieves all the price data for a ticker in the following format :

            Date || Open | High | Low | Close | Adj Close |
            -----||------|------|-----|-------|-----------|
            xxxx || xxxx | xxxx | xxx | xxxxx | xxxxxxxxx |

    args:
        [*] ticker    : The ticker for which historical data are retrieved
        [*] from_date : Date formatted as `dd/mm/yyyy`, since when data is going to be retrieved.
        [*] to_date   : Date formatted as `dd/mm/yyyy`, until when data is going to be retrieved.
    '''

    # 2 - check parameter validity
    # DATETIMES
    try:
        start = datetime.strptime(start, '%d/%m/%Y')
    except ValueError:
        raise ValueError(
            "ERR#0001: incorrect start date format, "
            "it should be 'dd/mm/yyyy'.")
    try:
        end = datetime.strptime(end, '%d/%m/%Y')
    except ValueError:
        raise ValueError(
            "ERR#0002: incorrect en dformat, it should be 'dd/mm/yyyy'.")

    if start >= end:
        raise ValueError(
            "ERR#0003: `end` should be greater than `start`, "
            "both formatted as 'dd/mm/yyyy'.")

    date_range = {'start': start, 'end': end}

    # 3 - Retrieve instrument data
    return Instrument(ticker, date_range)


def get_ticker_statistics(ticker, start, end):
    '''
    function:
        This function
        1. Uses `ticker_historical_data` to retrieve all the price data
            for a ticker in the following format :

            Date || Open | High | Low | Close | Adj Close |
            -----||------|------|-----|-------|-----------|
            xxxx || xxxx | xxxx | xxx | xxxxx | xxxxxxxxx |
        2. Calculate both RETURN & RISK descriptive statistics

    args:
        [*] ticker    : The ticker for which historical data are retrieved
        [*] start.    : Date formatted as `dd/mm/yyyy`, since when data is going to be retrieved.
        [*] end       : Date formatted as `dd/mm/yyyy`, until when data is going to be retrieved.
    '''

    # 2
    # 2.1 - Get the data
    instrument = get_ticker_historical_data(ticker, start, end)

    # 2.2 - Calculate [*] return
    #                 [*] log return
    #                 [*] expected daily return
    #                 [*] expected return
    instrument.calculate_statistics()

    return instrument


def get_tickers_statistics(tickers, start, end):
    '''
    function:
        This function
        1. Uses `get_ticker_statistics` N times, 1 for each ticker instrument. For each instrument
            1.1 Uses `ticker_historical_data` to retrieve all the price data for a ticker in the following format :

            Date || Open | High | Low | Close | Adj Close |
            -----||------|------|-----|-------|-----------|
            xxxx || xxxx | xxxx | xxx | xxxxx | xxxxxxxxx |
            1.2. Calculate both RETURN & RISK descriptive statistics

    args:
        [*] N             : Number of tickers
        [*] tickers : The ticker list for which historical data are retrieved
        [*] start       : Date formatted as `dd/mm/yyyy`, since when data is going to be retrieved.
        [*] end         : Date formatted as `dd/mm/yyyy`, until when data is going to be retrieved.
    '''

    printing = False

    # 2 - Retrieve Data & Calculate Descriptive statistics for each ticker:
    instruments = []
    expected_annual_return_list = []
    annual_std_list = []
    for ticker in tickers:
        instrument = get_ticker_statistics(ticker=ticker, start=start, end=end)
        instruments.append(instrument)
        expected_annual_return = instrument.return_statistics[4]
        expected_annual_return_list.append(expected_annual_return * 100)
        annual_std_list.append(instrument.risk_statistics[2])

    # 3 - Convert Descriptive statistics from list to dataframes
    descriptive_dict = {"Expected Annual Return": expected_annual_return_list,
                        "Annual Standard Deviation": annual_std_list
                        }
    descriptive_df = pd.DataFrame(descriptive_dict)
    descriptive_df.index = tickers

    # 4 - Display results
    utils.display(descriptive_df)

    return instruments, descriptive_df


def portfolio_optimization(num_portfolios, tickers, start, end):
    '''
    function:
        This function :
        1. Uses `get_tickers_statistics` N times, 1 for each ticker instrument
            to calculate the descriptive metrics
        2. Calls `portfolio_construction` for `num_port` times to
            construct `num_port` random portfolios
            2.1 Initialize random weights for the corresponding porftolio.
        3. Then by locating the one with the highest Sharpe ratio portfolio, it displays
                [*] Maximum Sharpe ratio portfolio as red star sign.
                [*] Minimum volatility portfolio as green start sign
            All the randomly generated portfolios will be also plotted
            with colour map applied to them based on the Sharpe ratio.
            The bluer, the higher Sharpe ratio.
        4. For these two optimal portfolios, it will also show how it allocates
            the budget within the portfolio.


    args:
        [*] P             : Number of portfolios
        [*] N             : Number of tickers
        [*] tickers : The ticker list of which portfolio will be constructed
        [*] start       : Date formatted as `dd/mm/yyyy`, since when data is going to be retrieved.
        [*] end         : Date formatted as `dd/mm/yyyy`, until when data is going to be retrieved.
    '''
    # 1 - check arguments
    printing = True

    # 2 - Get the instrument list along with their calculated descriptive statistics
    instruments, descriptive_df = get_tickers_statistics(
        tickers=tickers, start=start, end=end)
    stocks_idx = [idx for idx in range(
        len(tickers)) if tickers[idx] in stocks_tickers]
    risk_free = utils.risk_free_return(
        date_range=instruments[0].date_range)
    returns_merged = utils.merge_instrument_returns(
        instruments=instruments, tickers=tickers)

    print("instruments = ", instruments)

    # 3 - Portfolio simulation
    all_weights, ret_arr, std_arr, sharpe_arr = [], [], [], []
    for i in range(num_portfolios):
        if i % 100 == 0:
            print("{} out of {}\n".format(i, num_portfolios), end='')
        portfolio = Portfolio(instruments=instruments, returns_merged=returns_merged,
                              tickers=tickers, risk_free=risk_free)

        # weights
        portfolio.initialize_weights()
        w_stocks = sum([w[i] for i in portfolio.weights if i in stocks_idx])

        # return, std, sharpe ratio
        portfolio.calculate_statistics()
        portfolio_statistics = portfolio.statistics
        R_P = portfolio_statistics['portfolio_annual_return']
        STP_P = portfolio_statistics['portfolio_annual_std']
        SR_P = portfolio_statistics['portfolio_annual_sr']

        # append results
        all_weights.append(portfolio.weights)
        ret_arr.append(R_P)
        std_arr.append(STP_P)
        sharpe_arr.append(SR_P)

    # 4 - Calculate 2 most efficient portfolios.
    # [1] Max Sharpe Ratio Portfolio
    opt_idx = np.argmax(sharpe_arr)
    opt_sr, opt_ret, opt_std = sharpe_arr[opt_idx], ret_arr[opt_idx], std_arr[opt_idx]
    opt_weights = all_weights[opt_idx]
    messages = []
    messages.append(
        "          * Max Sharpe Ratio optimized Portfolio *          ")
    messages.append(" Portfolio Annual Return (252 days)  = {} % ".format(
        round(opt_ret * 100, 3)))
    messages.append(" Portfolio Annual Standard Deviation  (252 days)  = {}  ".format(
        round(opt_std, 3)))
    messages.append(
        " Portfolio Annual Sharpe Ratio  (252 days)  = {}  ".format(round(opt_sr, 3)))
    if printing:
        utils.pprint(messages)

    # [2] Min Standard Deviation Ratio portfolio
    min_idx = np.argmin(std_arr)
    min_sr, min_ret, min_std = sharpe_arr[min_idx], ret_arr[min_idx], std_arr[min_idx]
    min_weights = all_weights[min_idx]
    messages = []
    messages.append(
        "      * Min Standard Deviation optimized Portfolio *      ")
    messages.append(" Portfolio Annual Return (252 days)  = {} % ".format(
        round(min_ret * 100, 3)))
    messages.append(" Portfolio Annual Standard Deviation  (252 days)  = {}  ".format(
        round(min_std, 3)))
    messages.append(
        " Portfolio Annual Sharpe Ratio  (252 days)  = {}  ".format(round(min_sr, 3)))
    if printing:
        utils.pprint(messages)

    # [3] weight allocation for both efficient portfolios
    weights_dict = {"Max SR Allocation Weights": opt_weights *
                    100, 'Min σ Allocation Weights': min_weights * 100}
    weights_df = pd.DataFrame(weights_dict)
    weights_df.index = tickers
    if printing:
        utils.display(weights_df)

    # 5 - Plot the portfolios along with the 2 efficient portfolios.
    title = "{}_portfolio_simulation".format(num_portfolios)
    portfolio.plot_portfolio_simulation(
        title, instruments[0].date_range, std_arr, ret_arr, sharpe_arr, descriptive_df, returns_merged)

    return


def main():
    portfolio_optimization(
        5000,
        ["AAPL", "MSFT", "NVDA", "VRTX", "BTC-USD"],
        "01/01/2015",
        "23/04/2020"
    )


main()
