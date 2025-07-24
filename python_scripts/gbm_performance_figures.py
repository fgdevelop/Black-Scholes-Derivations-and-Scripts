# Import necessary packages
import datetime as dt
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import yfinance as yf
from pathlib import Path

class GatherData:

    @staticmethod
    def find_computer_path():
        """
        Description:
        Find through the computer path, where the project lies.

        Parameters:
        - None

        Returns:
        - string containing project folder path.
        """
        return Path(__file__).parent.__str__()

    @staticmethod
    def get_historical_equity_prices(ticker: str, start_date: dt.date, end_date: dt.date, path_prices: str) -> pd.DataFrame:
        """
        Description:
        Utilizes yahoo finance package to download the
        equity historical prices between the required dates.

        Parameters:
        - ticker: string with the stock label.
        - start_date: dt.date with the historical data starting date.
        - end_date: dt.date with the historical data last date.
        - path_prices: string containing path to access folder with the CSV
          files containing previously gathered historical prices from yfinance.

        Returns:
        - df_eq_price_data: pd.DataFrame containing historical stock prices.
        """
        if not os.path.exists(path_prices):
            # Use the yfinance API to gather the data if the file doesn't exist (Dropping possible NaN values)
            df_eq_price_data = yf.download(tickers=ticker,
                                           interval="1d",
                                           start=start_date.strftime("%Y-%m-%d"),
                                           end=end_date.strftime("%Y-%m-%d"),
                                           progress=False
                                           ).dropna().droplevel(level='Ticker', axis=1).reset_index(drop=True)
            # Save to Excel or CSV
            df_eq_price_data.to_csv(path_prices)
        else:
            # If the file already exists, just read it from the folder (removing the Ticker and Data rows, setting the Dates column as the index and dropping possible NaN values)
            df_eq_price_data = pd.read_csv(path_prices)[2:].dropna().set_index('Price').reset_index(drop=True)

        return df_eq_price_data

class DataComputation:

    @staticmethod
    def calculate_log_returns_from_data(df_eq_price_data: pd.DataFrame) -> np.ndarray:
        """
        Description:
        Utilizes the historical equity price data to compute their log-returns.

        Parameters:
        - df_eq_price_data: pd.DataFrame containing historical stock prices.

        Returns:
        - log_returns: array containing the log-returns from the historical stock prices.
        """
        # Get close prices
        prices = df_eq_price_data["Close"]
        prices = prices.astype('float64')
        # Compute daily log returns
        log_returns = np.log(prices / prices.shift(1)).dropna()

        return log_returns

    @staticmethod
    def calculate_historical_vol(log_returns: np.ndarray) -> None:
        """
        Description:
        Utilizes the log-return data to compute historical volatility printing it.

        Parameters:
        - log_returns: array containing the log-returns from the historical stock prices.

        Returns:
        - Print of the annual historical volatility (float).
        """
        # Print the historical volatility (σ_annual) from the stock log return
        sigma_hist_daily = log_returns.std()
        sigma_hist_annual = sigma_hist_daily * np.sqrt(252)
        print(f"Historical volatility: σ (daily) = {sigma_hist_daily}, σ (annual: 252) = {sigma_hist_annual}.")

    @staticmethod
    def fit_normal_dist(log_returns: np.ndarray) -> (float, float):
        """
        Description:
        Utilizes the log-returns to fit a normal distribution over the data,
        estimating the mean and variance parameters.

        Parameters:
        - log_returns: array containing the log-returns from the historical stock prices.

        Returns:
        - mu: float, mean of the fitted normal distribution over the historical log-returns.
        - sigma: float, variance of the fitted normal distribution over the historical log-returns.
        """
        # Fit a normal distribution (as implied by GBM)
        mean, variance = stats.norm.fit(log_returns)
        print(f"The estimated fitting parameters (mean and variance of normal dist): μ_log (daily) = {mean} and σ (daily) = {variance}.")

        return mean, variance

class Plotting:

    @staticmethod
    def histogram_vs_dist_plot(mean: float, variance: float, log_returns: np.ndarray) -> None:
        """
        Description:
        Plot a histogram of the log-returns with the fitted
        normal distribution over it for comparison purposes.

        Parameters:
        - mean: float, mean of the fitted normal distribution over the historical log-returns.
        - variance: float, variance of the fitted normal distribution over the historical log-returns.
        - log_returns: array containing the log-returns from the historical stock prices.

        Returns:
        - Figure of the described plot.
        """
        # Plot empirical histogram vs. fitted normal PDF
        x_vals = np.linspace(log_returns.min(), log_returns.max(), 100)
        pdf_vals = stats.norm.pdf(x_vals, mean, variance)
        plt.figure(figsize=(10, 6))
        plt.hist(log_returns, bins=50, density=True, alpha=0.6, color='blue', label="Empirical Log Returns")
        plt.plot(x_vals, pdf_vals, 'r--', label=f'Normal Fit\nμ = {mean:.4f}, σ = {variance:.4f}')
        plt.xlabel("AAPL Log Returns", fontsize=16)
        plt.ylabel("Density", fontsize=16)
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def quantile_quantile_plot(log_returns: np.ndarray) -> None:
        """
        Description:
        Plot a q-q plot using the quantiles of the log-returns comparing them with
        the theoretical quantiles of a normal distribution.

        Parameters:
        - log_returns: array containing the log-returns from the historical stock prices.

        Returns:
        - Figure of the described plot.
        """
        # Standardize log returns
        log_returns_std = (log_returns - log_returns.mean()) / log_returns.std()
        # Compute Q–Q data
        qq_data = stats.probplot(log_returns_std, dist="norm")
        theoretical_quantiles = qq_data[0][0]
        empirical_quantiles = qq_data[0][1]
        # Plot Q–Q with custom colors
        plt.figure(figsize=(8, 6))
        plt.scatter(theoretical_quantiles, empirical_quantiles, color='steelblue', edgecolor='black', s=20, label='Log Returns Quantiles')
        plt.plot(theoretical_quantiles, theoretical_quantiles, color='firebrick', linewidth=2, label='Normal Distribution Reference')
        # plt.title("Q–Q Plot: Standardized AAPL Log Returns vs Normal Distribution")
        plt.xlabel("Theoretical Quantiles of Log Returns (Normal)", fontsize=16)
        plt.ylabel("Empirical Quantiles of Log Returns", fontsize=16)
        plt.grid(True)
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # Defining the equity ticker
    ticker = "AAPL"
    # Defining start and end dates and setting them into string format
    start_date, end_date = dt.date(2022, 1, 1), dt.date(2024, 1, 2)

    # Getting the computer path to this folder (supposing .py file is right inside the project)
    computer_path = GatherData.find_computer_path()
    # Defining path and file names to save historical prices (supposing folder to save prices is right inside the project)
    folder_to_save_name = "yfin_hist_prices"
    prices_file_name = f"AAPL_historical_prices_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.xlsx"
    path_prices = os.path.join(computer_path, folder_to_save_name, prices_file_name)

    # Gather historical prices for the specific equity
    df_eq_price_data = GatherData.get_historical_equity_prices(ticker, start_date, end_date, path_prices)
    # Calculate the log returns from the historical stock prices
    log_returns = DataComputation.calculate_log_returns_from_data(df_eq_price_data)
    # Calculate the historical volatility according to the historical data
    DataComputation.calculate_historical_vol(log_returns)
    # Fit the historical log return data to a normal distribution
    mu, sigma = DataComputation.fit_normal_dist(log_returns)
    # Plot a histogram of the historical log returns together with the fitted normal distribution
    Plotting.histogram_vs_dist_plot(mu, sigma, log_returns)
    # Plot a quantile-quantile graphic using the historical data
    Plotting.quantile_quantile_plot(log_returns)
