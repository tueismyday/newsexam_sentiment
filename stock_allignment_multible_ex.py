import pandas as pd
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import sys


def main(sentiment_file_path, sentiment_analyzer):
    #######################################
    # Load processed sentiment data
    #######################################

    sentiment_analyzer = sentiment_analyzer
    
    # Load Reddit sentiment data
    try:
        print(f"\nLoading {sentiment_analyzer} sentiment data.")
        reddit_df = pd.read_csv(sentiment_file_path)
    except FileNotFoundError:
        print(f"Error: The file '{sentiment_file_path}' does not exist. Please check the file path and try again.")
        sys.exit(1)

    reddit_df['timestamp'] = pd.to_datetime(reddit_df['timestamp'])
    reddit_df = reddit_df.set_index('timestamp')

    
    
    #######################################
    # Get stock data for specific stocks
    #######################################

    # Define tickers to analyze
    tickers = ["GME", "AMC"]  # Add multiple tickers here
    stock_data = yf.download(tickers, start="2021-01-01", end="2021-08-1", interval="1d")['Adj Close']

    # Transform stock_data for analysis
    stock_data = stock_data.reset_index()
    stock_data = stock_data.melt(id_vars=['Date'], var_name='Ticker', value_name='Adj Close')
    stock_data['Ticker'] = stock_data['Ticker'].str.upper()

    #######################################
    # Allign the sentiment scores with the stock-data
    #######################################

    # Aggregate Reddit Sentiments
    for ticker in tickers:
        reddit_df[f'{ticker}_mention'] = reddit_df['mentioned_stocks'].apply(
            lambda x: ticker in x if isinstance(x, str) else False
        )

    # Ensure 'timestamp' is a column, not the index
    reddit_df = reddit_df.reset_index()

    # Melt the DataFrame for each ticker's mention flag
    reddit_melted = reddit_df.melt(
        id_vars=['timestamp', 'combined_sentiment'],
        value_vars=[f"{ticker}_mention" for ticker in tickers],
        var_name="Ticker", value_name="Mentioned"
    )

    # Filter and aggregate Reddit sentiment per day per ticker
    reddit_aggregated = (
        reddit_melted.query("Mentioned == True")
        .groupby([pd.Grouper(key='timestamp', freq='1D'), 'Ticker'])
        .agg(avg_sentiment=('combined_sentiment', 'mean'), sentiment_count=('combined_sentiment', 'count'))
        .reset_index()
    )
    reddit_aggregated['Ticker'] = reddit_aggregated['Ticker'].str.replace('_mention', '')

    # Combine Reddit and Stock Data
    combined_data = pd.merge(
        stock_data,
        reddit_aggregated,
        left_on=['Date', 'Ticker'],
        right_on=['timestamp', 'Ticker'],
        how='left'
    )
    combined_data['avg_sentiment'] = combined_data['avg_sentiment'].fillna(0)
    combined_data['sentiment_count'] = combined_data['sentiment_count'].fillna(0)

    # Save the combined dataset
    combined_data_path = f"data/combined_stock_reddit_data_{sentiment_analyzer}.csv"
    combined_data.to_csv(combined_data_path, index=False)

    #######################################
    # Visualize average change in stock and sentiment, and calculate correlation between stock- and reddit trends
    #######################################

    results = []
    for ticker_to_plot in tickers:
        plot_data = combined_data[combined_data['Ticker'] == ticker_to_plot]

        ########## For calculation and visualization of the rolling change in sentiment vs. the stock market ##########
        
        # Compute percentage change for stock price and average sentiment
        plot_data['Stock Price Change (%)'] = plot_data['Adj Close'].pct_change() * 100
        plot_data['Avg Sentiment Change'] = plot_data['avg_sentiment'].diff()

        window = 2 # Change window here to number of days for calculation
        # Rolling Averages
        plot_data['Stock Price Rolling Avg'] = plot_data['Stock Price Change (%)'].rolling(window=window).mean()
        plot_data['Sentiment Rolling Avg'] = plot_data['avg_sentiment'].rolling(window=window).mean()            

        # Visualization
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Stock Price Rolling Avg
        ax1.plot(plot_data['Date'], plot_data['Stock Price Rolling Avg'], label='Stock Price Rolling Avg', color='blue')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Stock Price Rolling Avg', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_ylim(-130, 130)

        # Add Sentiment Rolling Avg
        ax2 = ax1.twinx()
        ax2.plot(plot_data['Date'], plot_data['Sentiment Rolling Avg'], label='Sentiment Rolling Avg', color='orange')
        ax2.set_ylabel('Sentiment Rolling Avg', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax2.set_ylim(-1, 1)

        # Add Legends
        fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
        plt.title(f"Advanced Metrics for {ticker_to_plot}")
        plt.grid()
        
        # Save the plot as a PNG file
        output_path = f"results/{ticker_to_plot}_rolling_{window}_days_change_stocks+sentiment_{sentiment_analyzer}.png"
        plt.savefig(output_path)
        print(f"A visualization of the rolling average change in {window} days in sentiment vs. the {ticker_to_plot} stock has been saved to {output_path}.")
        plt.close()

        ########## For calculation and visualization of the daily change in sentiment vs. the stock market ##########
        
        # Compute percentage change for stock price and average sentiment
        plot_data['Stock Price Change (%)'] = plot_data['Adj Close'].pct_change() * 100  # Percentage change in stock price
        plot_data['Avg Sentiment Change'] = plot_data['avg_sentiment'].diff()  # Absolute change in sentiment

        # Create the main figure and axis
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot Stock Price Change on the primary y-axis
        ax1.plot(plot_data['Date'], plot_data['Stock Price Change (%)'], label='Stock Price Change (%)', color='blue')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Stock Price Change (%)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_title(f"{ticker_to_plot} Stock Price and Reddit Sentiment Changes")
        ax1.set_ylim(-130, 130)

        # Create a secondary y-axis for Avg Sentiment Change
        ax2 = ax1.twinx()
        ax2.bar(plot_data['Date'], plot_data['Avg Sentiment Change'], label='Avg Sentiment Change', color='orange', alpha=0.6)
        ax2.set_ylabel('Avg Sentiment Change', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax2.set_ylim(-1, 1)

        # Add legends
        fig.legend(loc="upper right", bbox_to_anchor=(0.1, 0.9))
        plt.grid()

        # Save the plot as a PNG file
        output_path = f"results/{ticker_to_plot}_daily_change_stocks+sentiment_{sentiment_analyzer}.png"
        plt.savefig(output_path)
        print(f"A visualization of the daily change in sentiment vs. the {ticker_to_plot} stock has been saved to {output_path}.")
        plt.close()
        
        # Summary of Metrics
        results_df = pd.DataFrame(results)
        print(results_df)

    #######################################
    # Exploring causality of sentiment reactions with the stock market
    # See whether the stocks significantly reacts to the market or the other way around
    #######################################

    for ticker_to_plot in tickers:
        plot_data = combined_data[combined_data['Ticker'] == ticker_to_plot].copy()

        # Compute percentage change for stock price and average sentiment
        plot_data['Stock Price Change (%)'] = plot_data['Adj Close'].pct_change() * 100
        plot_data['Avg Sentiment Change'] = plot_data['avg_sentiment'].diff()

        # Drop NaN values
        plot_data = plot_data.dropna(subset=['Stock Price Change (%)', 'Avg Sentiment Change'])

        # Granger Causality Test
        maxlag = 10  # Test up to 10 lags (days)
        data_for_test = plot_data[['Stock Price Change (%)', 'Avg Sentiment Change']]

        print(f"\nGranger Causality Tests for {ticker_to_plot}:")
        print("-" * 50)

        # Prepare lists to store F-statistics and p-values
        sentiment_to_stock = {"lags": [], "f_stat": [], "p_value": []}
        stock_to_sentiment = {"lags": [], "f_stat": [], "p_value": []}

        # Test: Does sentiment affect stock price?
        results = grangercausalitytests(data_for_test, maxlag=maxlag, verbose=False)
        for lag, res in results.items():
            sentiment_to_stock["lags"].append(lag)
            sentiment_to_stock["f_stat"].append(res[0]['ssr_ftest'][0])
            sentiment_to_stock["p_value"].append(res[0]['ssr_ftest'][1])

        # Test: Does stock price affect sentiment?
        results = grangercausalitytests(data_for_test[['Avg Sentiment Change', 'Stock Price Change (%)']], maxlag=maxlag, verbose=False)
        for lag, res in results.items():
            stock_to_sentiment["lags"].append(lag)
            stock_to_sentiment["f_stat"].append(res[0]['ssr_ftest'][0])
            stock_to_sentiment["p_value"].append(res[0]['ssr_ftest'][1])

        # Visualization
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot F-statistics
        ax1.plot(sentiment_to_stock["lags"], sentiment_to_stock["f_stat"], label="Sentiment -> Stock Price (F-stat)", marker='o')
        ax1.plot(stock_to_sentiment["lags"], stock_to_sentiment["f_stat"], label="Stock Price -> Sentiment (F-stat)", marker='x')
        ax1.set_xlabel("Lag (Days)")
        ax1.set_ylabel("F-statistic")
        ax1.set_title(f"Granger Causality for {ticker_to_plot}")
        ax1.grid(alpha=0.3)
        ax1.legend(loc="upper left")

        # Plot p-values on a secondary y-axis
        ax2 = ax1.twinx()
        ax2.plot(sentiment_to_stock["lags"], sentiment_to_stock["p_value"], linestyle="--", color="blue", alpha=0.7, label="Sentiment -> Stock Price (p-value)")
        ax2.plot(stock_to_sentiment["lags"], stock_to_sentiment["p_value"], linestyle="--", color="red", alpha=0.7, label="Stock Price -> Sentiment (p-value)")
        ax2.axhline(0.05, color='gray', linestyle='--', label="Significance Threshold (p=0.05)")
        ax2.set_ylabel("p-value")
        ax2.legend(loc="upper right")

        # Save the plot as a PNG file
        output_path = f"results/{ticker_to_plot}_grangercausalitytests_{sentiment_analyzer}.png"
        plt.savefig(output_path)
        print(f"A plot of the granger causality tests on the sentiment data and the {ticker_to_plot} stock has been saved to {output_path}.")
        plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python stock_allignment_multible_ex.py <sentiment_file_path> <sentiment_analyzer>")
        sys.exit(1)
    sentiment_file_path = sys.argv[1]
    sentiment_analyzer = sys.argv[2]
    main(sentiment_file_path, sentiment_analyzer)
