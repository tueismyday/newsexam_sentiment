# newsexam_sentiment



# Instruction:

To initialize, use "run.py" which serves as the main pipeline controller for the sentiment analysis and stock alignment workflow. It guides the user through sequential steps to load, clean, analyze, and align the data to the stock market.


# Explenation of the items in this repository


# The results folder:
-   This folder contains the outputs of the sentiment analysis and its comparison to the stock market. The sentiment analysis results include data on the emotional tone of textual data, and the comparison provides insights into how sentiment correlates with stock market trends.

# Data ZIP-folders:
- WSBdata.zip contains the original raw data provided for the project and is needed to run the "run.py" script.
  -  the script is meant to saved under folder name "data" in the python enviroment.
    
- cleaned_WSB_data.zip containt the cleaned data after run of data_load+cleaning
  - excluded_wallsteetbets_data.zip is the datarows not found to contain mention of stock tickers, which are meant for model-tuning
    - excluded_wallsteetbets_data_labels.zip is the same data, but 200 of the posts have gotten a subjective label of sentiment.
      
- The MODELNAME_sentiment.zip datafolders are resulting data from the sentiment analysis
  
- The combined_stock_reddit_data_MODELNAME.zip includes the stock-market values and sentiment analisys scores for the given model at a given date.

# data_load+cleaning.py:
- This script loads, cleans, and prepares data from the WallStreetBets subreddit for sentiment analysis and stock mentions.

    - Data Loading: Reads raw data (reddit_wsb.csv) and handles missing values, ensuring valid entries.

    - Stock Extraction: Identifies stock mentions (e.g., $TSLA, AAPL) in post titles and bodies. It performs two passes:
        - Extract mentions with the $ prefix.
        - Match previously found stocks without $.

    - Text Cleaning: Cleans titles and bodies by removing links, special characters, and converting text to lowercase.

    - Filtering Data: Excludes posts that don't mention any stocks and saves them separately (excluded_wallstreetbets_data.csv).

    - Stock Mention Analysis: Counts and plots the top 20 most mentioned stocks, saving the visualization to results/top_mentioned_stocks.png.

    - Text Analysis:
        - Computes and plots the word count distribution for titles and bodies (results/length_of_titles_and_bodies.png).
        - Generates Word Clouds for titles (results/wordcloud_titles.png) and bodies (results/wordcloud_bodies.png), excluding common stopwords.

    - Output:
        - Cleaned data saved to cleaned_wallstreetbets_data.csv.
        - Visualizations and word clouds are stored in the results folder.


# sentiment_vader.py:
- This script applies VADER (Valence Aware Dictionary and sEntiment Reasoner) for sentiment analysis on the cleaned WallStreetBets dataset.

    - Custom Sentiment Lexicon: Extends the VADER lexicon with WallStreetBets-specific phrases (e.g., "to the moon", "paper hands") and assigns custom sentiment scores to better capture subreddit jargon.
  
    - Data Loading: Reads the cleaned dataset (cleaned_wallstreetbets_data.csv) prepared in the previous step.
  
    - Sentiment Calculation:
        - Computes a combined sentiment score for each post based on the title and body text.
        - Combines title and body sentiment using a weighted average, giving equal importance to both.
  
    - Output:
        - Saves the dataset with sentiment scores to vader_sentiment_wallstreetbets_data.csv for further analysis.
  
    - Visualization:
        - Generates a histogram of sentiment scores, with the mean sentiment score marked for reference.
        - Saves the plot as results/sentiment_scores_vader_with_mean.png.


# sentiment_roberta.py:
- This script uses a RoBERTa-based sentiment analysis model to analyze the sentiment of WallStreetBets posts.

    - Data Loading:
        - Reads the cleaned dataset (cleaned_wallstreetbets_data.csv).
  
    - Model Initialization:
        - Leverages the cardiffnlp/twitter-roberta-base-sentiment model for accurate sentiment predictions.
        - Detects GPU availability for faster processing if applicable.
  
    - Sentiment Analysis:
        - Applies RoBERTa to compute sentiment scores for titles and bodies of posts.
        - Maps RoBERTa sentiment labels:
            - LABEL_2 → Positive (mapped to [0, 1])
            - LABEL_0 → Negative (mapped to [-1, 0])
            - LABEL_1 → Neutral (mapped to 0)
        - Combines title and body sentiment scores, weighted by their respective lengths.
  
    - Output:
        - Saves the sentiment-enhanced dataset to RoBERTa_sentiment_wallstreetbets_data.csv.
  
    - Visualization:
        - Creates a histogram of sentiment scores (ranging from -1 to 1), with the mean sentiment score highlighted.
        - Saves the plot as results/sentiment_scores_RoBERTa.png.


# sentiment_bart_large_zero_shot_classification.py:
- This script applies zero-shot sentiment classification using the BART-based facebook/bart-large-mnli model to analyze WallStreetBets posts.

    - Data Loading:
        - Reads the cleaned dataset (cleaned_wallstreetbets_data.csv).
  
    - Model Initialization:
        - Uses zero-shot classification with candidate labels: "bullish", "neutral", and "bearish".
        - Automatically detects GPU availability for faster inference.
  
    - Sentiment Analysis:
        - Analyzes sentiment for titles and bodies independently.
        - Maps the output labels to sentiment scores:
            - Bullish → Positive sentiment
            - Bearish → Negative sentiment
            - Neutral → No contribution to the score
        - Combines title and body scores, averaging them if both are available.
  
    - Output:
        - Saves the sentiment-enhanced dataset to bart_zero_shot_sentiment_wallstreetbets_data.csv.
  
    - Visualization:
        - Generates a histogram of sentiment scores ranging from -1 to 1, with the mean sentiment score highlighted.
        - Saves the plot as results/sentiment_scores_zero-shot_BART.png.


# roberta_finetuning_final.py:
- This script fine-tunes the RoBERTa model on a labeled dataset of excluded WallStreetBets posts for sentiment classification.

    - Data Preparation:
        - Loads labeled data (excluded_wallstreetbets_data_labels.csv) with sentiment labels: positive, neutral, and negative.
        - Combines the title and body into a single text input for training.
        - Maps sentiment labels to numerical values: positive → 2, neutral → 1, negative → 0.
        - Splits the data into training and evaluation sets (80/20) while maintaining label distribution.
  
    - Tokenization:
        - Tokenizes the text inputs using the roberta-base tokenizer with a maximum length of 128 tokens.
  
    - Fine-Tuning:
        - Uses Hugging Face's Trainer API for fine-tuning.
        - Optimized with:
            - Learning rate: 1e-5
            - Batch size: 4
            - Epochs: 20
        - Performs evaluation and saves checkpoints at the end of each epoch.
  
    - Output:
        - Saves the fine-tuned model and tokenizer to the directory ./roberta-finetuned-sentiment.


# sentiment_roberta_finetuned.py:
- This script applies a fine-tuned RoBERTa model for sentiment classification on the cleaned WallStreetBets dataset.

    - Data Loading:
        - Reads the cleaned dataset (cleaned_wallstreetbets_data.csv).
  
    - Model Initialization:
        - Loads a previously fine-tuned RoBERTa model from roberta-finetuned-sentiment.
        - Uses Hugging Face's pipeline for efficient inference.
        - Automatically detects GPU availability for faster processing.
  
    - Sentiment Analysis:
        - Analyzes sentiment for titles and bodies of posts.
        - Maps the model's labels (positive, neutral, negative) to sentiment scores:
            - Positive → [0, 1]
            - Negative → [-1, 0]
            - Neutral → 0
        - Combines title and body sentiment scores, weighted by text lengths.
  
    - Output:
        - Saves the sentiment-enhanced dataset to roberta_finetuned_sentiment_wallstreetbets_data.csv.
  
    - Visualization:
        - Generates a histogram of sentiment scores, with the mean sentiment score highlighted.
        - Saves the plot as results/sentiment_scores_finetuned_RoBERTa.png.


# stock_allignment_multible_ex.py:
- This script aligns sentiment analysis results with stock market data to explore correlations, visualize trends, and test for causal relationships.

    - Input:
        - Processed sentiment data file (e.g., vader_sentiment_wallstreetbets_data.csv).
        - Sentiment analyzer name for labeling results.
  
    - Steps:
        - Data Loading:
            - Loads Reddit sentiment data and stock price data for multiple tickers (e.g., GME, AMC) using yfinance.
        -Data Aggregation:
            - Aggregates sentiment scores for posts mentioning specific stocks on a daily basis.
            - Aligns Reddit sentiment with daily adjusted closing stock prices.
        - Visualization:
            - Generates rolling average plots (e.g., 2-day window) showing stock price changes and sentiment trends.
                - Output: Rolling average visualizations saved as results/<TICKER>_rolling_2_days_change_stocks+sentiment_<analyzer>.png.
            - Creates daily sentiment vs. stock price change comparison plots.
                - Output: Daily change visualizations saved as results/<TICKER>_daily_change_stocks+sentiment_<analyzer>.png.
        - Causality Analysis:
            - Uses Granger Causality Tests to explore whether sentiment influences stock prices or vice versa.
            - Visualizes F-statistics and p-values for different lags (up to 10 days).
                - Output: Causality plots saved as results/<TICKER>_grangercausalitytests_<analyzer>.png.
    - Output:
        - A combined dataset aligning sentiment and stock data saved to data/combined_stock_reddit_data_<analyzer>.csv.
        - Visualizations illustrating correlations and causality for each stock ticker.
    - Setup:
        - This script is set up to run with the "run.py" script, but can be called with "python stock_allignment_multible_ex.py <sentiment_file_path> <sentiment_analyzer>"



# run.py:
- This script serves as the main pipeline controller for the sentiment analysis and stock alignment workflow. It guides the user through sequential steps to load, clean, analyze, and align the data.

- Pipeline Steps:
    - Data Loading and Cleaning:
        - Checks if the data cleaning step (data_load+cleaning.py) has already been completed.
        - If not, runs the cleaning script to prepare the dataset.
    - Sentiment Analysis:
        - Allows the user to choose one of the following sentiment analysis methods:
            - RoBERTa: sentiment_roberta.py
            - VADER: sentiment_vader.py
            - BART-large: sentiment_bart_large_zero_shot_classification.py
            - Fine-Tuned RoBERTa: sentiment_roberta_finetuned.py
        - Runs the selected sentiment analysis script and prepares the sentiment-enhanced data file.
    - Stock Alignment Analysis:
        - Runs stock_allignment_multible_ex.py to align the sentiment data with stock market trends, analyze correlations, and produce visualization
        - 
  - How It Works:
    - The script uses subprocess to run other Python scripts sequentially.
    - Prompts the user for inputs to:
        - Confirm if data cleaning is needed.
        - Choose the sentiment analysis method.
    - Automatically passes the sentiment data file and analyzer name to the stock alignment script.
