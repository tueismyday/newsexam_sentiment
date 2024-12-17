import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np

# Download VADER's lexicon if not already downloaded
nltk.download('vader_lexicon')

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

custom_keywords = {
    "to the moon": 1.0,
    "diamond hands": 0.8,
    "hold the line": 0.8,
    "hold your ground": 0.8,
    "hold": 0.5,
    "bullish": 0.7,
    "stonks only go up": 1.0,
    "bag secured": 0.5,
    "all in": 0.4,
    "tendies": 1.0,
    "rockets": 0.8,
    "green candles": 0.5,
    "paper hands": -0.7,
    "bearish": -0.7,
    "bag holder": -1.0,
    "rekt": -0.8,
    "dump it": -0.6,
    "red candles": -0.5,
    "short squeeze": -0.3,
    "crash incoming": -1.0,
    "lost it all": -1.0,
    "buy the dip": 0.7,
    "sell high": 0.3,
    "apes together strong": 0.7,
    "stonks": 0.5,
    "we like the stock": 0.5,
    "printing tendies": 1.0,
    "bag secured": 0.5,
    "ath": 0.7,
    "fomo": -0.3,
    "jpow": 0.0,
    "guh": -0.8,
    "bang": 0.0,
    "lets go": 0.8
}

print("\nAdding custom sentences to the Vader-Lexicon.")
# Add each custom keyword to the VADER lexicon
for phrase, score in custom_keywords.items():
    sia.lexicon[phrase] = score

#######################################
# Load Cleaned Dataset
#######################################

file_path = "data/cleaned_wallstreetbets_data.csv"  # Update with the correct path
try:
    print("\nLoading the cleaned dataset.")
    # Attempt to load the dataset
    reddit_df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' does not exist. Please check the file path and try again.")
    exit(1)  # Exit the script with a non-zero status to indicate an error

#######################################
# Compute Sentiment for Each Row
#######################################

def compute_combined_sentiment(title, body):
    """
    Computes a combined sentiment score based on the title and body,
    leveraging the extended VADER lexicon.
    """
    if not isinstance(title, str):  # Handle non-string titles
        title = ""
    if not isinstance(body, str) or body.strip() == "":  # Handle missing or empty body
        body = None

    # Compute sentiment for title
    title_sentiment = sia.polarity_scores(title)['compound']
    
    if body is not None:  # Compute sentiment for body if it exists
        body_sentiment = sia.polarity_scores(body)['compound']

        # Combine title and body sentiment (weighted average)
        combined_sentiment = 0.5 * title_sentiment + 0.5 * body_sentiment
    else:  # Use only title sentiment
        combined_sentiment = title_sentiment

    return combined_sentiment

print("\nInitializing the sentiment analysis.")
# Apply sentiment computation to the cleaned dataset
reddit_df['combined_sentiment'] = reddit_df.apply(
    lambda row: compute_combined_sentiment(row['clean_title'], row['clean_body']), axis=1
)

# Verify the result
print(reddit_df[['clean_title', 'clean_body', 'combined_sentiment']].head())

#######################################
# Save Dataset with Sentiments
#######################################

sentiment_file_path = "data/vader_sentiment_wallstreetbets_data.csv"
reddit_df.to_csv(sentiment_file_path, index=False)
print(f"\nSentiment-enhanced data saved to {sentiment_file_path}.")

#######################################
# Visualize Sentiment Distribution
#######################################

# Calculate the mean of the sentiment scores
sentiment_mean = np.mean(reddit_df['combined_sentiment'])

plt.figure(figsize=(10, 6))
plt.hist(reddit_df['combined_sentiment'], bins=50, alpha=0.75, edgecolor='black')
plt.axvline(sentiment_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {sentiment_mean:.3f}')
plt.title('Sentiment Score Distribution')
plt.xlabel('Sentiment Score (Compound)')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()  # Add legend to display the mean value
# Save the plot as a PNG file
output_path = "results/sentiment_scores_vader_with_mean.png"
plt.savefig(output_path)
print(f"\nA plot of the sentiment scores with mean has been saved to {output_path}.")
plt.close()


print("\nSentiment analysis has been completed.\n")