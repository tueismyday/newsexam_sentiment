import pandas as pd
from transformers import pipeline, AutoTokenizer
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np

#######################################
# Load Cleaned Dataset
#######################################

file_path = "data/cleaned_wallstreetbets_data.csv"  # Update with the correct path
try:
    print("\nLoading the cleaned dataset.")
    reddit_df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' does not exist. Please check the file path and try again.")
    exit(1)  # Exit the script with a non-zero status to indicate an error

#######################################
# Check GPU Availability and Set Device
#######################################

device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, otherwise CPU

#######################################
# Initialize RoBERTa for Sentiment Analysis
#######################################

# Initialize tqdm for pandas
tqdm.pandas()

print("\nInitializing the RoBERTa model for sentiment analysis.")
model_name = "cardiffnlp/twitter-roberta-base-sentiment"  # Pretrained RoBERTa sentiment model
sentiment_analyzer = pipeline("sentiment-analysis", model=model_name, device=device)  # Use GPU if available
tokenizer = AutoTokenizer.from_pretrained(model_name)

MAX_LENGTH = 512  # Maximum token length for RoBERTa

def truncate_text(text, max_length=MAX_LENGTH):
    """
    Truncate text to the maximum length using tokenizer truncation.
    """
    if not isinstance(text, str):  # Handle non-string inputs
        return ""
    tokens = tokenizer.encode(text, max_length=max_length, truncation=True, return_tensors="pt")
    return tokenizer.decode(tokens[0], skip_special_tokens=True)

#######################################
# Define compute Sentiment for Each Row
#######################################

def compute_combined_sentiment(title, body):
    """
    Computes a combined sentiment score between -1 and 1 based on the title and body.
    """
    # Handle missing inputs
    title = truncate_text(title) if isinstance(title, str) else ""
    body = truncate_text(body) if isinstance(body, str) and body.strip() else None

    # Analyze title sentiment
    title_analysis = sentiment_analyzer(title)[0]
    title_sentiment_score = map_sentiment_to_score(title_analysis['label'], title_analysis['score'])

    # Analyze body sentiment if available
    if body:
        body_analysis = sentiment_analyzer(body)[0]
        body_sentiment_score = map_sentiment_to_score(body_analysis['label'], body_analysis['score'])

        # Dynamically adjust weights based on text presence
        total_length = len(title) + len(body)
        title_weight = len(title) / total_length
        body_weight = len(body) / total_length
        combined_sentiment = title_weight * title_sentiment_score + body_weight * body_sentiment_score
    else:
        combined_sentiment = title_sentiment_score

    return combined_sentiment

def map_sentiment_to_score(label, score):
    """
    Maps RoBERTa sentiment labels to a normalized sentiment score in the range [-1, 1].
    """
    if label == "LABEL_2":  # Positive sentiment
        return score  # Score is already in [0, 1]
    elif label == "LABEL_0":  # Negative sentiment
        return -score  # Negate the score to map it to [-1, 0]
    elif label == "LABEL_1":  # Neutral sentiment
        return 0  # Neutral sentiment is 0
    else:
        return 0  # Fallback to neutral if label is unexpected

#######################################
# Run Sentiment analysis
#######################################

# Apply sentiment computation with a progress bar
print("\nInitializing sentiment analysis\n")
reddit_df['combined_sentiment'] = reddit_df.progress_apply(
    lambda row: compute_combined_sentiment(row['clean_title'], row['clean_body']), axis=1
)

# Verify the result
print("\nResults:")
print(reddit_df[['clean_title', 'clean_body', 'combined_sentiment']].head())

#######################################
# Save Dataset with Sentiments
#######################################

sentiment_file_path = "data/RoBERTa_sentiment_wallstreetbets_data.csv"
reddit_df.to_csv(sentiment_file_path, index=False)
print(f"\nSentiment-enhanced data saved to {sentiment_file_path}.")

#######################################
# Visualize Sentiment Distribution
#######################################

mean_sentiment = np.mean(reddit_df['combined_sentiment'])
plt.figure(figsize=(10, 6))
plt.hist(reddit_df['combined_sentiment'], bins=50, alpha=0.75, edgecolor='black')
plt.axvline(mean_sentiment, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_sentiment:.2f}')
plt.title('Sentiment Score Distribution')
plt.xlabel('Sentiment Score (-1 to 1)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
# Save the plot as a PNG file
output_path = "results/sentiment_scores_RoBERTa.png"
plt.savefig(output_path)
print(f"A plot of the sentiment scores has been saved to {output_path}.")
plt.close()

print("\nSentiment analysis has been completed.\n")
