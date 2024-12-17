from transformers import pipeline
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
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

import torch
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, otherwise CPU

#######################################
# Initialize Zero-Shot Classification Pipeline
#######################################

print("\nInitializing the BART model for zero-shot classification.")
model_name = "facebook/bart-large-mnli"
classifier = pipeline("zero-shot-classification", model=model_name, device=device)

#######################################
# Candidate Labels for Sentiment
#######################################

candidate_labels = ["bullish", "neutral", "bearish"]
assert len(candidate_labels) > 0, "Candidate labels must not be empty."

#######################################
# Define Sentiment Analysis Function
#######################################

def compute_combined_sentiment(title, body):
    """
    Computes a combined sentiment score between -1 and 1 based on the title and body using zero-shot classification.
    """
    # Handle missing inputs
    title = title if isinstance(title, str) and title.strip() else None
    body = body if isinstance(body, str) and body.strip() else None

    # Ensure we have at least one non-empty input
    if not title and not body:
        return 0  # Neutral sentiment if both inputs are empty

    # Analyze title sentiment
    if title:
        title_analysis = classifier(title, candidate_labels)
        title_sentiment_score = map_sentiment_to_score(title_analysis["labels"], title_analysis["scores"])
    else:
        title_sentiment_score = 0

    # Analyze body sentiment
    if body:
        body_analysis = classifier(body, candidate_labels)
        body_sentiment_score = map_sentiment_to_score(body_analysis["labels"], body_analysis["scores"])
    else:
        body_sentiment_score = 0

    # Combine scores with equal weights
    combined_sentiment = (
        title_sentiment_score + body_sentiment_score
    ) / (2 if title and body else 1)  # Average if both exist, else single score

    return combined_sentiment


def map_sentiment_to_score(labels, scores):
    """
    Maps zero-shot classification labels and scores to a normalized sentiment score in the range [-1, 1].
    """
    score_dict = {label: score for label, score in zip(labels, scores)}
    # Positive sentiment is "bullish", negative is "bearish", and neutral is "neutral"
    sentiment_score = score_dict.get("bullish", 0) - score_dict.get("bearish", 0)
    return sentiment_score

#######################################
# Run Sentiment Analysis
#######################################

# Initialize tqdm for pandas
tqdm.pandas()

print("\nRunning sentiment analysis...")
reddit_df['combined_sentiment'] = reddit_df.progress_apply(
    lambda row: compute_combined_sentiment(row.get('clean_title'), row.get('clean_body')), axis=1
)

# Verify results
print("\nSample Results:")
print(reddit_df[['clean_title', 'clean_body', 'combined_sentiment']].head())

#######################################
# Save Results
#######################################

output_file_path = "data/bart_zero_shot_sentiment_wallstreetbets_data.csv"
reddit_df.to_csv(output_file_path, index=False)
print(f"\nSentiment-enhanced data saved to {output_file_path}.")


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
output_path = "results/sentiment_scores_zero-shot_BART.png"
plt.savefig(output_path)
print(f"A plot of the sentiment scores has been saved to {output_path}.")
plt.close()

print("\nSentiment analysis has been completed.\n")